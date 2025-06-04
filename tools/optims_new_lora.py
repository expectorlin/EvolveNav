import os
import torch
import glob
from transformers import get_constant_schedule_with_warmup
from peft import LoraConfig, get_peft_model


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    if 'embed_tokens' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('embed_tokens')
    if 'token_type_embeddings' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('token_type_embeddings')
    if 'gmap_pos_embeddings' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('gmap_pos_embeddings')
    if 'vp_pos_embeddings' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('vp_pos_embeddings')
    if 'og_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('og_head')
    if 'out_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('out_head')
    return list(lora_module_names)


def check_checkpoint(args, model, optimizer, lr_scheduler, logger) -> int:
    resume_from_epoch = 0
    if args.resume_from_checkpoint is not None:
        if args.rank == 0:
            logger.info(f"Loading checkpoint from {args.resume_from_checkpoint}")
        checkpoint = torch.load(args.resume_from_checkpoint, map_location="cpu")
        model_state_dict = model.state_dict()
        state_disk = {k.replace('module.', ''): v for k, v in checkpoint['model_state_dict'].items()}
        update_model_state = {}
        for key, val in state_disk.items():
            if key in model_state_dict and model_state_dict[key].shape == val.shape:
                update_model_state[key] = val
            else:
                logger.info(
                    'Ignore weight %s: %s' % (key, str(val.shape))
                )
        msg = model.load_state_dict(update_model_state, strict=False)
        logger.info(msg)

        if 'epoch' in checkpoint:
            resume_from_epoch = checkpoint['epoch'] + 1
            logger.info("Resume from Epoch {}".format(resume_from_epoch))
            optimizer.load_state_dict(checkpoint['optimizer'])


    return resume_from_epoch


def dist_models(args, model, logger):
    logger.info("*************** init model *************** ")
    # args.rank: global rank.
    total_gpus = torch.cuda.device_count()
    device_id = args.rank % total_gpus


    if (args.resume_from_checkpoint is not None and "lora" not in args.resume_from_checkpoint) or args.base_model is not None:
        if args.rank == 0:
            logger.info(f"Loading checkpoint from {args.resume_from_checkpoint}")
        checkpoint = torch.load(args.resume_from_checkpoint, map_location="cpu")
        model_state_dict = model.state_dict()
        state_disk = {k.replace('module.', ''): v for k, v in checkpoint['model_state_dict'].items()}
        update_model_state = {}
        for key, val in state_disk.items():
            if key in model_state_dict and model_state_dict[key].shape == val.shape:
                update_model_state[key] = val
            else:
                logger.info(
                    'Ignore weight %s: %s' % (key, str(val.shape))
                )
        msg = model.load_state_dict(update_model_state, strict=False)
        logger.info(msg)


    if args.enable_lora:

        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=find_all_linear_names(model.lang_model) if args.lora_target_modules != "none" else None,
            lora_dropout=args.lora_dropout,
            bias=args.lora_bias,
        )
        model.lang_model.resize_token_embeddings(len(model.lang_model.tokenizer))

        model.lang_model = get_peft_model(model.lang_model, lora_config)

    optimizer = torch.optim.AdamW([p for n, p in model.named_parameters() if p.requires_grad], lr=args.lr)

    lr_scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=args.num_warmup_steps)

    model.to(device_id)

    if args.resume_from_checkpoint is not None:
        if "lora" in args.resume_from_checkpoint:
            resume_from_epoch = check_checkpoint(
                args, model, optimizer, lr_scheduler, logger,
            )
        else:
            resume_from_epoch = 0
    else:
        resume_from_epoch = 0

    param_sums = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("model initialized with {:.2f} M trainable parameters".format(param_sums/1000**2))


    if args.distributed:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[device_id], find_unused_parameters=True)

        # args.batch_size: BATCH_SIZE_PER_GPU
        logger.info('Training in distributed mode : total_batch_size: %d' % (total_gpus * args.batch_size))
    else:
        total_gpus = 1
        logger.info('Training with a single process')

    return model, optimizer, resume_from_epoch, lr_scheduler


def save_checkpoint(model, model_path, optimizer=None, epoch: int=0, save_states: bool=False):
    if hasattr(model, 'module'):
        model = model.module
    
    state_dict = {
        "model_state_dict": model.state_dict()
    }
    if save_states:
        state_dict.update({
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
        })

    torch.save(state_dict, model_path)