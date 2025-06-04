# set mp3d path
# export PYTHONPATH=Matterport3DSimulator/build:$PYTHONPATH

# set java path
# export JAVA_HOME=$java_path
# export PATH=$JAVA_HOME/bin:$PATH
# export CLASSPATH=.:$JAVA_HOME/lib/dt.jar:$JAVA_HOME/lib/tools.jar

# activate environment
# conda activate navillm

# training for 30 epochs
torchrun --nnodes=1 --nproc_per_node=4 --master_port 41000 train_wlora_new.py \
    --stage finetune --cfg_file configs/cvdn.yaml \
    --data_dir data --pretrained_model_name_or_path data/models/Vicuna-7B --precision amp_bf16 \
    --batch_size 1 --gradient_accumulation_step 16 --num_steps_per_epoch 300 --lr 3e-5 --seed 0 --num_epochs 60 \
    --test_datasets CVDN \
    --enable_lora \
    --lora_r 128 --lora_alpha 256 --lora_dropout 0.05 \
    --lora_target_modules "all_linear" \
    --lora_bias "none" \
    --Stage_2_training \
    --max_saved_checkpoints 1 --output_dir output/cvdn_stage_2_training