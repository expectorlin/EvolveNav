# set mp3d path
# export PYTHONPATH=Matterport3DSimulator/build:$PYTHONPATH

# set java path
# export JAVA_HOME=$java_path
# export PATH=$JAVA_HOME/bin:$PATH
# export CLASSPATH=.:$JAVA_HOME/lib/dt.jar:$JAVA_HOME/lib/tools.jar

# activate environment
# conda activate navillm

torchrun --nnodes=1 --nproc_per_node=4 --master_port 41000 train_wlora_new.py \
    --stage finetune --mode test --data_dir data --cfg_file configs/r2r.yaml \
    --pretrained_model_name_or_path data/models/Vicuna-7B --precision amp_bf16 \
    --resume_from_checkpoint $stage_2_model_path \
    --test_datasets R2R \
    --enable_lora \
    --lora_r 128 --lora_alpha 256 --lora_dropout 0.05 \
    --lora_target_modules "all_linear" \
    --lora_bias "none" \
    --batch_size 2 --output_dir build/eval_r2r --validation_split test --save_pred_results
