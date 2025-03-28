@echo off
:: Train RoBERTa model with triplet loss and save in safetensors format
:: Optimized parameters for maximum performance

echo ==================================
echo Starting RoBERTa training with optimized parameters
echo ==================================

python run_triplet_pretraining.py ^
    --model_name FacebookAI/roberta-base ^
    --data_dir .\data ^
    --split_type random ^
    --max_length 512 ^
    --pretrain_batch_size 16 ^
    --pretrain_epochs 8 ^
    --pretrain_lr 1e-5 ^
    --pretrain_grad_accum 2 ^
    --finetune_batch_size 32 ^
    --finetune_epochs 5 ^
    --finetune_lr 2e-5 ^
    --finetune_grad_accum 1 ^
    --warmup_steps 500 ^
    --output_dir .\results ^
    --model_save_dir .\facebook-roberta-bias-detector-tlp ^
    --pretrained_weights_path .\pretrained_weights.safetensors ^
    --seed 42

echo Training completed!
pause