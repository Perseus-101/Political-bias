@echo off
:: Train RoBERTa baseline model for political bias detection
:: Optimized parameters for maximum performance

echo =============================================
echo Starting RoBERTa baseline training with optimized parameters
echo =============================================

python main.py ^
    --model_name roberta-base ^
    --model_type baseline ^
    --data_dir .\data ^
    --split_type random ^
    --max_length 512 ^
    --batch_size 16 ^
    --gradient_accumulation_steps 2 ^
    --learning_rate 2e-5 ^
    --num_epochs 7 ^
    --warmup_steps 200 ^
    --output_dir .\results ^
    --seed 42

echo =============================================
echo Baseline training complete! Results saved to .\results\consolidated_results.csv
echo =============================================

pause