@echo off
REM LoRA-native ablation on rank and fit steps (Windows CMD)
REM Usage: double-click or run from repo root

echo ========================================
echo LoRA-native Ablation: rank x fit_steps
echo ========================================
echo Sweeping lora_rank and lora_fit_steps in lora_native mode
echo Results will be grouped under a timestamped folder
echo ========================================
echo.

REM Assume Python environment already activated before running this script

python scripts\run_lora_native_ablation.py ^
    --model gpt2 ^
    --dataset counterfact_sampled_unique_cf_10_20000 ^
    --num_edits 200 ^
    --batch_size 1 ^
    --seed 42 ^
    --ranks 4 8 16 ^
    --fit_steps_list 0 5 10 ^
    --output_root results

echo.
echo Done. See the printed suite directory for this run.
echo.
