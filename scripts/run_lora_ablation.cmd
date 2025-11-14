@echo off
REM LoRA Ablation Experiments (Phase 3.2)
REM Compare EMMET vs EMMET+LoRA with different ranks

echo ========================================
echo LoRA Ablation Experiments
echo ========================================
echo Testing LoRA impact on EMMET editing
echo Configurations:
echo - EMMET baseline (no LoRA)
echo - EMMET + LoRA rank=4
echo - EMMET + LoRA rank=8
echo - EMMET + LoRA rank=16
echo ========================================
echo.

REM Activate conda environment
call conda activate emmet-edit

REM Set common parameters
set MODEL=gpt2
set NUM_EDITS=100
set BATCH_SIZE=10
set SEED=42

echo [1/4] Running EMMET baseline (no LoRA)...
python scripts\run_baseline.py ^
    --method emmet ^
    --model %MODEL% ^
    --num_edits %NUM_EDITS% ^
    --batch_size %BATCH_SIZE% ^
    --seed %SEED% ^
    --output_dir results\lora_ablation

if errorlevel 1 (
    echo ERROR: EMMET baseline failed
    exit /b 1
)
echo.

echo [2/4] Running EMMET + LoRA rank=4...
python scripts\run_baseline.py ^
    --method emmet ^
    --model %MODEL% ^
    --num_edits %NUM_EDITS% ^
    --batch_size %BATCH_SIZE% ^
    --seed %SEED% ^
    --use_lora ^
    --lora_rank 4 ^
    --lora_alpha 8 ^
    --output_dir results\lora_ablation

if errorlevel 1 (
    echo ERROR: LoRA rank=4 experiment failed
    exit /b 1
)
echo.

echo [3/4] Running EMMET + LoRA rank=8...
python scripts\run_baseline.py ^
    --method emmet ^
    --model %MODEL% ^
    --num_edits %NUM_EDITS% ^
    --batch_size %BATCH_SIZE% ^
    --seed %SEED% ^
    --use_lora ^
    --lora_rank 8 ^
    --lora_alpha 16 ^
    --output_dir results\lora_ablation

if errorlevel 1 (
    echo ERROR: LoRA rank=8 experiment failed
    exit /b 1
)
echo.

echo [4/4] Running EMMET + LoRA rank=16...
python scripts\run_baseline.py ^
    --method emmet ^
    --model %MODEL% ^
    --num_edits %NUM_EDITS% ^
    --batch_size %BATCH_SIZE% ^
    --seed %SEED% ^
    --use_lora ^
    --lora_rank 16 ^
    --lora_alpha 32 ^
    --output_dir results\lora_ablation

if errorlevel 1 (
    echo ERROR: LoRA rank=16 experiment failed
    exit /b 1
)
echo.

echo ========================================
echo All LoRA ablation experiments completed!
echo ========================================
echo.
echo Results saved to: results\lora_ablation\
echo.
echo To analyze results:
echo   python scripts\analyze_results.py --results_dir results\lora_ablation
echo.

pause
