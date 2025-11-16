@echo off
REM EMMET vs LoRA-native comparison (Windows CMD)
REM Usage: run from repo root after activating your Python environment

echo ========================================
echo EMMET vs LoRA-native comparison
echo ========================================
echo Running EMMET(raw) and EMMET+LoRA-native on a small dataset
echo Results will be grouped under a timestamped folder
echo ========================================
echo.

REM Delegate to the Python orchestrator, which will create an
REM emmet_lora_compare_* suite folder under results/ and run both configs.
python scripts\run_emmet_lora_compare.py ^
    --model gpt2 ^
    --dataset counterfact_500 ^
    --num_edits 200 ^
    --batch_size 16 ^
    --seed 42 ^
    --lora_rank 8 ^
    --lora_fit_steps 0 ^
    --output_root results

echo.
echo Done. See the printed suite directory for this run.
echo.
