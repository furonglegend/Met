@echo off
REM Run all baseline experiments (ROME/MEMIT/EMMET)

echo ========================================
echo Running All Baseline Experiments
echo ========================================

REM Activate conda environment
call conda activate emmet-replay

REM Run batch experiments
python scripts\run_batch_experiments.py ^
    --methods rome memit emmet ^
    --models gpt2-xl ^
    --num_edits 100 500 ^
    --seeds 42 43

echo.
echo All experiments completed!
echo.
echo Analyzing results...
python scripts\analyze_results.py --results_dir results\baseline

echo.
echo Done! Check results directory for outputs.
pause
