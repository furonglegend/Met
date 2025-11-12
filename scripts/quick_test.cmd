@echo off
REM Quick Test Script - Run a minimal experiment for testing

echo ========================================
echo Quick Test - Baseline Experiment
echo ========================================

REM Activate conda environment
call conda activate emmet-replay

REM Run single small experiment
python scripts\run_baseline.py ^
    --method emmet ^
    --model gpt2-xl ^
    --num_edits 10 ^
    --seed 42

echo.
echo Test completed!
pause
