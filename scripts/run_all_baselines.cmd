@echo off
REM Baseline Comparison Experiment (TODO 1.2) - ROME vs MEMIT
REM Objective: Compare ROME and MEMIT to validate the unified framework

echo ========================================
echo Baseline Comparison: ROME vs MEMIT
echo ========================================
echo Running ROME, MEMIT, and EMMET baselines
echo Results will be grouped under a timestamped folder
echo ========================================
echo.

REM Assume Python environment already activated before running this script

REM Delegate to the Python orchestrator, which will create an
REM all_baselines_* suite folder under results/ and run all configs.
python scripts\run_all_baselines.py ^
    --model gpt2-xl ^
    --num_edits 500 ^
    --dataset counterfact_500 ^
    --seed 42 ^
    --output_root results

echo.
echo Done. See the printed suite directory for this run.
echo.

pause
