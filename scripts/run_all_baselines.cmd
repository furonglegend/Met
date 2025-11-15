@echo off
REM Baseline Comparison Experiment (TODO 1.2) - ROME vs MEMIT
REM Objective: Compare ROME and MEMIT to validate the unified framework

echo ========================================
echo Baseline Comparison: ROME vs MEMIT
echo ========================================
echo Total: 2 experiments
echo - ROME: single edit (batch_size=1)
echo - MEMIT: batch edit (batch_size=32)
echo - Model: GPT-2 XL
echo - Num edits: 200
echo - Seed: 42
echo ========================================
echo.

REM Activate conda environment
call conda activate emmet-edit

REM Set common parameters
set MODEL=gpt2-xl
set NUM_EDITS=200
set SEED=42
set OUTPUT_DIR=results/baseline_comparison_rome_memit

echo [1/2] ROME: Single edit (batch_size=1)
echo Testing traditional single-edit approach...
python scripts\run_baseline.py --method rome --model %MODEL% --num_edits %NUM_EDITS% --batch_size 1 --seed %SEED% --output_dir %OUTPUT_DIR%
if errorlevel 1 (
    echo ERROR: ROME experiment failed!
    pause
    exit /b 1
)
echo.

echo [2/2] MEMIT: Batch edit (batch_size=32)
echo Testing batch editing with relaxed constraints...
python scripts\run_baseline.py --method memit --model %MODEL% --num_edits %NUM_EDITS% --batch_size 32 --seed %SEED% --output_dir %OUTPUT_DIR%
if errorlevel 1 (
    echo ERROR: MEMIT experiment failed!
    pause
    exit /b 1
)
echo.

echo ========================================
echo All baseline experiments completed!
echo ========================================
echo.
echo Results saved to: %OUTPUT_DIR%\
echo.
echo 2. Running automatic analysis and visualization...
python scripts\analyze_results.py --results_dir %OUTPUT_DIR% --output %OUTPUT_DIR%\baseline_comparison_rome_memit.csv
if errorlevel 1 (
    echo ERROR: Analysis failed!
    echo You can run: python scripts\analyze_results.py --results_dir %OUTPUT_DIR%
)
echo.
echo Aggregated results saved to: %OUTPUT_DIR%\baseline_comparison_rome_memit.csv (if analysis succeeded)
echo.
echo Key evaluation points (from TODO 1.2):
echo - Prove unified framework necessity
echo - Compare ROME and MEMIT performance
echo - Use same dataset and random seed
echo - Record intermediate edit states
echo.
pause
