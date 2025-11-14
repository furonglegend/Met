@echo off
REM Baseline Comparison Experiment (TODO 1.2) - ROME vs MEMIT vs EMMET
REM Objective: Demonstrate the necessity and superiority of the unified framework

echo ========================================
echo Baseline Comparison: ROME vs MEMIT vs EMMET
echo ========================================
echo Total: 3 experiments
echo - ROME: single edit (batch_size=1)
echo - MEMIT: batch edit (batch_size=32)
echo - EMMET: batch edit (batch_size=32)
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
set OUTPUT_DIR=results/baseline_comparison

echo [1/3] ROME: Single edit (batch_size=1)
echo Testing traditional single-edit approach...
python scripts\run_baseline.py --method rome --model %MODEL% --num_edits %NUM_EDITS% --batch_size 1 --seed %SEED% --output_dir %OUTPUT_DIR%
if errorlevel 1 (
    echo ERROR: ROME experiment failed!
    pause
    exit /b 1
)
echo.

echo [2/3] MEMIT: Batch edit (batch_size=32)
echo Testing batch editing with relaxed constraints...
python scripts\run_baseline.py --method memit --model %MODEL% --num_edits %NUM_EDITS% --batch_size 32 --seed %SEED% --output_dir %OUTPUT_DIR%
if errorlevel 1 (
    echo ERROR: MEMIT experiment failed!
    pause
    exit /b 1
)
echo.

echo [3/3] EMMET: Batch edit (batch_size=32)
echo Testing unified framework with closed-form solution...
python scripts\run_baseline.py --method emmet --model %MODEL% --num_edits %NUM_EDITS% --batch_size 32 --seed %SEED% --output_dir %OUTPUT_DIR%
if errorlevel 1 (
    echo ERROR: EMMET experiment failed!
    pause
    exit /b 1
)
echo.

echo ========================================
echo All 3 baseline experiments completed!
echo ========================================
echo.
echo Results saved to: %OUTPUT_DIR%\
echo.
echo Next steps:
echo 1. Compare ES/PS/NS metrics across methods
echo 2. Check time and memory overhead
echo 3. Run: python scripts\analyze_results.py --results_dir %OUTPUT_DIR%
echo 4. Generate baseline_comparison.csv
echo.
echo Key evaluation points (from TODO 1.2):
echo - Prove unified framework necessity
echo - Compare EMMET advantages over ROME/MEMIT
echo - Use same dataset and random seed
echo - Record intermediate edit states
echo.
pause
