@echo off
REM Combined Experiments: EMMET + Replay + LoRA (Phase 3.2)
REM Test all combinations to find optimal configuration

echo ========================================
echo Combined Configuration Experiments
echo ========================================
echo Testing: EMMET, Replay, LoRA, and combinations
echo ========================================
echo.

call conda activate emmet-edit

set MODEL=gpt2
set NUM_EDITS=200
set BATCH_SIZE=16
set SEED=42

echo [1/7] EMMET baseline (no enhancements)...
python scripts\run_baseline.py ^
    --method emmet --model %MODEL% ^
    --num_edits %NUM_EDITS% --batch_size %BATCH_SIZE% ^
    --seed %SEED% --output_dir results\combined

echo [2/7] EMMET + Replay (rate=0.3)...
python scripts\run_baseline.py ^
    --method emmet --model %MODEL% ^
    --num_edits %NUM_EDITS% --batch_size %BATCH_SIZE% ^
    --seed %SEED% --replay_rate 0.3 ^
    --output_dir results\combined

echo [3/7] EMMET + LoRA (rank=8)...
python scripts\run_baseline.py ^
    --method emmet --model %MODEL% ^
    --num_edits %NUM_EDITS% --batch_size %BATCH_SIZE% ^
    --seed %SEED% --use_lora --lora_rank 8 ^
    --output_dir results\combined

echo [4/7] EMMET + Replay + LoRA (rank=8)...
python scripts\run_baseline.py ^
    --method emmet --model %MODEL% ^
    --num_edits %NUM_EDITS% --batch_size %BATCH_SIZE% ^
    --seed %SEED% --replay_rate 0.3 ^
    --use_lora --lora_rank 8 ^
    --output_dir results\combined

echo [5/7] EMMET + Replay (rate=0.5) + LoRA (rank=4)...
python scripts\run_baseline.py ^
    --method emmet --model %MODEL% ^
    --num_edits %NUM_EDITS% --batch_size %BATCH_SIZE% ^
    --seed %SEED% --replay_rate 0.5 ^
    --use_lora --lora_rank 4 ^
    --output_dir results\combined

echo [6/7] EMMET + Replay (rate=0.3) + LoRA (rank=16)...
python scripts\run_baseline.py ^
    --method emmet --model %MODEL% ^
    --num_edits %NUM_EDITS% --batch_size %BATCH_SIZE% ^
    --seed %SEED% --replay_rate 0.3 ^
    --use_lora --lora_rank 16 ^
    --output_dir results\combined

echo [7/7] EMMET + Replay (rate=0.1) + LoRA (rank=8)...
python scripts\run_baseline.py ^
    --method emmet --model %MODEL% ^
    --num_edits %NUM_EDITS% --batch_size %BATCH_SIZE% ^
    --seed %SEED% --replay_rate 0.1 ^
    --use_lora --lora_rank 8 ^
    --output_dir results\combined

echo.
echo ========================================
echo All combined experiments completed!
echo ========================================
echo Results: results\combined\
echo.
echo Analyze with:
echo   python scripts\analyze_results.py --results_dir results\combined
echo.

pause
