@echo off
REM Replay ablation for EMMET (Windows CMD)
REM Varies replay_rate, strategy, and buffer size optionally.

echo ========================================
echo Replay Ablation: EMMET + Replay
echo ========================================
echo Sweeping replay_rate, strategy, and buffer size
echo Results will be grouped under a timestamped folder
echo ========================================
echo.

REM Assume Python environment already activated before running this script

python scripts\run_replay_ablation.py ^
    --model gpt2 ^
    --dataset counterfact_sampled_unique_cf_10_20000 ^
    --num_edits 200 ^
    --batch_size 1 ^
    --seed 42 ^
    --rates 0 0.1 0.3 0.5 ^
    --strategies random priority recent ^
    --buffer_sizes 100 200 ^
    --output_root results

echo.
echo Done. See the printed suite directory for this run.
echo.
