@echo off
REM Replay ablation for EMMET (Windows CMD)
REM Varies replay_rate, strategy, and buffer size optionally.

set MODEL=gpt2
set METHOD=emmet
set DATASET=counterfact_sampled_unique_cf_10_20000
set NUM_EDITS=200
set BATCH_SIZE=1
set SEED=42

REM Replay rates grid
set RATES=0 0.1 0.3 0.5
REM Strategies to compare (keep "random" only if you want fewer runs)
set STRATS=random priority recent
REM Buffer sizes (comment out others to reduce runs)
set BUFS=100 200

for %%S in (%STRATS%) do (
  for %%B in (%BUFS%) do (
    for %%R in (%RATES%) do (
      echo Running replay_rate=%%R strategy=%%S buffer=%%B...
      python scripts\run_baseline.py ^
        --method %METHOD% ^
        --model %MODEL% ^
        --num_edits %NUM_EDITS% ^
        --batch_size %BATCH_SIZE% ^
        --seed %SEED% ^
        --dataset %DATASET% ^
        --replay_rate %%R ^
        --replay_strategy %%S ^
        --replay_buffer_size %%B ^
        --replay_weight 1.0
    )
  )
)

echo Done.
