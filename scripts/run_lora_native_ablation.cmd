@echo off
REM LoRA-native ablation on rank and fit steps (Windows CMD)
REM Usage: double-click or run from repo root

set MODEL=gpt2
set METHOD=emmet
set DATASET=counterfact_sampled_unique_cf_10_20000
set NUM_EDITS=200
set BATCH_SIZE=1
set SEED=42

for %%R in (4 8 16) do (
  for %%S in (0 5 10) do (
    echo Running rank=%%R fit_steps=%%S...
    python scripts\run_baseline.py ^
      --method %METHOD% ^
      --model %MODEL% ^
      --num_edits %NUM_EDITS% ^
      --batch_size %BATCH_SIZE% ^
      --seed %SEED% ^
      --dataset %DATASET% ^
      --edit_mode lora_native ^
      --lora_rank %%R ^
      --lora_alpha %%R ^
      --lora_scale 1.0 ^
      --lora_fit_steps %%S ^
      --lora_use_svd
  )
)

echo Done.
