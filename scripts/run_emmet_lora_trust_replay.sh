#!/usr/bin/env bash
# EMMET + LoRA-native + Trust + Replay (Linux)

set -e

echo "========================================"
echo "EMMET + LoRA-native + Trust + Replay"
echo "========================================"
echo "Running baseline, replay, replay+LoRA, and replay+LoRA+Trust"
echo "Results will be grouped under a timestamped folder"
echo "========================================"
echo

# Assume conda/venv is already activated before running this script

# Delegate to the Python orchestrator, which will create a
# emmet_lora_trust_replay_* suite folder under results/ and run all configs.
python scripts/run_emmet_lora_trust_replay.py \
  --model gpt2 \
  --dataset counterfact_500 \
  --num_edits 200 \
  --batch_size 16 \
  --seed 42 \
  --output_root results

echo
echo "Done. See the printed suite directory, CSV, and combined_scores.png for this run."
echo
