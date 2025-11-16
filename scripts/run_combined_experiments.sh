#!/usr/bin/env bash
# Combined Experiments: EMMET + Replay + LoRA (Phase 3.2)
# Linux version

set -e

echo "========================================"
echo "Combined Configuration Experiments"
echo "========================================"
echo "Testing: EMMET, Replay, LoRA, and combinations"
echo "========================================"
echo

# Assume conda/venv is already activated before running this script

# Delegate to the Python orchestrator, which will create a
# timestamped suite folder under results/ and run all configs.
python scripts/run_combined_experiments.py \
  --model gpt2-xl \
  --num_edits 200 \
  --batch_size 16 \
  --seed 42 \
  --output_root results

echo
echo "Done. See the printed suite directory for this run."
echo
