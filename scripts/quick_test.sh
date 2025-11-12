#!/bin/bash
# Quick Test Script - Run a minimal experiment for testing

echo "========================================"
echo "Quick Test - Baseline Experiment"
echo "========================================"

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate emmet-replay

# Run single small experiment
python scripts/run_baseline.py \
    --method emmet \
    --model gpt2-xl \
    --num_edits 10 \
    --seed 42

echo ""
echo "Test completed!"
