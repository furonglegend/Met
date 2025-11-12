#!/bin/bash
# Quick Test Script - Run a minimal experiment for testing

echo "========================================"
echo "Quick Test - EMMET Baseline (10 samples)"
echo "========================================"

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate emmet-edit

# Run quick test using test_baseline.py
echo "Running quick test with GPT-2 and 10 samples..."
python scripts/test_baseline.py

echo ""
echo "Test completed!"
echo "Check results/test/ for outputs"
