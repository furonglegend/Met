#!/bin/bash
# MVP Experiment Matrix - Run all baseline experiments for the project
# Based on todo.md Section 7: 2 methods × 3 batch_sizes = 6 experiments

echo "========================================"
echo "MVP Experiment Matrix"
echo "========================================"
echo "Total: 6 experiments"
echo "- 2 methods: EMMET baseline, EMMET+Replay(0.3)"
echo "- 3 batch sizes: 1, 32, 256"
echo "- Model: GPT-2 (774M)"
echo "- Num edits: 500"
echo "========================================"
echo ""

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate emmet-edit

# Set common parameters
MODEL="gpt2"
NUM_EDITS=500
SEED=42

echo "[1/6] EMMET baseline, batch=1"
python scripts/run_baseline.py --method emmet --model $MODEL --num_edits $NUM_EDITS --batch_size 1 --seed $SEED
if [ $? -ne 0 ]; then
    echo "ERROR: Experiment 1 failed!"
    exit 1
fi
echo ""

echo "[2/6] EMMET baseline, batch=32"
python scripts/run_baseline.py --method emmet --model $MODEL --num_edits $NUM_EDITS --batch_size 32 --seed $SEED
if [ $? -ne 0 ]; then
    echo "ERROR: Experiment 2 failed!"
    exit 1
fi
echo ""

echo "[3/6] EMMET baseline, batch=256"
python scripts/run_baseline.py --method emmet --model $MODEL --num_edits $NUM_EDITS --batch_size 256 --seed $SEED
if [ $? -ne 0 ]; then
    echo "ERROR: Experiment 3 failed!"
    exit 1
fi
echo ""

echo "[4/6] EMMET+Replay(0.3), batch=1"
python scripts/run_baseline.py --method emmet --model $MODEL --num_edits $NUM_EDITS --batch_size 1 --replay_rate 0.3 --seed $SEED
if [ $? -ne 0 ]; then
    echo "ERROR: Experiment 4 failed!"
    exit 1
fi
echo ""

echo "[5/6] EMMET+Replay(0.3), batch=32"
python scripts/run_baseline.py --method emmet --model $MODEL --num_edits $NUM_EDITS --batch_size 32 --replay_rate 0.3 --seed $SEED
if [ $? -ne 0 ]; then
    echo "ERROR: Experiment 5 failed!"
    exit 1
fi
echo ""

echo "[6/6] EMMET+Replay(0.3), batch=256"
python scripts/run_baseline.py --method emmet --model $MODEL --num_edits $NUM_EDITS --batch_size 256 --replay_rate 0.3 --seed $SEED
if [ $? -ne 0 ]; then
    echo "ERROR: Experiment 6 failed!"
    exit 1
fi
echo ""

echo "========================================"
echo "All 6 experiments completed successfully!"
echo "========================================"
echo ""
echo "Results saved to: results/baseline/"
echo ""
echo "Next steps:"
echo "1. Run: python scripts/analyze_results.py"
echo "2. Check aggregated results in: results/baseline/aggregated_results.csv"
echo ""
