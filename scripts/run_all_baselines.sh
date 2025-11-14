#!/bin/bash
# Baseline Comparison Experiment - ROME vs MEMIT vs EMMET
# Based on TODO.md Section 1.2: Three baseline methods comparison
# Objective: Prove the necessity of unified framework and EMMET's advantages

echo "========================================"
echo "Baseline Comparison Experiment"
echo "========================================"
echo "Total: 3 experiments"
echo "- ROME: Single edit (batch_size=1)"
echo "- MEMIT: Batch edit (batch_size=32)"
echo "- EMMET: Batch edit (batch_size=32)"
echo "Model: GPT-2 XL (1.5B)"
echo "Num edits: 200"
echo "Dataset: CounterFact"
echo "========================================"
echo ""

# Activate conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate emmet-edit

# Set common parameters
MODEL="gpt2-xl"
NUM_EDITS=200
SEED=42
DATASET="counterfact_sampled_unique_cf_10_20000"
OUTPUT_DIR="results/baseline_comparison"

echo "Creating output directory..."
mkdir -p "$OUTPUT_DIR"
echo ""

echo "========================================"
echo "[1/3] Running ROME (batch_size=1)"
echo "========================================"
echo "ROME uses single-edit constraint optimization"
echo "Expected time: ~5-10 minutes"
echo ""
python scripts/run_baseline.py --method rome --model "$MODEL" --num_edits "$NUM_EDITS" --batch_size 1 --seed "$SEED" --dataset "$DATASET" --output_dir "$OUTPUT_DIR"
if [ $? -ne 0 ]; then
    echo "ERROR: ROME experiment failed!"
    echo "Check logs for details."
    exit 1
fi
echo ""
echo "✓ ROME experiment completed"
echo ""

echo "========================================"
echo "[2/3] Running MEMIT (batch_size=32)"
echo "========================================"
echo "MEMIT uses least-squares relaxation for batch editing"
echo "Expected time: ~3-5 minutes"
echo ""
python scripts/run_baseline.py --method memit --model "$MODEL" --num_edits "$NUM_EDITS" --batch_size 32 --seed "$SEED" --dataset "$DATASET" --output_dir "$OUTPUT_DIR"
if [ $? -ne 0 ]; then
    echo "ERROR: MEMIT experiment failed!"
    echo "Check logs for details."
    exit 1
fi
echo ""
echo "✓ MEMIT experiment completed"
echo ""

echo "========================================"
echo "[3/3] Running EMMET (batch_size=32)"
echo "========================================"
echo "EMMET uses unified constraint optimization framework"
echo "Expected time: ~3-5 minutes"
echo ""
python scripts/run_baseline.py --method emmet --model "$MODEL" --num_edits "$NUM_EDITS" --batch_size 32 --seed "$SEED" --dataset "$DATASET" --output_dir "$OUTPUT_DIR"
if [ $? -ne 0 ]; then
    echo "ERROR: EMMET experiment failed!"
    echo "Check logs for details."
    exit 1
fi
echo ""
echo "✓ EMMET experiment completed"
echo ""

echo "========================================"
echo "All baseline experiments completed!"
echo "========================================"
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Summary:"
echo "- ROME:  Single-edit constraint (slowest, most precise)"
echo "- MEMIT: Batch least-squares (faster, approximate)"
echo "- EMMET: Unified optimization (balanced, flexible)"
echo ""
echo "Next steps:"
echo "1. Compare metrics: ES, PS, NS, Composite Score"
echo "2. Analyze time and memory usage"
echo "3. Check detailed results in each experiment folder"
echo ""
echo "To aggregate results, run:"
echo "  python scripts/analyze_results.py --results_dir $OUTPUT_DIR"
echo ""
