#!/bin/bash
# LoRA Ablation Experiments (Phase 3.2)
# Compare EMMET vs EMMET+LoRA with different ranks

echo "========================================"
echo "LoRA Ablation Experiments"
echo "========================================"
echo "Testing LoRA impact on EMMET editing"
echo "Configurations:"
echo "- EMMET baseline (no LoRA)"
echo "- EMMET + LoRA rank=4"
echo "- EMMET + LoRA rank=8"
echo "- EMMET + LoRA rank=16"
echo "========================================"
echo ""

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate emmet-edit

# Set common parameters
MODEL="gpt2"
NUM_EDITS=100
BATCH_SIZE=10
SEED=42

echo "[1/4] Running EMMET baseline (no LoRA)..."
python scripts/run_baseline.py \
    --method emmet \
    --model $MODEL \
    --num_edits $NUM_EDITS \
    --batch_size $BATCH_SIZE \
    --seed $SEED \
    --output_dir results/lora_ablation

if [ $? -ne 0 ]; then
    echo "ERROR: EMMET baseline failed"
    exit 1
fi
echo ""

echo "[2/4] Running EMMET + LoRA rank=4..."
python scripts/run_baseline.py \
    --method emmet \
    --model $MODEL \
    --num_edits $NUM_EDITS \
    --batch_size $BATCH_SIZE \
    --seed $SEED \
    --use_lora \
    --lora_rank 4 \
    --lora_alpha 8 \
    --output_dir results/lora_ablation

if [ $? -ne 0 ]; then
    echo "ERROR: LoRA rank=4 experiment failed"
    exit 1
fi
echo ""

echo "[3/4] Running EMMET + LoRA rank=8..."
python scripts/run_baseline.py \
    --method emmet \
    --model $MODEL \
    --num_edits $NUM_EDITS \
    --batch_size $BATCH_SIZE \
    --seed $SEED \
    --use_lora \
    --lora_rank 8 \
    --lora_alpha 16 \
    --output_dir results/lora_ablation

if [ $? -ne 0 ]; then
    echo "ERROR: LoRA rank=8 experiment failed"
    exit 1
fi
echo ""

echo "[4/4] Running EMMET + LoRA rank=16..."
python scripts/run_baseline.py \
    --method emmet \
    --model $MODEL \
    --num_edits $NUM_EDITS \
    --batch_size $BATCH_SIZE \
    --seed $SEED \
    --use_lora \
    --lora_rank 16 \
    --lora_alpha 32 \
    --output_dir results/lora_ablation

if [ $? -ne 0 ]; then
    echo "ERROR: LoRA rank=16 experiment failed"
    exit 1
fi
echo ""

echo "========================================"
echo "All LoRA ablation experiments completed!"
echo "========================================"
echo ""
echo "Results saved to: results/lora_ablation/"
echo ""
echo "To analyze results:"
echo "  python scripts/analyze_results.py --results_dir results/lora_ablation"
echo ""
