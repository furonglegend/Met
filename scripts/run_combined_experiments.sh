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

MODEL="gpt2"
NUM_EDITS=200
BATCH_SIZE=16
SEED=42
OUTPUT_DIR="results/combined"

mkdir -p "$OUTPUT_DIR"

echo "[1/7] EMMET baseline (no enhancements)..."
python scripts/run_baseline.py \
  --method emmet --model "$MODEL" \
  --num_edits "$NUM_EDITS" --batch_size "$BATCH_SIZE" \
  --seed "$SEED" --output_dir "$OUTPUT_DIR"

echo "[2/7] EMMET + Replay (rate=0.3)..."
python scripts/run_baseline.py \
  --method emmet --model "$MODEL" \
  --num_edits "$NUM_EDITS" --batch_size "$BATCH_SIZE" \
  --seed "$SEED" --replay_rate 0.3 \
  --output_dir "$OUTPUT_DIR"

echo "[3/7] EMMET + LoRA (rank=8)..."
python scripts/run_baseline.py \
  --method emmet --model "$MODEL" \
  --num_edits "$NUM_EDITS" --batch_size "$BATCH_SIZE" \
  --seed "$SEED" --use_lora --lora_rank 8 \
  --output_dir "$OUTPUT_DIR"

echo "[4/7] EMMET + Replay + LoRA (rank=8)..."
python scripts/run_baseline.py \
  --method emmet --model "$MODEL" \
  --num_edits "$NUM_EDITS" --batch_size "$BATCH_SIZE" \
  --seed "$SEED" --replay_rate 0.3 \
  --use_lora --lora_rank 8 \
  --output_dir "$OUTPUT_DIR"

echo "[5/7] EMMET + Replay (rate=0.5) + LoRA (rank=4)..."
python scripts/run_baseline.py \
  --method emmet --model "$MODEL" \
  --num_edits "$NUM_EDITS" --batch_size "$BATCH_SIZE" \
  --seed "$SEED" --replay_rate 0.5 \
  --use_lora --lora_rank 4 \
  --output_dir "$OUTPUT_DIR"

echo "[6/7] EMMET + Replay (rate=0.3) + LoRA (rank=16)..."
python scripts/run_baseline.py \
  --method emmet --model "$MODEL" \
  --num_edits "$NUM_EDITS" --batch_size "$BATCH_SIZE" \
  --seed "$SEED" --replay_rate 0.3 \
  --use_lora --lora_rank 16 \
  --output_dir "$OUTPUT_DIR"

echo "[7/7] EMMET + Replay (rate=0.1) + LoRA (rank=8)..."
python scripts/run_baseline.py \
  --method emmet --model "$MODEL" \
  --num_edits "$NUM_EDITS" --batch_size "$BATCH_SIZE" \
  --seed "$SEED" --replay_rate 0.1 \
  --use_lora --lora_rank 8 \
  --output_dir "$OUTPUT_DIR"

echo
echo "========================================"
echo "All combined experiments completed!"
echo "========================================"
echo "Results: $OUTPUT_DIR/"
echo
echo "Analyze with:"
echo "  python scripts/analyze_results.py --results_dir $OUTPUT_DIR"
echo
