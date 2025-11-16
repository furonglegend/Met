#!/bin/bash
# Baseline Comparison Experiment - ROME vs MEMIT
# Based on TODO.md Section 1.2: Two baseline methods comparison
# Objective: Compare ROME and MEMIT to validate the unified framework
#!/bin/bash
# Baseline Comparison Experiment - ROME vs MEMIT vs EMMET
# Based on TODO.md Section 1.2: Three baseline methods comparison
# Objective: Prove the necessity of unified framework and EMMET's advantages

echo "========================================"
echo "Baseline Comparison Experiment"
echo "========================================"
echo "Running ROME, MEMIT, and EMMET baselines"
echo "Results will be grouped under a timestamped folder"
echo "========================================"
echo ""

# Assume Python environment already activated before running this script

# Delegate to the Python orchestrator, which will create an
# all_baselines_* suite folder under results/ and run all configs.
python scripts/run_all_baselines.py \
  --model gpt2-xl \
  --num_edits 500 \
  --dataset counterfact_500 \
  --seed 42 \
  --output_root results

echo ""
echo "Done. See the printed suite directory for this run."
echo ""
