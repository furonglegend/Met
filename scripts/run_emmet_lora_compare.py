"""Run a small EMMET vs LoRA-native comparison on a compact dataset.

This script runs two EMMET experiments under a single timestamped folder:

  1. Raw EMMET (edit_mode=raw)
  2. EMMET with native LoRA overlays (edit_mode=lora_native)

Both runs share the same model, dataset, num_edits, batch_size, and seed so
that you can directly compare ES/PS/NS metrics to verify whether LoRA-native
integration improves over plain EMMET on a small dataset.

Results are stored under, e.g.:

    results/emmet_lora_compare_20251117_010000/
        emmet_raw/
        emmet_lora_r8/
        emmet_lora_compare.csv

Each subfolder is produced by scripts/run_baseline.py using --run_name.
"""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


def run_emmet_lora_compare(
    model: str,
    dataset: str,
    num_edits: int,
    batch_size: int,
    seed: int,
    lora_rank: int,
    lora_fit_steps: int,
    output_root: Path,
) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suite_name = f"emmet_lora_compare_{ts}"
    suite_dir = output_root / suite_name
    suite_dir.mkdir(parents=True, exist_ok=True)

    print("========================================")
    print("EMMET vs LoRA-native comparison (small dataset)")
    print("Suite directory:", suite_dir)
    print("========================================")

    # 1) Raw EMMET (edit_mode=raw)
    raw_run_name = "emmet_raw"
    print()
    print("[1/2] Running plain EMMET (raw updates)...")
    print("Run name:", raw_run_name)

    cmd_raw: List[str] = [
        sys.executable,
        "scripts/run_baseline.py",
        "--method",
        "emmet",
        "--model",
        model,
        "--num_edits",
        str(num_edits),
        "--batch_size",
        str(batch_size),
        "--seed",
        str(seed),
        "--dataset",
        dataset,
        "--edit_mode",
        "raw",
        "--output_dir",
        str(suite_dir),
        "--run_name",
        raw_run_name,
    ]

    subprocess.run(cmd_raw, check=True)

    # 2) EMMET with native LoRA overlays
    lora_run_name = f"emmet_lora_r{lora_rank}"
    print()
    print("[2/2] Running EMMET with native LoRA overlays...")
    print("Run name:", lora_run_name)

    cmd_lora: List[str] = [
        sys.executable,
        "scripts/run_baseline.py",
        "--method",
        "emmet",
        "--model",
        model,
        "--num_edits",
        str(num_edits),
        "--batch_size",
        str(batch_size),
        "--seed",
        str(seed),
        "--dataset",
        dataset,
        "--edit_mode",
        "lora_native",
        "--lora_rank",
        str(lora_rank),
        "--lora_alpha",
        str(lora_rank),
        "--lora_scale",
        "1.0",
        "--lora_fit_steps",
        str(lora_fit_steps),
        "--lora_use_svd",
        "--output_dir",
        str(suite_dir),
        "--run_name",
        lora_run_name,
    ]

    subprocess.run(cmd_lora, check=True)

    # Aggregate ES/PS/NS metrics across the two runs
    print()
    print("Running analysis for EMMET vs LoRA-native suite...")
    analysis_csv = suite_dir / "emmet_lora_compare.csv"
    analyze_cmd: List[str] = [
        sys.executable,
        "scripts/analyze_results.py",
        "--results_dir",
        str(suite_dir),
        "--output",
        str(analysis_csv),
    ]
    try:
        subprocess.run(analyze_cmd, check=True)
        print("Aggregated CSV:", analysis_csv)
    except Exception as e:  # noqa: BLE001
        print("Analysis failed:", e)

    print()
    print("========================================")
    print("EMMET vs LoRA-native comparison completed!")
    print("Suite results:", suite_dir)
    print("========================================")

    return suite_dir


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a small EMMET vs LoRA-native comparison under a timestamped folder",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=str, default="gpt2", help="Model name")
    parser.add_argument(
        "--dataset",
        type=str,
        default="counterfact_500",
        help="Dataset name (without .json extension)",
    )
    parser.add_argument("--num_edits", type=int, default=200, help="Number of edits per run")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--lora_rank", type=int, default=8, help="LoRA rank")
    parser.add_argument(
        "--lora_fit_steps",
        type=int,
        default=0,
        help="Optional tiny fitting steps to refine LoRA factors",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="results",
        help="Root directory under which the emmet_lora_compare_* folder will be created",
    )

    args = parser.parse_args()
    suite_dir = run_emmet_lora_compare(
        model=args.model,
        dataset=args.dataset,
        num_edits=args.num_edits,
        batch_size=args.batch_size,
        seed=args.seed,
        lora_rank=args.lora_rank,
        lora_fit_steps=args.lora_fit_steps,
        output_root=Path(args.output_root),
    )

    print("Run finished. Suite directory:", suite_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
