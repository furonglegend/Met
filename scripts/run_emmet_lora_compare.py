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
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np

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

    # Create a simple ES/PS/NS/S comparison plot between raw and LoRA-native runs
    try:
        raw_metrics_path = suite_dir / raw_run_name / "metrics.json"
        lora_metrics_path = suite_dir / lora_run_name / "metrics.json"
        if raw_metrics_path.exists() and lora_metrics_path.exists():
            with open(raw_metrics_path, "r", encoding="utf-8") as f_raw:
                raw_metrics = json.load(f_raw)
            with open(lora_metrics_path, "r", encoding="utf-8") as f_lora:
                lora_metrics = json.load(f_lora)

            labels = ["ES", "PS", "NS", "S"]
            keys = [
                "efficacy_success",
                "paraphrase_success",
                "neighborhood_specificity",
                "composite_score",
            ]
            raw_vals = [raw_metrics.get(k, 0.0) for k in keys]
            lora_vals = [lora_metrics.get(k, 0.0) for k in keys]

            x = np.arange(len(labels))
            width = 0.35

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(x - width / 2, raw_vals, width, label="EMMET raw")
            ax.bar(x + width / 2, lora_vals, width, label=f"LoRA-native (r={lora_rank})")

            ax.set_ylabel("Score")
            ax.set_title("EMMET vs LoRA-native (ES/PS/NS/S)")
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.set_ylim(0.0, 1.05)
            ax.legend()
            fig.tight_layout()

            plot_path = suite_dir / "emmet_lora_compare.png"
            fig.savefig(plot_path, dpi=150)
            plt.close(fig)
            print("Saved comparison plot to", plot_path)
        else:
            print("metrics.json not found for one or both runs; skipping plot.")
    except Exception as e:  # noqa: BLE001
        print("Failed to generate comparison plot:", e)

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
