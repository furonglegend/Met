"""Orchestrate baseline comparison experiments (ROME, MEMIT, EMMET).

This script runs a small set of baseline methods and groups their
results under a single timestamped folder, e.g.:

    results/all_baselines_20251116_232000/
        rome_b1/
        memit_b32/
        emmet_b32/
        baseline_comparison_rome_memit.csv
        figs/...

Each subfolder is created by scripts/run_baseline.py using --run_name,
so the directory names stay short and informative.
"""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any


def build_run_name(method: str, batch_size: int) -> str:
    """Compact run name: rome_b1, memit_b32, emmet_b32."""
    return f"{method}_b{batch_size}"


def run_all_baselines(
    model: str,
    num_edits: int,
    dataset: str,
    seed: int,
    output_root: Path,
) -> Path:
    # Timestamped suite directory named after this script
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suite_name = f"all_baselines_{ts}"
    suite_dir = output_root / suite_name
    suite_dir.mkdir(parents=True, exist_ok=True)

    print("========================================")
    print("Baseline Comparison Experiments (ROME / MEMIT / EMMET)")
    print("Suite directory:", suite_dir)
    print("========================================")

    # Match the intent of todo.md 1.2 and existing shell scripts
    configs: List[Dict[str, Any]] = [
        {
            "label": "[1/3] ROME: single edit (batch_size=1)",
            "method": "rome",
            "batch_size": 1,
        },
        {
            "label": "[2/3] MEMIT: batch edit (batch_size=32)",
            "method": "memit",
            "batch_size": 32,
        },
        {
            "label": "[3/3] EMMET: batch edit (batch_size=32)",
            "method": "emmet",
            "batch_size": 32,
        },
    ]

    for cfg in configs:
        method = cfg["method"]
        batch_size = int(cfg["batch_size"])
        run_name = build_run_name(method, batch_size)

        print()
        print(cfg["label"])
        print("Run name:", run_name)

        cmd: List[str] = [
            sys.executable,
            "scripts/run_baseline.py",
            "--method",
            method,
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
            "--output_dir",
            str(suite_dir),
            "--run_name",
            run_name,
        ]

        subprocess.run(cmd, check=True)

    # Aggregate and visualize results into the same suite directory
    print()
    print("Running analysis for baseline suite...")
    analysis_csv = suite_dir / "baseline_comparison_rome_memit.csv"
    analyze_cmd = [
        sys.executable,
        "scripts/analyze_results.py",
        "--results_dir",
        str(suite_dir),
        "--output",
        str(analysis_csv),
    ]
    subprocess.run(analyze_cmd, check=True)

    print()
    print("========================================")
    print("All baseline experiments completed!")
    print("Suite results:", suite_dir)
    print("Aggregated CSV:", analysis_csv)
    print("========================================")

    return suite_dir


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run baseline comparison (ROME/MEMIT/EMMET) under a timestamped folder",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=str, default="gpt2-xl", help="Model name")
    parser.add_argument("--num_edits", type=int, default=500, help="Number of edits per run")
    parser.add_argument("--dataset", type=str, default="counterfact_500", help="Dataset name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output_root",
        type=str,
        default="results",
        help="Root directory under which the all_baselines_* folder will be created",
    )

    args = parser.parse_args()
    suite_dir = run_all_baselines(
        model=args.model,
        num_edits=args.num_edits,
        dataset=args.dataset,
        seed=args.seed,
        output_root=Path(args.output_root),
    )

    print("Run finished. Suite directory:", suite_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
