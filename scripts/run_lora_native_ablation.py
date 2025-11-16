"""Orchestrate LoRA-native ablation experiments (rank × fit_steps).

This script sweeps over lora_rank and lora_fit_steps for EMMET in
lora_native mode, grouping all runs under a timestamped folder, e.g.:

    results/lora_native_ablation_20251116_232000/
        r4_s0/
        r4_s5/
        r8_s0/
        ...

Each subfolder is created by scripts/run_baseline.py using --run_name,
so the directory names stay短且包含 rank / fit_steps 信息。
"""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List


def build_run_name(rank: int, fit_steps: int) -> str:
    """Compact run name, e.g. r8_s5 for rank=8, fit_steps=5."""
    return f"r{rank}_s{fit_steps}"


def run_lora_native_ablation(
    model: str,
    dataset: str,
    num_edits: int,
    batch_size: int,
    seed: int,
    ranks: List[int],
    fit_steps_list: List[int],
    output_root: Path,
) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suite_name = f"lora_native_ablation_{ts}"
    suite_dir = output_root / suite_name
    suite_dir.mkdir(parents=True, exist_ok=True)

    print("========================================")
    print("LoRA-native Ablation Experiments (rank × fit_steps)")
    print("Suite directory:", suite_dir)
    print("========================================")

    for rank in ranks:
        for fit_steps in fit_steps_list:
            run_name = build_run_name(rank, fit_steps)
            print()
            print(f"Running lora_rank={rank} lora_fit_steps={fit_steps}...")
            print("Run name:", run_name)

            cmd: List[str] = [
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
                str(rank),
                "--lora_alpha",
                str(rank),
                "--lora_scale",
                "1.0",
                "--lora_fit_steps",
                str(fit_steps),
                "--lora_use_svd",
                "--output_dir",
                str(suite_dir),
                "--run_name",
                run_name,
            ]

            subprocess.run(cmd, check=True)

    # Optional aggregation/visualization for this suite
    print()
    print("Running analysis for LoRA-native ablation suite...")
    analysis_csv = suite_dir / "lora_native_ablation.csv"
    analyze_cmd = [
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
    print("LoRA-native ablation experiments completed!")
    print("Suite results:", suite_dir)
    print("========================================")

    return suite_dir


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run LoRA-native ablation (rank × fit_steps) under a timestamped folder",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=str, default="gpt2", help="Model name")
    parser.add_argument(
        "--dataset",
        type=str,
        default="counterfact_sampled_unique_cf_10_20000",
        help="Dataset name",
    )
    parser.add_argument("--num_edits", type=int, default=200, help="Number of edits per run")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--ranks",
        nargs="*",
        type=int,
        default=[4, 8, 16],
        help="LoRA ranks to sweep",
    )
    parser.add_argument(
        "--fit_steps_list",
        nargs="*",
        type=int,
        default=[0, 5, 10],
        help="LoRA fit_steps values to sweep",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="results",
        help="Root directory under which the lora_native_ablation_* folder will be created",
    )

    args = parser.parse_args()
    suite_dir = run_lora_native_ablation(
        model=args.model,
        dataset=args.dataset,
        num_edits=args.num_edits,
        batch_size=args.batch_size,
        seed=args.seed,
        ranks=list(args.ranks),
        fit_steps_list=list(args.fit_steps_list),
        output_root=Path(args.output_root),
    )

    print("Run finished. Suite directory:", suite_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
