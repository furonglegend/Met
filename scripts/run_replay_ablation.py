"""Orchestrate EMMET + Replay ablation experiments.

This script sweeps over replay_rate, strategy, and buffer size, and
groups all runs under a timestamped folder, e.g.:

    results/replay_ablation_20251116_232000/
        b1_r0.0_srandom_buf100/
        b1_r0.1_srandom_buf100/
        ...

Each subfolder is created by scripts/run_baseline.py using --run_name,
so the directory names stay短且包含关键超参数。
"""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List


def build_run_name(batch_size: int, rate: float, strategy: str, buf_size: int) -> str:
    """Compact run name, e.g. b1_r0.3_srandom_buf100."""
    return f"b{batch_size}_r{rate}_s{strategy}_buf{buf_size}"


def run_replay_ablation(
    model: str,
    dataset: str,
    num_edits: int,
    batch_size: int,
    seed: int,
    rates: List[float],
    strategies: List[str],
    buffer_sizes: List[int],
    output_root: Path,
) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suite_name = f"replay_ablation_{ts}"
    suite_dir = output_root / suite_name
    suite_dir.mkdir(parents=True, exist_ok=True)

    print("========================================")
    print("Replay Ablation Experiments (EMMET + Replay)")
    print("Suite directory:", suite_dir)
    print("========================================")

    for strategy in strategies:
        for buf in buffer_sizes:
            for rate in rates:
                run_name = build_run_name(batch_size, rate, strategy, buf)
                print()
                print(f"Running replay_rate={rate} strategy={strategy} buffer={buf}...")
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
                    "--replay_rate",
                    str(rate),
                    "--replay_strategy",
                    strategy,
                    "--replay_buffer_size",
                    str(buf),
                    "--replay_weight",
                    "1.0",
                    "--output_dir",
                    str(suite_dir),
                    "--run_name",
                    run_name,
                ]

                subprocess.run(cmd, check=True)

    # Optional aggregation/visualization for this suite
    print()
    print("Running analysis for replay ablation suite...")
    analysis_csv = suite_dir / "replay_ablation.csv"
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
    print("Replay ablation experiments completed!")
    print("Suite results:", suite_dir)
    print("========================================")

    return suite_dir


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run EMMET+Replay ablation under a timestamped folder",
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
        "--rates",
        nargs="*",
        type=float,
        default=[0.0, 0.1, 0.3, 0.5],
        help="Replay rates to sweep",
    )
    parser.add_argument(
        "--strategies",
        nargs="*",
        type=str,
        default=["random", "priority", "recent"],
        help="Replay strategies to compare",
    )
    parser.add_argument(
        "--buffer_sizes",
        nargs="*",
        type=int,
        default=[100, 200],
        help="Replay buffer sizes to test",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="results",
        help="Root directory under which the replay_ablation_* folder will be created",
    )

    args = parser.parse_args()
    suite_dir = run_replay_ablation(
        model=args.model,
        dataset=args.dataset,
        num_edits=args.num_edits,
        batch_size=args.batch_size,
        seed=args.seed,
        rates=list(args.rates),
        strategies=list(args.strategies),
        buffer_sizes=list(args.buffer_sizes),
        output_root=Path(args.output_root),
    )

    print("Run finished. Suite directory:", suite_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
