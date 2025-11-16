"""Orchestrate combined EMMET + Replay + LoRA experiments.

This script runs a fixed set of EMMET configurations (baseline, replay,
LoRA, replay+LoRA) and groups them under a single timestamped suite
folder, e.g.:

    results/combined_experiments_20251116_232000/
        emmet_gpt2-xl_b16/                  # baseline
        emmet_gpt2-xl_b16_replay0.3/        # replay only
        emmet_gpt2-xl_b16_lora8/            # LoRA only
        emmet_gpt2-xl_b16_replay0.3_lora8/  # replay + LoRA
        ...
        combined_scores.png                 # ES/PS/NS/S summary plot

Each per-run subfolder is created by scripts/run_baseline.py using the
--run_name argument, so that the directory name reflects the important
hyperparameters (method/model/batch_size/replay_rate/lora_rank).
"""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional


def build_run_name(batch_size: int, replay_rate: float, use_lora: bool, lora_rank: Optional[int]) -> str:
    """Construct a compact human-readable run name.

    Examples (for batch_size=16):
      - baseline (no replay, no LoRA):   b16
      - replay only (r=0.3):             b16_r0.3
      - LoRA only (rank=8):              b16_l8
      - replay + LoRA (r=0.3, rank=8):   b16_r0.3_l8
    """

    parts: List[str] = [f"b{batch_size}"]
    if replay_rate and replay_rate > 0:
        parts.append(f"r{replay_rate}")
    if use_lora:
        lr = lora_rank if lora_rank is not None else 8
        parts.append(f"l{lr}")
    return "_".join(parts)


def run_combined_experiments(model: str, num_edits: int, batch_size: int, seed: int, output_root: Path) -> Path:
    # Timestamped suite directory
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suite_name = f"combined_experiments_{ts}"
    suite_dir = output_root / suite_name
    suite_dir.mkdir(parents=True, exist_ok=True)

    print("========================================")
    print("Combined Configuration Experiments (Python orchestrator)")
    print("Suite directory:", suite_dir)
    print("========================================")

    # Define the 7 configurations matching the original shell/cmd scripts
    configs: List[Dict[str, Any]] = [
        {
            "label": "[1/7] EMMET baseline (no enhancements)...",
            "replay_rate": 0.0,
            "use_lora": False,
            "lora_rank": None,
        },
        {
            "label": "[2/7] EMMET + Replay (rate=0.3)...",
            "replay_rate": 0.3,
            "use_lora": False,
            "lora_rank": None,
        },
        {
            "label": "[3/7] EMMET + LoRA (rank=8)...",
            "replay_rate": 0.0,
            "use_lora": True,
            "lora_rank": 8,
        },
        {
            "label": "[4/7] EMMET + Replay + LoRA (rank=8)...",
            "replay_rate": 0.3,
            "use_lora": True,
            "lora_rank": 8,
        },
        {
            "label": "[5/7] EMMET + Replay (rate=0.5) + LoRA (rank=4)...",
            "replay_rate": 0.5,
            "use_lora": True,
            "lora_rank": 4,
        },
        {
            "label": "[6/7] EMMET + Replay (rate=0.3) + LoRA (rank=16)...",
            "replay_rate": 0.3,
            "use_lora": True,
            "lora_rank": 16,
        },
        {
            "label": "[7/7] EMMET + Replay (rate=0.1) + LoRA (rank=8)...",
            "replay_rate": 0.1,
            "use_lora": True,
            "lora_rank": 8,
        },
    ]

    for idx, cfg in enumerate(configs, start=1):
        replay_rate = float(cfg["replay_rate"])
        use_lora = bool(cfg["use_lora"])
        lora_rank = int(cfg["lora_rank"]) if cfg["lora_rank"] is not None else None

        run_name = build_run_name(batch_size, replay_rate, use_lora, lora_rank)
        print()
        print(cfg["label"])
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
            "--output_dir",
            str(suite_dir),
            "--run_name",
            run_name,
        ]
        if replay_rate > 0:
            cmd.extend(["--replay_rate", str(replay_rate)])
        if use_lora:
            # Native LoRA editing mode with given rank
            lr = lora_rank if lora_rank is not None else 8
            cmd.extend([
                "--use_lora",
                "--lora_rank",
                str(lr),
                "--edit_mode",
                "lora_native",
            ])

        subprocess.run(cmd, check=True)

    # After all runs, create the combined bar chart inside the same suite dir
    print()
    print("Generating combined scores plot for suite...")
    plot_cmd = [
        sys.executable,
        "scripts/plot_combined_experiments.py",
        "--results_dir",
        str(suite_dir),
        "--output",
        str(suite_dir / "combined_scores.png"),
    ]
    subprocess.run(plot_cmd, check=True)

    print()
    print("========================================")
    print("All combined experiments completed!")
    print("Suite results:", suite_dir)
    print("Combined plot:", suite_dir / "combined_scores.png")
    print("========================================")

    return suite_dir


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run combined EMMET experiments and group results under a timestamped folder",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=str, default="gpt2-xl", help="Model name")
    parser.add_argument("--num_edits", type=int, default=200, help="Number of edits per run")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output_root",
        type=str,
        default="results",
        help="Root directory under which the combined_experiments_* folder will be created",
    )

    args = parser.parse_args()
    suite_dir = run_combined_experiments(
        model=args.model,
        num_edits=args.num_edits,
        batch_size=args.batch_size,
        seed=args.seed,
        output_root=Path(args.output_root),
    )

    print("Run finished. Suite directory:", suite_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
