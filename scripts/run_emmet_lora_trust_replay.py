"""Run a small suite of EMMET + LoRA-native + Trust + Replay experiments.

This orchestrator focuses on the "full stack" configuration for EMMET:

- Plain EMMET (baseline)
- EMMET + Replay
- EMMET + LoRA-native + Replay
- EMMET + LoRA-native + Replay + Trust

Each configuration is run via scripts/run_baseline.py and grouped under a
single timestamped directory, e.g.:

    results/emmet_lora_trust_replay_20251117_033000/
        emmet_b16/                      # baseline
        emmet_b16_r0.3/                 # replay only
        emmet_b16_r0.3_l16/             # replay + LoRA-native
        emmet_b16_r0.3_l16_trust/       # replay + LoRA-native + trust
        emmet_lora_trust_replay.csv
        combined_scores.png

The script then calls analyze_results.py and plot_combined_experiments.py
so that you get both aggregated CSVs and an ES/PS/NS/S bar chart.
"""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional


def build_run_name(
    batch_size: int,
    replay_rate: float,
    use_lora: bool,
    lora_rank: Optional[int],
    use_trust: bool,
) -> str:
    """Construct a compact human-readable run name.

    Examples (for batch_size=16):
      - baseline:                    emmet_b16
      - replay only (r=0.3):         emmet_b16_r0.3
      - replay + LoRA (r=0.3,l16):   emmet_b16_r0.3_l16
      - replay + LoRA + trust:       emmet_b16_r0.3_l16_trust
    """

    parts: List[str] = [f"emmet_b{batch_size}"]
    if replay_rate and replay_rate > 0:
        parts.append(f"r{replay_rate}")
    if use_lora:
        lr = lora_rank if lora_rank is not None else 16
        parts.append(f"l{lr}")
    if use_trust:
        parts.append("trust")
    return "_".join(parts)


def run_emmet_lora_trust_replay(
    model: str,
    dataset: str,
    num_edits: int,
    batch_size: int,
    seed: int,
    output_root: Path,
) -> Path:
    """Run a small suite of EMMET + LoRA + Trust + Replay experiments."""

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suite_name = f"emmet_lora_trust_replay_{ts}"
    suite_dir = output_root / suite_name
    suite_dir.mkdir(parents=True, exist_ok=True)

    print("========================================")
    print("EMMET + LoRA-native + Trust + Replay suite")
    print("Suite directory:", suite_dir)
    print("========================================")

    # Configurations: baseline, replay, replay+LoRA, replay+LoRA+trust
    configs: List[Dict[str, Any]] = [
        {
            "label": "[1/4] EMMET baseline (no replay, no LoRA, no trust)...",
            "replay_rate": 0.0,
            "use_lora": False,
            "lora_rank": None,
            "use_trust": False,
        },
        {
            "label": "[2/4] EMMET + Replay (rate=0.3)...",
            "replay_rate": 0.3,
            "use_lora": False,
            "lora_rank": None,
            "use_trust": False,
        },
        {
            "label": "[3/4] EMMET + LoRA-native (rank=16) + Replay (rate=0.3)...",
            "replay_rate": 0.3,
            "use_lora": True,
            "lora_rank": 16,
            "use_trust": False,
        },
        {
            "label": "[4/4] EMMET + LoRA-native (rank=16) + Replay (rate=0.3) + Trust...",
            "replay_rate": 0.3,
            "use_lora": True,
            "lora_rank": 16,
            "use_trust": True,
        },
    ]

    for cfg in configs:
        replay_rate = float(cfg["replay_rate"])
        use_lora = bool(cfg["use_lora"])
        lora_rank = int(cfg["lora_rank"]) if cfg["lora_rank"] is not None else None
        use_trust = bool(cfg["use_trust"])

        run_name = build_run_name(batch_size, replay_rate, use_lora, lora_rank, use_trust)
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
            "--dataset",
            dataset,
            "--output_dir",
            str(suite_dir),
            "--run_name",
            run_name,
        ]

        if replay_rate > 0:
            cmd.extend(["--replay_rate", str(replay_rate)])

        if use_lora:
            lr = lora_rank if lora_rank is not None else 16
            cmd.extend([
                "--edit_mode",
                "lora_native",
                "--lora_rank",
                str(lr),
                "--lora_alpha",
                str(lr),
                "--lora_scale",
                "1.0",
                "--lora_use_svd",
                "--allow_fallback",
                "--lora_residual_threshold",
                "0.3",
            ])

        if use_trust:
            cmd.extend([
                "--trust_enable",
                "--trust_threshold",
                "0.3",
                "--trust_action",
                "rollback",
                "--trust_scale",
                "0.5",
            ])

        subprocess.run(cmd, check=True)

    # After all runs, aggregate results and create plots inside the same suite dir
    print()
    print("Analyzing results and generating plots for suite...")

    analysis_csv = suite_dir / "emmet_lora_trust_replay.csv"
    analyze_cmd: List[str] = [
        sys.executable,
        "scripts/analyze_results.py",
        "--results_dir",
        str(suite_dir),
        "--output",
        str(analysis_csv),
    ]
    subprocess.run(analyze_cmd, check=True)

    plot_cmd: List[str] = [
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
    print("EMMET + LoRA + Trust + Replay suite completed!")
    print("Suite results:", suite_dir)
    print("Analysis CSV:", analysis_csv)
    print("Combined plot:", suite_dir / "combined_scores.png")
    print("========================================")

    return suite_dir


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run a small suite of EMMET + LoRA-native + Trust + Replay experiments "
            "and generate aggregated scores + visualization."
        ),
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
    parser.add_argument(
        "--output_root",
        type=str,
        default="results",
        help="Root directory under which the emmet_lora_trust_replay_* folder will be created",
    )

    args = parser.parse_args()
    suite_dir = run_emmet_lora_trust_replay(
        model=args.model,
        dataset=args.dataset,
        num_edits=args.num_edits,
        batch_size=args.batch_size,
        seed=args.seed,
        output_root=Path(args.output_root),
    )

    print("Run finished. Suite directory:", suite_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
