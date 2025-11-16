"""Plot combined experiment scores as bar charts.

Usage:
    python scripts/plot_combined_experiments.py \
        --results_dir results/combined \
        --output results/combined/combined_scores.png

This script scans a results directory for experiment runs that contain
`config.json` and `metrics.json`, extracts the final scores
  - efficacy_success (ES)
  - paraphrase_success (PS)
  - neighborhood_specificity (NS)
  - composite_score (S)
for each run, and generates a grouped bar chart where:

- X axis: experiment titles (run directory names)
- Y axis: scores in [0, 1]
- One bar per metric (ES/PS/NS/S) for each experiment.
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

import matplotlib.pyplot as plt
import numpy as np


def load_runs(results_dir: Path) -> List[Dict[str, Any]]:
    """Load all runs that have both config.json and metrics.json.

    Returns a list of dicts with keys:
        title, es, ps, ns, s, run_dir, timestamp
    """
    runs = []

    for config_path in results_dir.rglob("config.json"):
        run_dir = config_path.parent
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            continue

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics = json.load(f)
        except Exception:
            # Skip malformed entries
            continue

        es = metrics.get("efficacy_success")
        ps = metrics.get("paraphrase_success")
        ns = metrics.get("neighborhood_specificity")
        s = metrics.get("composite_score")

        # Require at least composite score to be present
        if s is None:
            continue

        run = {
            "title": run_dir.name,
            "es": es,
            "ps": ps,
            "ns": ns,
            "s": s,
            "run_dir": str(run_dir),
            "timestamp": config.get("timestamp", ""),
        }
        runs.append(run)

    # Sort by timestamp if available, otherwise by title
    runs.sort(key=lambda r: (r.get("timestamp") or "", r["title"]))
    return runs


def plot_scores(runs: List[Dict[str, Any]], output: Path) -> None:
    if not runs:
        print("No valid runs with metrics found. Nothing to plot.")
        return

    titles = [r["title"] for r in runs]
    es = [r["es"] for r in runs]
    ps = [r["ps"] for r in runs]
    ns = [r["ns"] for r in runs]
    s = [r["s"] for r in runs]

    x = np.arange(len(titles))
    width = 0.2

    # Make the figure width depend on number of experiments
    fig_width = max(8.0, len(titles) * 1.5)
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    ax.bar(x - 1.5 * width, es, width, label="ES (efficacy)")
    ax.bar(x - 0.5 * width, ps, width, label="PS (paraphrase)")
    ax.bar(x + 0.5 * width, ns, width, label="NS (neighborhood)")
    ax.bar(x + 1.5 * width, s, width, label="S (composite)")

    ax.set_ylabel("Score")
    ax.set_title("Combined experiment scores (ES / PS / NS / S)")
    ax.set_xticks(x)
    ax.set_xticklabels(titles, rotation=30, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.legend()
    fig.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    print(f"Saved combined scores plot to {output}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot combined experiment scores")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/combined",
        help="Directory containing combined experiment runs",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output image path (default: RESULTS_DIR/combined_scores.png)",
    )

    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return 1

    runs = load_runs(results_dir)
    if not runs:
        print(f"No runs with metrics.json found under {results_dir}")
        return 1

    output_path = Path(args.output) if args.output is not None else results_dir / "combined_scores.png"
    plot_scores(runs, output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
