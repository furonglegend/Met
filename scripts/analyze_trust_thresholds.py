"""Trust Threshold Sweep Analysis

Usage:
    python scripts/analyze_trust_thresholds.py --input results/trust_with_metrics.csv

This script reads the run-level joined CSV produced by analyze_results.py
(trust_with_metrics.csv or trust_with_metrics_key.csv) and performs a simple
threshold sweep over trust_score_mean. For each threshold, it reports:

- count of runs with trust_score_mean >= threshold
- mean efficacy_success / neighborhood_specificity / composite_score for those runs

This is a lightweight proxy for understanding how changing the trust threshold
might affect metrics, without re-running experiments.
"""

import argparse
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd


def setup_logging(log_file=None):
    handlers = [logging.StreamHandler()]
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers,
    )
    return logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Trust threshold sweep analysis")
    parser.add_argument(
        "--input",
        type=str,
        default="results/trust_with_metrics_key.csv",
        help="Input CSV file (trust_with_metrics.csv or trust_with_metrics_key.csv)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV prefix (default: alongside input, suffixed with _threshold_sweep)",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="*",
        default=[0.2, 0.3, 0.4, 0.5, 0.6],
        help="List of trust_score_mean thresholds to sweep",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default=None,
        help="Optional log file path; if set, logs are written to this file as well as console",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # If no log file is provided, default to logs/ directory with a timestamped name
    if args.log_file is None:
        input_stem = Path(args.input).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_log_dir = Path("logs")
        args.log_file = str(default_log_dir / f"trust_thresholds_{input_stem}_{timestamp}.log")

    logger = setup_logging(args.log_file)

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        return 1

    df = pd.read_csv(input_path)
    if "trust_score_mean" not in df.columns:
        logger.error("Column 'trust_score_mean' not found in %s", input_path)
        return 1

    # Ensure numeric
    df["trust_score_mean"] = pd.to_numeric(df["trust_score_mean"], errors="coerce")
    metrics = [
        c
        for c in [
            "efficacy_success",
            "paraphrase_success",
            "neighborhood_specificity",
            "composite_score",
        ]
        if c in df.columns
    ]
    for m in metrics:
        df[m] = pd.to_numeric(df[m], errors="coerce")

    rows = []
    for thr in args.thresholds:
        subset = df[df["trust_score_mean"] >= thr].copy()
        row = {"threshold": thr, "num_runs": int(len(subset))}
        if len(subset) > 0:
            for m in metrics:
                row[f"{m}_mean"] = float(subset[m].mean())
                row[f"{m}_std"] = float(subset[m].std()) if len(subset) > 1 else 0.0
        else:
            for m in metrics:
                row[f"{m}_mean"] = np.nan
                row[f"{m}_std"] = np.nan
        rows.append(row)

    sweep_df = pd.DataFrame(rows)

    if args.output:
        out_prefix = Path(args.output)
    else:
        out_prefix = input_path.with_suffix("")

    out_csv = out_prefix.with_name(out_prefix.name + "_threshold_sweep.csv")
    sweep_df.to_csv(out_csv, index=False, encoding="utf-8")
    logger.info("Threshold sweep CSV saved to %s", out_csv)

    # Optional: simple text summary
    txt_path = out_csv.with_suffix(".txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Trust threshold sweep summary\n")
        f.write(f"Source: {input_path}\n")
        f.write("\n")
        f.write(sweep_df.to_string(index=False))
        f.write("\n")
    logger.info("Threshold sweep summary saved to %s", txt_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
