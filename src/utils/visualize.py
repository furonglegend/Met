"""
Common visualization utilities for experiment results.

Provides functions to generate summary plots (bar plots and line plots)
for key metrics such as ES/PS/NS/Composite.

Dependencies: matplotlib, seaborn, pandas
"""
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def _ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)


def plot_grouped_bar(df: pd.DataFrame, group_col: str, metric: str, out_path: Path):
    """Plot a grouped bar plot for `metric` aggregated by `group_col`.

    df: DataFrame containing group_col and metric columns
    group_col: e.g. 'method' or 'batch_size'
    metric: column name to plot
    out_path: Path to save PNG
    """
    if metric not in df.columns or group_col not in df.columns:
        return None

    agg = df.groupby(group_col)[metric].agg(['mean', 'std', 'count']).reset_index()

    plt.figure(figsize=(8, 4))
    sns.barplot(data=agg, x=group_col, y='mean', yerr=agg['std'], palette='muted')
    plt.ylabel(f"{metric} (mean ± std)")
    plt.title(f"{metric} by {group_col}")
    plt.tight_layout()

    _ensure_dir(out_path)
    plt.savefig(out_path)
    plt.close()

    return out_path


def plot_metric_trends(df: pd.DataFrame, x_col: str, y_col: str, hue_col: str, out_path: Path):
    """Plot metric trends: lineplot of y_col vs x_col, colored by hue_col."""
    if x_col not in df.columns or y_col not in df.columns:
        return None

    plt.figure(figsize=(8, 4))
    sns.lineplot(data=df, x=x_col, y=y_col, hue=hue_col, marker='o')
    plt.ylabel(y_col)
    plt.title(f"{y_col} vs {x_col} by {hue_col}")
    plt.tight_layout()

    _ensure_dir(out_path)
    plt.savefig(out_path)
    plt.close()

    return out_path


def create_summary_plots(df: pd.DataFrame, output_dir: str):
    """Create a set of summary plots in `output_dir`.

    The function will generate:
      - bar plots of each main metric by `method`
      - bar plots of each main metric by `batch_size` (if available)
      - line plots of composite_score vs batch_size (if numeric)

    Returns list of saved file paths.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = [
        'efficacy_success',
        'paraphrase_success',
        'neighborhood_specificity',
        'composite_score'
    ]

    saved = []

    # Ensure numeric types where appropriate
    for m in metrics:
        if m in df.columns:
            try:
                df[m] = pd.to_numeric(df[m], errors='coerce')
            except Exception:
                pass

    # By method
    if 'method' in df.columns:
        for m in metrics:
            if m in df.columns:
                out = out_dir / f"{m}_by_method.png"
                res = plot_grouped_bar(df, 'method', m, out)
                if res:
                    saved.append(str(res))

    # By batch_size
    if 'batch_size' in df.columns:
        # convert batch_size to string for categorical plotting if needed
        try:
            df['batch_size'] = df['batch_size'].astype(str)
        except Exception:
            pass
        for m in metrics:
            if m in df.columns:
                out = out_dir / f"{m}_by_batch_size.png"
                res = plot_grouped_bar(df, 'batch_size', m, out)
                if res:
                    saved.append(str(res))

    # Composite vs batch_size numeric trend
    if 'composite_score' in df.columns and 'batch_size' in df.columns:
        # try numeric x
        try:
            df['batch_size_numeric'] = pd.to_numeric(df['batch_size'], errors='coerce')
            out = out_dir / f"composite_vs_batch_size.png"
            res = plot_metric_trends(df.dropna(subset=['batch_size_numeric','composite_score']),
                                     'batch_size_numeric', 'composite_score', 'method', out)
            if res:
                saved.append(str(res))
        except Exception:
            pass

    return saved
