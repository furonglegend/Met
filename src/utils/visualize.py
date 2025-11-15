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
import numpy as np


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


def _safe_numeric(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            try:
                df[c] = pd.to_numeric(df[c], errors='coerce')
            except Exception:
                pass
    return df


def create_lora_plots(df: pd.DataFrame, output_dir: str):
    """Create LoRA-native specific plots if data columns are present.

    - Histogram of lora_residual_rel_mean
    - Scatter of lora_residual_rel_mean vs ES/NS/Composite
    - Rank vs metrics bar/line plots (edit_mode == lora_native)
    - Heatmap of (rank x lora_fit_steps) vs composite_score (if both columns exist)
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    df = df.copy()
    # Ensure numeric types
    df = _safe_numeric(df, [
        'lora_rank', 'lora_fit_steps', 'lora_alpha', 'lora_scale',
        'lora_residual_rel_mean',
        'efficacy_success', 'paraphrase_success', 'neighborhood_specificity', 'composite_score'
    ])

    # Only LoRA-native runs
    if 'edit_mode' in df.columns:
        df_lora = df[df['edit_mode'] == 'lora_native'].copy()
    else:
        df_lora = df.copy()

    if df_lora.empty:
        return saved

    # Histogram of residuals
    if 'lora_residual_rel_mean' in df_lora.columns and df_lora['lora_residual_rel_mean'].notna().any():
        plt.figure(figsize=(6, 4))
        sns.histplot(df_lora['lora_residual_rel_mean'].dropna(), bins=20, kde=True)
        plt.xlabel('lora_residual_rel_mean')
        plt.title('LoRA residual (relative Frobenius) distribution')
        plt.tight_layout()
        p = out_dir / 'lora_residual_hist.png'
        _ensure_dir(p)
        plt.savefig(p)
        plt.close()
        saved.append(str(p))

    # Correlation scatter plots
    for metric in ['efficacy_success', 'neighborhood_specificity', 'composite_score']:
        if metric in df_lora.columns and 'lora_residual_rel_mean' in df_lora.columns:
            dsub = df_lora.dropna(subset=['lora_residual_rel_mean', metric])
            if not dsub.empty:
                plt.figure(figsize=(6, 4))
                sns.regplot(data=dsub, x='lora_residual_rel_mean', y=metric, scatter_kws={'s': 20}, line_kws={'color': 'red'})
                plt.title(f'{metric} vs LoRA residual')
                plt.tight_layout()
                p = out_dir / f'{metric}_vs_lora_residual.png'
                _ensure_dir(p)
                plt.savefig(p)
                plt.close()
                saved.append(str(p))

    # Rank vs metrics (bar)
    if 'lora_rank' in df_lora.columns and df_lora['lora_rank'].notna().any():
        for metric in ['efficacy_success', 'neighborhood_specificity', 'composite_score']:
            if metric in df_lora.columns:
                agg = df_lora.groupby('lora_rank')[metric].agg(['mean', 'std', 'count']).reset_index()
                plt.figure(figsize=(7, 4))
                sns.barplot(data=agg, x='lora_rank', y='mean', yerr=agg['std'], palette='Blues')
                plt.xlabel('lora_rank')
                plt.ylabel(f'{metric} (mean ± std)')
                plt.title(f'{metric} by LoRA rank')
                plt.tight_layout()
                p = out_dir / f'{metric}_by_lora_rank.png'
                _ensure_dir(p)
                plt.savefig(p)
                plt.close()
                saved.append(str(p))

    # Heatmap rank x fit_steps -> composite_score
    if 'lora_rank' in df_lora.columns and 'lora_fit_steps' in df_lora.columns and 'composite_score' in df_lora.columns:
        pivot = df_lora.pivot_table(index='lora_rank', columns='lora_fit_steps', values='composite_score', aggfunc='mean')
        if pivot.size > 0:
            plt.figure(figsize=(6, 5))
            sns.heatmap(pivot.sort_index().sort_index(axis=1), annot=True, fmt='.3f', cmap='viridis')
            plt.title('Composite score: rank x fit_steps')
            plt.tight_layout()
            p = out_dir / 'composite_heatmap_rank_fitsteps.png'
            _ensure_dir(p)
            plt.savefig(p)
            plt.close()
            saved.append(str(p))

    return saved


def create_replay_plots(df: pd.DataFrame, output_dir: str):
    """Create Replay ablation plots when replay columns are present.

    - Composite vs replay_rate (line, hue=method)
    - ES/PS/NS by replay_rate (bar, aggregated)
    - Heatmap of composite_score by (replay_rate x batch_size)
    - Strategy comparison: composite by strategy (if available)
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    df = df.copy()
    # Ensure numeric
    for col in ['replay_rate', 'batch_size', 'composite_score', 'efficacy_success', 'paraphrase_success', 'neighborhood_specificity']:
        if col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except Exception:
                pass

    if 'replay_rate' not in df.columns:
        return saved

    # Composite vs replay_rate
    if 'composite_score' in df.columns:
        plt.figure(figsize=(7, 4))
        hue = 'method' if 'method' in df.columns else None
        sns.lineplot(data=df.dropna(subset=['replay_rate', 'composite_score']), x='replay_rate', y='composite_score', hue=hue, marker='o')
        plt.title('Composite vs Replay Rate')
        plt.tight_layout()
        p = out_dir / 'composite_vs_replay_rate.png'
        _ensure_dir(p)
        plt.savefig(p)
        plt.close()
        saved.append(str(p))

    # Bars for ES/PS/NS by replay_rate
    for metric in ['efficacy_success', 'paraphrase_success', 'neighborhood_specificity']:
        if metric in df.columns:
            agg = df.groupby('replay_rate')[metric].agg(['mean', 'std', 'count']).reset_index()
            plt.figure(figsize=(7, 4))
            sns.barplot(data=agg, x='replay_rate', y='mean', yerr=agg['std'], palette='muted')
            plt.ylabel(f'{metric} (mean ± std)')
            plt.title(f'{metric} by Replay Rate')
            plt.tight_layout()
            p = out_dir / f'{metric}_by_replay_rate.png'
            _ensure_dir(p)
            plt.savefig(p)
            plt.close()
            saved.append(str(p))

    # Heatmap composite: replay_rate x batch_size
    if 'batch_size' in df.columns and 'composite_score' in df.columns:
        try:
            pivot = df.pivot_table(index='replay_rate', columns='batch_size', values='composite_score', aggfunc='mean')
            if pivot.size > 0:
                plt.figure(figsize=(6, 5))
                sns.heatmap(pivot.sort_index().sort_index(axis=1), annot=True, fmt='.3f', cmap='mako')
                plt.title('Composite: replay_rate x batch_size')
                plt.tight_layout()
                p = out_dir / 'composite_heatmap_replay_batch.png'
                _ensure_dir(p)
                plt.savefig(p)
                plt.close()
                saved.append(str(p))
        except Exception:
            pass

    # Strategy comparison if present
    if 'replay_strategy' in df.columns and 'composite_score' in df.columns:
        plt.figure(figsize=(7, 4))
        sns.barplot(data=df, x='replay_strategy', y='composite_score', estimator=np.mean, ci='sd', palette='Set2')
        plt.title('Composite by Replay Strategy')
        plt.tight_layout()
        p = out_dir / 'composite_by_replay_strategy.png'
        _ensure_dir(p)
        plt.savefig(p)
        plt.close()
        saved.append(str(p))

    return saved


def create_trust_plots(trust_df: pd.DataFrame, output_dir: str):
    """Create Trust mechanism plots from trust_events.jsonl aggregation.

        - Histogram of trust_score overall
        - Bar chart of action distribution (rollback vs scale)
        - Bar chart of trust_applied vs not
        - If columns from metrics join exist (e.g., composite_score_mean), scatter plots of
            trust_score_mean vs composite_score or ES/NS for quick sanity check.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    df = trust_df.copy()
    if df.empty:
        return saved

    # Ensure types
    for col in ['trust_score', 'trust_scale']:
        if col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except Exception:
                pass

    # Histogram of trust_score
    if 'trust_score' in df.columns and df['trust_score'].notna().any():
        plt.figure(figsize=(6, 4))
        sns.histplot(df['trust_score'].dropna(), bins=20, kde=True, color='teal')
        plt.xlabel('trust_score')
        plt.title('Trust score distribution')
        plt.tight_layout()
        p = out_dir / 'trust_score_hist.png'
        _ensure_dir(p)
        plt.savefig(p)
        plt.close()
        saved.append(str(p))

    # Action distribution
    if 'trust_action' in df.columns:
        plt.figure(figsize=(6, 4))
        sns.countplot(x='trust_action', data=df, palette='Set2')
        plt.xlabel('trust_action')
        plt.ylabel('count')
        plt.title('Trust action distribution')
        plt.tight_layout()
        p = out_dir / 'trust_action_counts.png'
        _ensure_dir(p)
        plt.savefig(p)
        plt.close()
        saved.append(str(p))

    # Applied vs not
    if 'trust_applied' in df.columns:
        plt.figure(figsize=(6, 4))
        sns.countplot(x=df['trust_applied'].fillna(False).map({True:'applied', False:'not_applied'}), palette='pastel')
        plt.xlabel('trust_applied')
        plt.ylabel('count')
        plt.title('Trust application (per-layer events)')
        plt.tight_layout()
        p = out_dir / 'trust_applied_counts.png'
        _ensure_dir(p)
        plt.savefig(p)
        plt.close()
        saved.append(str(p))

    # If this DataFrame already contains run-level metrics (e.g. from trust_with_metrics.csv),
    # provide simple correlation plots between mean trust score and core metrics.
    # We detect this by presence of 'trust_score_mean' and metric columns.
    if 'trust_score_mean' in df.columns:
        for metric in ['efficacy_success', 'neighborhood_specificity', 'composite_score']:
            if metric in df.columns:
                try:
                    x = pd.to_numeric(df['trust_score_mean'], errors='coerce')
                    y = pd.to_numeric(df[metric], errors='coerce')
                    dsub = pd.DataFrame({'trust_score_mean': x, metric: y}).dropna()
                    if dsub.empty:
                        continue
                    plt.figure(figsize=(6, 4))
                    sns.regplot(data=dsub, x='trust_score_mean', y=metric, scatter_kws={'s': 20}, line_kws={'color': 'red'})
                    plt.title(f'{metric} vs trust_score_mean (per run)')
                    plt.tight_layout()
                    p = out_dir / f'{metric}_vs_trust_score_mean.png'
                    _ensure_dir(p)
                    plt.savefig(p)
                    plt.close()
                    saved.append(str(p))
                except Exception:
                    continue

    return saved
