"""Plot forgetting curve from sequence_metrics.jsonl files.

Usage:
    python scripts/plot_forgetting_curve.py --results_dir results/baseline --output figs/forgetting_curves

It searches for sequence_metrics.jsonl under results_dir subfolders.
Generates ES/PS/NS vs cumulative_edits line plots.
"""
import argparse
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_sequence_metrics(results_dir: Path) -> pd.DataFrame:
    rows = []
    for fp in results_dir.rglob("sequence_metrics.jsonl"):
        run_dir = fp.parent
        try:
            with open(fp, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    obj['run_dir'] = str(run_dir)
                    rows.append(obj)
        except Exception:
            pass
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def plot_curves(df: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = ['es_mean', 'ps_mean', 'ns_mean']
    if df.empty:
        return []
    saved = []
    # Try add method from config if available (by parent config.json)
    method_map = {}
    for run in df['run_dir'].unique():
        cfg = Path(run) / 'config.json'
        if cfg.exists():
            try:
                with open(cfg, 'r', encoding='utf-8') as f:
                    c = json.load(f)
                    method_map[run] = c.get('method', '')
            except Exception:
                method_map[run] = ''
        else:
            method_map[run] = ''
    df['method'] = df['run_dir'].map(method_map)

    for metric in metrics:
        plt.figure(figsize=(7,4))
        sns.lineplot(data=df, x='cumulative_edits', y=metric, hue='method', marker='o')
        plt.title(f'{metric} vs edits')
        plt.tight_layout()
        out = out_dir / f'{metric}_forgetting_curve.png'
        plt.savefig(out)
        plt.close()
        saved.append(str(out))
    return saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results_dir', type=str, default='results', help='Root results directory')
    ap.add_argument('--output', type=str, default='results/figs', help='Output directory for plots')
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    df = load_sequence_metrics(results_dir)
    if df.empty:
        print('No sequence_metrics.jsonl found.')
        return 1
    saved = plot_curves(df, Path(args.output))
    print(f'Saved {len(saved)} forgetting curve plots to {args.output}')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
