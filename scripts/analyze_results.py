"""
Results Analysis Script
Analyze and aggregate experimental results

Usage:
    python scripts/analyze_results.py --results_dir results/baseline
    python scripts/analyze_results.py --results_dir results --output analysis.csv
"""

import argparse
import json
import pandas as pd
import sys
from pathlib import Path
import logging
from datetime import datetime

# Make sure project src/ is on sys.path so we can import visualization utilities
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
try:
    from utils import visualize as viz
except Exception:
    viz = None


class ResultsAnalyzer:
    """Analyze experimental results"""
    
    def __init__(self, results_dir, output_file=None, log_file=None):
        self.results_dir = Path(results_dir)
        self.output_file = output_file or f"results/analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self.setup_logging(log_file)
        
    def setup_logging(self, log_file=None):
        """Setup logging"""
        handlers = [logging.StreamHandler()]

        # Default log path under logs/ if not provided
        if log_file is None:
            results_name = self.results_dir.name or "results"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = Path("logs") / f"analyze_results_{results_name}_{timestamp}.log"
        else:
            log_path = Path(log_file)

        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=handlers,
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Logging to {log_path}")
        
    def find_result_dirs(self):
        """Find all result directories"""
        result_dirs = []
        
        for path in self.results_dir.rglob("config.json"):
            result_dirs.append(path.parent)
        
        self.logger.info(f"Found {len(result_dirs)} result directories")
        return result_dirs

    def _load_all_trust_events(self):
        """Load trust_events.jsonl from all detected run directories into a DataFrame.

        Returns a pandas DataFrame with columns like:
          run_dir, batch_idx, layer, weight_name, delta_norm,
          trust_enable, trust_score, trust_applied, trust_action, trust_scale
        Returns empty DataFrame or None if none found.
        """
        try:
            result_dirs = self.find_result_dirs()
            rows = []
            for rd in result_dirs:
                f = rd / 'trust_events.jsonl'
                if not f.exists():
                    continue
                try:
                    with open(f, 'r', encoding='utf-8') as fh:
                        for line in fh:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                rec = json.loads(line)
                            except Exception:
                                continue
                            rec['run_dir'] = str(rd)
                            rows.append(rec)
                except Exception:
                    continue
            if not rows:
                return None
            df = pd.DataFrame(rows)
            # Coerce numerics where applicable
            for col in ['batch_idx', 'delta_norm', 'trust_score', 'trust_scale']:
                if col in df.columns:
                    try:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    except Exception:
                        pass
            # Normalize action strings
            if 'trust_action' in df.columns:
                try:
                    df['trust_action'] = df['trust_action'].astype(str).str.lower()
                except Exception:
                    pass
            return df
        except Exception:
            return None
    
    def load_experiment_data(self, result_dir):
        """Load data from a single experiment"""
        config_file = result_dir / "config.json"
        metrics_file = result_dir / "metrics.json"
        edits_file = result_dir / "edit_results.json"
        lora_events_file = result_dir / "lora_native_events.jsonl"
        
        if not config_file.exists() or not metrics_file.exists():
            self.logger.warning(f"Incomplete results in {result_dir}")
            return None
        
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        # Optionally extract LoRA residual statistics from edit_results
        lora_residual_rel_mean = None
        lora_residual_rel_count = 0
        lora_fallback_rate = None
        lora_fallback_count = 0
        lora_event_count = 0
        if edits_file.exists():
            try:
                with open(edits_file, 'r') as f:
                    edits = json.load(f)
                vals = []
                for batch in edits:
                    ed = batch.get("edit_distances", {})
                    if isinstance(ed, dict):
                        for v in ed.values():
                            if isinstance(v, dict) and "lora_residual_rel" in v and v["lora_residual_rel"] is not None:
                                vals.append(float(v["lora_residual_rel"]))
                if vals:
                    import numpy as _np
                    lora_residual_rel_mean = float(_np.mean(vals))
                    lora_residual_rel_count = len(vals)
            except Exception:
                pass

        # Parse lora events if present to compute fallback rate and refined residual mean
        if lora_events_file.exists():
            try:
                residuals = []
                fb = 0
                total = 0
                with open(lora_events_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        total += 1
                        rec = json.loads(line)
                        r = rec.get('lora_residual_rel')
                        if r is not None:
                            try:
                                residuals.append(float(r))
                            except Exception:
                                pass
                        if rec.get('lora_fallback') is True:
                            fb += 1
                if total > 0:
                    lora_event_count = total
                    lora_fallback_count = fb
                    lora_fallback_rate = fb / total
                    # Prefer events-derived residuals if available
                    if residuals:
                        import numpy as _np
                        lora_residual_rel_mean = float(_np.mean(residuals))
                        lora_residual_rel_count = len(residuals)
            except Exception:
                pass

        # Combine config and metrics
        data = {
            "run_dir": str(result_dir),
            **config,
            **metrics,
            "lora_residual_rel_mean": lora_residual_rel_mean,
            "lora_residual_rel_count": lora_residual_rel_count,
            "lora_fallback_rate": lora_fallback_rate,
            "lora_fallback_count": lora_fallback_count,
            "lora_event_count": lora_event_count,
        }
        
        return data
    
    def aggregate_results(self):
        """Aggregate all results into a DataFrame"""
        result_dirs = self.find_result_dirs()
        
        all_data = []
        for result_dir in result_dirs:
            data = self.load_experiment_data(result_dir)
            if data:
                all_data.append(data)
        
        if not all_data:
            self.logger.error("No valid results found")
            return None
        
        df = pd.DataFrame(all_data)
        self.logger.info(f"Aggregated {len(df)} experiments")
        
        return df
    
    def compute_statistics(self, df):
        """Compute statistics across experiments"""
        if df is None or df.empty:
            return None
        
        # Group by method, batch_size, and replay_rate
        group_cols = []
        if "method" in df.columns:
            group_cols.append("method")
        if "batch_size" in df.columns:
            group_cols.append("batch_size")
        if "replay_rate" in df.columns:
            group_cols.append("replay_rate")
        
        if not group_cols:
            return None
        
        # Aggregate metrics
        agg_dict = {}
        for metric in ["efficacy_success", "paraphrase_success", "neighborhood_specificity", 
                       "composite_score", "generation_entropy", "success_rate"]:
            if metric in df.columns:
                agg_dict[metric] = ["mean", "std", "count"]
        
        if agg_dict:
            stats = df.groupby(group_cols).agg(agg_dict)
            self.logger.info("Statistics computed")
            return stats
        
        return None
    
    def save_results(self, df, stats):
        """Save aggregated results"""
        # Save detailed results
        output_path = Path(self.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_path, index=False, encoding="utf-8")
        self.logger.info(f"Detailed results saved to {output_path}")
        
        # Save statistics
        if stats is not None:
            stats_file = output_path.with_name(output_path.stem + "_stats.csv")
            stats.to_csv(stats_file, encoding="utf-8")
            self.logger.info(f"Statistics saved to {stats_file}")

        # Emit a general ablation matrix grouped by key config fields
        try:
            # Choose group keys that commonly appear in config.json; filter by existence
            candidate_group_cols = [
                'method', 'model', 'batch_size', 'replay_rate', 'replay_strategy', 'replay_buffer_size',
                'edit_mode', 'lora_rank', 'lora_alpha', 'trust_enable'
            ]
            group_cols = [c for c in candidate_group_cols if c in df.columns]
            metric_cols = [
                c for c in ['efficacy_success', 'paraphrase_success', 'neighborhood_specificity', 'composite_score']
                if c in df.columns
            ]
            if group_cols and metric_cols:
                ablation_matrix = df.groupby(group_cols)[metric_cols].mean().reset_index()
                ablation_file = output_path.with_name('ablation_matrix.csv')
                ablation_matrix.to_csv(ablation_file, index=False, encoding="utf-8")
                self.logger.info(f"General ablation matrix saved to {ablation_file}")

            # Preserve the more specific Replay-only ablation for convenience
            if 'replay_rate' in df.columns:
                r_group_cols = ['method', 'batch_size', 'replay_rate']
                if 'replay_strategy' in df.columns:
                    r_group_cols.append('replay_strategy')
                if 'replay_buffer_size' in df.columns:
                    r_group_cols.append('replay_buffer_size')
                r_group_cols = [c for c in r_group_cols if c in df.columns]
                if r_group_cols and metric_cols:
                    replay_ablation = df.groupby(r_group_cols)[metric_cols].mean().reset_index()
                    replay_file = output_path.with_name('replay_ablation.csv')
                    replay_ablation.to_csv(replay_file, index=False, encoding="utf-8")
                    self.logger.info(f"Replay ablation saved to {replay_file}")
        except Exception as e:
            self.logger.warning(f"Failed to save ablation CSVs: {e}")

        # Create figures folder next to output CSV
        figs_dir = output_path.parent / "figs"
        figs_dir.mkdir(parents=True, exist_ok=True)
        # If visualization utilities available, create summary plots
        if viz is not None:
            try:
                saved = viz.create_summary_plots(df, str(figs_dir))
                # Additional LoRA-native specific plots if fields available
                more = viz.create_lora_plots(df, str(figs_dir)) if hasattr(viz, 'create_lora_plots') else []
                replay_figs = viz.create_replay_plots(df, str(figs_dir)) if hasattr(viz, 'create_replay_plots') else []
                # Trust plots based on events, loaded below
                trust_figs = []
                if more:
                    saved.extend(more)
                if replay_figs:
                    saved.extend(replay_figs)
                # Load trust events across runs and plot
                trust_df = self._load_all_trust_events()
                if trust_df is not None and not trust_df.empty and hasattr(viz, 'create_trust_plots'):
                    trust_figs = viz.create_trust_plots(trust_df, str(figs_dir))
                    if trust_figs:
                        saved.extend(trust_figs)
                if saved:
                    self.logger.info(f"Saved {len(saved)} figures to {figs_dir}")
            except Exception as e:
                self.logger.warning(f"Visualization failed: {e}")

        # Save trust events aggregation CSVs if available, and join with run-level metrics
        try:
            trust_df = self._load_all_trust_events()
            if trust_df is not None and not trust_df.empty:
                trust_events_csv = output_path.with_name('trust_events.csv')
                trust_df.to_csv(trust_events_csv, index=False, encoding="utf-8")
                # Summary per run
                g = trust_df.groupby('run_dir')
                summary = pd.DataFrame({
                    'run_dir': g.size().index,
                    'events': g.size().values,
                    'trust_score_mean': g['trust_score'].mean().values,
                    'trust_score_std': g['trust_score'].std().values,
                    'rollback_count': g.apply(lambda x: (x.get('trust_action', pd.Series(dtype=str)) == 'rollback').sum()).values,
                    'scale_count': g.apply(lambda x: (x.get('trust_action', pd.Series(dtype=str)) == 'scale').sum()).values,
                    'applied_count': g.apply(lambda x: x.get('trust_applied', pd.Series(dtype=bool)).fillna(False).sum()).values
                })
                trust_summary_csv = output_path.with_name('trust_summary.csv')
                summary.to_csv(trust_summary_csv, index=False, encoding="utf-8")
                self.logger.info(f"Trust events saved to {trust_events_csv} and {trust_summary_csv}")

                # Join with run-level metrics for downstream correlation plots
                try:
                    # df contains run-level metrics and run_dir, ensure column exists
                    if 'run_dir' in df.columns:
                        trust_joined = df.merge(summary, on='run_dir', how='left')
                        trust_joined_csv = output_path.with_name('trust_with_metrics.csv')
                        trust_joined.to_csv(trust_joined_csv, index=False, encoding="utf-8")
                        self.logger.info(f"Trust + metrics joined CSV saved to {trust_joined_csv}")
                        # Save a thin version with just key metrics for easier plotting elsewhere
                        key_cols = [c for c in ['run_dir','method','batch_size','replay_rate',
                                                'efficacy_success','paraphrase_success','neighborhood_specificity','composite_score',
                                                'trust_score_mean','trust_score_std','events'] if c in trust_joined.columns]
                        if key_cols:
                            trust_key = trust_joined[key_cols]
                            trust_key_csv = output_path.with_name('trust_with_metrics_key.csv')
                            trust_key.to_csv(trust_key_csv, index=False, encoding="utf-8")
                except Exception as _e_join:
                    self.logger.warning(f"Failed to join trust summary with metrics: {_e_join}")
        except Exception as e:
            self.logger.warning(f"Failed to aggregate trust events: {e}")
    
    def generate_report(self, df):
        """Generate text report"""
        if df is None or df.empty:
            return
        
        report_file = Path(self.output_file).with_suffix('.txt')
        
        with open(report_file, 'w', encoding="utf-8") as f:
            f.write("="*80 + "\n")
            f.write("EXPERIMENTAL RESULTS REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Total Experiments: {len(df)}\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Summary by method and replay_rate
            if "method" in df.columns:
                f.write("-"*80 + "\n")
                f.write("Results by Method:\n")
                f.write("-"*80 + "\n")
                
                # Group by method and replay_rate if available
                group_cols = ["method"]
                if "replay_rate" in df.columns:
                    group_cols.append("replay_rate")
                if "batch_size" in df.columns:
                    group_cols.append("batch_size")
                
                for group_key, group in df.groupby(group_cols):
                    if isinstance(group_key, tuple):
                        group_name = "_".join([f"{k}={v}" for k, v in zip(group_cols, group_key)])
                    else:
                        group_name = f"{group_cols[0]}={group_key}"
                    
                    f.write(f"\n{group_name}:\n")
                    f.write(f"  Count: {len(group)}\n")
                    if "efficacy_success" in group.columns:
                        f.write(f"  Efficacy Success (ES): {group['efficacy_success'].mean():.4f} ± {group['efficacy_success'].std():.4f}\n")
                    if "paraphrase_success" in group.columns:
                        f.write(f"  Paraphrase Success (PS): {group['paraphrase_success'].mean():.4f} ± {group['paraphrase_success'].std():.4f}\n")
                    if "neighborhood_specificity" in group.columns:
                        f.write(f"  Neighborhood Specificity (NS): {group['neighborhood_specificity'].mean():.4f} ± {group['neighborhood_specificity'].std():.4f}\n")
                    if "composite_score" in group.columns:
                        f.write(f"  Composite Score (S): {group['composite_score'].mean():.4f} ± {group['composite_score'].std():.4f}\n")
        
        self.logger.info(f"Report saved to {report_file}")
    
    def run(self):
        """Run complete analysis"""
        self.logger.info("Starting results analysis...")
        
        # Aggregate results
        df = self.aggregate_results()
        
        if df is None or df.empty:
            self.logger.error("No results to analyze")
            return False
        
        # Compute statistics
        stats = self.compute_statistics(df)
        
        # Save results
        self.save_results(df, stats)
        
        # Generate report
        self.generate_report(df)
        # Also create plots (redundant with save_results but ensures figs exist)
        try:
            figs_dir = Path(self.output_file).parent / "figs"
            figs_dir.mkdir(parents=True, exist_ok=True)
            if viz is not None:
                viz.create_summary_plots(df, str(figs_dir))
                if hasattr(viz, 'create_lora_plots'):
                    viz.create_lora_plots(df, str(figs_dir))
                if hasattr(viz, 'create_replay_plots'):
                    viz.create_replay_plots(df, str(figs_dir))
                if hasattr(viz, 'create_trust_plots'):
                    trust_df = self._load_all_trust_events()
                    if trust_df is not None and not trust_df.empty:
                        viz.create_trust_plots(trust_df, str(figs_dir))
        except Exception:
            pass
        
        self.logger.info("Analysis completed!")
        return True


def main():
    parser = argparse.ArgumentParser(description="Analyze experimental results")
    
    parser.add_argument("--results_dir", type=str, default="results",
                       help="Results directory")
    parser.add_argument("--output", type=str, default=None,
                       help="Output CSV file")
    parser.add_argument("--log_file", type=str, default=None,
                       help="Optional log file path; if unset, logs/ with timestamped name is used")
    
    args = parser.parse_args()
    
    analyzer = ResultsAnalyzer(args.results_dir, args.output, args.log_file)
    success = analyzer.run()
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())

    
