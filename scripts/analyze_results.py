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
    
    def __init__(self, results_dir, output_file=None):
        self.results_dir = Path(results_dir)
        self.output_file = output_file or f"results/analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def find_result_dirs(self):
        """Find all result directories"""
        result_dirs = []
        
        for path in self.results_dir.rglob("config.json"):
            result_dirs.append(path.parent)
        
        self.logger.info(f"Found {len(result_dirs)} result directories")
        return result_dirs
    
    def load_experiment_data(self, result_dir):
        """Load data from a single experiment"""
        config_file = result_dir / "config.json"
        metrics_file = result_dir / "metrics.json"
        edits_file = result_dir / "edit_results.json"
        
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

        # Combine config and metrics
        data = {
            "run_dir": str(result_dir),
            **config,
            **metrics,
            "lora_residual_rel_mean": lora_residual_rel_mean,
            "lora_residual_rel_count": lora_residual_rel_count,
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
        
        df.to_csv(output_path, index=False)
        self.logger.info(f"Detailed results saved to {output_path}")
        
        # Save statistics
        if stats is not None:
            stats_file = output_path.with_name(output_path.stem + "_stats.csv")
            stats.to_csv(stats_file)
            self.logger.info(f"Statistics saved to {stats_file}")

        # If replay columns exist, emit a compact ablation CSV grouped by replay_rate/strategy/buffer
        try:
            if 'replay_rate' in df.columns:
                group_cols = ['method', 'batch_size', 'replay_rate']
                if 'replay_strategy' in df.columns:
                    group_cols.append('replay_strategy')
                if 'replay_buffer_size' in df.columns:
                    group_cols.append('replay_buffer_size')
                metrics = [c for c in ['efficacy_success','paraphrase_success','neighborhood_specificity','composite_score'] if c in df.columns]
                if metrics:
                    ablation = df.groupby(group_cols)[metrics].mean().reset_index()
                    ablation_file = output_path.with_name('replay_ablation.csv')
                    ablation.to_csv(ablation_file, index=False)
                    self.logger.info(f"Replay ablation saved to {ablation_file}")
        except Exception as e:
            self.logger.warning(f"Failed to save replay ablation CSV: {e}")

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
                if more:
                    saved.extend(more)
                if replay_figs:
                    saved.extend(replay_figs)
                if saved:
                    self.logger.info(f"Saved {len(saved)} figures to {figs_dir}")
            except Exception as e:
                self.logger.warning(f"Visualization failed: {e}")
    
    def generate_report(self, df):
        """Generate text report"""
        if df is None or df.empty:
            return
        
        report_file = Path(self.output_file).with_suffix('.txt')
        
        with open(report_file, 'w') as f:
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
    
    args = parser.parse_args()
    
    analyzer = ResultsAnalyzer(args.results_dir, args.output)
    success = analyzer.run()
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
