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
from pathlib import Path
import logging
from datetime import datetime


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
        
        if not config_file.exists() or not metrics_file.exists():
            self.logger.warning(f"Incomplete results in {result_dir}")
            return None
        
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        # Combine config and metrics
        data = {
            "run_dir": str(result_dir),
            **config,
            **metrics
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
        
        # Group by method and model
        group_cols = ["method", "model"]
        
        if all(col in df.columns for col in group_cols):
            stats = df.groupby(group_cols).agg({
                "efficacy_success": ["mean", "std", "count"],
                "paraphrase_success": ["mean", "std"],
                "neighborhood_specificity": ["mean", "std"],
                "generation_entropy": ["mean", "std"],
                "success_rate": ["mean", "std"]
            })
            
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
            
            # Summary by method
            if "method" in df.columns:
                f.write("-"*80 + "\n")
                f.write("Results by Method:\n")
                f.write("-"*80 + "\n")
                for method, group in df.groupby("method"):
                    f.write(f"\n{method.upper()}:\n")
                    f.write(f"  Count: {len(group)}\n")
                    if "efficacy_success" in group.columns:
                        f.write(f"  Efficacy Success: {group['efficacy_success'].mean():.4f} ± {group['efficacy_success'].std():.4f}\n")
                    if "neighborhood_specificity" in group.columns:
                        f.write(f"  Neighborhood Specificity: {group['neighborhood_specificity'].mean():.4f} ± {group['neighborhood_specificity'].std():.4f}\n")
        
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
