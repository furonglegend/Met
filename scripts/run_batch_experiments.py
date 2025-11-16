"""
Batch Experiment Runner
Run multiple experiments with different configurations

Usage:
    python scripts/run_batch_experiments.py --config configs/batch_config.json
    python scripts/run_batch_experiments.py --methods rome memit emmet --models gpt2-xl
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from itertools import product
import logging


class BatchExperimentRunner:
    """Run batch experiments with grid search"""
    
    def __init__(self, config_file=None, methods=None, models=None, 
                 num_edits_list=None, batch_sizes=None, replay_rates=None, seeds=None):
        self.setup_logging()
        
        if config_file:
            self.load_config(config_file)
        else:
            # Use command line arguments
            self.methods = methods or ["emmet"]
            self.models = models or ["gpt2"]
            self.num_edits_list = num_edits_list or [100]
            self.batch_sizes = batch_sizes or [1]
            self.replay_rates = replay_rates or [0.0]
            self.seeds = seeds or [42]
        
        self.results = []
        
    def setup_logging(self):
        """Setup logging"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"logs/batch_experiments_{timestamp}.log"
        
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def load_config(self, config_file):
        """Load configuration from JSON file"""
        self.logger.info(f"Loading config from {config_file}")
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        self.methods = config.get("methods", ["emmet"])
        self.models = config.get("models", ["gpt2"])
        self.num_edits_list = config.get("num_edits", [100])
        self.batch_sizes = config.get("batch_sizes", [1])
        self.replay_rates = config.get("replay_rates", [0.0])
        self.seeds = config.get("seeds", [42])
        self.dataset = config.get("dataset", "counterfact_sampled_unique_cf_10_20000")
        
    def generate_experiments(self):
        """Generate all experiment combinations"""
        experiments = []
        
        for method, model, num_edits, batch_size, replay_rate, seed in product(
            self.methods, self.models, self.num_edits_list, 
            self.batch_sizes, self.replay_rates, self.seeds
        ):
            exp = {
                "method": method,
                "model": model,
                "num_edits": num_edits,
                "batch_size": batch_size,
                "replay_rate": replay_rate,
                "seed": seed
            }
            experiments.append(exp)
        
        self.logger.info(f"Generated {len(experiments)} experiments")
        return experiments
    
    def run_experiment(self, exp):
        """Run a single experiment"""
        exp_name = f"{exp['method']}_{exp['model']}_n{exp['num_edits']}_b{exp['batch_size']}_r{exp['replay_rate']}_s{exp['seed']}"
        self.logger.info(f"Running experiment: {exp_name}")
        
        cmd = [
            "python", "scripts/run_baseline.py",
            "--method", exp["method"],
            "--model", exp["model"],
            "--num_edits", str(exp["num_edits"]),
            "--batch_size", str(exp["batch_size"]),
            "--replay_rate", str(exp["replay_rate"]),
            "--seed", str(exp["seed"])
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            self.logger.info(f"Experiment {exp_name} completed successfully")
            return {
                "experiment": exp,
                "status": "success",
                "stdout": result.stdout,
                "stderr": result.stderr
            }
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Experiment {exp_name} failed: {e}")
            return {
                "experiment": exp,
                "status": "failed",
                "stdout": e.stdout,
                "stderr": e.stderr
            }
    
    def run_all(self):
        """Run all experiments"""
        experiments = self.generate_experiments()
        
        self.logger.info(f"Starting batch experiments: {len(experiments)} total")
        
        for i, exp in enumerate(experiments, 1):
            self.logger.info(f"\n{'='*80}")
            self.logger.info(f"Experiment {i}/{len(experiments)}")
            self.logger.info(f"{'='*80}\n")
            
            result = self.run_experiment(exp)
            self.results.append(result)
        
        # Save summary
        self.save_summary()
        
    def save_summary(self):
        """Save experiment summary"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = f"results/batch_summary_{timestamp}.json"
        
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_experiments": len(self.results),
            "successful": sum(1 for r in self.results if r["status"] == "success"),
            "failed": sum(1 for r in self.results if r["status"] == "failed"),
            "results": self.results
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"\nBatch experiments completed!")
        self.logger.info(f"Summary saved to {summary_file}")
        self.logger.info(f"Total: {summary['total_experiments']}, "
                        f"Success: {summary['successful']}, "
                        f"Failed: {summary['failed']}")


def main():
    parser = argparse.ArgumentParser(description="Run batch experiments")
    
    parser.add_argument("--config", type=str,
                       help="Path to config JSON file")
    parser.add_argument("--methods", nargs="+", 
                       choices=["rome", "memit", "emmet"],
                       help="List of methods")
    parser.add_argument("--models", nargs="+",
                       help="List of models")
    parser.add_argument("--num_edits", nargs="+", type=int,
                       help="List of num_edits values")
    parser.add_argument("--batch_sizes", nargs="+", type=int,
                       help="List of batch sizes")
    parser.add_argument("--replay_rates", nargs="+", type=float,
                       help="List of replay rates (0.0-1.0)")
    parser.add_argument("--seeds", nargs="+", type=int,
                       help="List of random seeds")
    
    args = parser.parse_args()
    
    # Create and run batch experiments
    runner = BatchExperimentRunner(
        config_file=args.config,
        methods=args.methods,
        models=args.models,
        num_edits_list=args.num_edits,
        batch_sizes=args.batch_sizes,
        replay_rates=args.replay_rates,
        seeds=args.seeds
    )
    
    runner.run_all()


if __name__ == "__main__":
    main()
