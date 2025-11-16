"""
Data Preparation Script
Sample and prepare data for experiments

Usage:
    python scripts/prepare_data.py --num 200 --seed 42
    python scripts/prepare_data.py --num 500 --seed 42 --output data/sample_500.json
"""

import argparse
import json
import random
from pathlib import Path
from datetime import datetime
import logging

PROJECT_ROOT = Path(__file__).parent.parent


def setup_logging(log_file: str | None = None) -> logging.Logger:
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


def main():
    parser = argparse.ArgumentParser(description="Prepare data samples for experiments")
    
    parser.add_argument("--num", type=int, required=True,
                       help="Number of examples to sample")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--input", type=str,
                       default="data/counterfact.json",
                       help="Input data file")
    parser.add_argument("--output", type=str, default=None,
                       help="Output file (default: data/counterfact_{num}.json)")
    parser.add_argument("--log_file", type=str, default=None,
                       help="Optional log file path; if unset, logs/ with timestamped name is used")
    
    args = parser.parse_args()

    # Default log path if not provided
    if args.log_file is None:
        input_stem = Path(args.input).stem
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.log_file = str(Path('logs') / f'prepare_data_{input_stem}_{args.num}_{timestamp}.log')

    logger = setup_logging(args.log_file)
    
    # Set paths
    input_path = PROJECT_ROOT / args.input
    if args.output:
        output_path = PROJECT_ROOT / args.output
    else:
        output_path = PROJECT_ROOT / f"data/counterfact_{args.num}.json"
    
    logger.info("%s", "="*80)
    logger.info("Data Preparation")
    logger.info("%s", "="*80)
    logger.info("Input:  %s", input_path)
    logger.info("Output: %s", output_path)
    logger.info("Sample size: %d", args.num)
    logger.info("Seed: %d", args.seed)
    
    # Load data
    logger.info("Loading data from %s...", input_path)
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle dict format
    if isinstance(data, dict):
        data = list(data.values())
    
    logger.info("Total examples available: %d", len(data))
    
    # Sample
    if args.num > len(data):
        logger.warning("⚠️  Warning: Requested %d examples but only %d available", args.num, len(data))
        sampled_data = data
    else:
        random.seed(args.seed)
        sampled_data = random.sample(data, args.num)
        logger.info("✅ Sampled %d examples", len(sampled_data))
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sampled_data, f, indent=2, ensure_ascii=False)
    
    logger.info("✅ Saved to %s", output_path)
    
    # Show sample
    if sampled_data:
        logger.info("Sample record:")
        logger.info("%s", "-"*80)
        sample = sampled_data[0]
        if isinstance(sample, dict) and "requested_rewrite" in sample:
            rw = sample["requested_rewrite"]
            logger.info("Subject: %s", rw.get('subject', 'N/A'))
            logger.info("Prompt: %s", rw.get('prompt', 'N/A'))
            logger.info("Target (new): %s", rw.get('target_new', {}).get('str', 'N/A'))
            logger.info("Target (true): %s", rw.get('target_true', {}).get('str', 'N/A'))
        logger.info("%s", "-"*80)
    
    logger.info("✅ Data preparation complete!")


if __name__ == "__main__":
    main()
