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

PROJECT_ROOT = Path(__file__).parent.parent


def main():
    parser = argparse.ArgumentParser(description="Prepare data samples for experiments")
    
    parser.add_argument("--num", type=int, required=True,
                       help="Number of examples to sample")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--input", type=str,
                       default="data/counterfact_qna.json",
                       help="Input data file")
    parser.add_argument("--output", type=str, default=None,
                       help="Output file (default: data/counterfact_qna_sample_{num}.json)")
    
    args = parser.parse_args()
    
    # Set paths
    input_path = PROJECT_ROOT / args.input
    if args.output:
        output_path = PROJECT_ROOT / args.output
    else:
        output_path = PROJECT_ROOT / f"data/sample_{args.num}.json"
    
    print("="*80)
    print("Data Preparation")
    print("="*80)
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Sample size: {args.num}")
    print(f"Seed: {args.seed}")
    print()
    
    # Load data
    print(f"Loading data from {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle dict format
    if isinstance(data, dict):
        data = list(data.values())
    
    print(f"Total examples available: {len(data)}")
    
    # Sample
    if args.num > len(data):
        print(f"⚠️  Warning: Requested {args.num} examples but only {len(data)} available")
        sampled_data = data
    else:
        random.seed(args.seed)
        sampled_data = random.sample(data, args.num)
        print(f"✅ Sampled {len(sampled_data)} examples")
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sampled_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Saved to {output_path}")
    print()
    
    # Show sample
    if sampled_data:
        print("Sample record:")
        print("-"*80)
        sample = sampled_data[0]
        if isinstance(sample, dict) and "requested_rewrite" in sample:
            rw = sample["requested_rewrite"]
            print(f"Subject: {rw.get('subject', 'N/A')}")
            print(f"Prompt: {rw.get('prompt', 'N/A')}")
            print(f"Target (new): {rw.get('target_new', {}).get('str', 'N/A')}")
            print(f"Target (true): {rw.get('target_true', {}).get('str', 'N/A')}")
        print("-"*80)
    
    print("\n✅ Data preparation complete!")


if __name__ == "__main__":
    main()
