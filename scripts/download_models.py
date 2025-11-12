"""
Model Download Script

Usage:
    python scripts/download_models.py --model gpt2-xl
    python scripts/download_models.py --model llama3.2-3b --cache-dir ./models
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys

# Model configurations
MODEL_CONFIGS = {
    "gpt2": "gpt2",
    "gpt2-xl": "gpt2-xl",
    "llama3.2-1b": "meta-llama/Llama-3.2-1B-Instruct",
    "llama3.2-3b": "meta-llama/Llama-3.2-3B-Instruct",
    "llama2-7b": "meta-llama/Llama-2-7b-hf",
}

def download_model(model_name: str, cache_dir: str = None):
    """Download and cache model weights"""
    
    if model_name not in MODEL_CONFIGS:
        print(f"Error: Unknown model '{model_name}'")
        print(f"Available models: {', '.join(MODEL_CONFIGS.keys())}")
        return False
    
    hf_name = MODEL_CONFIGS[model_name]
    print(f"Downloading {hf_name}...")
    
    try:
        # Download tokenizer
        print("Loading tokenizer...")
        tok = AutoTokenizer.from_pretrained(hf_name, cache_dir=cache_dir)
        
        # Download model
        print("Loading model weights...")
        model = AutoModelForCausalLM.from_pretrained(
            hf_name,
            cache_dir=cache_dir,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        
        print(f"Success! Model downloaded to cache.")
        return True
        
    except Exception as e:
        print(f"Error: {str(e)}")
        if "authentication" in str(e).lower() or "gated" in str(e).lower():
            print("\nThis model requires authentication. Run:")
            print("  huggingface-cli login")
        return False

def main():
    parser = argparse.ArgumentParser(description="Download model weights")
    parser.add_argument("--model", type=str, required=True, 
                       help="Model name (gpt2, gpt2-xl, llama3.2-3b, etc.)")
    parser.add_argument("--cache-dir", type=str, default=None,
                       help="Cache directory (optional)")
    
    args = parser.parse_args()
    success = download_model(args.model, args.cache_dir)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
