"""
Quick Test for run_baseline.py
Test with minimal data (10 examples) to verify everything works

Usage:
    python scripts/test_baseline.py
"""

import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

print("="*80)
print("Quick Test: run_baseline.py")
print("="*80)
print("This will run a minimal test with 10 examples to verify the setup")
print()

# Test parameters
test_params = {
    "method": "emmet",
    "model": "gpt2",
    "num_edits": 10,
    "batch_size": 2,
    "seed": 42,
    "output_dir": "results/test"
}

print("Test parameters:")
for key, value in test_params.items():
    print(f"  {key}: {value}")
print()

# Build command
cmd = [
    sys.executable,
    "scripts/run_baseline.py",
    "--method", test_params["method"],
    "--model", test_params["model"],
    "--num_edits", str(test_params["num_edits"]),
    "--batch_size", str(test_params["batch_size"]),
    "--seed", str(test_params["seed"]),
    "--output_dir", test_params["output_dir"]
]

print("Running command:")
print(" ".join(cmd))
print()
print("="*80)
print()

# Run
try:
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
    print()
    print("="*80)
    print("✅ TEST PASSED!")
    print("="*80)
    print("The baseline script is working correctly.")
    print(f"Check results in: {test_params['output_dir']}/")
    sys.exit(0)
except subprocess.CalledProcessError as e:
    print()
    print("="*80)
    print("❌ TEST FAILED!")
    print("="*80)
    print(f"Error code: {e.returncode}")
    print("Please check the error messages above.")
    sys.exit(1)
