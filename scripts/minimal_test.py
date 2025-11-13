"""
Minimal Test Script - Environment Validation
验证环境配置是否正确，能否加载模型和数据

Usage:
    python scripts/minimal_test.py
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

print("=" * 80)
print("EMMET Stability Replay - Minimal Environment Test")
print("=" * 80)
print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Project root: {PROJECT_ROOT}")
print()

# Test results tracking
test_results = {
    "total": 0,
    "passed": 0,
    "failed": 0,
    "errors": []
}

def test_step(name, func):
    """Run a test step and track results"""
    test_results["total"] += 1
    print(f"[{test_results['total']}] Testing: {name}...", end=" ")
    try:
        func()
        print("✅ PASSED")
        test_results["passed"] += 1
        return True
    except Exception as e:
        print(f"❌ FAILED")
        print(f"    Error: {str(e)}")
        test_results["failed"] += 1
        test_results["errors"].append({
            "test": name,
            "error": str(e)
        })
        return False


# Test 1: Python version
def test_python_version():
    import sys
    version = sys.version_info
    print(f"\n    Python version: {version.major}.{version.minor}.{version.micro}")
    assert version.major == 3, f"Python 3.x required, got {version.major}.x"
    assert version.minor >= 9, f"Python 3.9+ required, got 3.{version.minor}"

test_step("Python Version (3.9+)", test_python_version)


# Test 2: PyTorch
def test_pytorch():
    import torch
    print(f"\n    PyTorch version: {torch.__version__}")
    print(f"    CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"    CUDA version: {torch.version.cuda}")
        print(f"    GPU count: {torch.cuda.device_count()}")
        print(f"    GPU name: {torch.cuda.get_device_name(0)}")
    else:
        print("    ⚠️  Warning: CUDA not available, will use CPU (slower)")

test_step("PyTorch Installation", test_pytorch)


# Test 3: Transformers
def test_transformers():
    import transformers
    print(f"\n    Transformers version: {transformers.__version__}")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("    AutoModelForCausalLM: OK")
    print("    AutoTokenizer: OK")

test_step("Transformers Library", test_transformers)


# Test 4: Other dependencies
def test_dependencies():
    import numpy as np
    import torch
    print(f"\n    NumPy version: {np.__version__}")
    
    # Test other key dependencies
    deps = ['datasets', 'nltk', 'matplotlib', 'scipy']
    for dep in deps:
        try:
            module = __import__(dep)
            version = getattr(module, '__version__', 'unknown')
            print(f"    {dep}: {version}")
        except ImportError:
            print(f"    {dep}: ⚠️  Not installed (optional)")
        except Exception as e:
            # Handle pyarrow compatibility issues with datasets
            if 'pyarrow' in str(e):
                print(f"    {dep}: ⚠️  Warning - pyarrow compatibility issue (can be ignored)")
            else:
                print(f"    {dep}: ⚠️  Error - {str(e)}")

test_step("Dependencies Check", test_dependencies)


# Test 5: Data files
def test_data_files():
    data_dir = PROJECT_ROOT / "data"
    print(f"\n    Data directory: {data_dir}")
    
    required_files = [
        "counterfact_sampled_unique_cf_10_20000.json",
        "counterfact_sampled_unique_mcf_10_20000.json"
    ]
    
    for filename in required_files:
        filepath = data_dir / filename
        if filepath.exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"    ✅ {filename}: {len(data)} examples")
        else:
            print(f"    ❌ {filename}: NOT FOUND")
            raise FileNotFoundError(f"Required data file not found: {filename}")

test_step("Data Files Availability", test_data_files)


# Test 6: Load small model (GPT-2)
def test_load_model():
    import torch
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    
    model_name = "gpt2"  # Smallest GPT-2 (124M parameters)
    print(f"\n    Loading model: {model_name}...")
    
    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    print(f"    ✅ Tokenizer loaded: {len(tokenizer)} tokens")
    
    # Load model
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    ✅ Model loaded: {total_params:,} parameters")
    print(f"    Trainable parameters: {trainable_params:,}")
    
    # Move to device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"    Device: {device}")
    
    # Test inference
    input_text = "The capital of France is"
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=20,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"    Test generation: '{generated_text}'")
    
    # Clean up
    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

test_step("Load GPT-2 Model and Generate", test_load_model)


# Test 7: Project structure
def test_project_structure():
    required_dirs = [
        "src/emmet",
        "src/memit",
        "src/rome",
        "src/utils",
        "src/hparams/EMMET",
        "scripts",
        "configs",
        "data"
    ]
    
    print()
    for dir_path in required_dirs:
        full_path = PROJECT_ROOT / dir_path
        if full_path.exists():
            print(f"    ✅ {dir_path}")
        else:
            print(f"    ❌ {dir_path} - NOT FOUND")
            raise FileNotFoundError(f"Required directory not found: {dir_path}")

test_step("Project Structure", test_project_structure)


# Test 8: Import project modules
def test_project_imports():
    print()
    
    # Test EMMET imports
    try:
        from emmet.emmet_hparams import EMMETHyperParams
        print("    ✅ emmet.emmet_hparams")
    except ImportError as e:
        print(f"    ❌ emmet.emmet_hparams: {e}")
        raise
    
    try:
        from emmet.emmet_main import apply_emmet_to_model
        print("    ✅ emmet.emmet_main")
    except ImportError as e:
        print(f"    ❌ emmet.emmet_main: {e}")
        raise
    
    # Test utils
    try:
        from utils.nethook import get_parameter
        print("    ✅ utils.nethook")
    except ImportError as e:
        print(f"    ❌ utils.nethook: {e}")
        raise
    
    try:
        from utils.globals import DATA_DIR, RESULTS_DIR
        print("    ✅ utils.globals")
    except ImportError as e:
        print(f"    ⚠️  utils.globals: {e} (may need configuration)")

test_step("Project Module Imports", test_project_imports)


# Test 9: Read sample data
def test_read_sample_data():
    data_file = PROJECT_ROOT / "data" / "counterfact_sampled_unique_cf_10_20000.json"
    
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"\n    Total examples: {len(data)}")
    
    # Check if data is dict or list
    if isinstance(data, dict):
        # Convert dict to list of values, take first 10
        data_list = list(data.values())
        sample = data_list[:10]
        print(f"    Data format: Dictionary (converted to list)")
    else:
        # Already a list
        sample = data[:10]
        print(f"    Data format: List")
    
    print(f"    Sample size: {len(sample)}")
    
    # Check data structure
    if len(sample) > 0:
        first_example = sample[0]
        print(f"    Example keys: {list(first_example.keys())}")
        
        if 'requested_rewrite' in first_example:
            rewrite = first_example['requested_rewrite']
            print(f"    Rewrite keys: {list(rewrite.keys())}")
            print(f"    Sample subject: {rewrite.get('subject', 'N/A')}")
            print(f"    Sample prompt: {rewrite.get('prompt', 'N/A')}")
            print(f"    Sample target: {rewrite.get('target_new', {}).get('str', 'N/A')}")
    
    return sample

sample_data = None
if test_step("Read Sample Data (10 examples)", test_read_sample_data):
    sample_data = test_read_sample_data()


# Print summary
print()
print("=" * 80)
print("TEST SUMMARY")
print("=" * 80)
print(f"Total tests: {test_results['total']}")
print(f"Passed: {test_results['passed']} ✅")
print(f"Failed: {test_results['failed']} ❌")
print()

if test_results['failed'] > 0:
    print("Failed tests:")
    for error in test_results['errors']:
        print(f"  - {error['test']}")
        print(f"    {error['error']}")
    print()
    print("❌ Environment validation FAILED")
    print("Please fix the issues above before proceeding.")
    sys.exit(1)
else:
    print("✅ All tests PASSED!")
    print()
    print("Environment is ready for EMMET experiments.")
    print()
    print("Next steps:")
    print("  1. Run baseline experiment:")
    print("     python scripts/run_baseline.py --method emmet --model gpt2 --num_edits 200")
    print()
    print("  2. Check available hyperparameters:")
    print("     ls src/hparams/EMMET/")
    print()
    
    # Save test results
    results_file = PROJECT_ROOT / "results" / "minimal_test_log.txt"
    results_file.parent.mkdir(exist_ok=True)
    
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write(f"Minimal Test Results\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Total tests: {test_results['total']}\n")
        f.write(f"Passed: {test_results['passed']}\n")
        f.write(f"Failed: {test_results['failed']}\n")
        f.write(f"\nStatus: {'PASSED' if test_results['failed'] == 0 else 'FAILED'}\n")
    
    print(f"Test log saved to: {results_file}")
    
    sys.exit(0)
