# EMMET Stability Replay

Stability-focused extensions to EMMET model-editing workflows.

## Project Structure

```plaintext
llm_project/
├── LICENSE
├── README.md
├── pyproject.toml        # Project configuration: dependencies, build tools
├── Makefile              # Automation tasks: make env, make install, make test
├── .gitignore
├── data/                 # Data directory (do not commit to Git)
│   ├── raw/              # Raw datasets
│   ├── processed/        # Processed datasets
│   └── embeddings/       # Precomputed vectors
├── configs/              # Configuration files
│   ├── model.yaml        # Model parameters (temperature, top_p)
│   └── prompts/          # Prompt templates
├── models/               # Model files
│   ├── checkpoints/      # Fine-tuned weights
│   └── download_scripts/
├── src/                  # Source package
│   ├── main.py
│   ├── core.py
│   ├── data/
│   │   └── make_dataset.py
│   ├── features/
│   │   └── build_embeddings.py
│   ├── models/
│   │   └── train_model.py
│   ├── integrations/
│   │   ├── huggingface.py
│   │   └── openai.py
│   └── utils/
│       └── logging.py
├── tests/
│   ├── test_core.py
│   └── conftest.py
├── docs/
│   └── api.md
└── notebooks/
    └── exploration.ipynb

## Environment Requirements

**All platforms:**

- Install [Miniforge](https://github.com/conda-forge/miniforge) (recommended; includes `mamba`).

**Linux / macOS:**

- ✅ `make` is typically available by default.

**Windows:**

- Install GNU Make:

```powershell
choco install make
# or
winget install GnuWin32.Make


## Quick Start

```bash
# 1. Clone the repo
git clone <repo-url>
cd emmet-stability-replay

# 2. Create the environment (auto-detects GPU/CPU)
# (e.g., `make env` or `make env-gpu` / `make env-cpu` depending on your setup)

# 3. Install project (development mode) + visualization deps
make install
pip install matplotlib seaborn pandas

# 4. Activate the environment
conda activate emmet-edit

# 5. Verify installation
python -c "from data import make_dataset; print('✅ Installation successful')"

```
# Running Experiments & Visualizations

## Three baseline comparison experiments (ROME / MEMIT / EMMET)

```bash
# Windows
scripts\run_all_baselines.cmd

# Linux / macOS
bash scripts/run_all_baselines.sh

```

**Auto-generated outputs:**
- `results/baseline_comparison/baseline_comparison.csv` – Aggregated results
- `results/baseline_comparison/figs/*.png` – Visualization charts

### View Visualization Results

After the experiment completes, charts are saved in the `results/baseline_comparison/figs/` directory:


# List all charts
ls results/baseline_comparison/figs/

# Windows: open folder
explorer results\baseline_comparison\figs

# Linux: open folder
xdg-open results/baseline_comparison/figs/



**Generated Charts:**
- `efficacy_success_by_method.png` – ES metric comparison
- `paraphrase_success_by_method.png` – PS metric comparison
- `neighborhood_specificity_by_method.png` – NS metric comparison
- `composite_score_by_method.png` – Composite score comparison
- `composite_score_by_batch_size.png` – Impact of batch size
- `composite_vs_batch_size.png` – Trend analysis

### Manual Result Analysis


# Analyze experiment results in a specific directory
python scripts/analyze_results.py --results_dir results/baseline_comparison

# Specify output file
python scripts/analyze_results.py --results_dir results/baseline --output my_analysis.csv


## Common Commands

| Command | Description |
|---------|-------------|
| `make env` | Create/update environment (auto-detect GPU/CPU) |
| `make env-cpu` | Force CPU version |
| `make env-gpu` | Force GPU version |
| `make install` | Install in development mode (code changes take effect immediately) |
| `make test` | Run tests |
| `make build` | Build distribution package |
| `make clean` | Clean build artifacts |

## Development Notes

- **Development mode**: After `make install`, code changes apply immediately
- **Module import**: `from data import make_dataset` (simple flat structure)
- **Environment variables**: `DEVICE=cpu make env` to manually specify device
- **Packaging**: `make build` creates wheel package in the `dist/` directory

## Dependency Management

### Using uv


# Add a package
uv add <package-name>

# Add a specific version
uv add <package-name>==<version>

# Remove a package
uv remove <package-name>



### Without uv

When adding dependencies, use **pip** or **conda** to manage them manually, and then update the `dependencies` section in `pyproject.toml` accordingly.  

