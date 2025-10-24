# Makefile for emmet-stability-replay project
# Cross-platform automation for environment setup, package installation, testing, and building
# 
# Platform Requirements:
#   - Linux/macOS: make (pre-installed)
#   - Windows: GNU Make (install via: winget install GnuWin32.Make or chocolatey)
#
# Usage:
#   make help        - Show all available commands
#   make env         - Create/update environment (auto-detect GPU/CPU)
#   make install     - Install project packages
#   make test        - Run unit tests
#   make build       - Build distribution packages

.PHONY: help env env-cpu env-gpu install test build clean lint format check-pyproject

# Detect OS
ifeq ($(OS),Windows_NT)
    DETECTED_OS := Windows
    RM := del /f /q
    RMDIR := rmdir /s /q
    MKDIR := mkdir
    PYTHON := python
else
    DETECTED_OS := $(shell uname -s)
    RM := rm -f
    RMDIR := rm -rf
    MKDIR := mkdir -p
    ifeq ($(DETECTED_OS),Darwin)
        PYTHON := python3
    else
        PYTHON := python3
    endif
endif

# Detect package manager: prefer mamba, fallback to conda
SHELL := /bin/bash
PKG_MGR := $(shell command -v mamba 2>/dev/null || command -v conda 2>/dev/null || echo "")
ifeq ($(PKG_MGR),)
    $(error Neither mamba nor conda is available in PATH. Please install Miniconda or Mambaforge)
endif

# Project configuration
PROJECT_ROOT := $(shell pwd)
PYPROJECT := pyproject.toml
ENV_NAME := emmet-edit
PYTHON_VERSION := 3.9.7
CHANNELS := pytorch conda-forge defaults

# Device detection: auto, cpu, or gpu
DEVICE ?= auto
ifeq ($(DEVICE),auto)
    GPU_DETECTED := $(shell command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1 && echo "yes" || echo "no")
    ifeq ($(GPU_DETECTED),yes)
        DEVICE := gpu
    else
        DEVICE := cpu
    endif
endif

# Default target: display help information
help:
	@echo "=========================================="
	@echo "  EMMET Stability Replay - Build System  "
	@echo "=========================================="
	@echo ""
	@echo "Detected OS: $(DETECTED_OS)"
	@echo "Package Manager: $(PKG_MGR)"
	@echo "Device Mode: $(DEVICE)"
	@echo ""
	@echo "Available targets:"
	@echo "  make env        - Create/update Conda/Mamba environment (auto-detect GPU/CPU)"
	@echo "  make env-cpu    - Create/update environment with CPU-only dependencies"
	@echo "  make env-gpu    - Create/update environment with GPU (CUDA) dependencies"
	@echo "  make install    - Install project in editable mode using uv or pip"
	@echo "  make test       - Run unit tests with pytest"
	@echo "  make build      - Build distribution packages (wheel/sdist)"
	@echo "  make clean      - Remove build artifacts and cache files"
	@echo "  make lint       - Run code linting checks"
	@echo "  make format     - Format code with black/isort"
	@echo ""
	@echo "Environment variables:"
	@echo "  DEVICE=cpu|gpu|auto  - Override device detection (default: auto)"
	@echo ""
	@echo "Examples:"
	@echo "  make env              # Auto-detect GPU and create environment"
	@echo "  DEVICE=cpu make env   # Force CPU-only environment"
	@echo "  make install          # Install project after environment is ready"
	@echo ""

# Check if pyproject.toml exists
check-pyproject:
	@if [ ! -f "$(PYPROJECT)" ]; then \
		echo "[make] Error: $(PYPROJECT) not found" >&2; \
		exit 1; \
	fi

# Environment setup target with auto device detection
env: check-pyproject
	@echo "[make] =========================================="
	@echo "[make] Setting up environment: $(ENV_NAME)"
	@echo "[make] Device mode: $(DEVICE)"
	@echo "[make] Package manager: $(PKG_MGR)"
	@echo "[make] =========================================="
	@$(MAKE) -s _create_env_file
	@$(MAKE) -s _setup_env
	@echo "[make] =========================================="
	@echo "[make] Environment setup complete!"
	@echo "[make] Activate with: conda activate $(ENV_NAME)"
	@echo "[make] =========================================="

# CPU-only environment
env-cpu: check-pyproject
	@$(MAKE) env DEVICE=cpu

# GPU environment with CUDA support
env-gpu: check-pyproject
	@$(MAKE) env DEVICE=gpu

# Internal: Create environment YAML from pyproject.toml
_create_env_file:
	@echo "[make] Parsing pyproject.toml and generating environment specification..."
	@$(PYTHON) -c '\
import sys; \
import pathlib; \
import ast; \
\
pyproject = pathlib.Path("$(PYPROJECT)"); \
device = "$(DEVICE)"; \
\
content = pyproject.read_text(encoding="utf-8"); \
sections = {}; \
current = None; \
for line in content.splitlines(): \
    stripped = line.strip(); \
    if not stripped or stripped.startswith("#"): \
        continue; \
    if stripped.startswith("[") and stripped.endswith("]"): \
        current = stripped.strip("[]"); \
        sections[current] = []; \
    elif current is not None: \
        sections[current].append(line); \
\
def parse_assignments(lines): \
    result = {}; \
    active_key = None; \
    buffer = []; \
    for raw in lines: \
        stripped = raw.strip(); \
        if not stripped or stripped.startswith("#"): \
            continue; \
        if active_key is not None: \
            buffer.append(stripped); \
            if stripped.endswith("]"): \
                result[active_key] = "\n".join(buffer); \
                active_key = None; \
                buffer = []; \
            continue; \
        if "=" not in stripped: \
            continue; \
        key, value = stripped.split("=", 1); \
        key = key.strip(); \
        value = value.strip(); \
        if value.startswith("[") and not value.endswith("]"): \
            active_key = key; \
            buffer = [value]; \
        else: \
            result[key] = value; \
    return result; \
\
def parse_value(raw): \
    raw = raw.strip(); \
    if not raw: \
        return ""; \
    try: \
        return ast.literal_eval(raw); \
    except: \
        return raw; \
\
env_data = parse_assignments(sections.get("tool.emmet.env", [])); \
env_name = parse_value(env_data.get("name", "\"emmet-edit\"")); \
python_version = parse_value(env_data.get("python", "\"3.9.7\"")); \
channels = list(parse_value(env_data.get("channels", "[\"pytorch\", \"conda-forge\", \"defaults\"]"))); \
\
if device == "gpu" and "nvidia" not in channels: \
    channels = [channels[0], "nvidia"] + channels[1:] if channels else ["nvidia"]; \
\
target_data = parse_assignments(sections.get(f"tool.emmet.env.{device}", [])); \
\
yaml_lines = [f"name: {env_name}", "channels:"]; \
for channel in channels: \
    yaml_lines.append(f"  - {channel}"); \
yaml_lines.append("dependencies:"); \
yaml_lines.append(f"  - python={python_version}"); \
yaml_lines.append("  - pip"); \
for pkg_name, version in target_data.items(): \
    version_str = parse_value(version); \
    if version_str: \
        yaml_lines.append(f"  - {pkg_name}={version_str}"); \
    else: \
        yaml_lines.append(f"  - {pkg_name}"); \
\
pathlib.Path(".env_temp.yml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8"); \
print(env_name); \
' > .env_name.tmp
	@echo "[make] Environment specification created: .env_temp.yml"

# Internal: Setup conda environment
_setup_env:
	@ENV_NAME=$$(cat .env_name.tmp); \
	if $(PKG_MGR) env list | grep -q "^$$ENV_NAME "; then \
		echo "[make] Updating existing environment: $$ENV_NAME"; \
		$(PKG_MGR) env update --name $$ENV_NAME --file .env_temp.yml --prune; \
	else \
		echo "[make] Creating new environment: $$ENV_NAME"; \
		$(PKG_MGR) env create --file .env_temp.yml; \
	fi
	@$(MAKE) -s _install_pip_deps
	@$(RM) .env_temp.yml .env_name.tmp

# Internal: Install pip dependencies from pyproject.toml
_install_pip_deps:
	@echo "[make] Installing pip dependencies from pyproject.toml..."
	@$(PYTHON) -c '\
import pathlib; \
import ast; \
\
pyproject = pathlib.Path("$(PYPROJECT)"); \
content = pyproject.read_text(encoding="utf-8"); \
sections = {}; \
current = None; \
for line in content.splitlines(): \
    stripped = line.strip(); \
    if not stripped or stripped.startswith("#"): \
        continue; \
    if stripped.startswith("[") and stripped.endswith("]"): \
        current = stripped.strip("[]"); \
        sections[current] = []; \
    elif current is not None: \
        sections[current].append(line); \
\
def parse_assignments(lines): \
    result = {}; \
    active_key = None; \
    buffer = []; \
    for raw in lines: \
        stripped = raw.strip(); \
        if not stripped or stripped.startswith("#"): \
            continue; \
        if active_key is not None: \
            buffer.append(stripped); \
            if stripped.endswith("]"): \
                result[active_key] = "\n".join(buffer); \
                active_key = None; \
                buffer = []; \
            continue; \
        if "=" not in stripped: \
            continue; \
        key, value = stripped.split("=", 1); \
        key = key.strip(); \
        value = value.strip(); \
        if value.startswith("[") and not value.endswith("]"): \
            active_key = key; \
            buffer = [value]; \
        else: \
            result[key] = value; \
    return result; \
\
def parse_value(raw): \
    raw = raw.strip(); \
    try: \
        return ast.literal_eval(raw); \
    except: \
        return raw; \
\
project_data = parse_assignments(sections.get("project", [])); \
dependencies = list(parse_value(project_data.get("dependencies", "[]"))); \
env_data = parse_assignments(sections.get("tool.emmet.env", [])); \
pip_exclude = set(parse_value(env_data.get("pip_exclude", "[]"))); \
pip_exclude.add("pip"); \
\
def normalize_pkg(spec): \
    for sep in ("==", ">=", "<=", "~=", "!=", ">", "<", "="): \
        if sep in spec: \
            return spec.split(sep, 1)[0].strip(); \
    return spec.strip(); \
\
pip_deps = [spec for spec in dependencies if normalize_pkg(spec) not in pip_exclude]; \
if pip_deps: \
    pathlib.Path(".pip_requirements.tmp").write_text("\n".join(pip_deps) + "\n", encoding="utf-8"); \
' && if [ -f .pip_requirements.tmp ]; then \
		if command -v uv >/dev/null 2>&1; then \
			ENV_PYTHON=$$($(PKG_MGR) run -n $(ENV_NAME) python -c "import sys; print(sys.executable)" 2>/dev/null); \
			echo "[make] Using uv for pip installation"; \
			uv pip install --python "$$ENV_PYTHON" --upgrade -r .pip_requirements.tmp; \
		else \
			echo "[make] Using pip for installation"; \
			$(PKG_MGR) run -n $(ENV_NAME) python -m pip install --upgrade -r .pip_requirements.tmp; \
		fi; \
		$(RM) .pip_requirements.tmp; \
	else \
		echo "[make] No additional pip dependencies to install"; \
	fi

# Install project in editable mode
install:
	@echo "[make] Installing project in editable mode..."
	@if command -v uv >/dev/null 2>&1; then \
		ENV_PYTHON=$$($(PKG_MGR) run -n $(ENV_NAME) python -c "import sys; print(sys.executable)" 2>/dev/null); \
		if [ -n "$$ENV_PYTHON" ]; then \
			echo "[make] Using uv for installation"; \
			uv pip install --python "$$ENV_PYTHON" -e .; \
		else \
			echo "[make] Error: Failed to locate Python in environment $(ENV_NAME)" >&2; \
			exit 1; \
		fi \
	else \
		echo "[make] Using pip for installation"; \
		$(PKG_MGR) run -n $(ENV_NAME) python -m pip install -e .; \
	fi
	@echo "[make] Project installed successfully!"

# Run unit tests
test:
	@echo "[make] Running unit tests..."
	@if [ -d "tests" ] || [ -d "test" ]; then \
		$(PKG_MGR) run -n $(ENV_NAME) python -m pytest -v; \
	else \
		echo "[make] Warning: No tests directory found"; \
	fi

# Build distribution packages
build:
	@echo "[make] Building distribution packages..."
	@if command -v uv >/dev/null 2>&1; then \
		echo "[make] Using uv build"; \
		uv build; \
	else \
		echo "[make] Using python -m build"; \
		$(PKG_MGR) run -n $(ENV_NAME) python -m build; \
	fi
	@echo "[make] Build complete! Packages are in dist/"

# Clean build artifacts and cache
clean:
	@echo "[make] Cleaning build artifacts..."
	@$(RMDIR) build 2>/dev/null || true
	@$(RMDIR) dist 2>/dev/null || true
	@$(RMDIR) src/*.egg-info 2>/dev/null || true
	@find . -type d -name "__pycache__" -exec $(RMDIR) {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec $(RMDIR) {} + 2>/dev/null || true
	@$(RM) .env_temp.yml .env_name.tmp .pip_requirements.tmp 2>/dev/null || true
	@echo "[make] Clean complete!"

# Code linting
lint:
	@echo "[make] Running linting checks..."
	@$(PKG_MGR) run -n $(ENV_NAME) python -m flake8 --version >/dev/null 2>&1 && \
		$(PKG_MGR) run -n $(ENV_NAME) python -m flake8 src/ tests/ || \
		echo "[make] flake8 not installed, skipping lint"

# Code formatting
format:
	@echo "[make] Formatting code..."
	@$(PKG_MGR) run -n $(ENV_NAME) python -m black --version >/dev/null 2>&1 && \
		$(PKG_MGR) run -n $(ENV_NAME) python -m black src/ tests/ || \
		echo "[make] black not installed, skipping format"
	@$(PKG_MGR) run -n $(ENV_NAME) python -m isort --version >/dev/null 2>&1 && \
		$(PKG_MGR) run -n $(ENV_NAME) python -m isort src/ tests/ || \
		echo "[make] isort not installed, skipping import sorting"
