# Makefile for emmet-stability-replay project
# Cross-platform automation using uv for Python package management
# 
# Platform Requirements:
#   - Linux/macOS: make (pre-installed)
#   - Windows: GNU Make (install via: winget install GnuWin32.Make or chocolatey)
#   - uv: pip install uv or curl -LsSf https://astral.sh/uv/install.sh | sh
#
# Usage:
#   make help        - Show all available commands
#   make venv        - Create Python virtual environment using uv
#   make install     - Install project and dependencies
#   make sync        - Sync dependencies with pyproject.toml
#   make test        - Run unit tests
#   make build       - Build distribution packages

.PHONY: help venv install sync test build clean lint format check-uv

# Detect OS
ifeq ($(OS),Windows_NT)
    DETECTED_OS := Windows
    RM := del /f /q
    RMDIR := rmdir /s /q
    PYTHON := python
    VENV_BIN := .venv\Scripts
    VENV_PYTHON := $(VENV_BIN)\python.exe
    VENV_ACTIVATE := $(VENV_BIN)\activate.bat
else
    DETECTED_OS := $(shell uname -s)
    RM := rm -f
    RMDIR := rm -rf
    PYTHON := python3
    VENV_BIN := .venv/bin
    VENV_PYTHON := $(VENV_BIN)/python
    VENV_ACTIVATE := $(VENV_BIN)/activate
    ifeq ($(DETECTED_OS),Darwin)
        PYTHON := python3
    endif
endif

# Project configuration
PROJECT_ROOT := $(shell pwd)
PYPROJECT := pyproject.toml
PYTHON_VERSION := 3.9

# Check if uv is installed
check-uv:
	@command -v uv >/dev/null 2>&1 || (echo "[make] Error: uv is not installed. Install it with: pip install uv" && exit 1)

# Default target: display help information
help:
	@echo "=========================================="
	@echo "  EMMET Stability Replay - Build System  "
	@echo "=========================================="
	@echo ""
	@echo "Detected OS: $(DETECTED_OS)"
	@echo "Python Version: $(PYTHON_VERSION)"
	@echo ""
	@echo "Available targets:"
	@echo "  make venv       - Create Python virtual environment using uv"
	@echo "  make install    - Install project and all dependencies in editable mode"
	@echo "  make sync       - Sync dependencies with pyproject.toml"
	@echo "  make test       - Run unit tests with pytest"
	@echo "  make build      - Build distribution packages (wheel/sdist)"
	@echo "  make clean      - Remove build artifacts, cache files, and virtual environment"
	@echo "  make lint       - Run code linting checks (if configured)"
	@echo "  make format     - Format code (if configured)"
	@echo ""
	@echo "Quick Start:"
	@echo "  1. make venv     # Create virtual environment"
	@echo "  2. make install  # Install project and dependencies"
	@echo "  3. make test     # Run tests"
	@echo ""
	@echo "Activate environment:"
ifeq ($(OS),Windows_NT)
	@echo "  $(VENV_ACTIVATE)"
else
	@echo "  source $(VENV_ACTIVATE)"
endif
	@echo ""

# Create virtual environment using uv
venv: check-uv
	@echo "[make] Creating virtual environment with Python $(PYTHON_VERSION)..."
	@uv venv --python $(PYTHON_VERSION) .venv
	@echo "[make] Virtual environment created: .venv"
	@echo "[make] Activate with:"
ifeq ($(OS),Windows_NT)
	@echo "  $(VENV_ACTIVATE)"
else
	@echo "  source $(VENV_ACTIVATE)"
endif

# Install project and all dependencies in editable mode
install: check-uv
	@echo "[make] Installing project and dependencies..."
	@if [ ! -d ".venv" ]; then \
		echo "[make] Virtual environment not found. Creating one..."; \
		$(MAKE) venv; \
	fi
	@uv pip install -e .
	@echo "[make] =========================================="
	@echo "[make] Installation complete!"
	@echo "[make] Project installed in editable mode"
	@echo "[make] =========================================="

# Sync dependencies with pyproject.toml
sync: check-uv
	@echo "[make] Syncing dependencies with pyproject.toml..."
	@if [ ! -d ".venv" ]; then \
		echo "[make] Virtual environment not found. Creating one..."; \
		$(MAKE) venv; \
	fi
	@uv pip sync pyproject.toml
	@echo "[make] Dependencies synced successfully!"

# Run unit tests
test:
	@echo "[make] Running unit tests..."
	@if [ ! -d ".venv" ]; then \
		echo "[make] Error: Virtual environment not found. Run 'make venv' first." >&2; \
		exit 1; \
	fi
	@if [ -d "tests" ] || [ -d "test" ]; then \
		$(VENV_PYTHON) -m pytest -v; \
	else \
		echo "[make] Warning: No tests directory found"; \
	fi

# Build distribution packages
build: check-uv
	@echo "[make] Building distribution packages..."
	@uv build
	@echo "[make] =========================================="
	@echo "[make] Build complete!"
	@echo "[make] Packages are in dist/"
	@echo "[make] =========================================="

# Clean build artifacts, cache, and virtual environment
clean:
	@echo "[make] Cleaning build artifacts and cache files..."
ifeq ($(OS),Windows_NT)
	@if exist build $(RMDIR) build
	@if exist dist $(RMDIR) dist
	@if exist .venv $(RMDIR) .venv
	@if exist src\*.egg-info $(RMDIR) src\*.egg-info
	@for /d /r . %%d in (__pycache__) do @if exist "%%d" $(RMDIR) "%%d"
	@for /d /r . %%d in (.pytest_cache) do @if exist "%%d" $(RMDIR) "%%d"
	@del /s /q *.pyc 2>nul
	@del /s /q *.pyo 2>nul
else
	@$(RMDIR) build 2>/dev/null || true
	@$(RMDIR) dist 2>/dev/null || true
	@$(RMDIR) .venv 2>/dev/null || true
	@$(RMDIR) src/*.egg-info 2>/dev/null || true
	@find . -type d -name "__pycache__" -exec $(RMDIR) {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec $(RMDIR) {} + 2>/dev/null || true
endif
	@echo "[make] Clean complete!"

# Code linting (requires flake8 to be installed)
lint:
	@echo "[make] Running linting checks..."
	@if [ ! -d ".venv" ]; then \
		echo "[make] Error: Virtual environment not found. Run 'make venv' first." >&2; \
		exit 1; \
	fi
	@$(VENV_PYTHON) -m flake8 --version >/dev/null 2>&1 && \
		$(VENV_PYTHON) -m flake8 src/ tests/ || \
		echo "[make] flake8 not installed, skipping lint"

# Code formatting (requires black and isort to be installed)
format:
	@echo "[make] Formatting code..."
	@if [ ! -d ".venv" ]; then \
		echo "[make] Error: Virtual environment not found. Run 'make venv' first." >&2; \
		exit 1; \
	fi
	@$(VENV_PYTHON) -m black --version >/dev/null 2>&1 && \
		$(VENV_PYTHON) -m black src/ tests/ || \
		echo "[make] black not installed, skipping format"
	@$(VENV_PYTHON) -m isort --version >/dev/null 2>&1 && \
		$(VENV_PYTHON) -m isort src/ tests/ || \
		echo "[make] isort not installed, skipping import sorting"
