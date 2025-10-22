#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "${PROJECT_ROOT}"

if command -v uv >/dev/null 2>&1; then
  echo "[build] Using uv to build project artifacts."
  uv build
  exit 0
fi

if command -v python3 >/dev/null 2>&1; then
  PY_CMD="python3"
elif command -v python >/dev/null 2>&1; then
  PY_CMD="python"
else
  echo "[build] Python interpreter not found." >&2
  exit 1
fi

if ! "${PY_CMD}" -m build --help >/dev/null 2>&1; then
  echo "[build] Installing missing 'build' module via pip." >&2
  "${PY_CMD}" -m pip install --upgrade build
fi

echo "[build] Using python -m build to create distributions."
"${PY_CMD}" -m build
