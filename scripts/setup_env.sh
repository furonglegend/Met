#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
PYPROJECT_PATH="${PROJECT_ROOT}/pyproject.toml"
if [[ ! -f "${PYPROJECT_PATH}" ]]; then
  echo "[setup] pyproject.toml not found." >&2
  exit 1
fi

if command -v mamba >/dev/null 2>&1; then
  PKG_MGR="mamba"
elif command -v conda >/dev/null 2>&1; then
  PKG_MGR="conda"
else
  echo "[setup] Neither mamba nor conda is available in PATH." >&2
  exit 1
fi

detect_gpu() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    if nvidia-smi >/dev/null 2>&1; then
      return 0
    fi
  fi
  return 1
}

OS_NAME="$(uname -s || echo unknown)"
REQUESTED_DEVICE="${DEVICE:-auto}"
if [[ "${REQUESTED_DEVICE}" == "auto" ]]; then
  if [[ "${OS_NAME}" == "Linux" || "${OS_NAME}" == "Windows" ]]; then
    if detect_gpu; then
      REQUESTED_DEVICE="gpu"
    else
      REQUESTED_DEVICE="cpu"
    fi
  else
    REQUESTED_DEVICE="cpu"
  fi
fi

if [[ "${REQUESTED_DEVICE}" != "cpu" && "${REQUESTED_DEVICE}" != "gpu" ]]; then
  echo "[setup] Unsupported DEVICE value: ${REQUESTED_DEVICE}." >&2
  exit 1
fi

if command -v python >/dev/null 2>&1 && python -c "import sys" >/dev/null 2>&1; then
  PY_CMD=(python)
elif command -v py >/dev/null 2>&1 && py -3 -c "import sys" >/dev/null 2>&1; then
  PY_CMD=(py -3)
elif command -v python3 >/dev/null 2>&1 && python3 -c "import sys" >/dev/null 2>&1; then
  PY_CMD=(python3)
else
  echo "[setup] Python interpreter not found." >&2
  exit 1
fi

TMP_ENV_FILE="$(mktemp)"
TMP_ENV_LIST="$(mktemp)"
cleanup() {
  if [[ "${DEBUG_SETUP:-0}" != "1" ]]; then
    rm -f "${TMP_ENV_FILE}" "${TMP_ENV_LIST}"
  fi
}
trap cleanup EXIT

set +e
ENV_NAME="$(${PY_CMD[@]} - "$PYPROJECT_PATH" "${REQUESTED_DEVICE}" "${TMP_ENV_FILE}" <<'PY'
import ast
import pathlib
import sys

pyproject = pathlib.Path(sys.argv[1])
device = sys.argv[2]
dest = pathlib.Path(sys.argv[3])

content = pyproject.read_text(encoding="utf-8")
sections: dict[str, list[str]] = {}
current = None
for line in content.splitlines():
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        continue
    if stripped.startswith("[") and stripped.endswith("]"):
        current = stripped.strip("[]")
        sections[current] = []
    elif current is not None:
        sections[current].append(line)

def parse_assignments(lines):
    result = {}
    active_key = None
    buffer = []
    for raw in lines:
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if active_key is not None:
            buffer.append(stripped)
            if stripped.endswith("]"):
                result[active_key] = "\n".join(buffer)
                active_key = None
                buffer = []
            continue
        if "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip()
        if value.startswith("[") and not value.endswith("]"):
            active_key = key
            buffer = [value]
        else:
            result[key] = value
    if active_key is not None:
        raise SystemExit(f"Unterminated array for key {active_key}.")
    return result

def parse_value(raw: str):
    raw = raw.strip()
    if not raw:
        return ""
    try:
        return ast.literal_eval(raw)
    except Exception:
        return raw

def normalize_pkg(spec: str) -> str:
    for separator in ("==", ">=", "<=", "~=", "!=", ">", "<", "="):
        if separator in spec:
            return spec.split(separator, 1)[0].strip()
    return spec.strip()

project_data = parse_assignments(sections.get("project", []))
dependencies = list(parse_value(project_data.get("dependencies", "[]")))

env_data = parse_assignments(sections.get("tool.emmet.env", []))
env_name = parse_value(env_data.get("name", '"emmet-edit"'))
python_version = parse_value(env_data.get("python", '"3.9"'))
channels = list(parse_value(env_data.get("channels", '["pytorch", "conda-forge", "defaults"]')))
pip_exclude = set(parse_value(env_data.get("pip_exclude", "[]")))

if device == "gpu" and "nvidia" not in channels:
    if channels:
        channels = [channels[0], "nvidia", *channels[1:]]
    else:
        channels = ["nvidia"]

target_data = parse_assignments(sections.get(f"tool.emmet.env.{device}", []))
target_table = {key: str(parse_value(value)).strip() for key, value in target_data.items()}

conda_deps = [f"python={python_version}", "pip"]
for pkg_name, version in target_table.items():
    entry = f"{pkg_name}={version}" if version else pkg_name
    conda_deps.append(entry)

pip_deps = [spec for spec in dependencies if normalize_pkg(spec) not in pip_exclude]

lines = [f"name: {env_name}", "channels:"]
for channel in channels:
    lines.append(f"  - {channel}")
lines.append("dependencies:")
for dep in conda_deps:
    lines.append(f"  - {dep}")
if pip_deps:
    lines.append("  - pip:")
    for dep in pip_deps:
        lines.append(f"    - {dep}")

dest.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(env_name)
PY
)"
PY_STATUS=$?
set -e
if [[ ${PY_STATUS} -ne 0 ]]; then
  echo "[setup] Failed to parse pyproject.toml (exit ${PY_STATUS})." >&2
  exit ${PY_STATUS}
fi
ENV_NAME="${ENV_NAME//$'\r'/}"
ENV_NAME="${ENV_NAME//$'\n'/}"
if [[ -z "${ENV_NAME}" ]]; then
  echo "[setup] Failed to resolve environment name from pyproject.toml." >&2
  exit 1
fi

if [[ "${DEBUG_SETUP:-0}" == "1" ]]; then
  echo "[setup] Generated environment specification (${TMP_ENV_FILE}):"
  cat "${TMP_ENV_FILE}"
fi

env_exists=false
if "${PKG_MGR}" env list --json >"${TMP_ENV_LIST}" 2>/dev/null; then
  if ${PY_CMD[@]} - "$ENV_NAME" "${TMP_ENV_LIST}" <<'PY'
import json
import pathlib
import sys

env_name = sys.argv[1]
path = pathlib.Path(sys.argv[2])
data = json.loads(path.read_text(encoding="utf-8"))
if any(pathlib.Path(p).name == env_name for p in data.get("envs", [])):
    sys.exit(0)
sys.exit(1)
PY
  then
    env_exists=true
  fi
else
  if "${PKG_MGR}" env list | awk '{print $1}' | tr -d '*' | grep -Fx "${ENV_NAME}" >/dev/null 2>&1; then
    env_exists=true
  fi
fi

if [[ "${env_exists}" == true ]]; then
  echo "[setup] Updating environment ${ENV_NAME} with ${PKG_MGR}."
  "${PKG_MGR}" env update --name "${ENV_NAME}" --file "${TMP_ENV_FILE}" --prune
else
  echo "[setup] Creating environment ${ENV_NAME} with ${PKG_MGR}."
  "${PKG_MGR}" env create --name "${ENV_NAME}" --file "${TMP_ENV_FILE}"
fi

echo "[setup] Done. Activate the environment via: conda activate ${ENV_NAME}"
