from pathlib import Path
import yaml


# Find the project root (where globals.yml is located)
# This file is in src/utils/, so go up two levels to project root
_current_file = Path(__file__).resolve()
_project_root = _current_file.parent.parent.parent
_globals_file = _project_root / "globals.yml"

with open(_globals_file, "r") as stream:
    data = yaml.safe_load(stream)

# Convert relative paths to absolute paths based on project root
(RESULTS_DIR, DATA_DIR, STATS_DIR, HPARAMS_DIR, KV_DIR) = (
    _project_root / z if not Path(z).is_absolute() else Path(z)
    for z in [
        data["RESULTS_DIR"],
        data["DATA_DIR"],
        data["STATS_DIR"],
        data["HPARAMS_DIR"],
        data["KV_DIR"],
    ]
)

REMOTE_ROOT_URL = data["REMOTE_ROOT_URL"]
