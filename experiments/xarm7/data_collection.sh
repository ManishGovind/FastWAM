#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ROOT="${CONDA_ROOT:-/home/skapse/workspace/miniconda3}"
PYTHON_BIN="${CONDA_ROOT}/envs/fastwam_xarm7/bin/python"

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/xarm_data_collection.py" "$@"
