#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ROOT="${CONDA_ROOT:-/home/skapse/workspace/miniconda3}"
PYTHON_BIN="${CONDA_ROOT}/envs/fastwam_xarm7/bin/python"

cd "${SCRIPT_DIR}"
exec "${PYTHON_BIN}" xarm_fastwam_inference.py "$@"
