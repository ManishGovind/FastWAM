#!/usr/bin/env bash
# Robot client for serve_piwam_xarm7.py (same env and robot behavior as ../xarm7/run_inference.sh).
#   ./run_piwam_inference.sh --task insert_bottle [--ckpt_tag piwam_xarm7_20k] [--name ...]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ROOT="${CONDA_ROOT:-/home/skapse/workspace/miniconda3}"
PYTHON_BIN="${CLIENT_PYTHON:-${CONDA_ROOT}/envs/fastwam_xarm7/bin/python}"
cd "${SCRIPT_DIR}"
exec "${PYTHON_BIN}" xarm_piwam_inference.py "$@"
