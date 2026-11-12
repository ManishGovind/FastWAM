#!/usr/bin/env bash
# pi-WAM (router) policy server on 127.0.0.1:8780, code from the origin/dev worktree.
#   ./serve_piwam.sh [--checkpoint step_030000.pt] [--active-future rgb|depth|flow|random]
# One-time setup: git -C ~/workspace/FastWAM worktree add --detach ~/workspace/FastWAM_dev origin/dev
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ROOT="${CONDA_ROOT:-/home/skapse/workspace/miniconda3}"
PYTHON_BIN="${SERVER_PYTHON:-${CONDA_ROOT}/envs/fastwam/bin/python}"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
cd "${SCRIPT_DIR}"
exec "${PYTHON_BIN}" -u serve_piwam_xarm7.py --code-root "${CODE_ROOT:-$HOME/workspace/FastWAM_dev}" \
    --device cuda:0 --port "${PORT:-8780}" "$@"
