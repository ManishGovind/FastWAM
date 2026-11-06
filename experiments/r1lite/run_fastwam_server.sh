#!/usr/bin/env bash
# Launch the workstation-side FastWAM R1 Lite policy server.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
FASTWAM_CONDA_ENV="${FASTWAM_CONDA_ENV:-fastwam}"
DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-${REPO_ROOT}/checkpoints}"
export DIFFSYNTH_MODEL_BASE_PATH

# If this script was started from an active target environment, reuse it.
# Otherwise source Conda's shell hook before calling `conda activate`.
if [[ "${CONDA_DEFAULT_ENV:-}" != "$FASTWAM_CONDA_ENV" ]]; then
  if [[ "$(type -t conda 2>/dev/null || true)" != "function" ]]; then
    for conda_setup in \
      "${CONDA_EXE:+$(dirname "$(dirname "$CONDA_EXE")")/etc/profile.d/conda.sh}" \
      "$HOME/miniconda3/etc/profile.d/conda.sh" \
      "$HOME/anaconda3/etc/profile.d/conda.sh"; do
      if [[ -n "$conda_setup" && -f "$conda_setup" ]]; then
        # shellcheck disable=SC1090
        source "$conda_setup"
        break
      fi
    done
  fi
  if [[ "$(type -t conda 2>/dev/null || true)" != "function" ]]; then
    echo "Conda shell integration is unavailable. Source conda.sh first." >&2
    exit 1
  fi
  conda activate "$FASTWAM_CONDA_ENV"
fi
cd "$REPO_ROOT"
exec python experiments/r1lite/serve_fastwam_r1lite.py "$@"
