#!/usr/bin/env bash
# usage: source scripts/env.sh
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/.venv/bin/activate"
export DIFFSYNTH_MODEL_BASE_PATH="${ROOT}/checkpoints"
export DIFFSYNTH_DOWNLOAD_SOURCE="${DIFFSYNTH_DOWNLOAD_SOURCE:-huggingface}"
export PATH="${ROOT}/.venv/bin:${PATH}"

# Triton / DeepSpeed autotune cache (avoid home NFS ~/.triton)
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/work/mgovind1/hf_cache/triton}"
mkdir -p "${TRITON_CACHE_DIR}/autotune"

# CUDA toolkit (uv: nvidia-cuda-nvcc). cu12 wheels ship only ptxas; cu13 provides nvcc.
CUDA_NVCC_HOME="${ROOT}/.venv/lib/python3.10/site-packages/nvidia/cu13"
if [[ -x "${CUDA_NVCC_HOME}/bin/nvcc" ]]; then
  export CUDA_HOME="${CUDA_NVCC_HOME}"
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${LD_LIBRARY_PATH:-}"
  # torch is cu128 (12.8); pip nvcc is 13.x. Needed so DeepSpeed can import.
  export DS_SKIP_CUDA_CHECK=1
fi

cd "${ROOT}"
echo "[fastwam] venv + CUDA_HOME=${CUDA_HOME:-unset} + TRITON_CACHE_DIR=${TRITON_CACHE_DIR} + DIFFSYNTH_MODEL_BASE_PATH=${DIFFSYNTH_MODEL_BASE_PATH}"
