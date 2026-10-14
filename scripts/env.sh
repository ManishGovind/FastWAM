#!/usr/bin/env bash
# usage: source scripts/env.sh
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/.venv/bin/activate"
export DIFFSYNTH_MODEL_BASE_PATH="${ROOT}/checkpoints"
export DIFFSYNTH_DOWNLOAD_SOURCE="${DIFFSYNTH_DOWNLOAD_SOURCE:-huggingface}"
export PATH="${ROOT}/.venv/bin:${PATH}"

# Keep uv cache on /data rather than $HOME or /tmp.
export UV_CACHE_DIR="${UV_CACHE_DIR:-/data/mgovind/.cache/uv}"
mkdir -p "${UV_CACHE_DIR}"

# Triton / DeepSpeed autotune cache (avoid home NFS ~/.triton)
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/work/mgovind1/hf_cache/triton}"
mkdir -p "${TRITON_CACHE_DIR}/autotune"

# Prepend a lib dir to LD_LIBRARY_PATH at most once (idempotent across re-source).
_fastwam_prepend_lib() {
  local d="$1"
  [[ -n "${d}" && -d "${d}" ]] || return 0
  case ":${LD_LIBRARY_PATH:-}:" in
    *":${d}:"*) ;;
    *) export LD_LIBRARY_PATH="${d}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" ;;
  esac
}

# TorchCodec needs CUDA-12 NPP + FFmpeg 6 shared libs (.ffmpeg-env from micromamba).
NPP_SO="$(find "${ROOT}/.venv" -name 'libnppicc.so.12' 2>/dev/null | head -1 || true)"
NPP_LIB=""
if [[ -n "${NPP_SO}" ]]; then
  NPP_LIB="$(cd "$(dirname "${NPP_SO}")" && pwd)"
elif [[ -e /data/mgovind/npp_libs/libnppicc.so.12 ]]; then
  # Fallback: symlinks to the CUDA-12 NPP libs only (not the whole ltm env, whose FFmpeg 4.2
  # has no AV1 decoder). Only needed by torchcodec builds that link NPP.
  NPP_LIB="/data/mgovind/npp_libs"
fi
FFMPEG_LIB="${ROOT}/.ffmpeg-env/lib"
_fastwam_prepend_lib "${NPP_LIB}"
_fastwam_prepend_lib "${FFMPEG_LIB}"
if [[ ! -e "${FFMPEG_LIB}/libdav1d.so.6" ]]; then
  echo "[fastwam] warning: ${FFMPEG_LIB}/libdav1d.so.6 missing; TorchCodec may fall back to pyav." >&2
  echo "[fastwam] hint: micromamba create -y -p ${ROOT}/.ffmpeg-env -c conda-forge 'ffmpeg=6.*'" >&2
fi
if [[ -z "${NPP_LIB}" ]]; then
  echo "[fastwam] warning: libnppicc.so.12 not found under .venv; pip/uv install nvidia-npp-cu12" >&2
fi

# ImageMagick C library for LIBERO-plus sensor-noise (`wand`). Python stays in .venv.
if [[ -d "${ROOT}/.deps/imagemagick/lib" ]]; then
  export MAGICK_HOME="${ROOT}/.deps/imagemagick"
  _fastwam_prepend_lib "${MAGICK_HOME}/lib"
fi

# CUDA toolkit (uv: nvidia-cuda-nvcc). cu12 wheels ship only ptxas; cu13 provides nvcc.
CUDA_NVCC_HOME="${ROOT}/.venv/lib/python3.10/site-packages/nvidia/cu13"
if [[ -x "${CUDA_NVCC_HOME}/bin/nvcc" ]]; then
  export CUDA_HOME="${CUDA_NVCC_HOME}"
  export PATH="${CUDA_HOME}/bin:${PATH}"
  _fastwam_prepend_lib "${CUDA_HOME}/lib"
  # torch is cu128 (12.8); pip nvcc is 13.x. Needed so DeepSpeed can import.
  export DS_SKIP_CUDA_CHECK=1
fi

cd "${ROOT}"
echo "[fastwam] venv + CUDA_HOME=${CUDA_HOME:-unset} + TRITON_CACHE_DIR=${TRITON_CACHE_DIR} + DIFFSYNTH_MODEL_BASE_PATH=${DIFFSYNTH_MODEL_BASE_PATH}"
echo "[fastwam] TorchCodec libs: ffmpeg=${FFMPEG_LIB} npp=${NPP_LIB:-MISSING}"
