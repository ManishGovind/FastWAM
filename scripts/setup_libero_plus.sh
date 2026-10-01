#!/usr/bin/env bash
# Clone LIBERO-plus, write a project-local libero config, and optionally download assets.
#
# Do NOT `pip install -e third_party/LIBERO-plus`. That would replace the original
# `libero` package. FastWAM eval prepends this checkout via PYTHONPATH instead.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PLUS_DIR="${ROOT}/third_party/LIBERO-plus"
PLUS_REPO="${LIBERO_PLUS_REPO:-https://github.com/sylvestf/LIBERO-plus.git}"
DOWNLOAD_ASSETS=0

for arg in "$@"; do
  case "${arg}" in
    --download-assets) DOWNLOAD_ASSETS=1 ;;
    -h|--help)
      cat <<EOF
Usage: bash scripts/setup_libero_plus.sh [--download-assets]

  Clones https://github.com/sylvestf/LIBERO-plus into third_party/LIBERO-plus
  and writes .libero_plus/config.yaml so FastWAM eval can keep original LIBERO
  installed.

  --download-assets
      Download assets.zip from Hugging Face (Sylvest/LIBERO-plus, ~6.4GB)
      and unzip it into third_party/LIBERO-plus/libero/libero/assets.

System packages required for sensor-noise eval (once per machine):
  sudo apt-get install -y libexpat1 libfontconfig1-dev libpython3-stdlib libmagickwand-dev

If you cannot sudo (typical on Hopper), keep using the uv venv and point wand at
a local MagickWand prefix via MAGICK_HOME (scripts/env.sh does this when
.deps/imagemagick exists).
EOF
      exit 0
      ;;
    *)
      echo "Unknown argument: ${arg}" >&2
      exit 1
      ;;
  esac
done

if [[ ! -f "${PLUS_DIR}/setup.py" ]]; then
  echo "[libero-plus] cloning ${PLUS_REPO} -> ${PLUS_DIR}"
  git clone --depth 1 "${PLUS_REPO}" "${PLUS_DIR}"
else
  echo "[libero-plus] using existing checkout: ${PLUS_DIR}"
fi

BENCH="${PLUS_DIR}/libero/libero"
CONFIG_DIR="${ROOT}/.libero_plus"
mkdir -p "${CONFIG_DIR}"
cat > "${CONFIG_DIR}/config.yaml" <<EOF
assets: ${BENCH}/assets
bddl_files: ${BENCH}/bddl_files
benchmark_root: ${BENCH}
datasets: ${PLUS_DIR}/libero/datasets
init_states: ${BENCH}/init_files
EOF
echo "[libero-plus] wrote ${CONFIG_DIR}/config.yaml"

PYTHON_BIN="${ROOT}/.venv/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="$(command -v python)"
fi

echo "[libero-plus] installing wand and scikit-image into the uv venv"
UV_BIN="$(command -v uv || true)"
if [[ -n "${UV_BIN}" ]]; then
  # This repo's uv venv does not ship pip. Prefer uv, and copy if cache is on another FS.
  export UV_CACHE_DIR="${UV_CACHE_DIR:-/data/mgovind/.cache/uv}"
  mkdir -p "${UV_CACHE_DIR}"
  export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
  "${UV_BIN}" pip install --python "${PYTHON_BIN}" wand scikit-image
elif [[ -x "${ROOT}/.venv/bin/pip" ]]; then
  "${ROOT}/.venv/bin/pip" install wand scikit-image
else
  "${PYTHON_BIN}" -m pip install wand scikit-image
fi

MAGICK_PREFIX="${ROOT}/.deps/imagemagick"
if [[ -d "${MAGICK_PREFIX}/lib" ]]; then
  export MAGICK_HOME="${MAGICK_PREFIX}"
  export LD_LIBRARY_PATH="${MAGICK_HOME}/lib:${LD_LIBRARY_PATH:-}"
  echo "[libero-plus] MAGICK_HOME=${MAGICK_HOME} (uv venv unchanged)"
else
  echo "[libero-plus] WARNING: MagickWand C library not found."
  echo "  wand is installed in the uv venv, but it needs libMagickWand."
  echo "  Prefer: sudo apt-get install -y libmagickwand-dev"
  echo "  Or create ${MAGICK_PREFIX} and source scripts/env.sh"
fi

if [[ "${DOWNLOAD_ASSETS}" == "1" ]]; then
  echo "[libero-plus] downloading assets.zip from Hugging Face (Sylvest/LIBERO-plus)"
  HF_CLI="${ROOT}/.venv/bin/huggingface-cli"
  if [[ ! -x "${HF_CLI}" ]]; then
    HF_CLI="$(command -v huggingface-cli)"
  fi
  "${HF_CLI}" download Sylvest/LIBERO-plus assets.zip --repo-type dataset --local-dir "${PLUS_DIR}"
  echo "[libero-plus] unzipping assets into ${BENCH}/assets"
  UNPACK_DIR="${PLUS_DIR}/_assets_unpack"
  rm -rf "${UNPACK_DIR}"
  mkdir -p "${UNPACK_DIR}"
  unzip -q -o "${PLUS_DIR}/assets.zip" -d "${UNPACK_DIR}"
  ASSETS_SRC="$(find "${UNPACK_DIR}" -type d -name new_objects -print -quit 2>/dev/null || true)"
  if [[ -n "${ASSETS_SRC}" ]]; then
    ASSETS_SRC="$(dirname "${ASSETS_SRC}")"
  else
    ASSETS_SRC="$(find "${UNPACK_DIR}" -type d -name assets -print -quit 2>/dev/null || true)"
  fi
  if [[ -z "${ASSETS_SRC}" ]]; then
    echo "[libero-plus] ERROR: unzipped archive has no assets/ directory" >&2
    exit 1
  fi
  mkdir -p "${BENCH}/assets"
  cp -a "${ASSETS_SRC}/." "${BENCH}/assets/"
  rm -rf "${UNPACK_DIR}"
fi

if [[ ! -d "${BENCH}/assets/new_objects" ]]; then
  echo "[libero-plus] WARNING: assets/new_objects is missing."
  echo "  Download them before eval: bash scripts/setup_libero_plus.sh --download-assets"
else
  echo "[libero-plus] assets OK: ${BENCH}/assets/new_objects"
fi

echo "[libero-plus] setup complete."
echo "Evaluate unimodal FastWAM (RGB / depth / flow) on LIBERO-plus:"
echo "  python experiments/libero_plus/run_libero_plus_manager.py \\"
echo "    task=libero_uncond_2cam224_1e-4_depth ckpt=/path/to/ckpt.pt \\"
echo "    EVALUATION.dataset_stats_path=/path/to/dataset_stats.json \\"
echo "    EVALUATION.sigma_shift=5.0 \\"
echo "    MULTIRUN.perturbation_categories='[Noise,Camera]'"
