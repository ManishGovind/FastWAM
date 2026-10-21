#!/usr/bin/env bash
# Run LIBERO-plus eval on three seeds.
#
# Usage:
#   bash scripts/eval_libero_plus_3seeds.sh \
#     --task libero_uncond_no_cotrain_2cam224_1e-4 \
#     --ckpt ./runs/.../step_021700.pt \
#     --stats ./runs/.../dataset_stats.json \
#     [--action-mode rgb_flow] \
#     [--categories Noise,Layout,Background,Camera] \
#     [--max-tasks-per-category 5] \
#     [--num-gpus 4] \
#     [--num-trials 1] \
#     [--seeds 42,43,44]

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"


export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export LD_LIBRARY_PATH=/data/mgovind/miniconda3/envs/ltm/lib:${LD_LIBRARY_PATH:-}

export MAGICK_HOME=/data/mgovind/FastWAM/.deps/imagemagick
export LD_LIBRARY_PATH="${MAGICK_HOME}/lib:${LD_LIBRARY_PATH:-}"



TASK=""
CKPT=""
STATS=""
ACTION_MODE=""
ACTIVE_FUTURE=""
ACTIVE_FUTURE_PER_TASK=""
CATEGORIES=""
MAX_TASKS_PER_CATEGORY=""
NUM_GPUS=4
NUM_TRIALS=1
SEEDS="42 43 44"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/eval_libero_plus_3seeds.sh \
    --task <task> \
    --ckpt <ckpt.pt> \
    --stats <dataset_stats.json> \
    [--action-mode <mode>] \
    [--active-future rgb|depth|flow|all] \
    [--active-future-per-task <map.json>] \
    [--categories Noise,Layout,Background,Camera] \
    [--max-tasks-per-category 5] \
    [--num-gpus 4] \
    [--num-trials 1] \
    [--seeds 42,43,44]

Required:
  --task          Hydra task config
  --ckpt          Checkpoint .pt path
  --stats         dataset_stats.json path

Optional:
  --action-mode              EVALUATION.action_attend_mode (multimodal ablations)
  --active-future            Fixed MoT future for MultimodalSelect (rgb|depth|flow),
                             or all. Prefer rgb for GT/random (matches training).
                             Task yaml may already set this (e.g. modality_gt).
  --categories               Comma-separated plus dims:
                             Camera,Robot,Language,Light,Background,Noise,Layout
                             Omit = all 7 dimensions (~10k tasks)
  --max-tasks-per-category   Cap variants per (suite, category, base task).
                             Omit = all variants
  --active-future-per-task   JSON {suite: {task_id|base_task: rgb|depth|flow}}; sets the stream
                             per task (EVALUATION.active_future_per_task).
  --num-gpus                 Default 4
  --num-trials               Default 1 (official LIBERO-plus protocol)
  --seeds                    Comma-separated. Default 42,43,44
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --task) TASK="${2:?}"; shift 2 ;;
    --ckpt) CKPT="${2:?}"; shift 2 ;;
    --stats) STATS="${2:?}"; shift 2 ;;
    --action-mode) ACTION_MODE="${2:?}"; shift 2 ;;
    --active-future) ACTIVE_FUTURE="${2:?}"; shift 2 ;;
    --active-future-per-task) ACTIVE_FUTURE_PER_TASK="$(realpath "${2:?}")"; shift 2 ;;
    --categories) CATEGORIES="${2:?}"; shift 2 ;;
    --max-tasks-per-category) MAX_TASKS_PER_CATEGORY="${2:?}"; shift 2 ;;
    --num-gpus) NUM_GPUS="${2:?}"; shift 2 ;;
    --num-trials) NUM_TRIALS="${2:?}"; shift 2 ;;
    --seeds) SEEDS="${2//,/ }"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "${TASK}" || -z "${CKPT}" || -z "${STATS}" ]]; then
  echo "Missing required arguments." >&2
  usage >&2
  exit 1
fi
if [[ ! -f "${CKPT}" ]]; then
  echo "Checkpoint not found: ${CKPT}" >&2
  exit 1
fi
if [[ ! -f "${STATS}" ]]; then
  echo "dataset_stats.json not found: ${STATS}" >&2
  exit 1
fi

EXTRA_ARGS=()
if [[ -n "${ACTION_MODE}" ]]; then
  EXTRA_ARGS+=( "EVALUATION.action_attend_mode=${ACTION_MODE}" )
fi
if [[ -n "${ACTIVE_FUTURE}" ]]; then
  EXTRA_ARGS+=( "++EVALUATION.active_future=${ACTIVE_FUTURE}" )
fi
if [[ -n "${ACTIVE_FUTURE_PER_TASK}" ]]; then
  [[ -f "${ACTIVE_FUTURE_PER_TASK}" ]] || { echo "map not found: ${ACTIVE_FUTURE_PER_TASK}" >&2; exit 1; }
  EXTRA_ARGS+=( "EVALUATION.active_future_per_task=${ACTIVE_FUTURE_PER_TASK}" )
fi
if [[ -n "${CATEGORIES}" ]]; then
  # Noise,Layout -> [Noise,Layout] for Hydra
  EXTRA_ARGS+=( "MULTIRUN.perturbation_categories=[${CATEGORIES}]" )
fi
if [[ -n "${MAX_TASKS_PER_CATEGORY}" ]]; then
  EXTRA_ARGS+=( "MULTIRUN.max_tasks_per_category=${MAX_TASKS_PER_CATEGORY}" )
fi

echo "task=${TASK}"
echo "ckpt=${CKPT}"
echo "stats=${STATS}"
echo "num_gpus=${NUM_GPUS} num_trials=${NUM_TRIALS}"
echo "seeds=${SEEDS}"
[[ -n "${ACTION_MODE}" ]] && echo "action_attend_mode=${ACTION_MODE}"
[[ -n "${ACTIVE_FUTURE}" ]] && echo "active_future=${ACTIVE_FUTURE}"
[[ -n "${ACTIVE_FUTURE_PER_TASK}" ]] && echo "active_future_per_task=${ACTIVE_FUTURE_PER_TASK}"
[[ -n "${CATEGORIES}" ]] && echo "perturbation_categories=${CATEGORIES}" || echo "perturbation_categories=all"
[[ -n "${MAX_TASKS_PER_CATEGORY}" ]] && echo "max_tasks_per_category=${MAX_TASKS_PER_CATEGORY}" || echo "max_tasks_per_category=all"

for seed in ${SEEDS}; do
  echo
  echo "======== seed=${seed} ========"
  python experiments/libero_plus/run_libero_plus_manager.py \
    task="${TASK}" \
    ckpt="${CKPT}" \
    seed="${seed}" \
    EVALUATION.dataset_stats_path="${STATS}" \
    MULTIRUN.num_gpus="${NUM_GPUS}" \
    EVALUATION.num_trials="${NUM_TRIALS}" \
    "${EXTRA_ARGS[@]}"
done

echo
echo "Done. All seeds finished: ${SEEDS}"
