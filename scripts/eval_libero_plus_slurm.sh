#!/bin/bash
#SBATCH -t 7-00:00:00
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:h200:8
#SBATCH --output=./slurm_jobs/%j.out
#SBATCH --error=./error_logs/%j.err
#SBATCH --mail-user=mgovind@charlotte.edu
#SBATCH --mail-type=END,FAIL
#SBATCH -J fastwam_eval_libero_plus
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=192


set -euo pipefail
cd /work/mgovind1/projects/FastWAM


# shellcheck disable=SC1091
source scripts/env.sh

# Headless MuJoCo rendering on Slurm GPU nodes
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

# Unimodal FastWAM variants (eval still sees RGB from the perturbed sim):
#   TASK=libero_uncond_2cam224_1e-4          # RGB future
#   TASK=libero_uncond_2cam224_1e-4_depth    # depth future
#   TASK=libero_uncond_2cam224_1e-4_flow     # flow future
# Joint counterparts: libero_joint_2cam224_1e-4{,_depth,_flow}
TASK=libero_uncond_2cam224_1e-4_depth
RUN_DIR=./runs/libero_uncond_2cam224_1e-4_depth/2026-08-19_17-12-39
CKPT="${RUN_DIR}/checkpoints/weights/step_021700.pt"
STATS="${RUN_DIR}/dataset_stats.json"
# Keep in sync with #SBATCH --gres gpu count
NUM_GPUS=8
# Match the run's action infer_shift. Depth/flow uncond runs used 5.0; current
# configs/model/fastwam.yaml defaults action infer_shift to 1.0.
SIGMA_SHIFT=5.0
# Optional: restrict to perturbation dimensions, e.g. '[Noise,Camera]'
# Leave empty for the full 7-dimension LIBERO-plus benchmark.
PERTURB_CATS=""
# Avoid Slurm's CUDA_VISIBLE_DEVICES remapping confusing the tmux launcher GPU ids.
unset CUDA_VISIBLE_DEVICES

if [[ ! -f "${CKPT}" ]]; then
  echo "Checkpoint not found: ${CKPT}" >&2
  exit 1
fi
if [[ ! -f "${STATS}" ]]; then
  echo "dataset_stats.json not found: ${STATS}" >&2
  exit 1
fi
if [[ ! -d third_party/LIBERO-plus/libero/libero/assets/new_objects ]]; then
  echo "LIBERO-plus assets missing. Run: bash scripts/setup_libero_plus.sh --download-assets" >&2
  exit 1
fi

PLUS_ARGS=(
  task="${TASK}"
  ckpt="${CKPT}"
  EVALUATION.dataset_stats_path="${STATS}"
  MULTIRUN.num_gpus="${NUM_GPUS}"
)
if [[ -n "${SIGMA_SHIFT}" ]]; then
  PLUS_ARGS+=("EVALUATION.sigma_shift=${SIGMA_SHIFT}")
fi
if [[ -n "${PERTURB_CATS}" ]]; then
  PLUS_ARGS+=("MULTIRUN.perturbation_categories=${PERTURB_CATS}")
fi
if [[ -n "${MAX_PER_CATEGORY}" ]]; then
  PLUS_ARGS+=("MULTIRUN.max_tasks_per_category=${MAX_PER_CATEGORY}")
fi

python experiments/libero_plus/run_libero_plus_manager.py "${PLUS_ARGS[@]}"
