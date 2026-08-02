#!/bin/bash
#SBATCH -t 1-00:00:00
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:h200:2
#SBATCH --output=./slurm_jobs/%j.out
#SBATCH --error=./error_logs/%j.err
#SBATCH --mail-user=mgovind@charlotte.edu
#SBATCH --mail-type=END,FAIL
#SBATCH -J fastwam_eval_libero
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

TASK=libero_joint_2cam224_1e-4
RUN_DIR=./runs/libero_joint_2cam224_1e-4/2026-07-22_00-54-34
CKPT="${RUN_DIR}/checkpoints/weights/step_014470.pt"
STATS="${RUN_DIR}/dataset_stats.json"
# Keep in sync with #SBATCH --gres gpu count
NUM_GPUS=2
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

python experiments/libero/run_libero_manager.py \
  task="${TASK}" \
  ckpt="${CKPT}" \
  EVALUATION.dataset_stats_path="${STATS}" \
  MULTIRUN.num_gpus="${NUM_GPUS}"
