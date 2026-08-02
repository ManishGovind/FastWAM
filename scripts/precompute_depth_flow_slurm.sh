#!/bin/bash
#SBATCH -t 2-00:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH -J fastwam_dav2_vitl_flow
#SBATCH --output=./slurm_jobs/%j.out
#SBATCH -e ./error_logs/%j.err

set -euo pipefail
cd /work/mgovind1/projects/FastWAM
mkdir -p slurm_jobs error_logs
source scripts/env.sh

# Depth Anything V2 (vitl) + RAFT flow with FlowWAM postprocess:
#   magnitude cap 25 px, noise threshold 0.5 px, white-bg HSV.
# Use --overwrite (or delete old flow dirs) to regenerate noisy flow videos.
# --skip-existing allows resume for missing episodes only.
python scripts/precompute_depth_flow.py \
  --dataset-dir \
    data/libero_mujoco3.3.2/libero_spatial_no_noops_lerobot \
    data/libero_mujoco3.3.2/libero_object_no_noops_lerobot \
    data/libero_mujoco3.3.2/libero_goal_no_noops_lerobot \
    data/libero_mujoco3.3.2/libero_10_no_noops_lerobot \
  --depth-encoder vitl \
  --flow-clip-px 25 \
  --flow-noise-thresh-px 0.5 \
  --device cuda \
  --skip-existing \
  --depth-batch-size 4 \
  --flow-batch-size 4 \
  "$@"
