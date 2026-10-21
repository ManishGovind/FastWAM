#!/bin/bash
#SBATCH -t 2-00:00:00
#SBATCH --partition=gpu-hp
#SBATCH --qos=charlotte_h200_hp
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=96G
#SBATCH -J robotwin_dav2_flow
#SBATCH --output=./slurm_jobs/%x_%j.out
#SBATCH -e ./error_logs/%x_%j.err
#
# RoboTwin 2.0 depth (Depth Anything V2 vitl) + RAFT flow for the 3 cameras.
# One process (= one shard) per GPU of the job; with an array, every array task takes the next
# block of shards. --skip-existing makes everything resumable: resubmit to fill gaps.
#
#   sbatch scripts/precompute_depth_flow_robotwin_slurm.sh                  # 1 job  x 4 GPUs =  4 shards
#   sbatch --array=0-3 scripts/precompute_depth_flow_robotwin_slurm.sh      # 4 jobs x 4 GPUs = 16 shards

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p slurm_jobs error_logs

GPUS_PER_JOB="${GPUS_PER_JOB:-4}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
NUM_JOBS="${SLURM_ARRAY_TASK_COUNT:-1}"
NUM_SHARDS=$((GPUS_PER_JOB * NUM_JOBS))

for g in $(seq 0 $((GPUS_PER_JOB - 1))); do
  SHARD=$((TASK_ID * GPUS_PER_JOB + g))
  echo "GPU ${g}: shard ${SHARD}/${NUM_SHARDS}"
  CUDA_VISIBLE_DEVICES=$g python scripts/precompute_depth_flow.py \
    --dataset-dir data/robotwin2.0/robotwin2.0 \
    --source-key observation.images.cam_high observation.images.cam_left_wrist observation.images.cam_right_wrist \
    --depth-encoder vitl \
    --flow-clip-px 25 \
    --flow-noise-thresh-px 0.5 \
    --depth-batch-size 32 \
    --flow-batch-size 16 \
    --device cuda \
    --skip-existing \
    --shard-id "${SHARD}" \
    --num-shards "${NUM_SHARDS}" \
    "$@" > "slurm_jobs/robotwin_depth_flow_shard${SHARD}.log" 2>&1 &
done
wait
echo "all ${GPUS_PER_JOB} shards of task ${TASK_ID} finished"
