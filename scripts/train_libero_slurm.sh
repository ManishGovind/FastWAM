#!/bin/bash
#SBATCH -t 7-00:00:00
#SBATCH --partition=gpu-hp
#SBATCH --qos=charlotte_h200_hp
#SBATCH --gres=gpu:h200:8
#SBATCH --output=./slurm_jobs/%j.out
#SBATCH --error=./error_logs/%j.err
#SBATCH --mail-user=mgovind@charlotte.edu
#SBATCH --mail-type=END,FAIL
#SBATCH -J fastwam_uncond_libero
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=192
#SBATCH --mem=2000G


set -euo pipefail
cd /work/mgovind1/projects/FastWAM
mkdir -p slurm_jobs error_logs

# shellcheck disable=SC1091
source scripts/env.sh

# One-time (or if missing): ActionDiT backbone from Wan2.2 DiT
if [[ ! -f checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt ]]; then
  python scripts/preprocess_action_dit_backbone.py \
    --model-config configs/model/fastwam_joint.yaml \
    --output checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt \
    --device cuda \
    --dtype bfloat16
fi

# One-time: T5 text embedding cache
if [[ ! -d data/text_embeds_cache/libero ]] || [[ -z "$(ls -A data/text_embeds_cache/libero 2>/dev/null || true)" ]]; then
  torchrun --standalone --nproc_per_node=8 scripts/precompute_text_embeds.py \
    task=libero_joint_2cam224_1e-4
fi

bash scripts/train_zero1.sh 8 task=libero_uncond_2cam224_1e-4
