#!/usr/bin/env bash
# LIBERO then LIBERO-Plus eval of the router short m-context-video model (router picks the stream per replan).
cd /data/mgovind/FastWAM && source .venv/bin/activate
R=runs/libero_joint_2cam224_1e-4_router_short_m_context_video/2026-10-10_03-23-54
T=libero_joint_2cam224_1e-4_router_short_m_context_video
L=notebooks/router_analysis/logs
bash scripts/eval_libero_3seeds.sh --task $T --ckpt $R/checkpoints/weights/step_021700.pt \
  --stats $R/dataset_stats.json --seeds 42 > $L/eval_router_m_context_video_libero.log 2>&1
bash scripts/eval_libero_plus_3seeds.sh --task $T --ckpt $R/checkpoints/weights/step_021700.pt \
  --stats $R/dataset_stats.json --categories Noise,Layout,Background,Camera \
  --max-tasks-per-category 5 --seeds 42 > $L/eval_router_m_context_video_plus.log 2>&1
