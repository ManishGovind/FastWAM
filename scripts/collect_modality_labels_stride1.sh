#!/usr/bin/env bash
# Collect modality action labels with frame-stride 1 (training-aligned windows).
# Sequential models: finish RGB (4 GPU shards), then Depth, then Flow, then analyze.
# Resume-safe per shard .pt.
set -euo pipefail

cd /data/mgovind/FastWAM
if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

FRAME_STRIDE=1
N_EPISODES=0
SEEDS=(42 43 44)
NUM_SHARDS=4
COMMON=(--frame-stride "${FRAME_STRIDE}" --n-episodes "${N_EPISODES}" --seeds "${SEEDS[@]}" --num-shards "${NUM_SHARDS}")

run_model() {
  local model="$1"
  echo "=== Collect ${model}: launching ${NUM_SHARDS} shards ==="
  local pids=()
  for shard in $(seq 0 $((NUM_SHARDS - 1))); do
    CUDA_VISIBLE_DEVICES="${shard}" \
      python notebooks/modality_action_labels.py --model "${model}" "${COMMON[@]}" \
        --shard "${shard}" &
    pids+=("$!")
    echo "  ${model} shard ${shard} -> GPU ${shard} pid=${pids[-1]}"
  done
  local status=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      echo "ERROR: pid ${pid} failed for ${model}" >&2
      status=1
    fi
  done
  if [[ "${status}" -ne 0 ]]; then
    echo "Aborting: ${model} collect had failures." >&2
    exit "${status}"
  fi
  echo "=== ${model} done ==="
}

run_model RGB
run_model Depth
run_model Flow

echo "=== Analyze ==="
python notebooks/modality_action_labels.py --analyze \
  --frame-stride "${FRAME_STRIDE}" \
  --n-episodes "${N_EPISODES}" \
  --seeds "${SEEDS[@]}"

echo "=== Verify ==="
python notebooks/modality_action_labels.py --verify \
  --frame-stride "${FRAME_STRIDE}" \
  --n-episodes "${N_EPISODES}" \
  --seeds "${SEEDS[@]}"

echo "All done."
