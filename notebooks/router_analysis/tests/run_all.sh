#!/usr/bin/env bash
# Run the router test suite. From the repo root:
#   bash notebooks/router_analysis/tests/run_all.sh          # GPU if available
#   TEST_DEVICE=cpu bash notebooks/router_analysis/tests/run_all.sh   # CPU only (safe next to training)
set -u
cd "$(dirname "$0")/../../.."
source scripts/env.sh >/dev/null 2>&1
export LD_LIBRARY_PATH=/data/mgovind/npp_libs:${LD_LIBRARY_PATH:-}
T=notebooks/router_analysis/tests
L=notebooks/router_analysis/logs/test_run_$(date +%Y%m%d_%H%M%S)
mkdir -p "$L"
if [ "${TEST_DEVICE:-}" = "cpu" ]; then export CUDA_VISIBLE_DEVICES=""; fi
DT=$([ "${TEST_DEVICE:-}" = "cpu" ] && echo fp32 || echo bf16)
run() {  # name, command...
  local name=$1; shift
  local t0=$(date +%s)
  "$@" > "$L/$name.log" 2>&1; local rc=$?
  printf "%-26s exit=%d  %4ds  -> %s\n" "$name" "$rc" "$(( $(date +%s) - t0 ))" "$L/$name.log"
  grep -E "^ *[0-9]\.|^\[train\]|^\[infer\]|^window|^[A-D] |vs |identical|joint context|context \[|infer_action|prompts distinct|Traceback|Error" "$L/$name.log" | grep -v -i warning | sed 's/^/    /' | cut -c1-260
}
# fast CPU checks
run smoke_router_new   python $T/smoke_router_new.py
run smoke_warmup       python $T/smoke_warmup.py
run smoke_gumbel_cpu   python $T/smoke_gumbel_cpu.py
run smoke_gumbel_real  python $T/smoke_gumbel_real.py
run eval_prompt_parity python $T/test_eval_prompt_parity.py
run masks              python $T/test_masks.py
run distill            python $T/test_distill.py   # real student + 3 teachers on CPU, ~5 min
run stream_embedding   python $T/test_stream_embedding.py   # real model on CPU, ~5 min
run router_embed       python $T/test_router_embed.py       # real model on CPU, ~5 min
run router_transformer python $T/test_router_transformer.py # real model on CPU, ~5 min
run robotwin_tile      python $T/test_multimodal_robotwin_tile.py   # R1-Lite data on CPU, ~1 min
run image_history      python $T/test_image_history.py      # LIBERO data on CPU, ~2 min
run router_history     python $T/test_router_history.py     # real model on CPU, ~10 min
run history_invariance python $T/test_history_video_invariance.py   # real model on CPU, ~10 min
run mm_stream_embedding python $T/test_mm_stream_embedding.py   # real MM model on CPU, ~8 min
# real model (5B; several minutes each on CPU)
run test_prompt_target python $T/test_prompt_target.py
run test_trainer_eval  python $T/test_trainer_eval.py
run test_joint         python $T/test_joint.py $DT
run test_infer_full2   python $T/test_infer_full2.py $DT
run test_prompt_layout python $T/test_prompt_layout.py
echo "logs: $L"
