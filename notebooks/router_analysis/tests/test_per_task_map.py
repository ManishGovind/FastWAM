"""EVALUATION.active_future_per_task helper (CPU, no model).

    PYTHONPATH=. python notebooks/router_analysis/tests/test_per_task_map.py
"""
import glob, json, os
from omegaconf import OmegaConf
from experiments.libero.eval_infer import apply_per_task_active_future, plus_base_task

os.chdir("/data/mgovind/FastWAM")
MAP = "notebooks/router_analysis/outputs/per_task_stream_map_libero_ctx.json"
m = json.load(open(MAP))
n = 0
for suite, tasks in m.items():
    for tid, stream in tasks.items():
        cfg = OmegaConf.create({"EVALUATION": {"active_future": "random", "active_future_per_task": MAP}})
        assert apply_per_task_active_future(cfg, suite, [str(tid)]) == stream
        assert cfg.EVALUATION.active_future == stream  # overrides the global setting
        n += 1
print(f"LIBERO: {n} tasks resolve to the map")
# unset -> untouched
cfg = OmegaConf.create({"EVALUATION": {"active_future": "depth", "active_future_per_task": None}})
assert apply_per_task_active_future(cfg, "libero_10", ["3"]) is None and cfg.EVALUATION.active_future == "depth"
# missing task -> error (no silent fallback)
try:
    apply_per_task_active_future(OmegaConf.create({"EVALUATION": {"active_future_per_task": MAP}}), "libero_10", ["99"])
    raise AssertionError("missing task should raise")
except KeyError as e:
    print("missing task raises:", str(e)[:80])
# LIBERO-Plus names -> 40 base tasks
names = {json.load(open(f))["task_name"] for f in glob.glob(
    "evaluate_results/libero_plus/libero_joint_2cam224_1e-4_random_short_m_context_video/20261011_032922/**/*_results.json", recursive=True)}
print(f"LIBERO-Plus: {len(names)} variant names -> {len({plus_base_task(x) for x in names})} base tasks")
print("OK")

# --- top-2 map: list values -> "random:a,b", drawn per call by the model ---
import collections
import torch
import torch.nn as nn
from fastwam.models.wan22.fastwam_joint_multimodal_select_new import FastWAMJointMultimodalSelectNew

MAP2 = "notebooks/router_analysis/outputs/per_task_stream_map_libero_ctx_top2.json"
m2 = json.load(open(MAP2))
cfg = OmegaConf.create({"EVALUATION": {"active_future": "random", "active_future_per_task": MAP2}})
af = apply_per_task_active_future(cfg, "libero_10", ["6"])
assert af == "random:" + ",".join(m2["libero_10"]["6"]), af
print("top-2 map: libero_10 / 6 ->", af)

model = object.__new__(FastWAMJointMultimodalSelectNew)
nn.Module.__init__(model)
model.enabled_video_streams = ("rgb", "depth", "flow")
torch.manual_seed(0)
picks = collections.Counter(
    model._resolve_infer_active_future(active_future=af, action_attend_mode="all",
                                       first_frame_latents=None, context=None, context_mask=None)
    for _ in range(2000))
assert set(picks) == set(m2["libero_10"]["6"]), picks
assert all(800 < v < 1200 for v in picks.values()), picks
print("model draws only the listed streams, ~uniform:", dict(picks))
try:
    model._resolve_infer_active_future(active_future="random:depth,foo", action_attend_mode="all",
                                       first_frame_latents=None, context=None, context_mask=None)
    raise AssertionError("unknown stream should raise")
except ValueError as e:
    print("bad stream raises:", str(e)[:70])
print("OK top-2")
