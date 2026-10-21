"""Closed-loop eval (infer_action) runs the same trajectory as infer_joint (decoded in the
modality-decoding check): same seed -> same actions per active_future, and infer_joint's
decoded future is the requested modality. So with active_future=depth the eval's hidden
future latents are depth latents.

    CUDA_VISIBLE_DEVICES=0 python notebooks/router_analysis/tests/test_infer_action_vs_joint.py
"""
import os, sys
import numpy as np, torch
sys.path.insert(0, "notebooks/router_analysis/scripts")
from verify_m_context import build, CTX_TASK, CTX_RUN
from hydra.utils import instantiate
from fastwam.utils.video_metrics import pil_frames_to_video_tensor, video_psnr

os.chdir("/data/mgovind/FastWAM")
cfg, model = build(CTX_TASK, CTX_RUN, "cuda", torch.bfloat16)
ds = instantiate(cfg.data.train, pretrained_norm_stats=f"{CTX_RUN}/dataset_stats.json")
for i in [0, 138856, 242998]:
    s = ds[i]
    kw = dict(prompt=s["prompt"], input_image=s["video_rgb"][:, 0].cuda().to(torch.bfloat16),
              action_horizon=int(s["action"].shape[0]), num_video_frames=int(s["video_rgb"].shape[1]),
              proprio=s["proprio"][0].cuda(), num_inference_steps=10, seed=42)
    for st in ("rgb", "depth", "flow"):
        with torch.no_grad():
            a = model.infer_action(active_future=st, **kw)["action"]
            j = model.infer_joint(active_future=st, **kw)
        gen = pil_frames_to_video_tensor(j["video"])[:, 1:]
        ps = {g: video_psnr(pred=gen, target=((s[f"video_{g}"].float().clamp(-1, 1) + 1) / 2)[:, 1:]) for g in ("rgb", "depth", "flow")}
        best = max(ps, key=ps.get)
        print(f"window {i:6d} active_future={st:5s} | max|a_infer_action - a_infer_joint| = {float((a - j['action']).abs().max()):.2e} "
              f"| infer_joint future looks like: {best} (PSNR {ps[best]:.1f})", flush=True)
        assert best == st
print("OK")
