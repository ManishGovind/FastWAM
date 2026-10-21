"""MM model: which modality does each future slot (rgb / depth / flow) actually generate?
Slots share weights and RoPE positions and have no stream identity; this checks the decoded
futures against all three GT streams (6 windows, seed 42, 10 steps).

    CUDA_VISIBLE_DEVICES=0 python notebooks/router_analysis/scripts/mm_slot_identity.py
"""
import os, sys, numpy as np, torch
sys.path.insert(0, "notebooks/router_analysis/scripts")
from verify_m_context import build, BASELINES, CTX_RUN
from hydra.utils import instantiate
from fastwam.utils.video_metrics import pil_frames_to_video_tensor, video_psnr
os.chdir("/data/mgovind/FastWAM")
cfg, m = build(*BASELINES["MM (all futures)"], "cuda", torch.bfloat16)
ds = instantiate(cfg.data.train, pretrained_norm_stats=f"{CTX_RUN}/dataset_stats.json")
print(type(m).__name__, "enabled", m._enabled_streams() if hasattr(m, "_enabled_streams") else m.enabled_video_streams)
for i in np.linspace(0, len(ds) - 1, 6).round().astype(int):
    s = ds[int(i)]
    with torch.no_grad():
        out = m.infer_joint(prompt=s["prompt"], input_image=s["video_rgb"][:, 0].cuda().to(torch.bfloat16),
                            num_video_frames=int(s["video_rgb"].shape[1]), action_horizon=int(s["action"].shape[0]),
                            proprio=s["proprio"][0].cuda(), num_inference_steps=10, seed=42)
    row = []
    for slot in ("rgb", "depth", "flow"):
        v = out.get(f"video_{slot}")
        if v is None: row.append(f"{slot}: none"); continue
        g = pil_frames_to_video_tensor(v)[:, 1:]
        ps = {t: float(video_psnr(pred=g, target=((s[f'video_{t}'].float().clamp(-1, 1) + 1) / 2)[:, 1:])) for t in ("rgb", "depth", "flow")}
        b = max(ps, key=ps.get); row.append(f"slot {slot} -> looks like {b} ({ps[b]:.1f} dB; vs GT {slot} {ps[slot]:.1f})")
    print(f"window {int(i):6d}: " + " | ".join(row), flush=True)
