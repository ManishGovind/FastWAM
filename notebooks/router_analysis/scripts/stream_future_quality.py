"""Future prediction quality per stream: context model (active_future=<stream>) vs the
unimodal joint model trained on that stream (RGB input frame -> <stream> future).

Same dataset windows, same input frame / instruction / proprio, same noise seed and steps.
Decoded future frames 1..T vs the ground-truth <stream> future: PSNR / SSIM.

    CUDA_VISIBLE_DEVICES=0 python notebooks/router_analysis/scripts/stream_future_quality.py --stream depth --n 24
"""

import argparse
import inspect
import json
import os
import sys
import time

import numpy as np
import torch
from hydra.utils import instantiate

sys.path.insert(0, os.path.dirname(__file__))
from verify_m_context import CTX_RUN, CTX_TASK, build  # noqa: E402

from fastwam.utils.video_metrics import pil_frames_to_video_tensor, video_psnr, video_ssim  # noqa: E402

# (task, run, checkpoint step) -- the checkpoints used in the LIBERO / LIBERO-Plus evals
UNIMODAL = {
    "rgb": ("libero_joint_2cam224_1e-4", "runs/libero_joint_2cam224_1e-4/2026-08-23_17-22-15", 21700),
    "depth": ("libero_joint_2cam224_1e-4_depth", "runs/libero_joint_2cam224_1e-4_depth/2026-08-21_23-21-22", 21700),
    "flow": ("libero_joint_2cam224_1e-4_flow", "runs/libero_joint_2cam224_1e-4_flow/2026-08-22_14-11-53", 21000),
}


def to01(v):
    return ((v.detach().float().cpu().clamp(-1, 1) + 1) * 0.5).contiguous()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", choices=list(UNIMODAL), required=True)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    os.chdir("/data/mgovind/FastWAM")
    dt = torch.bfloat16
    models = {f"ctx model ({a.stream})": (CTX_TASK, CTX_RUN, 21700, {"active_future": a.stream}),
              f"unimodal {a.stream}": (*UNIMODAL[a.stream], {})}
    samples, idxs, res = None, None, {}
    for name, (task, run, step, extra) in models.items():
        t0 = time.time()
        cfg, m = build(task, run, "cuda", dt, step=step)
        if samples is None:  # multimodal dataset: has video_rgb / video_depth / video_flow
            ds = instantiate(cfg.data.train, pretrained_norm_stats=f"{CTX_RUN}/dataset_stats.json")
            idxs = np.linspace(0, len(ds) - 1, a.n).round().astype(int)
            samples = [ds[int(i)] for i in idxs]
        sig = inspect.signature(m.infer_joint).parameters
        ps, ss = [], []
        for s in samples:
            kw = dict(prompt=s["prompt"], input_image=s["video_rgb"][:, 0].cuda().to(dt),
                      num_video_frames=int(s["video_rgb"].shape[1]), action_horizon=int(s["action"].shape[0]),
                      proprio=s["proprio"][0].cuda(), num_inference_steps=a.steps, seed=a.seed, **extra)
            with torch.no_grad():
                out = m.infer_joint(**{k: v for k, v in kw.items() if k in sig})
            gen = pil_frames_to_video_tensor(out["video"])[:, 1:]
            gt = to01(s[f"video_{a.stream}"])[:, 1:]
            ps.append(float(video_psnr(pred=gen, target=gt)))
            ss.append(float(video_ssim(pred=gen, target=gt)))
        res[name] = {"psnr": ps, "ssim": ss}
        print(f"{name:20s} PSNR {np.mean(ps):6.2f} (median {np.median(ps):.2f})  SSIM {np.mean(ss):.3f}   "
              f"[{len(ps)} windows, {time.time() - t0:.0f}s]", flush=True)
        del m
        torch.cuda.empty_cache()
    names = list(res)
    d = np.array(res[names[0]]["psnr"]) - np.array(res[names[1]]["psnr"])
    print(f"{names[0]} - {names[1]}: mean {d.mean():+.2f} dB, better in {int((d > 0).sum())}/{len(d)} windows")
    out = f"notebooks/router_analysis/outputs/future_quality_{a.stream}.json"
    json.dump({"windows": idxs.tolist(), **res}, open(out, "w"))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
