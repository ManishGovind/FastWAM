"""RGB future prediction quality: context model (active_future=rgb) vs unimodal RGB vs MM.

Same dataset windows, same input frame / instruction / proprio, same noise seed and steps.
Each model's decoded RGB future (frames 1..T, frame 0 is the shared input) is compared with
the ground-truth RGB future: PSNR / SSIM (the trainer's evaluate() metrics).

    CUDA_VISIBLE_DEVICES=0 python notebooks/router_analysis/scripts/rgb_future_quality.py --n 24
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
from verify_m_context import BASELINES, CTX_RUN, CTX_TASK, build  # noqa: E402

from fastwam.utils.video_metrics import pil_frames_to_video_tensor, video_psnr, video_ssim  # noqa: E402


def gt01(v):
    return ((v.detach().float().cpu().clamp(-1, 1) + 1) * 0.5).contiguous()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    os.chdir("/data/mgovind/FastWAM")
    dt = torch.bfloat16

    models = {"ctx model (rgb)": (CTX_TASK, CTX_RUN, {"active_future": "rgb"}),
              "unimodal RGB": (*BASELINES["unimodal RGB"], {}),
              "MM (rgb stream)": (*BASELINES["MM (all futures)"], {})}
    ds = None
    res = {}
    for name, (task, run, extra) in models.items():
        t0 = time.time()
        cfg, m = build(task, run, "cuda", dt)
        if ds is None:
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
            frames = out.get("video_rgb") or out["video"]
            gen = pil_frames_to_video_tensor(frames)[:, 1:]
            gt = gt01(s["video_rgb"])[:, 1:]
            ps.append(float(video_psnr(pred=gen, target=gt)))
            ss.append(float(video_ssim(pred=gen, target=gt)))
        res[name] = {"psnr": ps, "ssim": ss}
        print(f"{name:18s} PSNR {np.mean(ps):6.2f} (median {np.median(ps):.2f})  SSIM {np.mean(ss):.3f}   "
              f"[{len(ps)} windows, {time.time() - t0:.0f}s]", flush=True)
        del m
        torch.cuda.empty_cache()

    base = np.array(res["unimodal RGB"]["psnr"])
    for name in res:
        if name == "unimodal RGB":
            continue
        d = np.array(res[name]["psnr"]) - base
        print(f"{name:18s} - unimodal RGB: mean {d.mean():+.2f} dB, better in {int((d > 0).sum())}/{len(d)} windows")
    out = "notebooks/router_analysis/outputs/rgb_future_quality.json"
    json.dump({"windows": idxs.tolist(), **res}, open(out, "w"))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
