"""Implementation check of the modality-context model before drawing conclusions.

B. Train/eval parity, end to end: ``infer_action`` on the eval path (prompt string -> online
   plain instruction + joint prompt from the cache) vs the training-style path (dataset
   ``context`` + ``prompt_text``), same seed. The only expected difference is T5 bf16
   numerics of the plain instruction (cached in batches vs encoded alone).
C. Open-loop action error on dataset windows (normalized space; all runs share identical
   dataset stats): the context model per stream (rgb / depth / flow) vs the unimodal RGB
   joint model and the MM model, same windows, same noise seed. A bug in the context model
   (wrong condition, wrong stream, broken action context) would show as a clearly larger
   error than the baselines on the data it was trained on.

    CUDA_VISIBLE_DEVICES=0 python notebooks/router_analysis/scripts/verify_m_context.py --n 24
"""

import argparse
import inspect
import json
import os
import time

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

REPO = "/data/mgovind/FastWAM"
CTX_TASK = "libero_joint_2cam224_1e-4_random_short_m_context_video"
CTX_RUN = "runs/libero_joint_2cam224_1e-4_random_short_m_context_video/2026-10-10_09-57-20"
BASELINES = {
    "unimodal RGB": ("libero_joint_2cam224_1e-4", "runs/libero_joint_2cam224_1e-4/2026-08-23_17-22-15"),
    "MM (all futures)": ("libero_joint_2cam224_1e-4_rgb_depth_flow_modality_specific", "runs/libero_joint_2cam224_1e-4_rgb_depth_flow/2026-08-28_09-13-18"),
}
STREAMS = ("rgb", "depth", "flow")


def build(task, run, device, dtype, step=21700):
    with initialize_config_dir(version_base="1.3", config_dir=f"{REPO}/configs"):
        cfg = compose(config_name="sim_libero", overrides=[f"task={task}"])  # eval config: loads T5
    model = instantiate(cfg.model, model_dtype=dtype, device=device)
    model.load_checkpoint(f"{run}/checkpoints/weights/step_{step:06d}.pt")
    return cfg, model.to(device).eval()


def call(model, **kw):
    sig = inspect.signature(model.infer_action).parameters
    with torch.no_grad():
        return model.infer_action(**{k: v for k, v in kw.items() if k in sig})["action"].float()


def act_err(pred, s):
    gt = s["action"].float()
    valid = ~s["action_is_pad"].bool() if "action_is_pad" in s else torch.ones(gt.shape[0], dtype=torch.bool)
    d = (pred - gt)[valid] ** 2
    return float(d.mean()), float(d[:, :-1].mean()), float(d[:, -1].mean())  # all, arm, gripper


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    os.chdir(REPO)
    dt = torch.bfloat16

    cfg, model = build(CTX_TASK, CTX_RUN, a.device, dt)
    ds = instantiate(cfg.data.train, pretrained_norm_stats=f"{CTX_RUN}/dataset_stats.json")
    idxs = np.linspace(0, len(ds) - 1, a.n).round().astype(int)
    samples = [ds[int(i)] for i in idxs]
    print(f"{type(model).__name__}: modality_prompt={model.modality_prompt} mode={model.modality_prompt_mode} "
          f"target={model.modality_prompt_target} select={model.modality_select} layout={model.router_layout}", flush=True)

    def kw(s, **extra):
        base = dict(input_image=s["video_rgb"][:, 0].to(a.device, dt), action_horizon=int(s["action"].shape[0]),
                    num_video_frames=int(s["video_rgb"].shape[1]), proprio=s["proprio"][0].to(a.device),
                    num_inference_steps=a.steps, seed=a.seed)
        base.update(extra)
        return base

    res = {"windows": idxs.tolist()}
    # B: parity (eval path vs training-style path), per stream
    print("\n=== B. eval path vs training-style path (same seed) ===", flush=True)
    for st in STREAMS:
        diffs, errs_eval, errs_train = [], [], []
        for s in samples:
            pe = call(model, **kw(s, prompt=s["prompt"], active_future=st))
            pt = call(model, **kw(s, prompt=None, context=s["context"].to(a.device), context_mask=s["context_mask"].to(a.device),
                                  prompt_text=s["prompt"], active_future=st))
            diffs.append(float((pe - pt).abs().max()))
            errs_eval.append(act_err(pe, s)[0]); errs_train.append(act_err(pt, s)[0])
        print(f"  {st:5s} max|a_eval - a_train| median {np.median(diffs):.4f} max {max(diffs):.4f} | "
              f"MSE eval {np.mean(errs_eval):.4f} vs train-path {np.mean(errs_train):.4f}", flush=True)
        res[f"B_{st}"] = {"maxdiff": diffs, "mse_eval": errs_eval, "mse_train": errs_train}

    # C: open-loop action error, context model per stream (eval path)
    print("\n=== C. open-loop action MSE (normalized; all / arm / gripper) ===", flush=True)
    table = {}
    for st in STREAMS:
        e = np.array([act_err(call(model, **kw(s, prompt=s["prompt"], active_future=st)), s) for s in samples])
        table[f"ctx {st}"] = e
    del model
    torch.cuda.empty_cache()
    for name, (task, run) in BASELINES.items():
        t0 = time.time()
        _, m = build(task, run, a.device, dt)
        table[name] = np.array([act_err(call(m, **kw(s, prompt=s["prompt"])), s) for s in samples])
        print(f"  ({name} done in {time.time() - t0:.0f}s)", flush=True)
        del m
        torch.cuda.empty_cache()
    for name, e in table.items():
        m = e.mean(0)
        print(f"  {name:18s} {m[0]:.4f} / {m[1]:.4f} / {m[2]:.4f}   median all {np.median(e[:, 0]):.4f}")
        res[f"C_{name}"] = e.tolist()
    out = "notebooks/router_analysis/outputs/verify_m_context.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, "w"))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
