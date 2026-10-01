"""Modality labels from action error, on the full LIBERO training data.

For every window each unimodal model (RGB / Depth / Flow) samples action chunks exactly as
at eval (``infer_action``: 10-step flow-matching sampler, RGB frame-0 condition, T5 context +
proprio) for several seeds, and the label is the model whose chunk is closest to the demo
action on the 6 delta-EEF dims. Initial noise for seed ``s`` is generated the same way as
``FastWAMJoint.infer_action(seed=s)``, so the three models are compared on identical noise
and results match the single-window pilot in ``modality_labels.py``.

Sampling is batched over windows x seeds, which is what makes the full dataset affordable.

Usage::

    CUDA_VISIBLE_DEVICES=0 python notebooks/modality_action_labels.py --model RGB --frame-stride 4 --shard 0 --num-shards 4
    ...
    python notebooks/modality_action_labels.py --analyze --frame-stride 4
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent))

from modality_labels import MODELS, SUITES, spread_episodes, suite_of  # noqa: E402
from modality_loss_labels import _router_features, _text_mean, episode_windows  # noqa: E402
from repr_analysis import JOINT_UNIMODAL, REPO, compose_task_cfg, load_dataset, load_joint_model  # noqa: E402

OUT_DIR = REPO / "notebooks" / "outputs" / "modality_action_labels"
DELTA = slice(0, 6)
GRIPPER = 6


def _tag(n_episodes: int, n_windows: int, frame_stride: int, seeds: tuple[int, ...]) -> str:
    ep = "all" if n_episodes <= 0 else str(n_episodes)
    win = f"stride{frame_stride}" if frame_stride > 0 else f"win{n_windows}"
    return f"ep{ep}_{win}_s{'-'.join(map(str, seeds))}"


@torch.no_grad()
def sample_actions(model, samples: list[dict[str, Any]], seeds: tuple[int, ...], cfg) -> tuple[np.ndarray, torch.Tensor]:
    """Batched ``infer_action``. Returns actions [B, S, H, D] and frame-0 latents [B, C, h, w]."""
    n, s_count = len(samples), len(seeds)
    num_frames = int(cfg.data.train.num_frames)
    horizon = num_frames - 1
    num_video_frames = (num_frames - 1) // int(cfg.data.train.action_video_freq_ratio) + 1
    steps = int(cfg.EVALUATION.num_inference_steps)
    shift = cfg.EVALUATION.get("sigma_shift")
    shift = None if shift is None else float(shift)
    dtype, dev = model.torch_dtype, model.device

    first = torch.cat([
        model._encode_input_image_latents_tensor(s["video"][:, 0].unsqueeze(0).to(dev, dtype)) for s in samples
    ])  # [B, C, 1, h, w]
    context = torch.stack([s["context"] for s in samples]).to(dev, dtype)
    context_mask = torch.stack([s["context_mask"] for s in samples]).to(dev, torch.bool)
    proprio = torch.stack([s["proprio"][0] for s in samples]).to(dev, dtype)
    context, context_mask = model._append_proprio_to_context(context, context_mask, proprio)

    _, c, _, h, w = first.shape
    lt = (num_video_frames - 1) // model.vae.temporal_downsample_factor + 1
    # Same generators as FastWAMJoint.infer_action(seed=s): separate video/action streams.
    nv = torch.stack([torch.randn((c, lt, h, w), generator=torch.Generator().manual_seed(s)) for s in seeds])
    na = torch.stack([torch.randn((horizon, model.action_expert.action_dim),
                                  generator=torch.Generator().manual_seed(s)) for s in seeds])
    rep = lambda x: x.repeat_interleave(s_count, dim=0)  # noqa: E731  [B*S, ...], window-major
    latents_video = nv.repeat(n, 1, 1, 1, 1).to(dev, dtype)
    latents_action = na.repeat(n, 1, 1).to(dev, dtype)
    first_r = rep(first)
    latents_video[:, :, 0:1] = first_r
    context, context_mask = rep(context), rep(context_mask)
    fuse = bool(getattr(model.video_expert, "fuse_vae_embedding_in_latents", False))

    tv, dv = model.infer_video_scheduler.build_inference_schedule(steps, dev, latents_video.dtype, shift_override=shift)
    ta, da = model.infer_action_scheduler.build_inference_schedule(steps, dev, latents_action.dtype, shift_override=shift)
    b = latents_video.shape[0]
    for t_v, d_v, t_a, d_a in zip(tv, dv, ta, da):
        pv, pa = model._predict_joint_noise(
            latents_video=latents_video,
            latents_action=latents_action,
            timestep_video=t_v.expand(b),
            timestep_action=t_a.expand(b),
            context=context,
            context_mask=context_mask,
            fuse_vae_embedding_in_latents=fuse,
            gt_action=None,
        )
        latents_video = model.infer_video_scheduler.step(pv, d_v, latents_video)
        latents_action = model.infer_action_scheduler.step(pa, d_a, latents_action)
        latents_video[:, :, 0:1] = first_r
    actions = latents_action.float().view(n, s_count, horizon, -1).cpu().numpy()
    return actions, first[:, :, 0].float()


def collect(model_name: str, *, n_episodes: int, n_windows: int, frame_stride: int, seeds: tuple[int, ...],
            shard: int, num_shards: int, batch_windows: int, save_every: int) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = _tag(n_episodes, n_windows, frame_stride, seeds)
    out_path = OUT_DIR / f"{model_name}_{tag}_shard{shard}of{num_shards}.pt"
    spec = JOINT_UNIMODAL[model_name]
    cfg = compose_task_cfg(spec["task"])
    dataset = load_dataset(cfg, Path(spec["run_dir"]) / "dataset_stats.json")
    episodes = spread_episodes(dataset, n_episodes)[shard::num_shards]

    state: dict[str, Any] = {"model": model_name, "episodes": episodes, "seeds": list(seeds),
                             "ckpt": str(spec["ckpt"]), "records": [], "text_features": {}}
    if out_path.exists():
        state = torch.load(out_path, map_location="cpu", weights_only=False)
    done = {(r["episode"], r["window"]) for r in state["records"]}

    items = []
    for ep in episodes:
        idxs = episode_windows(dataset, ep, n_windows, frame_stride)
        span = max(idxs[-1] - idxs[0], 1)
        items += [(ep, w, int(i), (i - idxs[0]) / span) for w, i in enumerate(idxs) if (ep, w) not in done]
    print(f"[{model_name} {shard}/{num_shards}] {len(episodes)} episodes, {len(done)} windows done, "
          f"{len(items)} to go -> {out_path}", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_joint_model(cfg, Path(spec["ckpt"]), device=device)
    loader = DataLoader(Subset(dataset, [it[2] for it in items]), batch_size=batch_windows,
                        num_workers=6, collate_fn=lambda x: x, prefetch_factor=4)

    want_features = model_name == "RGB"
    t0, n_seen, since_save, pos = time.time(), 0, 0, 0
    for samples in loader:
        meta = items[pos: pos + len(samples)]
        pos += len(samples)
        actions, r0 = sample_actions(model, samples, seeds, cfg)
        for j, (ep, w, idx, prog) in enumerate(meta):
            pad = samples[j].get("action_is_pad")
            rec = {
                "episode": ep, "window": w, "frame_idx": idx, "progress": float(prog),
                "pred": actions[j].astype(np.float16),  # [S, H, D]
                "gt": samples[j]["action"].float().numpy(),
                "action_is_pad": None if pad is None else pad.numpy(),
            }
            if want_features:
                prompt = samples[j].get("prompt")
                rec["features"] = _router_features(samples[j], r0[j])
                rec["prompt"] = prompt
                if prompt not in state["text_features"]:
                    state["text_features"][prompt] = _text_mean(samples[j])
            state["records"].append(rec)
        n_seen += len(samples)
        since_save += len(samples)
        if since_save >= save_every:
            torch.save(state, out_path)
            since_save = 0
            rate = n_seen / (time.time() - t0)
            print(f"[{model_name} {shard}/{num_shards}] {n_seen}/{len(items)} windows, {rate:.2f} win/s, "
                  f"~{(len(items) - n_seen) / max(rate, 1e-9) / 3600:.1f} h left", flush=True)
    torch.save(state, out_path)
    print(f"[{model_name} {shard}/{num_shards}] done", flush=True)
    del model
    gc.collect()
    return out_path


# --------------------------------------------------------------------------------------
# analysis + label export (same export format as modality_loss_labels.py, so
# train_modality_router.py consumes it unchanged)
# --------------------------------------------------------------------------------------


def _load(model: str, tag: str):
    files = sorted(OUT_DIR.glob(f"{model}_{tag}_shard*of*.pt"))
    if not files:
        raise FileNotFoundError(f"No action-label cache for {model} {tag} in {OUT_DIR}")
    recs, text = {}, {}
    for f in files:
        st = torch.load(f, map_location="cpu", weights_only=False)
        text.update(st.get("text_features", {}))
        for r in st["records"]:
            recs[(r["episode"], r["window"])] = r
    return recs, text


def analyze(n_episodes: int, n_windows: int, frame_stride: int, seeds: tuple[int, ...],
            gripper_weight: float = 0.0) -> dict[str, Any]:
    """gripper_weight < 0 balances the gripper term to the same mean as the delta-EEF error."""
    tag = _tag(n_episodes, n_windows, frame_stride, seeds)
    loaded = [_load(m, tag) for m in MODELS]
    per_model = [r for r, _ in loaded]
    text = loaded[0][1]
    keys = sorted(set.intersection(*[set(r) for r in per_model]))
    if not keys:
        raise ValueError("No windows shared by all three models yet.")
    n, s_count = len(keys), len(seeds)
    if s_count < 2:
        raise ValueError("Need >= 2 seeds for the held-out-seed evaluation.")

    pred = np.stack([np.stack([per_model[m][k]["pred"] for k in keys]) for m in range(3)]).astype(np.float32)
    gt = np.stack([per_model[0][k]["gt"] for k in keys])  # [N, H, D]
    pad = per_model[0][keys[0]]["action_is_pad"]
    valid = (np.ones(gt.shape[:2]) if pad is None
             else ~np.stack([per_model[0][k]["action_is_pad"] for k in keys])).astype(np.float32)  # [N, H]

    def masked_mean(x):  # [..., H] -> [...]
        return (x * valid[None, :, None]).sum(-1) / valid.sum(-1).clip(min=1)[None, :, None]

    delta_err = masked_mean(((pred[..., DELTA] - gt[None, :, None, :, DELTA]) ** 2).mean(-1))  # [M, N, S]
    # Gripper commands are near-binary, so score open/close agreement rather than MSE.
    grip_err = masked_mean((np.sign(pred[..., GRIPPER]) != np.sign(gt[None, :, None, :, GRIPPER])).astype(np.float32))
    if gripper_weight < 0:
        gripper_weight = float(delta_err.mean() / max(grip_err.mean(), 1e-8))
    err = delta_err + gripper_weight * grip_err
    ha, hb = slice(0, None, 2), slice(1, None, 2)
    e_all, e_a, e_b = err.mean(2), err[:, :, ha].mean(2), err[:, :, hb].mean(2)
    label, lab_a, lab_b = e_all.argmin(0), e_a.argmin(0), e_b.argmin(0)
    counts = np.bincount(label, minlength=3)
    freq = counts / n
    stability = float((lab_a == lab_b).mean())
    chance = float(((np.bincount(lab_a, minlength=3) / n) * (np.bincount(lab_b, minlength=3) / n)).sum())
    best_i = int(e_b.mean(1).argmin())
    pick = float(e_b[lab_a, np.arange(n)].mean())
    best = float(e_b[best_i].mean())

    # soft_mean on the half-B chunks: the training-free bar.
    fused = pred[:, :, hb].mean(axis=(0, 2))  # average over models and half-B seeds -> [N, H, D]
    fused_step = (((fused[..., DELTA] - gt[..., DELTA]) ** 2).mean(-1)
                  + gripper_weight * (np.sign(fused[..., GRIPPER]) != np.sign(gt[..., GRIPPER])))
    fused_err = float(((fused_step * valid).sum(-1) / valid.sum(-1).clip(min=1)).mean())

    if gripper_weight > 0:
        tag = f"{tag}_grip{gripper_weight:.4g}"

    episodes = np.array([k[0] for k in keys])
    progress = np.array([per_model[0][k]["progress"] for k in keys])
    suites = np.array([suite_of(int(e)) for e in episodes])

    print(f"\n=== action-error labels: {n} windows, {len(set(episodes))} episodes, seeds {list(seeds)} ===")
    print("labels: " + ", ".join(f"{MODELS[m]} {counts[m]} ({100 * freq[m]:.0f}%)" for m in range(3)))
    print(f"gripper weight: {gripper_weight:.4g}")
    print("delta-EEF MSE:          " + ", ".join(f"{MODELS[m]} {delta_err[m].mean():.5f}" for m in range(3)))
    print("gripper mismatch rate:  " + ", ".join(f"{MODELS[m]} {grip_err[m].mean():.4f}" for m in range(3)))
    print("mean action error (all seeds): " + ", ".join(f"{MODELS[m]} {e_all[m].mean():.5f}" for m in range(3)))
    print(f"label stability (seeds {list(seeds[ha])} vs {list(seeds[hb])}): {stability:.2f} (chance {chance:.2f})")
    print(f"pick per window (label from half A, scored on half B): {pick:.5f} vs always {MODELS[best_i]} "
          f"{best:.5f} ({100 * (1 - pick / best):+.1f}%)")
    print(f"average of all models' chunks (half B): {fused_err:.5f} ({100 * (1 - fused_err / best):+.1f}%)")
    print("\nby suite:")
    for name, _ in SUITES:
        sel = suites == name
        if sel.any():
            c = np.bincount(label[sel], minlength=3)
            print(f"  {name:15s} n={sel.sum():6d} labels R/D/F = {c.tolist()}  err R/D/F = "
                  + " / ".join(f"{e_all[m, sel].mean():.5f}" for m in range(3)))
    print("\nby position in episode:")
    bins = np.minimum((progress * 8).astype(int), 7)
    for bi in range(8):
        sel = bins == bi
        if sel.any():
            c = np.bincount(label[sel], minlength=3) / sel.sum() * 100
            print(f"  {100 * bi / 8:3.0f}-{100 * (bi + 1) / 8:3.0f}%: R {c[0]:4.0f}%  D {c[1]:4.0f}%  F {c[2]:4.0f}%  (n={sel.sum()})")

    feats = [per_model[0][k].get("features") for k in keys]
    export = {
        "keys": keys, "episode": episodes, "suite": suites, "progress": progress,
        "label": label, "label_half_a": lab_a,
        "weighted_action_loss": e_all.T, "loss_half_a": e_a.T, "loss_half_b": e_b.T,
        "models": list(MODELS), "label_source": "action_error", "gripper_weight": gripper_weight,
        "delta_err": delta_err.mean(2).T, "gripper_mismatch": grip_err.mean(2).T,
    }
    if all(f is not None for f in feats):
        export["r0_pool"] = np.stack([f["r0_pool"] for f in feats]).astype(np.float32)
        export["proprio"] = np.stack([f["proprio"] for f in feats]).astype(np.float32)
        export["text_mean"] = np.stack([text[per_model[0][k]["prompt"]] for k in keys]).astype(np.float32)
    out = OUT_DIR / f"labels_{tag}.pt"
    torch.save(export, out)
    summary = {"n_windows": n, "gripper_weight": gripper_weight, "label_counts": {MODELS[m]: int(counts[m]) for m in range(3)},
               "stability": stability, "chance": chance, "pick_per_window": pick,
               "best_single": MODELS[best_i], "best_single_err": best, "average_chunks_err": fused_err}
    with open(OUT_DIR / f"summary_{tag}.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    print(f"\nwrote {out}")
    return summary


if __name__ == "__main__":
    os.chdir(REPO)
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=MODELS)
    p.add_argument("--analyze", action="store_true")
    p.add_argument("--n-episodes", type=int, default=0, help="0 = all episodes")
    p.add_argument("--n-windows", type=int, default=8)
    p.add_argument("--frame-stride", type=int, default=4, help="window every N frames; 0 = --n-windows evenly spaced")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--batch-windows", type=int, default=4)
    p.add_argument("--save-every", type=int, default=2000, help="windows between checkpoints")
    p.add_argument("--gripper-weight", type=float, default=-1.0,
                   help="weight on gripper open/close mismatch rate; -1 = balance to delta-EEF error, 0 = ignore")
    a = p.parse_args()
    seeds = tuple(a.seeds)
    if a.analyze:
        analyze(a.n_episodes, a.n_windows, a.frame_stride, seeds, gripper_weight=a.gripper_weight)
    elif a.model:
        collect(a.model, n_episodes=a.n_episodes, n_windows=a.n_windows, frame_stride=a.frame_stride, seeds=seeds,
                shard=a.shard, num_shards=a.num_shards, batch_windows=a.batch_windows, save_every=a.save_every)
    else:
        p.error("pass --model NAME to collect or --analyze")
