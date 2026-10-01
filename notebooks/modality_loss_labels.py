"""Proxy modality labels from the training loss of the unimodal Joint checkpoints.

For every window, each unimodal model (RGB / Depth / Flow = the ``R0 + fut_k -> action``
path) is scored with its own flow-matching training loss. The label is the model with the
lowest action loss. Compared with full-inference action error this needs one forward pass
per noise draw instead of a 10-step sampler, and matches what an in-model router would see
during training.

To make the three losses comparable, each window gets K noise draws that are *identical
across models*: the same video/action noise tensors (seeded by the dataset index) and the
same stratified timesteps. Loss differences then come from the modality, not from which
noise a model happened to draw.

Usage::

    # one process per (model, shard); every GPU can hold several
    CUDA_VISIBLE_DEVICES=0 python notebooks/modality_loss_labels.py --model RGB --shard 0 --num-shards 4
    ...
    python notebooks/modality_loss_labels.py --analyze
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent))

from modality_labels import MODELS, SUITES, spread_episodes, suite_of  # noqa: E402
from repr_analysis import (  # noqa: E402
    JOINT_UNIMODAL,
    REPO,
    compose_task_cfg,
    episode_window_indices,
    load_dataset,
    load_joint_model,
)

OUT_DIR = REPO / "notebooks" / "outputs" / "modality_loss_labels"
STACK_KEYS = ("video", "action", "action_is_pad", "image_is_pad", "context", "context_mask", "proprio")


def _tag(n_episodes: int, n_windows: int, k: int, frame_stride: int = 0) -> str:
    ep = "all" if n_episodes <= 0 else str(n_episodes)
    win = f"stride{frame_stride}" if frame_stride > 0 else f"win{n_windows}"
    return f"ep{ep}_{win}_K{k}"


def episode_windows(dataset, episode: int, n_windows: int, frame_stride: int) -> list[int]:
    """Window start indices: every ``frame_stride`` frames (training-style), else ``n_windows`` evenly spaced."""
    if frame_stride <= 0:
        return episode_window_indices(dataset, episode, n_windows=n_windows)[0]
    epi = dataset.lerobot_dataset.episode_data_index
    start, end = int(epi["from"][episode].item()), int(epi["to"][episode].item())
    last = end - int(dataset.lerobot_dataset.obs_size)
    return list(range(start, last + 1, frame_stride))


def _cache_path(model: str, tag: str, shard: int, num_shards: int) -> Path:
    return OUT_DIR / f"{model}_{tag}_shard{shard}of{num_shards}.pt"


def stratified_timesteps(scheduler, k: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """K (t_video, t_action) pairs covering the training distribution, fixed for every window.

    Training draws u ~ U(0,1) independently for video and action and maps it through the
    shift warp. Stratifying u and pairing video/action strata in a fixed shuffled order keeps
    the same coverage without sampling noise.
    """
    u = (torch.arange(k, dtype=torch.float32) + 0.5) / k
    perm = torch.randperm(k, generator=torch.Generator().manual_seed(1234))
    to_t = lambda uu: scheduler._phi(uu, scheduler.shift) * float(scheduler.num_train_timesteps)  # noqa: E731
    return to_t(u[perm]).to(device), to_t(u).to(device)


@torch.no_grad()
def window_losses(model, samples: list[dict[str, Any]], dataset_idx: list[int], k: int) -> dict[str, np.ndarray]:
    """Per-window, per-draw action and video losses. Returns arrays shaped [n_windows, K]."""
    batch = {key: torch.stack([s[key] for s in samples]) for key in STACK_KEYS if key in samples[0]}
    inputs = model.build_inputs(batch)
    n = len(samples)
    rep = lambda x: None if x is None else x.repeat_interleave(k, dim=0)  # noqa: E731

    latents = rep(inputs["input_latents"])
    first = rep(inputs["first_frame_latents"])
    context, context_mask = rep(inputs["context"]), rep(inputs["context_mask"])
    action = rep(inputs["action"])
    action_is_pad, image_is_pad = rep(inputs["action_is_pad"]), rep(inputs["image_is_pad"])

    # Noise seeded by dataset index -> identical across the three models' processes.
    nv, na = [], []
    for idx in dataset_idx:
        g = torch.Generator().manual_seed(int(idx))
        nv.append(torch.randn((k, *inputs["input_latents"].shape[1:]), generator=g))
        na.append(torch.randn((k, *inputs["action"].shape[1:]), generator=g))
    noise_video = torch.cat(nv).to(device=latents.device, dtype=latents.dtype)
    noise_action = torch.cat(na).to(device=action.device, dtype=action.dtype)

    tv, ta = stratified_timesteps(model.train_video_scheduler, k, latents.device)
    t_video = tv.repeat(n).to(latents.dtype)
    t_action = ta.repeat(n).to(action.dtype)

    noisy_video = model.train_video_scheduler.add_noise(latents, noise_video, t_video)
    target_video = model.train_video_scheduler.training_target(latents, noise_video, t_video)
    if first is not None:
        noisy_video[:, :, 0:1] = first
    noisy_action = model.train_action_scheduler.add_noise(action, noise_action, t_action)
    target_action = model.train_action_scheduler.training_target(action, noise_action, t_action)

    patch_t, patch_h, patch_w = (int(s) for s in model.video_expert.patch_size)
    lt, lh, lw = noisy_video.shape[-3:]
    tpf = (lh // patch_h) * (lw // patch_w)
    mask = model._build_mot_attention_mask(
        video_seq_len=(lt // patch_t) * tpf,
        action_seq_len=noisy_action.shape[1],
        video_tokens_per_frame=tpf,
        device=noisy_video.device,
    )
    pred_video, pred_action = model._joint_denoise_core(
        latents_video=noisy_video,
        latents_action=noisy_action,
        timestep_video=t_video,
        timestep_action=t_action,
        context=context,
        context_mask=context_mask,
        attention_mask=mask,
        fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
        action_condition=action,
    )

    if first is not None:
        pred_video, target_video = pred_video[:, :, 1:], target_video[:, :, 1:]
    video_loss = model._compute_video_loss_per_sample(
        pred_video=pred_video, target_video=target_video, image_is_pad=image_is_pad,
        include_initial_video_step=first is None,
    )

    tok = F.mse_loss(pred_action.float(), target_action.float(), reduction="none")  # [B, H, D]
    valid = torch.ones(tok.shape[:2], device=tok.device) if action_is_pad is None else (~action_is_pad).float()
    denom = valid.sum(1).clamp(min=1.0)
    action_loss = (tok.mean(2) * valid).sum(1) / denom
    action_loss_delta = (tok[..., :6].mean(2) * valid).sum(1) / denom

    shape = (n, k)
    return {
        "action_loss": action_loss.view(shape).cpu().numpy(),
        "action_loss_delta": action_loss_delta.view(shape).cpu().numpy(),
        "video_loss": video_loss.float().view(shape).cpu().numpy(),
        "w_action": model.train_action_scheduler.training_weight(t_action).float().view(shape).cpu().numpy(),
        "w_video": model.train_video_scheduler.training_weight(t_video).float().view(shape).cpu().numpy(),
        "r0_latent": inputs["input_latents"][:, :, 0].float(),
    }


def _router_features(sample: dict[str, Any], r0_latent: torch.Tensor) -> dict[str, np.ndarray]:
    return {
        "r0_pool": F.adaptive_avg_pool2d(r0_latent[None], 2).flatten().cpu().numpy().astype(np.float16),
        "proprio": sample["proprio"][0].float().numpy(),
    }


def _text_mean(sample: dict[str, Any]) -> np.ndarray:
    ctx = sample["context"].float()
    m = sample["context_mask"].float().unsqueeze(-1)
    return ((ctx * m).sum(0) / m.sum(0).clamp(min=1.0)).numpy().astype(np.float16)


def collect(model_name: str, *, n_episodes: int, n_windows: int, k: int, shard: int, num_shards: int,
            batch_windows: int, save_every: int, frame_stride: int = 0) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = _tag(n_episodes, n_windows, k, frame_stride)
    out_path = _cache_path(model_name, tag, shard, num_shards)
    spec = JOINT_UNIMODAL[model_name]
    cfg = compose_task_cfg(spec["task"])
    dataset = load_dataset(cfg, Path(spec["run_dir"]) / "dataset_stats.json")
    # Same selection as modality_labels.py, so --n-episodes 50 --n-windows 8 reproduces the
    # pilot's windows exactly.
    episodes = spread_episodes(dataset, n_episodes)[shard::num_shards]

    state: dict[str, Any] = {"model": model_name, "episodes": episodes, "k": k, "n_windows": n_windows,
                             "frame_stride": frame_stride, "ckpt": str(spec["ckpt"]),
                             "records": [], "text_features": {}}
    if out_path.exists():
        state = torch.load(out_path, map_location="cpu", weights_only=False)
        state.setdefault("text_features", {})
    done = {(r["episode"], r["window"]) for r in state["records"]}

    # (episode, window, dataset index, progress in episode) for every remaining window.
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
    t0, n_seen, since_save = time.time(), 0, 0
    pos = 0
    for samples in loader:
        meta = items[pos: pos + len(samples)]
        pos += len(samples)
        out = window_losses(model, samples, [it[2] for it in meta], k)
        for j, (ep, w, idx, prog) in enumerate(meta):
            rec = {
                "episode": ep, "window": w, "frame_idx": idx, "progress": float(prog),
                **{key: out[key][j] for key in ("action_loss", "action_loss_delta", "video_loss",
                                                "w_action", "w_video")},
            }
            if want_features:
                prompt = samples[j].get("prompt")
                rec["features"] = _router_features(samples[j], out["r0_latent"][j])
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
            left = (len(items) - n_seen) / max(rate, 1e-9) / 3600
            print(f"[{model_name} {shard}/{num_shards}] {n_seen}/{len(items)} windows, "
                  f"{rate:.2f} win/s, ~{left:.1f} h left", flush=True)
    torch.save(state, out_path)
    print(f"[{model_name} {shard}/{num_shards}] done", flush=True)
    del model
    gc.collect()
    return out_path


# --------------------------------------------------------------------------------------
# analysis + label export
# --------------------------------------------------------------------------------------


def _load(model: str, tag: str) -> tuple[dict[tuple[int, int], dict], dict[str, np.ndarray]]:
    files = sorted(OUT_DIR.glob(f"{model}_{tag}_shard*of*.pt"))
    if not files:
        raise FileNotFoundError(f"No loss-label cache for {model} {tag} in {OUT_DIR}")
    recs, text = {}, {}
    for f in files:
        state = torch.load(f, map_location="cpu", weights_only=False)
        text.update(state.get("text_features", {}))
        for r in state["records"]:
            recs[(r["episode"], r["window"])] = r
    return recs, text


def _action_error_labels(keys, n_episodes: int, n_windows: int) -> np.ndarray | None:
    """Labels from modality_labels.py (full-inference action error) for the same windows, if cached."""
    from modality_labels import DELTA, _load_aligned

    try:
        d = _load_aligned(n_episodes, n_windows, (0, 1, 2))
    except (FileNotFoundError, ValueError):
        return None
    err = ((d["pred"][..., DELTA] - d["gt"][None, :, None, :, DELTA]) ** 2).mean(axis=(-1, -2)).mean(2)
    lookup = dict(zip(d["keys"], err.argmin(0)))
    if not all(key in lookup for key in keys):
        return None
    return np.array([lookup[key] for key in keys])


def analyze(n_episodes: int, n_windows: int, k: int, frame_stride: int = 0) -> dict[str, Any]:
    tag = _tag(n_episodes, n_windows, k, frame_stride)
    loaded = [_load(m, tag) for m in MODELS]
    per_model = [recs for recs, _ in loaded]
    text_features = loaded[0][1]
    keys = sorted(set.intersection(*[set(r) for r in per_model]))
    if not keys:
        raise ValueError("No windows shared by all three models yet.")
    n = len(keys)

    def stack(field: str) -> np.ndarray:  # [M, N, K]
        return np.stack([np.stack([per_model[m][key][field] for key in keys]) for m in range(3)])

    a_loss, a_delta, v_loss = stack("action_loss"), stack("action_loss_delta"), stack("video_loss")
    w_a = stack("w_action")
    weighted = (a_loss * w_a).mean(axis=2)  # training objective, per window  [M, N]
    label = weighted.argmin(axis=0)
    counts = np.bincount(label, minlength=3)

    # Stability: split the K draws into two interleaved halves; does each half pick the same
    # model? Action timesteps are stored in increasing order, so a contiguous split would
    # compare low-noise against high-noise draws.
    ha, hb = slice(0, None, 2), slice(1, None, 2)
    lab_a = (a_loss[:, :, ha] * w_a[:, :, ha]).mean(2).argmin(0)
    lab_b = (a_loss[:, :, hb] * w_a[:, :, hb]).mean(2).argmin(0)
    split_agree = float((lab_a == lab_b).mean())
    freq = counts / n
    chance = float((freq ** 2).sum())

    # Paired comparison per model pair: how often is the gap consistent across draws?
    sign_consistency = {}
    for i in range(3):
        for j in range(i + 1, 3):
            diff = a_loss[i] - a_loss[j]  # [N, K]
            sign_consistency[f"{MODELS[i]}<{MODELS[j]}"] = float((diff < 0).mean())

    # Gain if we could pick per window (choose on half A, score on half B).
    score_b = (a_loss[:, :, hb] * w_a[:, :, hb]).mean(2)
    best_i = int(score_b.mean(1).argmin())
    honest = float(score_b[lab_a, np.arange(n)].mean())
    best_single = float(score_b[best_i].mean())

    episodes = np.array([key[0] for key in keys])
    progress = np.array([per_model[0][key].get("progress", key[1] / max(n_windows - 1, 1)) for key in keys])
    suites = np.array([suite_of(int(e)) for e in episodes])

    print(f"\n=== loss proxy labels: {n} windows, {len(set(episodes))} episodes, K={k} ===")
    print(f"labels (argmin weighted action loss): " + ", ".join(f"{MODELS[m]} {counts[m]} ({100 * freq[m]:.0f}%)" for m in range(3)))
    print(f"mean weighted action loss: " + ", ".join(f"{MODELS[m]} {weighted[m].mean():.5f}" for m in range(3)))
    print(f"mean video loss (own stream): " + ", ".join(f"{MODELS[m]} {v_loss[m].mean():.5f}" for m in range(3)))
    print(f"label stability (K/2 vs K/2 draws): {split_agree:.2f}  (chance given frequencies {chance:.2f})")
    print(f"per-draw win rate: {sign_consistency}")
    print(f"pick-per-window (chosen on half A, scored on half B): {honest:.5f} vs best single "
          f"{MODELS[best_i]} {best_single:.5f} ({100 * (1 - honest / best_single):+.1f}%)")
    err_label = _action_error_labels(keys, n_episodes, n_windows) if frame_stride <= 0 else None
    agree_err = None
    if err_label is not None:
        agree_err = float((err_label == label).mean())
        ef = np.bincount(err_label, minlength=3) / n
        conf = np.zeros((3, 3), dtype=int)
        for a, b in zip(label, err_label):
            conf[a, b] += 1
        print(f"\nagreement with action-error labels (modality_labels.py): {agree_err:.2f} "
              f"(chance given both frequencies {float((freq * ef).sum()):.2f})")
        print("  rows = loss label, cols = action-error label (RGB, Depth, Flow)")
        for m in range(3):
            print(f"  {MODELS[m]:6s} {conf[m].tolist()}")
    print("\nby suite:")
    for name, _ in SUITES:
        sel = suites == name
        if sel.any():
            c = np.bincount(label[sel], minlength=3)
            print(f"  {name:15s} n={sel.sum():5d} labels R/D/F = {c.tolist()}  "
                  f"loss R/D/F = {' / '.join(f'{weighted[m, sel].mean():.5f}' for m in range(3))}")
    print("\nby position in episode (progress bins):")
    bins = np.minimum((progress * 8).astype(int), 7)
    for b in range(8):
        sel = bins == b
        if sel.any():
            c = np.bincount(label[sel], minlength=3) / sel.sum() * 100
            print(f"  {100 * b / 8:3.0f}-{100 * (b + 1) / 8:3.0f}%: R {c[0]:4.0f}%  D {c[1]:4.0f}%  F {c[2]:4.0f}%"
                  f"   (n={sel.sum()})")

    # Export labels + router features for training. Per-half losses let the router be
    # trained on labels from one half of the draws and scored on the other.
    feats = [per_model[0][key].get("features") for key in keys]
    export = {
        "keys": keys,
        "episode": episodes,
        "suite": suites,
        "progress": progress,
        "label": label,
        "label_half_a": lab_a,
        "weighted_action_loss": weighted.T,  # [N, M]
        "loss_half_a": (a_loss[:, :, ha] * w_a[:, :, ha]).mean(2).T,
        "loss_half_b": score_b.T,
        "soft_target": torch.softmax(torch.from_numpy(-weighted.T / max(weighted.std(), 1e-8)), dim=1).numpy(),
        "models": list(MODELS),
    }
    if all(f is not None for f in feats):
        export["r0_pool"] = np.stack([f["r0_pool"] for f in feats]).astype(np.float32)
        export["proprio"] = np.stack([f["proprio"] for f in feats]).astype(np.float32)
        if "text_mean" in feats[0]:
            export["text_mean"] = np.stack([f["text_mean"] for f in feats]).astype(np.float32)
        else:
            prompts = [per_model[0][key]["prompt"] for key in keys]
            export["text_mean"] = np.stack([text_features[p] for p in prompts]).astype(np.float32)
    if err_label is not None:
        export["action_error_label"] = err_label
    out = OUT_DIR / f"labels_{tag}.pt"
    torch.save(export, out)
    summary = {
        "n_windows": n, "label_counts": {MODELS[m]: int(counts[m]) for m in range(3)},
        "split_agreement": split_agree, "chance": chance, "win_rates": sign_consistency,
        "pick_per_window": honest, "best_single": MODELS[best_i], "best_single_loss": best_single,
        "agreement_with_action_error_labels": agree_err,
    }
    with open(OUT_DIR / f"summary_{tag}.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    print(f"\nwrote {out}")
    return summary


if __name__ == "__main__":
    os.chdir(REPO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--n-episodes", type=int, default=50,
                        help="evenly spaced episodes; 50 matches the modality_labels.py pilot, 0 = all")
    parser.add_argument("--n-windows", type=int, default=8)
    parser.add_argument("--k", type=int, default=8, help="noise/timestep draws per window")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--batch-windows", type=int, default=2)
    parser.add_argument("--frame-stride", type=int, default=0,
                        help="window every N frames of every episode (training-style); 0 = --n-windows evenly spaced")
    parser.add_argument("--save-every", type=int, default=2000, help="windows between checkpoints")
    args = parser.parse_args()

    if args.analyze:
        analyze(args.n_episodes, args.n_windows, args.k, args.frame_stride)
    elif args.model:
        collect(args.model, n_episodes=args.n_episodes, n_windows=args.n_windows, k=args.k, shard=args.shard,
                num_shards=args.num_shards, batch_windows=args.batch_windows, save_every=args.save_every,
                frame_stride=args.frame_stride)
    else:
        parser.error("pass --model NAME to collect or --analyze")
