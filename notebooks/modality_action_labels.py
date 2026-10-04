"""Offline GT modality labels from unimodal action error.

For each FastWAM training window start, run RGB / Depth / Flow ``infer_action``
(several seeds) and label the modality with lowest action error vs the demo.

Training alignment
------------------
Window starts match ``RobotVideoDataset.__getitem__(idx)``:

    range(ep_from, ep_to - obs_size + 1, global_sample_stride)

All episodes with ``length >= obs_size`` are included.

Usage
-----
    # Collect one model (one GPU per shard)
    CUDA_VISIBLE_DEVICES=0 python notebooks/modality_action_labels.py \\
        --model RGB --frame-stride 1 --seeds 42 43 44 --shard 0 --num-shards 4

    # After RGB + Depth + Flow shards exist:
    python notebooks/modality_action_labels.py --analyze --frame-stride 1 --seeds 42 43 44
    python notebooks/modality_action_labels.py --verify  --frame-stride 1 --seeds 42 43 44
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

from repr_analysis import (  # noqa: E402
    JOINT_UNIMODAL,
    REPO,
    compose_task_cfg,
    load_dataset,
    load_joint_model,
)

OUT_DIR = REPO / "notebooks" / "outputs" / "modality_action_labels"
MODELS = ("RGB", "Depth", "Flow")
# Episode counts in dataset_dirs order (configs/data/libero_2cam*.yaml).
SUITES = (
    ("libero_spatial", 434),
    ("libero_object", 457),
    ("libero_goal", 433),
    ("libero_10", 388),
)
DELTA = slice(0, 6)
GRIPPER = 6


# ---------------------------------------------------------------------------
# Indexing (must match FastWAM training)
# ---------------------------------------------------------------------------


def suite_of(episode: int) -> str:
    start = 0
    for name, count in SUITES:
        if episode < start + count:
            return name
        start += count
    return "unknown"


def run_tag(n_episodes: int, frame_stride: int, seeds: tuple[int, ...]) -> str:
    ep = "all" if n_episodes <= 0 else str(n_episodes)
    return f"ep{ep}_stride{frame_stride}_s{'-'.join(map(str, seeds))}"


def _tag(n_episodes: int, n_windows: int, frame_stride: int, seeds: tuple[int, ...]) -> str:
    """Alias for plot_modality_labels.py (n_windows ignored)."""
    del n_windows
    return run_tag(n_episodes, frame_stride, seeds)


def training_window_starts(dataset, episode: int, stride: int) -> list[int]:
    """Valid full-window dataset indices for one episode (same as training)."""
    epi = dataset.lerobot_dataset.episode_data_index
    start = int(epi["from"][episode].item())
    end = int(epi["to"][episode].item())
    obs = int(dataset.lerobot_dataset.obs_size)
    last = end - obs
    if last < start:
        return []
    return list(range(start, last + 1, max(int(stride), 1)))


def training_episodes(dataset, n_episodes: int = 0) -> list[int]:
    """Episodes with at least one full window (``length >= obs_size``)."""
    epi = dataset.lerobot_dataset.episode_data_index
    obs = int(dataset.lerobot_dataset.obs_size)
    lengths = (epi["to"] - epi["from"]).tolist()
    usable = [i for i, length in enumerate(lengths) if int(length) >= obs]
    if n_episodes <= 0 or len(usable) <= n_episodes:
        return usable
    pick = np.linspace(0, len(usable) - 1, num=n_episodes).round().astype(int)
    return [usable[i] for i in pick]


def expected_windows(dataset, n_episodes: int, stride: int) -> dict[tuple[int, int], int]:
    """Map ``(episode, window) -> frame_idx`` for the full training label set."""
    out: dict[tuple[int, int], int] = {}
    for ep in training_episodes(dataset, n_episodes):
        for w, idx in enumerate(training_window_starts(dataset, ep, stride)):
            out[(ep, w)] = int(idx)
    return out


def load_rgb_dataset():
    """LIBERO train set used by unimodal RGB (same dirs / obs_size as multimodal)."""
    spec = JOINT_UNIMODAL["RGB"]
    cfg = compose_task_cfg(spec["task"])
    dataset = load_dataset(cfg, Path(spec["run_dir"]) / "dataset_stats.json")
    return cfg, dataset, spec


def load_shard_records(model: str, tag: str) -> tuple[dict[tuple[int, int], dict], dict[str, Any]]:
    files = sorted(OUT_DIR.glob(f"{model}_{tag}_shard*of*.pt"))
    if not files:
        raise FileNotFoundError(f"No shards for {model} tag={tag} in {OUT_DIR}")
    recs: dict[tuple[int, int], dict] = {}
    text: dict[str, Any] = {}
    for path in files:
        state = torch.load(path, map_location="cpu", weights_only=False)
        text.update(state.get("text_features", {}))
        for rec in state["records"]:
            recs[(int(rec["episode"]), int(rec["window"]))] = rec
    return recs, text


def _load(model: str, tag: str):
    """Alias for plot_modality_labels.py."""
    return load_shard_records(model, tag)


def _router_features(sample: dict[str, Any], r0_latent: torch.Tensor) -> dict[str, np.ndarray]:
    return {
        "r0_pool": F.adaptive_avg_pool2d(r0_latent[None], 2).flatten().cpu().numpy().astype(np.float16),
        "proprio": sample["proprio"][0].float().numpy(),
    }


def _text_mean(sample: dict[str, Any]) -> np.ndarray:
    """Text-only mean (no proprio token). Kept for task hashing / gt_knn assets."""
    ctx = sample["context"].float()
    mask = sample["context_mask"].float().unsqueeze(-1)
    return ((ctx * mask).sum(0) / mask.sum(0).clamp(min=1.0)).numpy().astype(np.float16)


def _context_mean(model, sample: dict[str, Any]) -> np.ndarray:
    """Same as in-graph ``_build_router_features``: mean of MoT context after proprio append."""
    device, dtype = model.device, model.torch_dtype
    context = sample["context"].unsqueeze(0).to(device=device, dtype=dtype)
    context_mask = sample["context_mask"].unsqueeze(0).to(device=device, dtype=torch.bool)
    proprio = sample["proprio"][0].unsqueeze(0).to(device=device, dtype=dtype)
    context, context_mask = model._append_proprio_to_context(context, context_mask, proprio)
    w = context_mask.to(dtype=torch.float32).unsqueeze(-1)
    mean = (context.float() * w).sum(1) / w.sum(1).clamp(min=1.0)
    return mean.squeeze(0).cpu().numpy().astype(np.float16)


# ---------------------------------------------------------------------------
# Collect
# ---------------------------------------------------------------------------


@torch.no_grad()
def sample_actions(
    model, samples: list[dict[str, Any]], seeds: tuple[int, ...], cfg
) -> tuple[np.ndarray, torch.Tensor]:
    """Batched ``infer_action``. Returns ``actions [B, S, H, D]``, R0 latents ``[B, C, h, w]``."""
    n, n_seeds = len(samples), len(seeds)
    num_frames = int(cfg.data.train.num_frames)
    horizon = num_frames - 1
    num_video_frames = (num_frames - 1) // int(cfg.data.train.action_video_freq_ratio) + 1
    steps = int(cfg.EVALUATION.num_inference_steps)
    shift = cfg.EVALUATION.get("sigma_shift")
    shift = None if shift is None else float(shift)
    dtype, device = model.torch_dtype, model.device

    first = torch.cat([
        model._encode_input_image_latents_tensor(
            s["video"][:, 0].unsqueeze(0).to(device, dtype)
        )
        for s in samples
    ])
    context = torch.stack([s["context"] for s in samples]).to(device, dtype)
    context_mask = torch.stack([s["context_mask"] for s in samples]).to(device, torch.bool)
    proprio = torch.stack([s["proprio"][0] for s in samples]).to(device, dtype)
    context, context_mask = model._append_proprio_to_context(context, context_mask, proprio)

    _, c, _, h, w = first.shape
    lt = (num_video_frames - 1) // model.vae.temporal_downsample_factor + 1
    # Same RNG as FastWAMJoint.infer_action(seed=s).
    noise_v = torch.stack([
        torch.randn((c, lt, h, w), generator=torch.Generator().manual_seed(s)) for s in seeds
    ])
    noise_a = torch.stack([
        torch.randn(
            (horizon, model.action_expert.action_dim),
            generator=torch.Generator().manual_seed(s),
        )
        for s in seeds
    ])

    def repeat_seeds(x):
        return x.repeat_interleave(n_seeds, dim=0)

    latents_video = noise_v.repeat(n, 1, 1, 1, 1).to(device, dtype)
    latents_action = noise_a.repeat(n, 1, 1).to(device, dtype)
    first_r = repeat_seeds(first)
    latents_video[:, :, 0:1] = first_r
    context, context_mask = repeat_seeds(context), repeat_seeds(context_mask)
    fuse = bool(getattr(model.video_expert, "fuse_vae_embedding_in_latents", False))

    t_v, d_v = model.infer_video_scheduler.build_inference_schedule(
        steps, device, latents_video.dtype, shift_override=shift
    )
    t_a, d_a = model.infer_action_scheduler.build_inference_schedule(
        steps, device, latents_action.dtype, shift_override=shift
    )
    batch = latents_video.shape[0]
    for tv, dv, ta, da in zip(t_v, d_v, t_a, d_a):
        pred_v, pred_a = model._predict_joint_noise(
            latents_video=latents_video,
            latents_action=latents_action,
            timestep_video=tv.expand(batch),
            timestep_action=ta.expand(batch),
            context=context,
            context_mask=context_mask,
            fuse_vae_embedding_in_latents=fuse,
            gt_action=None,
        )
        latents_video = model.infer_video_scheduler.step(pred_v, dv, latents_video)
        latents_action = model.infer_action_scheduler.step(pred_a, da, latents_action)
        latents_video[:, :, 0:1] = first_r

    actions = latents_action.float().view(n, n_seeds, horizon, -1).cpu().numpy()
    return actions, first[:, :, 0].float()


def collect(
    model_name: str,
    *,
    n_episodes: int,
    frame_stride: int,
    seeds: tuple[int, ...],
    shard: int,
    num_shards: int,
    batch_windows: int,
    save_every: int,
) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = run_tag(n_episodes, frame_stride, seeds)
    out_path = OUT_DIR / f"{model_name}_{tag}_shard{shard}of{num_shards}.pt"

    spec = JOINT_UNIMODAL[model_name]
    cfg = compose_task_cfg(spec["task"])
    dataset = load_dataset(cfg, Path(spec["run_dir"]) / "dataset_stats.json")
    obs = int(dataset.lerobot_dataset.obs_size)
    train_stride = int(getattr(dataset.lerobot_dataset, "global_sample_stride", 1))
    stride = frame_stride if frame_stride > 0 else train_stride
    if stride != train_stride:
        print(
            f"[{model_name}] WARNING: --frame-stride={stride} != "
            f"training global_sample_stride={train_stride}",
            flush=True,
        )

    all_eps = training_episodes(dataset, n_episodes)
    episodes = all_eps[shard::num_shards]

    state: dict[str, Any] = {
        "model": model_name,
        "episodes": episodes,
        "seeds": list(seeds),
        "ckpt": str(spec["ckpt"]),
        "obs_size": obs,
        "frame_stride": stride,
        "global_sample_stride": train_stride,
        "records": [],
        "text_features": {},
    }
    if out_path.exists():
        state = torch.load(out_path, map_location="cpu", weights_only=False)
    # Resume across all shards (episode→shard mapping can change if episode
    # filters change; analyze merges every shard anyway).
    done: set[tuple[int, int]] = set()
    for path in OUT_DIR.glob(f"{model_name}_{tag}_shard*of*.pt"):
        prev = torch.load(path, map_location="cpu", weights_only=False)
        done.update((r["episode"], r["window"]) for r in prev.get("records", []))
        if path.resolve() != out_path.resolve():
            state.setdefault("text_features", {}).update(prev.get("text_features", {}))

    # (episode, window, frame_idx, progress)
    todo: list[tuple[int, int, int, float]] = []
    n_starts = 0
    for ep in episodes:
        idxs = training_window_starts(dataset, ep, stride)
        n_starts += len(idxs)
        if not idxs:
            continue
        span = max(idxs[-1] - idxs[0], 1)
        for w, idx in enumerate(idxs):
            if (ep, w) not in done:
                todo.append((ep, w, int(idx), (idx - idxs[0]) / span))

    print(
        f"[{model_name} {shard}/{num_shards}] "
        f"eps={len(episodes)}/{len(all_eps)} obs={obs} stride={stride} "
        f"starts={n_starts} done={len(done)} todo={len(todo)} -> {out_path}",
        flush=True,
    )
    if not todo:
        return out_path

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_joint_model(cfg, Path(spec["ckpt"]), device=device)
    loader = DataLoader(
        Subset(dataset, [item[2] for item in todo]),
        batch_size=batch_windows,
        num_workers=6,
        collate_fn=lambda x: x,
        prefetch_factor=4,
    )

    save_features = model_name == "RGB"
    t0, n_seen, since_save, pos = time.time(), 0, 0, 0
    for samples in loader:
        meta = todo[pos: pos + len(samples)]
        pos += len(samples)
        actions, r0 = sample_actions(model, samples, seeds, cfg)
        for j, (ep, w, idx, progress) in enumerate(meta):
            pad = samples[j].get("action_is_pad")
            rec: dict[str, Any] = {
                "episode": ep,
                "window": w,
                "frame_idx": idx,
                "progress": float(progress),
                "pred": actions[j].astype(np.float16),
                "gt": samples[j]["action"].float().numpy(),
                "action_is_pad": None if pad is None else pad.numpy(),
            }
            if save_features:
                prompt = samples[j].get("prompt")
                rec["features"] = _router_features(samples[j], r0[j])
                # Inline router feature: masked mean of context after proprio append.
                rec["features"]["context_mean"] = _context_mean(model, samples[j])
                rec["prompt"] = prompt
                if prompt not in state["text_features"]:
                    state["text_features"][prompt] = _text_mean(samples[j])
            state["records"].append(rec)

        n_seen += len(samples)
        since_save += len(samples)
        if since_save >= save_every:
            torch.save(state, out_path)
            since_save = 0
            rate = n_seen / max(time.time() - t0, 1e-9)
            eta_h = (len(todo) - n_seen) / max(rate, 1e-9) / 3600
            print(
                f"[{model_name} {shard}/{num_shards}] "
                f"{n_seen}/{len(todo)}  {rate:.2f} win/s  ~{eta_h:.1f} h left",
                flush=True,
            )

    torch.save(state, out_path)
    print(f"[{model_name} {shard}/{num_shards}] done", flush=True)
    del model
    gc.collect()
    return out_path


# ---------------------------------------------------------------------------
# Analyze → labels_*.pt
# ---------------------------------------------------------------------------


@torch.no_grad()
def _backfill_context_mean(frame_idxs: np.ndarray, *, batch_size: int = 64) -> np.ndarray:
    """Recompute in-graph ``context_mean`` for labeled window starts (RGB model + dataset)."""
    spec = JOINT_UNIMODAL["RGB"]
    cfg = compose_task_cfg(spec["task"])
    dataset = load_dataset(cfg, Path(spec["run_dir"]) / "dataset_stats.json")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_joint_model(cfg, Path(spec["ckpt"]), device=device)
    out = np.zeros((len(frame_idxs), int(getattr(model, "text_dim", 4096))), dtype=np.float32)
    for start in range(0, len(frame_idxs), batch_size):
        idxs = frame_idxs[start : start + batch_size]
        samples = [dataset[int(i)] for i in idxs]
        for j, sample in enumerate(samples):
            out[start + j] = _context_mean(model, sample).astype(np.float32)
        if (start // batch_size) % 20 == 0:
            print(f"  context_mean backfill {min(start + batch_size, len(frame_idxs))}/{len(frame_idxs)}", flush=True)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def analyze(
    n_episodes: int,
    frame_stride: int,
    seeds: tuple[int, ...],
    gripper_weight: float = -1.0,
) -> dict[str, Any]:
    """Label = argmin action error over RGB/Depth/Flow. Writes ``labels_*.pt``."""
    tag = run_tag(n_episodes, frame_stride, seeds)
    loaded = [load_shard_records(m, tag) for m in MODELS]
    per_model = [recs for recs, _ in loaded]
    text = loaded[0][1]

    keys = sorted(set.intersection(*(set(r) for r in per_model)))
    if not keys:
        raise ValueError("No windows shared by all three models yet.")

    n = len(keys)
    pred = np.stack([
        np.stack([per_model[m][k]["pred"] for k in keys]) for m in range(3)
    ]).astype(np.float32)  # [M, N, S, H, D]
    gt = np.stack([per_model[0][k]["gt"] for k in keys])  # [N, H, D]
    pad0 = per_model[0][keys[0]]["action_is_pad"]
    if pad0 is None:
        valid = np.ones(gt.shape[:2], dtype=np.float32)
    else:
        valid = (~np.stack([per_model[0][k]["action_is_pad"] for k in keys])).astype(np.float32)

    def masked_mean(x: np.ndarray) -> np.ndarray:
        return (x * valid[None, :, None]).sum(-1) / valid.sum(-1).clip(min=1)[None, :, None]

    delta_err = masked_mean(((pred[..., DELTA] - gt[None, :, None, :, DELTA]) ** 2).mean(-1))
    grip_err = masked_mean(
        (np.sign(pred[..., GRIPPER]) != np.sign(gt[None, :, None, :, GRIPPER])).astype(np.float32)
    )
    if gripper_weight < 0:
        gripper_weight = float(delta_err.mean() / max(float(grip_err.mean()), 1e-8))
    err = delta_err + gripper_weight * grip_err  # [M, N, S]
    e_all = err.mean(2)  # mean over seeds → [M, N]
    label = e_all.argmin(0)
    counts = np.bincount(label, minlength=3)

    frame_idxs = np.array([int(per_model[0][k]["frame_idx"]) for k in keys], dtype=np.int64)
    for m, name in enumerate(MODELS):
        other = np.array([int(per_model[m][k]["frame_idx"]) for k in keys], dtype=np.int64)
        if not np.array_equal(frame_idxs, other):
            raise ValueError(f"frame_idx mismatch: RGB vs {name}")

    episodes = np.array([k[0] for k in keys])
    progress = np.array([per_model[0][k]["progress"] for k in keys])
    suites = np.array([suite_of(int(e)) for e in episodes])

    best_i = int(e_all.mean(1).argmin())
    best = float(e_all[best_i].mean())
    out_tag = tag if gripper_weight <= 0 else f"{tag}_grip{gripper_weight:.4g}"

    print(f"\n=== labels: {n} windows, {len(set(episodes))} episodes, seeds {list(seeds)} ===")
    print(f"stride={frame_stride}  frame_idx=[{frame_idxs.min()}, {frame_idxs.max()}]")
    print("labels: " + ", ".join(f"{MODELS[m]} {counts[m]} ({100 * counts[m] / n:.0f}%)" for m in range(3)))
    print(f"gripper_weight={gripper_weight:.4g}")
    print("delta-EEF MSE:         " + ", ".join(f"{MODELS[m]} {delta_err[m].mean():.5f}" for m in range(3)))
    print("gripper mismatch:      " + ", ".join(f"{MODELS[m]} {grip_err[m].mean():.4f}" for m in range(3)))
    print(f"mean action err:       " + ", ".join(f"{MODELS[m]} {e_all[m].mean():.5f}" for m in range(3)))
    print(f"best single modality:  {MODELS[best_i]} ({best:.5f})")
    print("by suite:")
    for name, _ in SUITES:
        sel = suites == name
        if not sel.any():
            continue
        c = np.bincount(label[sel], minlength=3)
        errs = " / ".join(f"{e_all[m, sel].mean():.5f}" for m in range(3))
        print(f"  {name:15s} n={sel.sum():6d}  R/D/F={c.tolist()}  err={errs}")

    def _as_tensor(x: Any) -> Any:
        if isinstance(x, np.ndarray):
            return torch.from_numpy(np.ascontiguousarray(x))
        return x

    export: dict[str, Any] = {
        "keys": keys,
        "episode": _as_tensor(np.asarray(episodes)),
        "suite": suites,
        "progress": _as_tensor(np.asarray(progress, dtype=np.float32)),
        "frame_idx": _as_tensor(np.asarray(frame_idxs)),
        "frame_stride": int(frame_stride),
        "label": _as_tensor(np.asarray(label, dtype=np.int64)),
        "weighted_action_loss": _as_tensor(np.asarray(e_all.T, dtype=np.float32)),
        "models": list(MODELS),
        "label_source": "action_error",
        "gripper_weight": gripper_weight,
        "delta_err": _as_tensor(np.asarray(delta_err.mean(2).T, dtype=np.float32)),
        "gripper_mismatch": _as_tensor(
            np.asarray(grip_err.mean(2).T, dtype=np.float32)
        ),
    }
    feats = [per_model[0][k].get("features") for k in keys]
    if all(f is not None for f in feats):
        export["r0_pool"] = _as_tensor(
            np.stack([f["r0_pool"] for f in feats]).astype(np.float32)
        )
        export["proprio"] = _as_tensor(
            np.stack([f["proprio"] for f in feats]).astype(np.float32)
        )
        export["text_mean"] = _as_tensor(
            np.stack([text[per_model[0][k]["prompt"]] for k in keys]).astype(np.float32)
        )
        if all(f.get("context_mean") is not None for f in feats):
            export["context_mean"] = _as_tensor(
                np.stack([f["context_mean"] for f in feats]).astype(np.float32)
            )
        else:
            # Older RGB shards lack context_mean; backfill with the same MoT context mean.
            print("context_mean missing in RGB features; backfilling via RGB model + dataset…")
            export["context_mean"] = _as_tensor(
                _backfill_context_mean(frame_idxs).astype(np.float32)
            )

    out = OUT_DIR / f"labels_{out_tag}.pt"
    # Features for ~200k windows can exceed 4 GiB; protocol>=4 is required.
    torch.save(export, out, pickle_protocol=4)
    summary = {
        "n_windows": n,
        "frame_stride": int(frame_stride),
        "gripper_weight": gripper_weight,
        "label_counts": {MODELS[m]: int(counts[m]) for m in range(3)},
        "best_single": MODELS[best_i],
        "best_single_err": best,
        "labels_path": str(out),
    }
    with open(OUT_DIR / f"summary_{out_tag}.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    print(f"\nwrote {out}")
    return summary


# ---------------------------------------------------------------------------
# Verify collect vs training index
# ---------------------------------------------------------------------------


def verify(n_episodes: int, frame_stride: int, seeds: tuple[int, ...]) -> None:
    """Check shard ``(episode, window, frame_idx)`` against training window starts."""
    tag = run_tag(n_episodes, frame_stride, seeds)
    _, dataset, _ = load_rgb_dataset()
    obs = int(dataset.lerobot_dataset.obs_size)
    train_stride = int(getattr(dataset.lerobot_dataset, "global_sample_stride", 1))
    stride = frame_stride if frame_stride > 0 else train_stride
    expected = expected_windows(dataset, n_episodes, stride)
    expected_frames = set(expected.values())
    epi = dataset.lerobot_dataset.episode_data_index
    n_ep = len(epi["from"])
    short = [
        i for i in range(n_ep)
        if int((epi["to"][i] - epi["from"][i]).item()) < obs
    ]

    print("=== training reference ===")
    print(
        f"dataset_len={len(dataset)}  episodes={n_ep}  "
        f"labelable={len(training_episodes(dataset, n_episodes))}"
    )
    print(f"obs_size={obs}  train_stride={train_stride}  collect_stride={stride}")
    print(f"expected_window_starts={len(expected)}  too_short={len(short)}")

    ok = True
    per_model_frame: list[dict[tuple[int, int], int]] = []
    for model in MODELS:
        try:
            recs, _ = load_shard_records(model, tag)
        except FileNotFoundError:
            print(f"\n[{model}] MISSING shards tag={tag}")
            ok = False
            continue

        got = {k: int(recs[k]["frame_idx"]) for k in recs}
        missing = sorted(set(expected) - set(got))
        extra = sorted(set(got) - set(expected))
        mismatch = [
            (k, got[k], expected[k])
            for k in got
            if k in expected and got[k] != expected[k]
        ]
        miss_f = sorted(expected_frames - set(got.values()))

        print(f"\n[{model}] records={len(got)}")
        print(
            f"  missing_keys={len(missing)}  extra={len(extra)}  "
            f"frame_idx_mismatch={len(mismatch)}"
        )
        print(f"  missing_frame_idx={len(miss_f)}")
        if missing[:3]:
            print(f"  e.g. missing {missing[:3]}")
        if mismatch[:3]:
            print(f"  e.g. mismatch {mismatch[:3]}")
        if missing or mismatch or miss_f:
            ok = False
        per_model_frame.append(got)

    if len(per_model_frame) == 3:
        shared = set(per_model_frame[0]) & set(per_model_frame[1]) & set(per_model_frame[2])
        disagree = sum(
            1
            for k in shared
            if not (
                per_model_frame[0][k]
                == per_model_frame[1][k]
                == per_model_frame[2][k]
                == expected.get(k)
            )
        )
        print("\n=== cross-model ===")
        print(f"shared={len(shared)}/{len(expected)}  frame_idx_disagree={disagree}")
        if len(shared) < len(expected) or disagree:
            ok = False

    print("\nRESULT:", "PASS" if ok else "FAIL")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    os.chdir(REPO)
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model", choices=MODELS, help="collect this unimodal model")
    p.add_argument("--analyze", action="store_true", help="merge shards → labels_*.pt")
    p.add_argument("--verify", action="store_true", help="check shards vs training windows")
    p.add_argument("--n-episodes", type=int, default=0, help="0 = all labelable episodes")
    p.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="window stride; use 1 to match training global_sample_stride",
    )
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--batch-windows", type=int, default=4)
    p.add_argument("--save-every", type=int, default=2000)
    p.add_argument(
        "--gripper-weight",
        type=float,
        default=-1.0,
        help="-1 = auto-balance to delta-EEF; 0 = ignore gripper",
    )
    a = p.parse_args()
    seeds = tuple(a.seeds)

    if a.verify:
        verify(a.n_episodes, a.frame_stride, seeds)
    elif a.analyze:
        analyze(a.n_episodes, a.frame_stride, seeds, gripper_weight=a.gripper_weight)
    elif a.model:
        collect(
            a.model,
            n_episodes=a.n_episodes,
            frame_stride=a.frame_stride,
            seeds=seeds,
            shard=a.shard,
            num_shards=a.num_shards,
            batch_windows=a.batch_windows,
            save_every=a.save_every,
        )
    else:
        p.error("pass --model NAME, --analyze, or --verify")


if __name__ == "__main__":
    main()
