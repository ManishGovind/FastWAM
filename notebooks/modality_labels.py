"""Offline go/no-go for per-window modality selection with the unimodal Joint checkpoints.

Each unimodal model (RGB / Depth / Flow) is the ``R0 + fut_k -> action`` path of the
in-model router design, so the three checkpoints give the three counterfactual actions
without retraining anything. Per demo window and per model this collects:

* the action chunk for several sampler seeds (-> per-window "best modality" label),
* the seed spread of the predicted future video latents (-> uncertainty-based selection),
* router input features (pooled R0 VAE latent, pooled T5 context, proprio).

One process per model so the three can run on separate GPUs; results are saved after
every episode and a rerun resumes from where it stopped.

Usage::

    # collect (one per GPU)
    CUDA_VISIBLE_DEVICES=0 python notebooks/modality_labels.py --model RGB   --n-episodes 50
    CUDA_VISIBLE_DEVICES=1 python notebooks/modality_labels.py --model Depth --n-episodes 50
    CUDA_VISIBLE_DEVICES=2 python notebooks/modality_labels.py --model Flow  --n-episodes 50

    # analyze
    python notebooks/modality_labels.py --analyze --n-episodes 50
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from action_agreement import _episode_phase_labels, _infer_kwargs_from_sample  # noqa: E402
from repr_analysis import (  # noqa: E402
    JOINT_UNIMODAL,
    REPO,
    compose_task_cfg,
    episode_window_indices,
    load_dataset,
    load_joint_model,
)

OUT_DIR = REPO / "notebooks" / "outputs" / "modality_labels"
MODELS = ("RGB", "Depth", "Flow")
# Episode counts in `dataset_dirs` order (configs/data/libero_2cam*.yaml); the multi-dataset
# concatenates them, so a global episode index maps to a suite by these boundaries.
SUITES = (("libero_spatial", 434), ("libero_object", 457), ("libero_goal", 433), ("libero_10", 388))


def suite_of(episode: int) -> str:
    start = 0
    for name, count in SUITES:
        if episode < start + count:
            return name
        start += count
    return "unknown"
DELTA = slice(0, 6)
GRIPPER = 6


def _run_tag(n_episodes: int, n_windows: int, seeds: tuple[int, ...]) -> str:
    ep = "all" if n_episodes <= 0 else str(n_episodes)
    return f"ep{ep}_win{n_windows}_s{'-'.join(str(s) for s in seeds)}"


def _cache_path(model: str, n_episodes: int, n_windows: int, seeds: tuple[int, ...],
                shard: int = 0, num_shards: int = 1) -> Path:
    suffix = "" if num_shards == 1 else f"_shard{shard}of{num_shards}"
    return OUT_DIR / f"{model}_{_run_tag(n_episodes, n_windows, seeds)}{suffix}.pt"


def _cache_files(model: str, n_episodes: int, n_windows: int, seeds: tuple[int, ...]) -> list[Path]:
    tag = _run_tag(n_episodes, n_windows, seeds)
    files = sorted(OUT_DIR.glob(f"{model}_{tag}_shard*of*.pt"))
    single = OUT_DIR / f"{model}_{tag}.pt"
    return files or ([single] if single.exists() else [])


def spread_episodes(dataset, n: int, min_len: int = 80) -> list[int]:
    """``n`` usable episodes evenly spaced over the whole dataset (all of them if ``n <= 0``).

    ``action_agreement.pick_episodes`` takes the first ``n``, which all come from one suite.
    """
    epi = dataset.lerobot_dataset.episode_data_index
    obs = int(dataset.lerobot_dataset.obs_size)
    need = max(int(min_len), obs + 1)
    lengths = (epi["to"] - epi["from"]).tolist()
    usable = [i for i, length in enumerate(lengths) if length >= need]
    if n <= 0 or len(usable) <= n:
        return usable
    pick = np.linspace(0, len(usable) - 1, num=n).round().astype(int)
    return [usable[i] for i in pick]


def _video_spread(lat: torch.Tensor) -> dict[str, Any]:
    """Seed spread of predicted future latents. lat: [S, C, T, H, W]."""
    var = lat.var(dim=0, unbiased=False)  # [C, T, H, W]
    flat = F.normalize(lat.flatten(1), dim=1)
    cos = flat @ flat.T
    s = lat.shape[0]
    off = cos[~torch.eye(s, dtype=torch.bool)]
    return {
        "var": float(var.mean()),
        "var_per_frame": var.mean(dim=(0, 2, 3)).numpy(),
        "cos_disp": float((1.0 - off).mean()),
        "rms": float(lat.pow(2).mean().sqrt()),
    }


@torch.no_grad()
def _router_features(model, sample: dict[str, Any], kwargs: dict[str, Any]) -> dict[str, np.ndarray]:
    img = kwargs["input_image"].unsqueeze(0).to(device=model.device, dtype=model.torch_dtype)
    z = model._encode_input_image_latents_tensor(input_image=img)  # [1, C, 1, h, w]
    z_pool = F.adaptive_avg_pool2d(z[:, :, 0].float(), 2).flatten(1)[0]
    ctx = sample["context"].float()
    mask = sample["context_mask"].float().unsqueeze(-1)
    ctx_mean = (ctx * mask).sum(0) / mask.sum(0).clamp(min=1.0)
    return {
        "r0_pool": z_pool.cpu().numpy(),
        "text_mean": ctx_mean.cpu().numpy(),
        "proprio": sample["proprio"][0].float().cpu().numpy(),
    }


@torch.no_grad()
def collect(model_name: str, *, n_episodes: int, n_windows: int, seeds: tuple[int, ...],
            shard: int = 0, num_shards: int = 1) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _cache_path(model_name, n_episodes, n_windows, seeds, shard, num_shards)
    spec = JOINT_UNIMODAL[model_name]
    cfg = compose_task_cfg(spec["task"])
    dataset = load_dataset(cfg, Path(spec["run_dir"]) / "dataset_stats.json")
    episodes = spread_episodes(dataset, n_episodes)[shard::num_shards]

    state: dict[str, Any] = {"model": model_name, "episodes": episodes, "seeds": list(seeds),
                             "ckpt": str(spec["ckpt"]), "records": []}
    if out_path.exists():
        state = torch.load(out_path, map_location="cpu", weights_only=False)
        if state["episodes"] != episodes or state["seeds"] != list(seeds):
            raise ValueError(f"{out_path} was written with different episodes/seeds; delete it to restart.")
    done = {r["episode"] for r in state["records"]}
    print(f"[{model_name}] {len(episodes)} episodes, {len(done)} already done -> {out_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_joint_model(cfg, Path(spec["ckpt"]), device=device)

    for ep in episodes:
        if ep in done:
            continue
        indices, _ = episode_window_indices(dataset, ep, n_windows=n_windows)
        phases = _episode_phase_labels(dataset, ep, indices)
        for w, idx in enumerate(indices):
            sample = dataset[int(idx)]
            kwargs = _infer_kwargs_from_sample(sample, cfg, model)
            actions, lats = [], []
            for s in seeds:
                out = model.infer_action(**kwargs, seed=int(s), return_video_latents=True)
                actions.append(out["action"])
                lats.append(out["video_latents"])
            state["records"].append({
                "episode": ep,
                "window": w,
                "frame_idx": int(idx),
                "phase": int(phases[w]),
                "prompt": sample.get("prompt"),
                "pred": torch.stack(actions).numpy(),  # [S, H, D]
                "gt": sample["action"].float().cpu().numpy(),  # [H, D]
                "video": _video_spread(torch.stack(lats)),
                "features": _router_features(model, sample, kwargs),
            })
        torch.save(state, out_path)
        n_done = len({r["episode"] for r in state["records"]})
        print(f"[{model_name}] episode {ep} done ({n_done}/{len(episodes)})", flush=True)

    del model
    gc.collect()
    return out_path


# --------------------------------------------------------------------------------------
# analysis
# --------------------------------------------------------------------------------------


def _load_records(model: str, n_episodes: int, n_windows: int, seeds: tuple[int, ...]) -> list[dict]:
    files = _cache_files(model, n_episodes, n_windows, seeds)
    if not files:
        raise FileNotFoundError(f"No cache for {model} {_run_tag(n_episodes, n_windows, seeds)} in {OUT_DIR}")
    records: list[dict] = []
    for f in files:
        records += torch.load(f, map_location="cpu", weights_only=False)["records"]
    return records


def _load_aligned(n_episodes: int, n_windows: int, seeds: tuple[int, ...]) -> dict[str, Any]:
    keys = [{(r["episode"], r["window"]): r for r in _load_records(m, n_episodes, n_windows, seeds)}
            for m in MODELS]
    common = sorted(set(keys[0]) & set(keys[1]) & set(keys[2]))
    if not common:
        raise ValueError("No windows shared by all three models yet.")
    for k in common:
        fi = {keys[i][k]["frame_idx"] for i in range(3)}
        if len(fi) != 1:
            raise ValueError(f"Window {k} is misaligned across models: frame_idx {fi}")

    pred = np.stack([np.stack([keys[i][k]["pred"] for k in common]) for i in range(3)])  # [M, N, S, H, D]
    gt = np.stack([keys[0][k]["gt"] for k in common])  # [N, H, D]
    video = {
        key: np.array([[keys[i][k]["video"][key] for k in common] for i in range(3)])
        for key in ("var", "cos_disp")
    }
    return {
        "keys": common,
        "episode": np.array([k[0] for k in common]),
        "phase": np.array([keys[0][k]["phase"] for k in common]),
        "prompt": [keys[0][k]["prompt"] for k in common],
        "pred": pred,
        "gt": gt,
        "video": video,
    }


def _pick_err(err: np.ndarray, choice: np.ndarray) -> float:
    """err: [M, N], choice: [N] -> mean error of the chosen model."""
    return float(err[choice, np.arange(err.shape[1])].mean())


def analyze(n_episodes: int, n_windows: int, seeds: tuple[int, ...]) -> dict[str, Any]:
    sys.path.insert(0, str(REPO))
    from experiments.libero.action_selection import select

    d = _load_aligned(n_episodes, n_windows, seeds)
    pred, gt = d["pred"], d["gt"]
    m_count, n, s_count = pred.shape[:3]
    if s_count < 3:
        raise ValueError("Need at least 3 seeds (two to choose, one held out).")

    # err[m, n, s]: delta-EEF MSE of one sample vs ground truth.
    err = ((pred[..., DELTA] - gt[None, :, None, :, DELTA]) ** 2).mean(axis=(-1, -2))
    err_mean = err.mean(axis=2)  # [M, N]
    label = err_mean.argmin(axis=0)
    counts = np.bincount(label, minlength=m_count)
    chance_marginal = float(((counts / n) ** 2).sum())

    # 1) Label stability: does the best model under one seed agree with another seed?
    seed_agree = [float((err[:, :, i].argmin(0) == err[:, :, j].argmin(0)).mean())
                  for i in range(s_count) for j in range(i + 1, s_count)]

    # 2) Honest oracle: choose with seeds[:-1], score on the held-out last seed.
    choose = err[:, :, :-1].mean(axis=2).argmin(axis=0)
    held = err[:, :, -1]  # [M, N]
    per_model_held = held.mean(axis=1)
    best_i = int(per_model_held.argmin())
    best_single_held = float(per_model_held[best_i])
    honest_oracle = _pick_err(held, choose)
    naive_oracle = float(held.min(axis=0).mean())

    # Noise-only "oracle": pick the best seed of the *same* model. Gain here is pure
    # selection-on-noise, the bar the cross-model naive oracle has to be read against.
    noise_gain = {MODELS[m]: float(1.0 - err[m].min(axis=1).mean() / err[m].mean())
                  for m in range(m_count)}
    naive_gain = float(1.0 - naive_oracle / best_single_held)
    honest_gain = float(1.0 - honest_oracle / best_single_held)

    # 3) Training-free rules on the held-out seed's chunks.
    chunks = pred[:, :, -1]  # [M, N, H, D]
    rule_err = {}
    for rule in ("soft_median", "soft_mean", "hard_medoid"):
        e = []
        for i in range(n):
            fused, _ = select(chunks[:, i], rule)
            e.append(float(((fused[:, DELTA] - gt[i, :, DELTA]) ** 2).mean()))
        rule_err[rule] = float(np.mean(e))

    # 4) Uncertainty selection: calibrate on the first half of episodes, test on the rest.
    eps = np.unique(d["episode"])
    calib_eps = set(eps[: len(eps) // 2].tolist())
    is_cal = np.array([e in calib_eps for e in d["episode"]])
    test = ~is_cal
    unc = {}
    for key, raw in d["video"].items():
        med = np.median(raw[:, is_cal], axis=1, keepdims=True)
        cal = raw / np.maximum(med, 1e-12)
        for name, score in (("raw", raw), ("calibrated", cal)):
            pick = score[:, test].argmin(axis=0)
            unc[f"video_{key}_{name}"] = {
                "mse": _pick_err(held[:, test], pick),
                "match_label": float((pick == label[test]).mean()),
                "counts": np.bincount(pick, minlength=m_count).tolist(),
            }
    a_spread = pred[..., DELTA].var(axis=2).mean(axis=(-1, -2))  # [M, N]
    a_med = np.median(a_spread[:, is_cal], axis=1, keepdims=True)
    pick = (a_spread / np.maximum(a_med, 1e-12))[:, test].argmin(axis=0)
    unc["action_spread_calibrated"] = {
        "mse": _pick_err(held[:, test], pick),
        "match_label": float((pick == label[test]).mean()),
        "counts": np.bincount(pick, minlength=m_count).tolist(),
    }
    test_ref = {
        "best_single": float(held[best_i, test].mean()),
        "honest_oracle": _pick_err(held[:, test], choose[test]),
    }

    # 5) Label and headroom by phase and by suite.
    by_phase = {}
    for p in np.unique(d["phase"]):
        sel = d["phase"] == p
        by_phase[int(p)] = np.bincount(label[sel], minlength=m_count).tolist()
    suite = np.array([suite_of(int(e)) for e in d["episode"]])
    by_suite = {}
    for name, _ in SUITES:
        sel = suite == name
        if not sel.any():
            continue
        pm = held[:, sel].mean(axis=1)
        by_suite[name] = {
            "n": int(sel.sum()),
            "labels": np.bincount(label[sel], minlength=m_count).tolist(),
            "per_model_mse": {MODELS[m]: float(pm[m]) for m in range(m_count)},
            "honest_oracle_mse": _pick_err(held[:, sel], choose[sel]),
        }

    summary = {
        "n_windows": int(n),
        "n_episodes": int(len(eps)),
        "seeds": list(seeds),
        "label_counts": {MODELS[m]: int(counts[m]) for m in range(m_count)},
        "label_seed_agreement": seed_agree,
        "label_chance_uniform": 1.0 / m_count,
        "label_chance_marginal": chance_marginal,
        "per_model_mse_heldout_seed": {MODELS[m]: float(per_model_held[m]) for m in range(m_count)},
        "best_single": MODELS[best_i],
        "best_single_mse": best_single_held,
        "naive_oracle_mse": naive_oracle,
        "naive_oracle_gain": naive_gain,
        "honest_oracle_mse": honest_oracle,
        "honest_oracle_gain": honest_gain,
        "noise_only_oracle_gain": noise_gain,
        "rules_mse": rule_err,
        "uncertainty_test_half": unc,
        "reference_test_half": test_ref,
        "label_by_phase": by_phase,
        "by_suite": by_suite,
    }

    print(f"\n=== {n} windows from {len(eps)} episodes, seeds {list(seeds)} ===")
    print(f"label counts: {summary['label_counts']}")
    print("\n1) label stability across seeds")
    print(f"   agreement per seed pair: {[f'{a:.2f}' for a in seed_agree]}")
    print(f"   chance: uniform {1.0 / m_count:.2f}, given label frequencies {chance_marginal:.2f}")
    print("\n2) oracle headroom (MSE on held-out seed)")
    for m in range(m_count):
        print(f"   {MODELS[m]:6s} {per_model_held[m]:.6f}")
    print(f"   best single ({MODELS[best_i]}): {best_single_held:.6f}")
    print(f"   naive oracle  {naive_oracle:.6f}  gain {100 * naive_gain:.1f}%  (picks on the scored seed)")
    print(f"   honest oracle {honest_oracle:.6f}  gain {100 * honest_gain:.1f}%  (picks on other seeds)")
    print(f"   noise-only gain (best seed of one model): "
          + ", ".join(f"{k} {100 * v:.1f}%" for k, v in noise_gain.items()))
    print("\n3) training-free rules (held-out seed)")
    for k, v in rule_err.items():
        print(f"   {k:12s} {v:.6f}  vs best single {100 * (1 - v / best_single_held):+.1f}%")
    print("\n4) uncertainty selection (test half; calibrated on the other half)")
    print(f"   best single {test_ref['best_single']:.6f}   honest oracle {test_ref['honest_oracle']:.6f}")
    for k, v in unc.items():
        print(f"   {k:28s} {v['mse']:.6f}  match label {v['match_label']:.2f}  picks {v['counts']}")
    print(f"\n5) label by phase (RGB, Depth, Flow): {by_phase}")
    print("   by suite:")
    for name, v in by_suite.items():
        pm = " ".join(f"{k}={x:.5f}" for k, x in v["per_model_mse"].items())
        print(f"   {name:15s} n={v['n']:3d} labels={v['labels']}  {pm}  oracle={v['honest_oracle_mse']:.5f}")

    out = OUT_DIR / f"summary_ep{n_episodes}_win{n_windows}.json"
    with open(out, "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    print(f"\nwrote {out}")
    return summary


if __name__ == "__main__":
    os.chdir(REPO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--n-episodes", type=int, default=50)
    parser.add_argument("--n-windows", type=int, default=8)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()

    seeds = tuple(args.seeds)
    if args.analyze:
        analyze(args.n_episodes, args.n_windows, seeds)
    elif args.model:
        collect(args.model, n_episodes=args.n_episodes, n_windows=args.n_windows, seeds=seeds)
    else:
        parser.error("pass --model NAME to collect or --analyze")
