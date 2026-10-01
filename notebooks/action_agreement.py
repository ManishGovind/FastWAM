"""Action-agreement analysis for LIBERO joint unimodal FastWAM (RGB / Depth / Flow).

The three unimodal models are all RGB-conditioned and differ only in the future-prediction
target, so they can be driven from the *same* observation and their action chunks compared
directly. That pairing is the whole point: three independent rollouts diverge in state within
a few steps, after which action differences reflect different states rather than different
modalities.

Pipeline:

1. ``extract_actions``   -- replay dataset episodes; every model predicts from the identical
                            condition frame / proprio / text context. Ground truth comes along,
                            which is what makes a per-timestep *oracle* modality label possible.
2. ``seed_noise_floor``  -- same model, several seeds. ``infer_action`` is a diffusion sampler,
                            so this is the null distribution every cross-model distance is read
                            against.
3. ``analyze``           -- pairwise distances, the oracle ceiling, whether the oracle choice is
                            structured (phase / persistence), whether it is predictable online,
                            and the ensembling baseline any router has to beat.

Usage::

    python notebooks/action_agreement.py --n-episodes 1          # smoke test
    python notebooks/action_agreement.py --n-episodes 10
    python notebooks/action_agreement.py --analysis-only
"""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from hydra.utils import instantiate

from repr_analysis import (
    JOINT_UNIMODAL,
    REPO,
    _episode_action_state,
    compose_task_cfg,
    episode_window_indices,
    load_dataset,
    load_joint_model,
    pick_device,
    segment_episode_phases,
)

CACHE_DIR = REPO / "notebooks" / "outputs" / "action_agreement"

PHASE_NAMES = ("free_space", "pregrasp_contact", "post_manip")

# LIBERO action layout: 6 delta-EEF dims + 1 absolute gripper command.
# `delta_action_dim_mask = [True x 6, False]` in the data config.
ACTION_DIM_NAMES = ("dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper")
GRIPPER_DIM = 6


# --------------------------------------------------------------------------------------
# episode selection
# --------------------------------------------------------------------------------------


def pick_episodes(dataset, n: int = 10, min_len: int = 80) -> list[int]:
    """First ``n`` episodes long enough to hold a full model window.

    ``repr_analysis.pick_episode`` returns only the first usable episode; this is the same
    filter applied across the dataset.
    """
    epi = dataset.lerobot_dataset.episode_data_index
    n_ep = int(epi["from"].shape[0])
    obs = int(dataset.lerobot_dataset.obs_size)
    need = max(int(min_len), obs + 1)

    out: list[int] = []
    for i in range(n_ep):
        length = int(epi["to"][i].item()) - int(epi["from"][i].item())
        if length >= need:
            out.append(i)
        if len(out) >= n:
            break

    if not out:
        raise ValueError(
            f"No episode reaches the model window (obs_size={obs}, min_len={min_len})."
        )
    if len(out) < n:
        print(f"[warn] only {len(out)}/{n} episodes are long enough (obs_size={obs})")
    return out


def _episode_phase_labels(dataset, episode_idx: int, frame_indices: list[int]) -> np.ndarray:
    """Phase label per window, from the episode's own action/state trace.

    Falls back to -1 when the trace is too short for ``segment_episode_phases``.
    """
    epi = dataset.lerobot_dataset.episode_data_index
    start = int(epi["from"][episode_idx].item())

    try:
        raw = dataset.lerobot_dataset.multi_dataset.get_episode_data(episode_idx)
        action, state = _episode_action_state(raw)
        labels, _meta = segment_episode_phases(action, state)
    except Exception as exc:  # noqa: BLE001 - phase labels are a nice-to-have
        print(f"[warn] phase segmentation failed for episode {episode_idx}: {exc}")
        return np.full(len(frame_indices), -1, dtype=np.int64)

    rel = np.clip(np.asarray(frame_indices) - start, 0, len(labels) - 1)
    return labels[rel].astype(np.int64)


# --------------------------------------------------------------------------------------
# inference
# --------------------------------------------------------------------------------------


def _infer_kwargs_from_sample(sample: dict[str, Any], cfg, model) -> dict[str, Any]:
    """Build ``infer_action`` kwargs from a dataset sample.

    Two things that are easy to get wrong:

    * ``compose_task_cfg`` disables the text encoder, and ``prompt`` is mutually exclusive
      with ``context``/``context_mask``, so the cached T5 embeddings must be passed instead.
    * ``proprio`` is a *single* state ``[D]``, not the sample's ``[T-1, D]`` trace.
    """
    if "context" not in sample or "context_mask" not in sample:
        raise KeyError(
            "Dataset sample has no cached text context. The model was loaded with "
            "load_text_encoder=false, so `text_embedding_cache_dir` must be populated "
            "(scripts/precompute_text_embeds.py)."
        )

    num_frames = int(cfg.data.train.num_frames)
    freq_ratio = int(cfg.data.train.action_video_freq_ratio)

    kwargs: dict[str, Any] = {
        "prompt": None,
        "context": sample["context"],
        "context_mask": sample["context_mask"],
        "input_image": sample["video"][:, 0],  # [C, T, H, W] -> [C, H, W]
        "action_horizon": num_frames - 1,
        "proprio": sample["proprio"][0],
        "num_inference_steps": int(cfg.EVALUATION.num_inference_steps),
        "sigma_shift": (
            None
            if cfg.EVALUATION.get("sigma_shift") is None
            else float(cfg.EVALUATION.get("sigma_shift"))
        ),
        "rand_device": str(cfg.EVALUATION.get("rand_device", "cpu")),
        "tiled": bool(cfg.EVALUATION.get("tiled", False)),
    }

    # FastWAMJoint.infer_action needs num_video_frames; the base class does not.
    import inspect

    if "num_video_frames" in inspect.signature(model.infer_action).parameters:
        kwargs["num_video_frames"] = (num_frames - 1) // freq_ratio + 1
    return kwargs


@torch.no_grad()
def _predict_one_model(
    name: str,
    *,
    episodes: list[int],
    n_windows: int,
    seeds: tuple[int, ...],
    device: str,
    want_ground_truth: bool,
) -> dict[str, Any]:
    """Run one model over every (episode, window, seed) and return its action chunks."""
    spec = JOINT_UNIMODAL[name]
    cfg = compose_task_cfg(spec["task"])
    dataset = load_dataset(cfg, Path(spec["run_dir"]) / "dataset_stats.json")
    model = load_joint_model(cfg, Path(spec["ckpt"]), device=device)

    preds: list[list[list[torch.Tensor]]] = []  # [ep][win][seed]
    gts: list[list[torch.Tensor]] = []
    phases: list[np.ndarray] = []
    frame_idx: list[list[int]] = []

    try:
        for ep in episodes:
            indices, _meta = episode_window_indices(dataset, ep, n_windows=n_windows)
            ep_preds: list[list[torch.Tensor]] = []
            ep_gts: list[torch.Tensor] = []

            for idx in indices:
                sample = dataset[int(idx)]
                kwargs = _infer_kwargs_from_sample(sample, cfg, model)
                ep_preds.append(
                    [
                        model.infer_action(**kwargs, seed=int(s))["action"].float()
                        for s in seeds
                    ]
                )
                if want_ground_truth:
                    ep_gts.append(sample["action"].detach().cpu().float())

            preds.append(ep_preds)
            if want_ground_truth:
                gts.append(ep_gts)
            phases.append(_episode_phase_labels(dataset, ep, indices))
            frame_idx.append([int(i) for i in indices])
            print(f"[{name}] episode {ep}: {len(indices)} windows x {len(seeds)} seed(s)")
    finally:
        del model
        gc.collect()
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()

    # [n_ep, n_win, n_seed, horizon, dim]
    stacked = torch.stack(
        [torch.stack([torch.stack(w) for w in ep]) for ep in preds]
    )
    out: dict[str, Any] = {
        "pred": stacked,
        "phase": np.stack(phases),
        "frame_idx": frame_idx,
        "ckpt": str(spec["ckpt"]),
        "task": spec["task"],
    }
    if want_ground_truth:
        out["gt"] = torch.stack([torch.stack(ep) for ep in gts])
    return out


def extract_actions(
    *,
    n_episodes: int = 10,
    n_windows: int = 8,
    seed: int = 0,
    models: Optional[list[str]] = None,
    device: Optional[str] = None,
    cache_dir: Path = CACHE_DIR,
    reuse_cache: bool = True,
) -> dict[str, Any]:
    """Paired action chunks: every model predicts from the same observations."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    names = models or list(JOINT_UNIMODAL.keys())
    out_path = cache_dir / f"actions_ep{n_episodes}_win{n_windows}_seed{seed}.pt"

    if reuse_cache and out_path.exists():
        print(f"loading action cache {out_path}")
        return torch.load(out_path, map_location="cpu", weights_only=False)

    device = device or pick_device()
    print(f"extract_actions device={device} models={names} seed={seed}")

    # Episode indices come from the first model's dataset; all three share the same
    # dataset_dirs and sampling config, so the windows line up across models.
    spec0 = JOINT_UNIMODAL[names[0]]
    cfg0 = compose_task_cfg(spec0["task"])
    ds0 = load_dataset(cfg0, Path(spec0["run_dir"]) / "dataset_stats.json")
    episodes = pick_episodes(ds0, n=n_episodes)
    del ds0
    gc.collect()
    print(f"episodes={episodes}")

    per_model: dict[str, Any] = {}
    for i, name in enumerate(names):
        per_model[name] = _predict_one_model(
            name,
            episodes=episodes,
            n_windows=n_windows,
            seeds=(seed,),
            device=device,
            want_ground_truth=(i == 0),
        )

    first = per_model[names[0]]
    payload = {
        # [n_models, n_ep, n_win, horizon, dim] -- seed axis squeezed out
        "pred": torch.stack([per_model[n]["pred"][:, :, 0] for n in names]),
        "gt": first["gt"],
        "phase": first["phase"],
        "frame_idx": first["frame_idx"],
        "models": names,
        "episodes": episodes,
        "seed": seed,
        "space": "normalized",
        "ckpts": {n: per_model[n]["ckpt"] for n in names},
    }
    torch.save(payload, out_path)
    print(f"saved {out_path}  pred={tuple(payload['pred'].shape)}")
    return payload


def seed_noise_floor(
    *,
    model_name: str = "RGB",
    n_episodes: int = 10,
    n_windows: int = 8,
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
    device: Optional[str] = None,
    cache_dir: Path = CACHE_DIR,
    reuse_cache: bool = True,
) -> dict[str, Any]:
    """One model, many seeds -- the null distribution for cross-model distance.

    ``infer_action`` is a diffusion sampler. Without this, sampler variance reads as
    modality signal.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / f"noise_floor_{model_name.lower()}_ep{n_episodes}_win{n_windows}.pt"
    if reuse_cache and out_path.exists():
        print(f"loading noise-floor cache {out_path}")
        return torch.load(out_path, map_location="cpu", weights_only=False)

    device = device or pick_device()
    print(f"seed_noise_floor model={model_name} seeds={seeds} device={device}")

    spec = JOINT_UNIMODAL[model_name]
    cfg = compose_task_cfg(spec["task"])
    ds = load_dataset(cfg, Path(spec["run_dir"]) / "dataset_stats.json")
    episodes = pick_episodes(ds, n=n_episodes)
    del ds
    gc.collect()

    res = _predict_one_model(
        model_name,
        episodes=episodes,
        n_windows=n_windows,
        seeds=tuple(int(s) for s in seeds),
        device=device,
        want_ground_truth=False,
    )
    payload = {
        "pred": res["pred"],  # [n_ep, n_win, n_seed, horizon, dim]
        "model": model_name,
        "seeds": list(seeds),
        "episodes": episodes,
        "space": "normalized",
    }
    torch.save(payload, out_path)
    print(f"saved {out_path}  pred={tuple(payload['pred'].shape)}")
    return payload


# --------------------------------------------------------------------------------------
# eval rollouts (experiments/libero/eval_action_agreement.py)
# --------------------------------------------------------------------------------------


def load_eval_rollouts(eval_dir: Path) -> dict[str, Any]:
    """Load paired-rollout dumps produced by ``eval_action_agreement.py``.

    Unlike demo replay, sim has no ground-truth action, so there is no per-timestep
    oracle. What it has instead is per-episode success under each driver, plus
    disagreement measured on the *on-policy* state distribution -- the states a policy
    actually visits, which is what ultimately matters.
    """
    eval_dir = Path(eval_dir)
    with open(eval_dir / "index.json") as fh:
        index = json.load(fh)

    rollouts = []
    for row in index["episodes"]:
        d = torch.load(eval_dir / row["file"], map_location="cpu", weights_only=False)
        rollouts.append(d)
    return {"index": index, "rollouts": rollouts, "models": index["models"],
            "drivers": index["drivers"], "dir": str(eval_dir)}


def eval_success_matrix(data: dict[str, Any]) -> dict[str, Any]:
    """Per-episode success under each driver, and which episodes routing could help.

    An episode where every driver agrees (all succeed or all fail) is not routable --
    no per-timestep choice changes the outcome. The split ones are the whole opportunity.
    """
    drivers = data["drivers"]
    by_ep: dict[int, dict[str, bool]] = {}
    meta: dict[int, dict[str, Any]] = {}
    for r in data["rollouts"]:
        by_ep.setdefault(r["episode_idx"], {})[r["driver"]] = bool(r["success"])
        meta[r["episode_idx"]] = {
            "suite": r["suite"], "task_id": r["task_id"],
            "task_description": r["task_description"],
        }

    complete = {e: v for e, v in by_ep.items() if len(v) == len(drivers)}
    all_ok = [e for e, v in complete.items() if all(v.values())]
    all_bad = [e for e, v in complete.items() if not any(v.values())]
    split = [e for e in complete if e not in all_ok and e not in all_bad]

    per_driver = {d: sum(v.get(d, False) for v in complete.values()) for d in drivers}
    return {
        "by_episode": by_ep, "meta": meta, "n_complete": len(complete),
        "per_driver_successes": per_driver,
        "all_success": all_ok, "all_fail": all_bad, "routable": split,
        "oracle_successes": len(all_ok) + len(split),
    }


def eval_disagreement(data: dict[str, Any]) -> dict[str, Any]:
    """Cross-model action distance per replan, on the on-policy state distribution."""
    names = data["models"]
    pairs: dict[str, list[np.ndarray]] = {}
    floor: list[np.ndarray] = []
    per_rollout: list[dict[str, Any]] = []

    for r in data["rollouts"]:
        c = r["chunks"]  # [n_replan, n_models, horizon, dim]
        if c is None:
            continue
        row: dict[str, Any] = {
            "episode_idx": r["episode_idx"], "driver": r["driver"],
            "success": bool(r["success"]), "suite": r["suite"],
            "replan_t": r["replan_t"],
        }
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                key = f"{names[i]}-{names[j]}"
                d = _per_dim_l1(c[:, i], c[:, j])  # [n_replan, dim]
                pairs.setdefault(key, []).append(d)
                row[key] = d
        # Three-way spread per replan: one scalar "how much do they disagree here".
        spread = np.stack(
            [_per_dim_l1(c[:, i], c[:, j]).mean(axis=-1)
             for i in range(len(names)) for j in range(i + 1, len(names))]
        ).mean(axis=0)
        row["spread"] = spread
        per_rollout.append(row)

        nc = r.get("noise_chunks")
        if nc is not None:
            ns = nc.shape[1]
            floor.extend(
                _per_dim_l1(nc[:, a], nc[:, b])
                for a in range(ns) for b in range(a + 1, ns)
            )

    return {
        "pairs": {k: np.concatenate(v) for k, v in pairs.items()},
        "floor": np.concatenate(floor) if floor else None,
        "per_rollout": per_rollout,
        "dims": list(ACTION_DIM_NAMES),
    }


def eval_disagreement_vs_success(dis: dict[str, Any]) -> dict[str, Any]:
    """Does disagreement predict failure?

    If the three models diverge exactly where the driver goes wrong, disagreement is a
    usable online routing trigger -- it needs no ground truth. If failed and successful
    rollouts have the same disagreement profile, it is not.
    """
    ok = [r["spread"] for r in dis["per_rollout"] if r["success"]]
    bad = [r["spread"] for r in dis["per_rollout"] if not r["success"]]

    def _stats(xs: list[np.ndarray]) -> dict[str, float]:
        if not xs:
            return {"n": 0, "mean": float("nan"), "max": float("nan")}
        flat = np.concatenate(xs)
        return {
            "n": len(xs),
            "mean": float(flat.mean()),
            "max": float(np.mean([x.max() for x in xs])),
        }

    return {"success": _stats(ok), "failure": _stats(bad)}


def analyze_eval(data: dict[str, Any], *, cache_dir: Path = CACHE_DIR) -> dict[str, Any]:
    """Report for paired eval rollouts (no ground-truth action available)."""
    import csv

    cache_dir.mkdir(parents=True, exist_ok=True)
    names = data["models"]
    succ = eval_success_matrix(data)
    dis = eval_disagreement(data)
    vs = eval_disagreement_vs_success(dis)
    dims = dis["dims"]

    print("\n=== 1. per-episode success by driver ===")
    for d, n in succ["per_driver_successes"].items():
        print(f"  {d:8s} {n}/{succ['n_complete']}")
    print(f"  all drivers succeed : {len(succ['all_success'])}")
    print(f"  all drivers fail    : {len(succ['all_fail'])}")
    print(f"  split (ROUTABLE)    : {len(succ['routable'])}  episodes {succ['routable']}")
    best = max(succ["per_driver_successes"].values()) if succ["per_driver_successes"] else 0
    print(f"  best single driver  : {best}/{succ['n_complete']}")
    print(f"  per-episode oracle  : {succ['oracle_successes']}/{succ['n_complete']}")

    print("\n=== 2. on-policy cross-model disagreement (mean |diff| per dim) ===")
    print(f"{'pair':16s}" + "".join(f"{d:>10s}" for d in dims))
    for k, v in dis["pairs"].items():
        print(f"{k:16s}" + "".join(f"{x:>10.4f}" for x in v.mean(axis=0)))
    if dis["floor"] is not None:
        fm = dis["floor"].mean(axis=0)
        print(f"{'[seed floor]':16s}" + "".join(f"{x:>10.4f}" for x in fm))
        ratio = np.mean([v.mean(axis=0) for v in dis["pairs"].values()], axis=0) / np.maximum(fm, 1e-12)
        print(f"{'ratio x floor':16s}" + "".join(f"{x:>10.2f}" for x in ratio))
    else:
        print("[seed floor] not collected -- rerun the rollout with --noise-seeds")

    print("\n=== 3. does disagreement predict failure? ===")
    print(f"  successful rollouts (n={vs['success']['n']}): "
          f"mean spread {vs['success']['mean']:.4f}, peak {vs['success']['max']:.4f}")
    print(f"  failed rollouts     (n={vs['failure']['n']}): "
          f"mean spread {vs['failure']['mean']:.4f}, peak {vs['failure']['max']:.4f}")

    out_csv = cache_dir / "eval_rollout_disagreement.csv"
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["episode_idx", "suite", "driver", "success", "replan_idx",
                    "replan_t", "spread", *dis["pairs"].keys()])
        for r in dis["per_rollout"]:
            for i, tt in enumerate(r["replan_t"]):
                w.writerow([
                    r["episode_idx"], r["suite"], r["driver"], r["success"], i, tt,
                    f"{r['spread'][i]:.6f}",
                    *[f"{r[k][i].mean():.6f}" for k in dis["pairs"].keys()],
                ])

    summary = {
        "dir": data["dir"], "models": names, "drivers": data["drivers"],
        "success": {k: v for k, v in succ.items() if k not in ("by_episode", "meta")},
        "disagreement_mean": {k: v.mean(axis=0).tolist() for k, v in dis["pairs"].items()},
        "seed_floor_mean": (dis["floor"].mean(axis=0).tolist()
                            if dis["floor"] is not None else None),
        "disagreement_vs_success": vs,
        "dims": dims,
    }
    with open(cache_dir / "eval_rollout_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    print(f"\nwrote {out_csv}\n      {cache_dir / 'eval_rollout_summary.json'}")
    return {"success": succ, "disagreement": dis, "vs_success": vs, "summary": summary}


# --------------------------------------------------------------------------------------
# analysis
# --------------------------------------------------------------------------------------


def _per_dim_l1(a: torch.Tensor, b: torch.Tensor) -> np.ndarray:
    """Mean |a - b| over the action horizon, keeping the dimension axis.

    Per-dimension on purpose: the gripper dim carries different units from the six
    delta-EEF dims, so a single pooled norm is not interpretable.
    """
    return (a - b).abs().mean(dim=-2).numpy()


def pairwise_distances(payload: dict[str, Any]) -> dict[str, Any]:
    """Cross-model action distance per window, per dimension."""
    names = payload["models"]
    pred = payload["pred"]  # [M, E, W, H, D]
    pairs: dict[str, np.ndarray] = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            key = f"{names[i]}-{names[j]}"
            pairs[key] = _per_dim_l1(pred[i], pred[j])  # [E, W, D]
    return {"pairs": pairs, "dims": list(ACTION_DIM_NAMES[: pred.shape[-1]])}


def noise_floor_distances(floor: dict[str, Any]) -> np.ndarray:
    """Seed-to-seed distance for a single model -- same metric as ``pairwise_distances``."""
    pred = floor["pred"]  # [E, W, S, H, D]
    n_seed = pred.shape[2]
    if n_seed < 2:
        raise ValueError("Noise floor needs at least 2 seeds.")
    dists = [
        _per_dim_l1(pred[:, :, i], pred[:, :, j])
        for i in range(n_seed)
        for j in range(i + 1, n_seed)
    ]
    return np.stack(dists)  # [n_pairs, E, W, D]


def oracle_ceiling(payload: dict[str, Any]) -> dict[str, Any]:
    """The go/no-go number: perfect per-timestep selection vs the best fixed model.

    A selector that cannot beat the best single model by a meaningful margin means the
    routing idea has no headroom, regardless of how well a router could be trained.
    """
    names = payload["models"]
    pred = payload["pred"]  # [M, E, W, H, D]
    gt = payload["gt"].unsqueeze(0)  # [1, E, W, H, D]

    # Mean squared error per model per window, pooled over horizon and dims.
    err = ((pred - gt) ** 2).mean(dim=(-1, -2)).numpy()  # [M, E, W]

    per_model = err.reshape(len(names), -1).mean(axis=1)
    best_i = int(np.argmin(per_model))
    choice = err.argmin(axis=0)  # [E, W]
    oracle_err = err.min(axis=0).mean()

    best_single = float(per_model[best_i])
    rgb_i = names.index("RGB") if "RGB" in names else best_i

    counts = {names[k]: int((choice == k).sum()) for k in range(len(names))}
    return {
        "per_model_mse": {names[k]: float(per_model[k]) for k in range(len(names))},
        "best_single": names[best_i],
        "best_single_mse": best_single,
        "always_rgb_mse": float(per_model[rgb_i]),
        "oracle_mse": float(oracle_err),
        "gain_vs_best_single": float(1.0 - oracle_err / best_single) if best_single > 0 else 0.0,
        "choice": choice,
        "choice_counts": counts,
        "err": err,
    }


def oracle_structure(payload: dict[str, Any], oracle: dict[str, Any]) -> dict[str, Any]:
    """Is the oracle choice signal or noise -- by phase, and by temporal persistence."""
    names = payload["models"]
    choice = oracle["choice"]  # [E, W]
    phase = payload["phase"]  # [E, W]

    by_phase: dict[str, dict[str, int]] = {}
    for p, pname in enumerate(PHASE_NAMES):
        mask = phase == p
        if not mask.any():
            continue
        sel = choice[mask]
        by_phase[pname] = {names[k]: int((sel == k).sum()) for k in range(len(names))}

    # A choice that flips at the sampling-noise timescale is not routable.
    if choice.shape[1] > 1:
        same = (choice[:, 1:] == choice[:, :-1]).mean()
    else:
        same = float("nan")
    chance = 1.0 / max(len(names), 1)

    return {
        "by_phase": by_phase,
        "persistence": float(same),
        "persistence_chance": chance,
    }


def score_selection_rules(
    payload: dict[str, Any],
    oracle: dict[str, Any],
    *,
    rules: Optional[list[str]] = None,
    temperature: float = 0.02,
) -> dict[str, Any]:
    """Score every hard/soft selection rule against ground truth.

    This is the measurement the rules cannot get on sim: demo replay carries the true
    action, so each rule's fused chunk can be scored directly and ranked against both the
    best fixed model and the per-timestep oracle. ``hard_oracle`` is the ceiling; nothing
    that runs online can beat it.
    """
    import sys

    sys.path.insert(0, str(REPO))
    from experiments.libero.action_selection import SELECTION_RULES, select

    names = payload["models"]
    pred = payload["pred"].numpy()  # [M, E, W, H, D]
    gt = payload["gt"].numpy()  # [E, W, H, D]
    rules = rules or (list(SELECTION_RULES) + ["hard_oracle"])

    n_e, n_w = gt.shape[0], gt.shape[1]
    out: dict[str, Any] = {}
    for rule in rules:
        errs = []
        picks = np.zeros(len(names))
        for e in range(n_e):
            for w in range(n_w):
                chunks = pred[:, e, w]  # [M, H, D]
                fused, wt = select(
                    chunks, rule, temperature=temperature,
                    ground_truth=gt[e, w] if rule == "hard_oracle" else None,
                )
                errs.append(float(((fused - gt[e, w]) ** 2).mean()))
                picks += wt
        out[rule] = {
            "mse": float(np.mean(errs)),
            "weights": (picks / max(picks.sum(), 1e-9)).tolist(),
        }

    best = oracle["best_single_mse"]
    for rule, v in out.items():
        v["vs_best_single_pct"] = float(100.0 * (1.0 - v["mse"] / best)) if best > 0 else 0.0
    return {"rules": out, "best_single": oracle["best_single"], "best_single_mse": best,
            "models": names}


def report_selection_rules(scored: dict[str, Any]) -> None:
    names = scored["models"]
    print(f"\n=== selection rules vs ground truth "
          f"(best single = {scored['best_single']}, MSE {scored['best_single_mse']:.6f}) ===")
    print(f"{'rule':18s}{'MSE':>12s}{'vs best':>10s}   weight share")
    rows = sorted(scored["rules"].items(), key=lambda kv: kv[1]["mse"])
    for rule, v in rows:
        share = " ".join(f"{n}={w:.2f}" for n, w in zip(names, v["weights"]))
        tag = "  <- ceiling" if rule == "hard_oracle" else ""
        print(f"{rule:18s}{v['mse']:>12.6f}{v['vs_best_single_pct']:>9.1f}%   {share}{tag}")


def ensemble_baseline(payload: dict[str, Any], oracle: dict[str, Any]) -> dict[str, Any]:
    """Averaging needs no selection signal at all -- a router must beat this."""
    pred = payload["pred"]
    gt = payload["gt"]

    mean_err = ((pred.mean(dim=0) - gt) ** 2).mean().item()
    med_err = ((pred.median(dim=0).values - gt) ** 2).mean().item()

    best = oracle["best_single_mse"]
    orc = oracle["oracle_mse"]
    span = best - orc

    def _captured(e: float) -> float:
        return float((best - e) / span) if span > 0 else float("nan")

    return {
        "mean_mse": mean_err,
        "median_mse": med_err,
        "mean_captures_of_oracle_gain": _captured(mean_err),
        "median_captures_of_oracle_gain": _captured(med_err),
    }


def router_feasibility(payload: dict[str, Any], oracle: dict[str, Any]) -> dict[str, Any]:
    """Can the oracle label be recovered from what a router can actually see?

    The only ground-truth-free signal available here is inter-model disagreement: for each
    model, how far it sits from the other two. If picking the most "central" model
    recovers the oracle choice above chance, a router has something to learn from.
    """
    names = payload["models"]
    pred = payload["pred"]  # [M, E, W, H, D]
    m = pred.shape[0]

    spread = torch.stack(
        [
            torch.stack(
                [((pred[i] - pred[j]) ** 2).mean(dim=(-1, -2)) for j in range(m) if j != i]
            ).mean(dim=0)
            for i in range(m)
        ]
    ).numpy()  # [M, E, W]

    centroid_choice = spread.argmin(axis=0)
    agree = float((centroid_choice == oracle["choice"]).mean())

    # Error of the model the centroid rule picks, vs oracle and best-single.
    err = oracle["err"]
    ei, wi = np.meshgrid(
        np.arange(err.shape[1]), np.arange(err.shape[2]), indexing="ij"
    )
    centroid_mse = float(err[centroid_choice, ei, wi].mean())

    return {
        "centroid_agreement_with_oracle": agree,
        "chance_agreement": 1.0 / max(m, 1),
        "centroid_mse": centroid_mse,
        "oracle_mse": oracle["oracle_mse"],
        "best_single_mse": oracle["best_single_mse"],
    }


def analyze(
    payload: dict[str, Any],
    floor: Optional[dict[str, Any]] = None,
    *,
    cache_dir: Path = CACHE_DIR,
) -> dict[str, Any]:
    """Run every analysis in order and write the CSV/JSON exports."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    names = payload["models"]
    dims = list(ACTION_DIM_NAMES[: payload["pred"].shape[-1]])

    pw = pairwise_distances(payload)
    oracle = oracle_ceiling(payload)
    struct = oracle_structure(payload, oracle)
    ens = ensemble_baseline(payload, oracle)
    router = router_feasibility(payload, oracle)
    scored = score_selection_rules(payload, oracle)

    floor_d = noise_floor_distances(floor) if floor is not None else None

    # ---- report -------------------------------------------------------------------
    print("\n=== 1. cross-model action distance (mean |diff| per dim) ===")
    header = f"{'pair':16s}" + "".join(f"{d:>10s}" for d in dims)
    print(header)
    for key, arr in pw["pairs"].items():
        print(f"{key:16s}" + "".join(f"{v:>10.4f}" for v in arr.mean(axis=(0, 1))))
    if floor_d is not None:
        fm = floor_d.mean(axis=(0, 1, 2))
        print(f"{'[seed floor]':16s}" + "".join(f"{v:>10.4f}" for v in fm))
        ratio = np.mean([a.mean(axis=(0, 1)) for a in pw["pairs"].values()], axis=0) / np.maximum(fm, 1e-12)
        print(f"{'ratio x floor':16s}" + "".join(f"{v:>10.2f}" for v in ratio))
    else:
        print("[seed floor] not computed -- cross-model distances have no null to be read against")

    print("\n=== 2. oracle ceiling (MSE vs ground truth) ===")
    for k, v in oracle["per_model_mse"].items():
        print(f"  {k:8s} {v:.6f}")
    print(f"  best single ({oracle['best_single']}): {oracle['best_single_mse']:.6f}")
    print(f"  always RGB:                {oracle['always_rgb_mse']:.6f}")
    print(f"  oracle:                    {oracle['oracle_mse']:.6f}")
    print(f"  oracle gain vs best single: {100.0 * oracle['gain_vs_best_single']:.1f}%")
    print(f"  oracle choice counts: {oracle['choice_counts']}")

    print("\n=== 3. is the oracle choice structured? ===")
    for pname, counts in struct["by_phase"].items():
        total = max(sum(counts.values()), 1)
        share = {k: f"{100.0 * v / total:.0f}%" for k, v in counts.items()}
        print(f"  {pname:18s} {share}")
    print(
        f"  temporal persistence: {struct['persistence']:.3f} "
        f"(chance {struct['persistence_chance']:.3f})"
    )

    print("\n=== 4. ensembling baseline ===")
    print(f"  mean   MSE {ens['mean_mse']:.6f}  "
          f"captures {100.0 * ens['mean_captures_of_oracle_gain']:.0f}% of oracle gain")
    print(f"  median MSE {ens['median_mse']:.6f}  "
          f"captures {100.0 * ens['median_captures_of_oracle_gain']:.0f}% of oracle gain")

    print("\n=== 5. router feasibility (no ground truth) ===")
    print(f"  centroid rule agrees with oracle: {100.0 * router['centroid_agreement_with_oracle']:.1f}% "
          f"(chance {100.0 * router['chance_agreement']:.1f}%)")
    print(f"  centroid MSE {router['centroid_mse']:.6f} vs oracle {router['oracle_mse']:.6f} "
          f"vs best single {router['best_single_mse']:.6f}")

    report_selection_rules(scored)

    # ---- exports ------------------------------------------------------------------
    import csv

    dist_csv = cache_dir / "action_pairwise_distance.csv"
    with open(dist_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["pair", *dims])
        for key, arr in pw["pairs"].items():
            w.writerow([key, *[f"{v:.6f}" for v in arr.mean(axis=(0, 1))]])
        if floor_d is not None:
            w.writerow(["seed_floor", *[f"{v:.6f}" for v in floor_d.mean(axis=(0, 1, 2))]])

    choice_csv = cache_dir / "oracle_choice.csv"
    with open(choice_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["episode", "window", "frame_idx", "phase", "oracle_model",
                    *[f"mse_{n}" for n in names]])
        for e, ep in enumerate(payload["episodes"]):
            for win in range(oracle["choice"].shape[1]):
                p = int(payload["phase"][e, win])
                w.writerow([
                    ep, win, payload["frame_idx"][e][win],
                    PHASE_NAMES[p] if 0 <= p < len(PHASE_NAMES) else "unknown",
                    names[int(oracle["choice"][e, win])],
                    *[f"{oracle['err'][k, e, win]:.6f}" for k in range(len(names))],
                ])

    summary = {
        "models": names,
        "episodes": payload["episodes"],
        "space": payload.get("space", "normalized"),
        "ckpts": payload.get("ckpts", {}),
        "per_model_mse": oracle["per_model_mse"],
        "best_single": oracle["best_single"],
        "oracle_mse": oracle["oracle_mse"],
        "oracle_gain_vs_best_single": oracle["gain_vs_best_single"],
        "oracle_choice_counts": oracle["choice_counts"],
        "by_phase": struct["by_phase"],
        "persistence": struct["persistence"],
        "ensemble": ens,
        "router": {k: v for k, v in router.items()},
        "selection_rules": scored["rules"],
    }
    summary_path = cache_dir / "action_agreement_summary.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2, default=float)

    print(f"\nwrote {dist_csv}\n      {choice_csv}\n      {summary_path}")
    return {
        "pairwise": pw,
        "selection": scored,
        "oracle": oracle,
        "structure": struct,
        "ensemble": ens,
        "router": router,
        "summary": summary,
    }


if __name__ == "__main__":
    import argparse
    import os

    # RobotVideoDataset writes dataset_stats.json into a cwd-relative work_dir
    # (`./runs/...` from the Hydra config), so run from the repo root.
    os.chdir(REPO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--n-episodes", type=int, default=10)
    parser.add_argument("--n-windows", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--noise-seeds", type=int, nargs="*", default=[0])
    parser.add_argument("--noise-model", type=str, default="RGB")
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no-reuse-cache", action="store_true")
    parser.add_argument("--eval-dir", type=str, default=None,
                        help="Analyze paired LIBERO rollouts from "
                             "experiments/libero/eval_action_agreement.py instead of "
                             "replaying demo episodes.")
    parser.add_argument("--noise-floor-only", action="store_true")
    parser.add_argument("--analysis-only", action="store_true")
    parser.add_argument("--skip-noise-floor", action="store_true")
    args = parser.parse_args()

    if args.eval_dir:
        analyze_eval(load_eval_rollouts(Path(args.eval_dir)))
        raise SystemExit(0)

    reuse = not args.no_reuse_cache
    common = dict(
        n_episodes=args.n_episodes,
        n_windows=args.n_windows,
        device=args.device,
        reuse_cache=True if args.analysis_only else reuse,
    )

    if args.noise_floor_only:
        seed_noise_floor(
            model_name=args.noise_model,
            seeds=tuple(args.noise_seeds),
            **common,
        )
        raise SystemExit(0)

    payload = extract_actions(seed=args.seed, models=args.models, **common)

    floor = None
    if not args.skip_noise_floor:
        floor = seed_noise_floor(
            model_name=args.noise_model,
            seeds=tuple(args.noise_seeds),
            **common,
        )

    analyze(payload, floor)
