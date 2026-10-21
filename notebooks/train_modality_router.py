"""Train the offline modality router on action-error oracle labels.

``labels_*.pt`` from ``modality_action_labels.py --analyze`` supplies the oracle
(``label``, ``weighted_action_loss``) and index metadata (``frame_idx``,
``episode``, ``suite``).

Router features come from the **same tensors FastWAM training feeds the MoT**
(``FastWAM.build_inputs``), with no extra pooling:

* ``r0`` — ``input_latents[:, :, 0]`` flattened ``[B, C*H*W]``
* ``context`` — full text+proprio token sequence ``[B, L, D]``
  (proprio already appended as in training)

Hard CE on argmin action-error label. Episode-stratified train/val/test per suite.

Usage::

    python notebooks/train_modality_router.py \\
        --labels notebooks/outputs/modality_action_labels/labels_....pt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent))

from modality_action_labels import load_label_dataset  # noqa: E402
from repr_analysis import (  # noqa: E402
    JOINT_UNIMODAL,
    compose_task_cfg,
    load_joint_model,
    pick_device,
)

# Prefer the in-graph router so offline CE and FastWAM training share one class.
try:
    from fastwam.models.wan22.modality_router import (  # noqa: E402
        ModalityRouter,
        collapse_kl,
    )
except ImportError:  # pragma: no cover — notebook without package install
    ModalityRouter = None  # type: ignore
    collapse_kl = None  # type: ignore

MODELS = ("RGB", "Depth", "Flow")
# Match training inputs (not the old pooled r0_pool / context_mean summaries).
FEATURE_GROUPS = ("r0", "context")


def _as_np(x):
    """Labels export may store torch tensors (protocol-4) or numpy arrays."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def collate_train_samples(samples: list[dict]) -> dict[str, torch.Tensor]:
    """Stack fields needed by ``FastWAM.build_inputs``."""
    keys = ("video", "action", "proprio", "context", "context_mask", "action_is_pad", "image_is_pad")
    batch: dict[str, torch.Tensor] = {}
    for key in keys:
        if key not in samples[0]:
            continue
        batch[key] = torch.stack([s[key] for s in samples], dim=0)
    return batch


@torch.no_grad()
def router_features_from_batch(model, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Training tensors from ``build_inputs`` — no adaptive pool, no token mean.

    Returns
    -------
    r0 : [B, C*H*W]
        Full first latent frame, flattened.
    context : [B, L, D]
        Full MoT context after proprio append (same as training).
    context_mask : [B, L]
        Bool mask aligned with ``context``.
    text_mean : [B, D]
        Text-only mean (no proprio) for the per-task majority baseline only.
    """
    # Baseline hash feature only (not a router input).
    ctx0 = batch["context"].float()
    m0 = batch["context_mask"].float().unsqueeze(-1)
    text_mean = (ctx0 * m0).sum(1) / m0.sum(1).clamp(min=1.0)

    inputs = model.build_inputs(batch, tiled=False)
    # Same R0 MoT conditions on — full spatial map, not pooled to 2x2.
    r0 = inputs["input_latents"][:, :, 0].float().flatten(1)
    # Same context sequence MoT attends to — keep all tokens.
    context = inputs["context"].float()
    context_mask = inputs["context_mask"].bool()
    return {
        "r0": r0,
        "context": context,
        "context_mask": context_mask,
        "text_mean": text_mean,
    }


@torch.no_grad()
def extract_features_from_dataset(
    *,
    frame_idxs: np.ndarray,
    dataset,
    model,
    batch_size: int,
    num_workers: int = 4,
) -> dict[str, np.ndarray]:
    """Encode every labeled window via the training sample dict + ``build_inputs``."""
    loader = DataLoader(
        Subset(dataset, frame_idxs.tolist()),
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_train_samples,
        prefetch_factor=2 if num_workers > 0 else None,
        pin_memory=str(model.device).startswith("cuda"),
    )
    outs: dict[str, list[np.ndarray]] = {
        g: [] for g in (*FEATURE_GROUPS, "context_mask", "text_mean")
    }
    t0, n_seen = time.time(), 0
    n_total = len(frame_idxs)
    for batch in loader:
        feats = router_features_from_batch(model, batch)
        for g, tensor in feats.items():
            outs[g].append(tensor.float().cpu().numpy())
        n_seen += int(batch["video"].shape[0])
        if n_seen == n_total or n_seen % max(batch_size * 20, 1) < batch_size:
            rate = n_seen / max(time.time() - t0, 1e-9)
            eta = (n_total - n_seen) / max(rate, 1e-9) / 60.0
            print(f"  features {n_seen}/{n_total}  {rate:.1f} win/s  ~{eta:.1f} min left", flush=True)
    return {g: np.concatenate(v, axis=0) for g, v in outs.items()}


if ModalityRouter is None:  # pragma: no cover

    class ModalityRouter(nn.Module):  # type: ignore[no-redef]
        """Fallback copy of ``fastwam.models.wan22.modality_router.ModalityRouter``."""

        def __init__(
            self,
            *,
            r0_dim: int,
            context_dim: int,
            context_len: int,
            num_classes: int = len(MODELS),
            d: int = 128,
            hidden: int = 256,
            dropout: float = 0.1,
        ):
            super().__init__()
            self.r0_dim = int(r0_dim)
            self.context_dim = int(context_dim)
            self.context_len = int(context_len)
            self.num_classes = int(num_classes)
            self.r0_proj = nn.Sequential(nn.LayerNorm(r0_dim), nn.Linear(r0_dim, d))
            self.context_proj = nn.Sequential(
                nn.LayerNorm(context_dim), nn.Linear(context_dim, d)
            )
            self.head = nn.Sequential(
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(d + d * self.context_len, hidden),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, self.num_classes),
            )
            for linear in (self.r0_proj[1], self.context_proj[1], self.head[2], self.head[5]):
                nn.init.kaiming_normal_(linear.weight, nonlinearity="relu")
                if linear.bias is not None:
                    nn.init.zeros_(linear.bias)

        def forward(self, feats: dict[str, torch.Tensor]) -> torch.Tensor:
            r0, context = feats["r0"], feats["context"]
            if "context_mask" in feats:
                context = context * feats["context_mask"].to(dtype=context.dtype).unsqueeze(-1)
            return self.head(
                torch.cat(
                    [self.r0_proj(r0), self.context_proj(context).flatten(1)], dim=-1
                )
            )

        @torch.no_grad()
        def route_offline(self, feats: dict[str, torch.Tensor]) -> torch.Tensor:
            was_training = self.training
            self.eval()
            out = self.forward(feats).argmax(dim=-1)
            if was_training:
                self.train()
            return out

    def collapse_kl(logits: torch.Tensor) -> torch.Tensor:  # type: ignore[no-redef]
        p_bar = F.softmax(logits, dim=-1).mean(0).clamp_min(1e-8)
        uniform = torch.full_like(p_bar, 1.0 / p_bar.numel())
        return F.kl_div(p_bar.log(), uniform, reduction="sum")


def split_by_episode(episodes: np.ndarray, suites: np.ndarray, seed: int, frac=(0.8, 0.1)):
    rng = np.random.default_rng(seed)
    train, val, test = [], [], []
    for s in np.unique(suites):
        eps = np.unique(episodes[suites == s])
        rng.shuffle(eps)
        a, b = int(len(eps) * frac[0]), int(len(eps) * (frac[0] + frac[1]))
        train += eps[:a].tolist()
        val += eps[a:b].tolist()
        test += eps[b:].tolist()
    return (np.isin(episodes, train), np.isin(episodes, val), np.isin(episodes, test))


def evaluate(pick: np.ndarray, loss: np.ndarray, label: np.ndarray, sel: np.ndarray) -> dict:
    idx = np.where(sel)[0]
    return {
        "loss": float(loss[idx, pick[idx]].mean()),
        "acc": float((pick[idx] == label[idx]).mean()),
        "picks": np.bincount(pick[idx], minlength=3).tolist(),
    }


def print_confusion(name: str, pick: np.ndarray, label: np.ndarray, sel: np.ndarray) -> None:
    cm = np.zeros((3, 3), dtype=int)
    np.add.at(cm, (label[sel], pick[sel]), 1)
    print(f"\n{name}: label (rows) vs router pick (cols), test windows")
    print(f"  {'':12s}" + "".join(f"{'pick ' + m:>12s}" for m in MODELS) + f"{'recall':>9s}")
    for i, m in enumerate(MODELS):
        rec = cm[i, i] / max(cm[i].sum(), 1)
        print(f"  {'label ' + m:12s}" + "".join(f"{v:12d}" for v in cm[i]) + f"{rec:9.2f}")
    prec = [cm[j, j] / max(cm[:, j].sum(), 1) for j in range(3)]
    print(
        f"  {'precision':12s}" + "".join(f"{p:12.2f}" for p in prec)
        + f"   accuracy {np.trace(cm) / max(cm.sum(), 1):.2f}"
    )


def _batch_feats(x: dict[str, torch.Tensor], idx: torch.Tensor) -> dict[str, torch.Tensor]:
    out = {"r0": x["r0"][idx], "context": x["context"][idx]}
    if "context_mask" in x:
        out["context_mask"] = x["context_mask"][idx]
    return out


def train_router(
    feats,
    target,
    weights,
    masks,
    *,
    epochs: int,
    lr: float,
    wd: float,
    batch: int,
    seed: int,
    device: str,
    kl: float = 0.0,
):
    torch.manual_seed(seed)
    tr, va, _ = masks
    r0_dim = int(feats["r0"].shape[1])
    context_len = int(feats["context"].shape[1])
    context_dim = int(feats["context"].shape[2])
    model = ModalityRouter(
        r0_dim=r0_dim,
        context_dim=context_dim,
        context_len=context_len,
        num_classes=len(MODELS),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    x = {
        "r0": torch.from_numpy(feats["r0"]).float().to(device),
        "context": torch.from_numpy(feats["context"]).float().to(device),
        "context_mask": torch.from_numpy(feats["context_mask"]).bool().to(device),
    }
    y = torch.from_numpy(target).long().to(device)
    cw = torch.from_numpy(weights).float().to(device)
    tr_idx = torch.from_numpy(np.where(tr)[0]).to(device)
    va_idx = torch.from_numpy(np.where(va)[0]).to(device)

    def loss_fn(logits, yy):
        fit = F.cross_entropy(logits, yy, weight=cw)
        if kl <= 0:
            return fit
        return fit + kl * collapse_kl(logits)

    best, best_state, bad, best_ep = float("inf"), None, 0, 0
    ep = 0
    for ep in range(epochs):
        model.train()
        perm = tr_idx[torch.randperm(len(tr_idx), device=device)]
        for i in range(0, len(perm), batch):
            b = perm[i: i + batch]
            loss = loss_fn(model(_batch_feats(x, b)), y[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = float(loss_fn(model(_batch_feats(x, va_idx)), y[va_idx]))
        if vl < best - 1e-5:
            best, best_state, bad, best_ep = (
                vl,
                {k: v.detach().clone() for k, v in model.state_dict().items()},
                0,
                ep,
            )
        else:
            bad += 1
            if bad >= 15:
                break
    print(f"  stopped after {ep + 1} epochs; best val loss {best:.4f} at epoch {best_ep + 1}")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        # Score in chunks to limit peak memory with full context sequences.
        logits_chunks = []
        all_idx = torch.arange(len(y), device=device)
        for i in range(0, len(all_idx), batch):
            b = all_idx[i: i + batch]
            logits_chunks.append(model(_batch_feats(x, b)).cpu())
        logits = torch.cat(logits_chunks, dim=0).numpy()
    return model, logits, best


def _default_feature_cache(labels_path: Path) -> Path:
    return labels_path.with_name(labels_path.stem + "_router_feats_nopool.pt")


def load_or_extract_features(
    *,
    labels_path: Path,
    frame_idxs: np.ndarray,
    cache_path: Path | None,
    force_extract: bool,
    feature_batch: int,
    device: str,
) -> dict[str, np.ndarray]:
    if cache_path is None:
        cache_path = _default_feature_cache(labels_path)
    need = (*FEATURE_GROUPS, "context_mask", "text_mean")
    if cache_path.is_file() and not force_extract:
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        cached_idx = _as_np(cached["frame_idx"]).astype(np.int64)
        if np.array_equal(cached_idx, frame_idxs) and all(k in cached for k in need):
            print(f"loaded feature cache {cache_path}", flush=True)
            return {g: _as_np(cached[g]) for g in need}
        print(f"feature cache stale/mismatch; re-extracting -> {cache_path}", flush=True)

    if device == "cuda":
        device = pick_device()
    spec = JOINT_UNIMODAL["RGB"]
    model_cfg = compose_task_cfg(spec["task"])
    _, dataset = load_label_dataset(Path(spec["run_dir"]) / "dataset_stats.json")
    print(f"feature extract device={device}  dataset_len={len(dataset)}", flush=True)
    model = load_joint_model(model_cfg, Path(spec["ckpt"]), device=device)
    print(
        f"extracting training features (r0 flatten + full context) "
        f"n={len(frame_idxs)} batch={feature_batch}…",
        flush=True,
    )
    feats = extract_features_from_dataset(
        frame_idxs=frame_idxs,
        dataset=dataset,
        model=model,
        batch_size=feature_batch,
    )
    payload = {"frame_idx": torch.from_numpy(frame_idxs.astype(np.int64))}
    for g, arr in feats.items():
        payload[g] = torch.from_numpy(np.ascontiguousarray(arr))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache_path, pickle_protocol=4)
    print(f"wrote feature cache {cache_path}", flush=True)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return feats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1e-2)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--feature-batch", type=int, default=4,
                        help="VAE / build_inputs batch size when extracting features from samples")
    parser.add_argument("--feature-cache", type=str, default=None,
                        help="optional path for extracted features "
                             "(default: <labels>_router_feats_nopool.pt)")
    parser.add_argument("--force-extract", action="store_true",
                        help="ignore existing feature cache and re-encode from samples")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--kl", "--kl-uniform", type=float, default=0.0, dest="kl",
                        help="weight on KL(Uniform || mean softmax); >0 blocks always-Depth collapse")
    parser.add_argument("--tag", type=str, default="hard",
                        help="suffix for output files (default hard -> routers_hard.pt)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    labels_path = Path(args.labels)
    d = torch.load(labels_path, map_location="cpu", weights_only=False)
    for key in ("label", "weighted_action_loss", "episode", "suite", "frame_idx", "keys"):
        if key not in d:
            raise KeyError(f"{labels_path} missing `{key}` (from modality_action_labels --analyze)")

    loss_score = _as_np(d["weighted_action_loss"]).astype(np.float32)
    label_train = label_score = _as_np(d["label"]).astype(np.int64)
    frame_idxs = _as_np(d["frame_idx"]).astype(np.int64)
    oracle_name = "oracle (all-seed label)"
    n = len(label_train)
    if len(frame_idxs) != n:
        raise ValueError(f"frame_idx/label length mismatch: {len(frame_idxs)} vs {n}")

    masks = split_by_episode(_as_np(d["episode"]), _as_np(d["suite"]), args.seed)
    tr, va, te = masks

    raw = load_or_extract_features(
        labels_path=labels_path,
        frame_idxs=frame_idxs,
        cache_path=Path(args.feature_cache) if args.feature_cache else None,
        force_extract=bool(args.force_extract),
        feature_batch=args.feature_batch,
        device=args.device,
    )

    # Standardize on train split only. context: per-dim over (N,L) jointly.
    feats: dict[str, np.ndarray] = {}
    feat_norm: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    r0 = raw["r0"].astype(np.float32)
    mu, sd = r0[tr].mean(0), r0[tr].std(0) + 1e-6
    feat_norm["r0"] = (mu, sd)
    feats["r0"] = (r0 - mu) / sd

    ctx = raw["context"].astype(np.float32)  # [N, L, D]
    mu = ctx[tr].reshape(-1, ctx.shape[-1]).mean(0)
    sd = ctx[tr].reshape(-1, ctx.shape[-1]).std(0) + 1e-6
    feat_norm["context"] = (mu, sd)
    feats["context"] = (ctx - mu) / sd
    feats["context_mask"] = raw["context_mask"].astype(bool)

    counts = np.bincount(label_train[tr], minlength=3)
    class_w = (counts.sum() / np.maximum(counts, 1) / 3).astype(np.float32)

    print(f"{n} windows | train {tr.sum()} / val {va.sum()} / test {te.sum()} (split by episode)")
    print(f"hard CE | train label counts R/D/F = {counts.tolist()}")
    print(
        f"features from build_inputs: r0 {feats['r0'].shape[1]}-d "
        f"(C*H*W) + context {feats['context'].shape[1]}x{feats['context'].shape[2]} "
        f"(no pool / no mean)"
    )
    if args.kl > 0:
        print(f"loss = CE + {args.kl:g} * KL(Uniform || mean_p)  (anti-collapse)")

    rows = {}
    best_single = int(loss_score[tr].mean(0).argmin())
    rows["always " + MODELS[best_single]] = evaluate(np.full(n, best_single), loss_score, label_score, te)
    for m in range(3):
        if m != best_single:
            rows["always " + MODELS[m]] = evaluate(np.full(n, m), loss_score, label_score, te)
    rows[oracle_name] = evaluate(label_train, loss_score, label_score, te)

    task_feat = raw["text_mean"].astype(np.float32)
    task_key = np.array([hash(v.tobytes()) for v in task_feat])
    prior = {}
    for t in np.unique(task_key[tr]):
        prior[t] = int(np.bincount(label_train[tr & (task_key == t)], minlength=3).argmax())
    rows["per-task majority"] = evaluate(
        np.array([prior.get(t, best_single) for t in task_key]), loss_score, label_score, te
    )

    deploy_name = "router hard CE (r0 + context)"
    print(deploy_name)
    model, logits, vloss = train_router(
        feats, label_train, class_w, masks,
        epochs=args.epochs, lr=args.lr, wd=args.wd, batch=args.batch, seed=args.seed,
        device=args.device, kl=args.kl,
    )
    rows[deploy_name] = evaluate(logits.argmax(1), loss_score, label_score, te)
    rows[deploy_name]["val_loss"] = vloss
    picks = {
        deploy_name: {
            "pick": logits.argmax(1),
            "prob": torch.softmax(torch.from_numpy(logits), -1).numpy(),
        }
    }
    print_confusion(deploy_name, picks[deploy_name]["pick"], label_score, te)

    base = rows["always " + MODELS[best_single]]["loss"]
    print(f"\n{('test set (held-out episodes)'):46s} {'loss':>10s} {'vs best':>8s} {'acc':>6s}  picks R/D/F")
    for name, r in rows.items():
        print(
            f"{name:46s} {r['loss']:10.6f} {100 * (1 - r['loss'] / base):+7.1f}% "
            f"{r['acc']:6.2f}  {r['picks']}"
        )

    out_dir = labels_path.parent
    suffix = "" if args.kl <= 0 else f"_kl{args.kl:g}"
    tag = args.tag.strip("_")
    if tag:
        suffix = f"{suffix}_{tag}" if suffix else f"_{tag}"

    torch.save({
        "routers": {
            deploy_name: {
                "state_dict": model.state_dict(),
                "groups": list(FEATURE_GROUPS),
                "r0_dim": int(feats["r0"].shape[1]),
                "context_dim": int(feats["context"].shape[2]),
                "context_len": int(feats["context"].shape[1]),
            }
        },
        "feature_norm": feat_norm,
        "models": list(MODELS),
        "kl": args.kl,
        "loss": "hard_ce",
        "feature_source": "build_inputs_nopool",
    }, out_dir / f"routers{suffix}.pt")
    split = np.where(tr, "train", np.where(va, "val", "test"))
    torch.save(
        {
            "keys": d["keys"],
            "split": split,
            "label": label_score,
            "models": list(MODELS),
            "routers": picks,
            "loss": "hard_ce",
        },
        out_dir / f"router_predictions{suffix}.pt",
    )
    with open(out_dir / f"router_results{suffix}.json", "w") as fh:
        json.dump(
            {
                "labels": str(labels_path),
                "kl": args.kl,
                "loss": "hard_ce",
                "feature_source": "build_inputs_nopool",
                "rows": rows,
                "best": deploy_name,
                "deploy": deploy_name,
            },
            fh,
            indent=2,
        )
    print(f"\ndeploy router: {deploy_name}")
    print(f"all trained -> {out_dir / f'routers{suffix}.pt'}")


if __name__ == "__main__":
    main()
