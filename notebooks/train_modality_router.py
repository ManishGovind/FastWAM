"""Train the offline modality router on loss-based proxy labels.

Input is ``labels_<tag>.pt`` from ``modality_action_labels.py --analyze``: per window the
router features (pooled R0, T5 text, proprio) and each expert's weighted action error.

All routers use **hard CE** on the argmin action-error label (no soft targets). Feature
ablations (all / R0+proprio / text / …) are trained and written to ``routers_hard.pt``
by default. Optional ``--label-seeds half-a`` trains on even seeds and scores on odd.
Train/val/test are always split by episode, stratified per suite.

Usage::

    python notebooks/train_modality_router.py --labels notebooks/outputs/modality_action_labels/labels_....pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MODELS = ("RGB", "Depth", "Flow")


class ModalityRouter(nn.Module):
    """Each feature group gets its own LayerNorm + projection so the 4096-d text vector
    cannot drown out the 192-d image and 8-d proprio parts."""

    def __init__(self, dims: dict[str, int], d: int = 128, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.names = list(dims)
        self.proj = nn.ModuleDict({k: nn.Sequential(nn.LayerNorm(v), nn.Linear(v, d)) for k, v in dims.items()})
        self.head = nn.Sequential(
            nn.SiLU(), nn.Dropout(dropout), nn.Linear(d * len(dims), hidden), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(hidden, len(MODELS)),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, feats: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.head(torch.cat([self.proj[k](feats[k]) for k in self.names], dim=-1))


def split_by_episode(episodes: np.ndarray, suites: np.ndarray, seed: int, frac=(0.8, 0.1)):
    rng = np.random.default_rng(seed)
    train, val, test = [], [], []
    for s in np.unique(suites):
        eps = np.unique(episodes[suites == s])
        rng.shuffle(eps)
        a, b = int(len(eps) * frac[0]), int(len(eps) * (frac[0] + frac[1]))
        train += eps[:a].tolist(); val += eps[a:b].tolist(); test += eps[b:].tolist()
    return (np.isin(episodes, train), np.isin(episodes, val), np.isin(episodes, test))


def evaluate(pick: np.ndarray, loss_b: np.ndarray, label_b: np.ndarray, sel: np.ndarray) -> dict:
    """Loss of the picked model on held-out draws, plus accuracy vs the half-B label."""
    idx = np.where(sel)[0]
    return {
        "loss": float(loss_b[idx, pick[idx]].mean()),
        "acc": float((pick[idx] == label_b[idx]).mean()),
        "picks": np.bincount(pick[idx], minlength=3).tolist(),
    }


def print_confusion(name: str, pick: np.ndarray, label: np.ndarray, sel: np.ndarray) -> None:
    """Rows = action-error label, columns = router pick, on the windows in `sel`."""
    cm = np.zeros((3, 3), dtype=int)
    np.add.at(cm, (label[sel], pick[sel]), 1)
    print(f"\n{name}: label (rows) vs router pick (cols), test windows")
    print(f"  {'':12s}" + "".join(f"{'pick ' + m:>12s}" for m in MODELS) + f"{'recall':>9s}")
    for i, m in enumerate(MODELS):
        rec = cm[i, i] / max(cm[i].sum(), 1)
        print(f"  {'label ' + m:12s}" + "".join(f"{v:12d}" for v in cm[i]) + f"{rec:9.2f}")
    prec = [cm[j, j] / max(cm[:, j].sum(), 1) for j in range(3)]
    print(f"  {'precision':12s}" + "".join(f"{p:12.2f}" for p in prec)
          + f"   accuracy {np.trace(cm) / max(cm.sum(), 1):.2f}")


def fit_kl(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """KL(target || softmax). Same gradient as soft CE; target is a 3-way distribution."""
    return F.kl_div(F.log_softmax(logits, dim=-1), target, reduction="batchmean")


def collapse_kl(logits: torch.Tensor) -> torch.Tensor:
    """KL(Uniform || batch-mean softmax). Zero if all three modalities are used equally.

    Reverse KL: if any class has mean mass ~0, the term blows up, so the router cannot
    collapse to always-Depth.
    """
    p_bar = F.softmax(logits, dim=-1).mean(0).clamp_min(1e-8)
    uniform = torch.full_like(p_bar, 1.0 / p_bar.numel())
    return F.kl_div(p_bar.log(), uniform, reduction="sum")


def train_router(feats, target, weights, masks, *, groups, soft: bool, epochs: int, lr: float,
                 wd: float, batch: int, seed: int, device: str, kl: float = 0.0):
    torch.manual_seed(seed)
    tr, va, _ = masks
    dims = {g: feats[g].shape[1] for g in groups}
    model = ModalityRouter(dims).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    x = {g: torch.from_numpy(feats[g]).float().to(device) for g in groups}
    y = torch.from_numpy(target).float().to(device) if soft else torch.from_numpy(target).long().to(device)
    cw = torch.from_numpy(weights).float().to(device)
    tr_idx = torch.from_numpy(np.where(tr)[0]).to(device)
    va_idx = torch.from_numpy(np.where(va)[0]).to(device)

    def loss_fn(logits, yy):
        if soft:
            fit = fit_kl(logits, yy)
        else:
            fit = F.cross_entropy(logits, yy, weight=cw)
        if kl <= 0:
            return fit
        return fit + kl * collapse_kl(logits)

    best, best_state, bad = float("inf"), None, 0
    for ep in range(epochs):
        model.train()
        perm = tr_idx[torch.randperm(len(tr_idx), device=device)]
        for i in range(0, len(perm), batch):
            b = perm[i: i + batch]
            loss = loss_fn(model({g: x[g][b] for g in groups}), y[b])
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = float(loss_fn(model({g: x[g][va_idx] for g in groups}), y[va_idx]))
        if vl < best - 1e-5:
            best, best_state, bad, best_ep = vl, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0, ep
        else:
            bad += 1
            if bad >= 15:
                break
    print(f"  stopped after {ep + 1} epochs; best val loss {best:.4f} at epoch {best_ep + 1}")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(x).cpu().numpy()
    return model, logits, best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1e-2)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--kl", "--kl-uniform", type=float, default=0.0, dest="kl",
                   help="weight on KL(Uniform || mean softmax); >0 blocks always-Depth collapse")
    parser.add_argument("--label-seeds", choices=("all", "half-a"), default="all",
                        help="all = mean error over seeds 0-3 (default); half-a = train on even, score on odd")
    parser.add_argument("--tag", type=str, default="hard",
                        help="suffix for output files (default hard -> routers_hard.pt); empty string to omit")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    d = torch.load(args.labels, map_location="cpu", weights_only=False)
    if args.label_seeds == "all":
        # Mean action error over all collected seeds → one hard label per window.
        loss_train = loss_score = d["weighted_action_loss"]  # [N, M]
        label_train = label_score = d["label"]
        oracle_name = "oracle (all-seed label)"
    else:
        loss_train, loss_score = d["loss_half_a"], d["loss_half_b"]
        label_train, label_score = loss_train.argmin(1), loss_score.argmin(1)
        oracle_name = "oracle (half-A label)"
    n = len(label_train)
    masks = split_by_episode(d["episode"], d["suite"], args.seed)
    tr, va, te = masks

    # Standardize features on train only.
    feats = {}
    for g in ("r0_pool", "text_mean", "proprio"):
        x = d[g].astype(np.float32)
        mu, sd = x[tr].mean(0), x[tr].std(0) + 1e-6
        feats[g] = (x - mu) / sd

    counts = np.bincount(label_train[tr], minlength=3)
    class_w = (counts.sum() / np.maximum(counts, 1) / 3).astype(np.float32)

    print(f"{n} windows | train {tr.sum()} / val {va.sum()} / test {te.sum()} (split by episode)")
    print(f"label-seeds={args.label_seeds} | hard CE | train label counts R/D/F = {counts.tolist()}")
    if args.kl > 0:
        print(f"loss = CE + {args.kl:g} * KL(Uniform || mean_p)  (anti-collapse)")

    rows = {}
    best_single = int(loss_score[tr].mean(0).argmin())
    rows["always " + MODELS[best_single]] = evaluate(np.full(n, best_single), loss_score, label_score, te)
    for m in range(3):
        if m != best_single:
            rows["always " + MODELS[m]] = evaluate(np.full(n, m), loss_score, label_score, te)
    rows[oracle_name] = evaluate(label_train, loss_score, label_score, te)

    # Per-task prior: most frequent train label of the instruction in train (text only, no model).
    task_key = np.array([hash(v.tobytes()) for v in d["text_mean"]])
    prior = {}
    for t in np.unique(task_key[tr]):
        prior[t] = int(np.bincount(label_train[tr & (task_key == t)], minlength=3).argmax())
    rows["per-task majority"] = evaluate(
        np.array([prior.get(t, best_single) for t in task_key]), loss_score, label_score, te)

    # All feature ablations use hard CE on the argmin action-error label (no soft targets).
    configs = {
        "router hard CE (all feats)": dict(groups=("r0_pool", "text_mean", "proprio"), soft=False, target=label_train),
        "router hard, R0+proprio": dict(groups=("r0_pool", "proprio"), soft=False, target=label_train),
        "router hard, text only": dict(groups=("text_mean",), soft=False, target=label_train),
        "router hard, text+proprio": dict(groups=("text_mean", "proprio"), soft=False, target=label_train),
        "router hard, R0 only": dict(groups=("r0_pool",), soft=False, target=label_train),
        "router hard, proprio only": dict(groups=("proprio",), soft=False, target=label_train),
    }

    saved, picks = {}, {}
    for name, cfg in configs.items():
        print(name)
        model, logits, vloss = train_router(
            feats, cfg["target"], class_w, masks, groups=cfg["groups"], soft=cfg["soft"],
            epochs=args.epochs, lr=args.lr, wd=args.wd, batch=args.batch, seed=args.seed,
            device=args.device, kl=args.kl,
        )
        rows[name] = evaluate(logits.argmax(1), loss_score, label_score, te)
        rows[name]["val_loss"] = vloss
        saved[name] = model
        picks[name] = {"pick": logits.argmax(1), "prob": torch.softmax(torch.from_numpy(logits), -1).numpy()}

    for name in configs:
        print_confusion(name, picks[name]["pick"], label_score, te)

    base = rows["always " + MODELS[best_single]]["loss"]
    score_note = "held-out episodes" + ("" if args.label_seeds == "all" else ", held-out draws")
    print(f"\n{('test set (' + score_note + ')'):46s} {'loss':>10s} {'vs best':>8s} {'acc':>6s}  picks R/D/F")
    for name, r in rows.items():
        print(f"{name:46s} {r['loss']:10.6f} {100 * (1 - r['loss'] / base):+7.1f}% {r['acc']:6.2f}  {r['picks']}")

    out_dir = Path(args.labels).parent
    suffix = "" if args.kl <= 0 else f"_kl{args.kl:g}"
    if args.label_seeds != "all":
        suffix = f"{suffix}_halfa" if suffix else "_halfa"
    tag = args.tag.strip("_")
    if tag:
        suffix = f"{suffix}_{tag}" if suffix else f"_{tag}"

    best_name = min((k for k in rows if k.startswith("router")), key=lambda k: rows[k]["loss"])
    deploy_name = "router hard CE (all feats)" if "router hard CE (all feats)" in saved else best_name
    feat_norm = {g: (d[g][tr].mean(0), d[g][tr].std(0) + 1e-6) for g in ("r0_pool", "text_mean", "proprio")}
    torch.save({
        "state_dict": saved[deploy_name].state_dict(),
        "config": deploy_name,
        "groups": list(configs[deploy_name]["groups"]),
        "dims": {g: feats[g].shape[1] for g in configs[deploy_name]["groups"]},
        "feature_norm": feat_norm,
        "models": list(MODELS),
        "label_seeds": args.label_seeds,
        "best_by_loss": best_name,
        "loss": "hard_ce",
        "kl": args.kl,
    }, out_dir / f"router{suffix}.pt")
    torch.save({
        "routers": {name: {"state_dict": saved[name].state_dict(), "groups": list(configs[name]["groups"]),
                           "dims": {g: feats[g].shape[1] for g in configs[name]["groups"]}} for name in configs},
        "feature_norm": feat_norm,
        "models": list(MODELS),
        "kl": args.kl,
        "label_seeds": args.label_seeds,
        "loss": "hard_ce",
    }, out_dir / f"routers{suffix}.pt")
    split = np.where(tr, "train", np.where(va, "val", "test"))
    torch.save({"keys": d["keys"], "split": split, "label": label_score, "models": list(MODELS),
                "routers": picks, "label_seeds": args.label_seeds, "loss": "hard_ce"},
               out_dir / f"router_predictions{suffix}.pt")
    with open(out_dir / f"router_results{suffix}.json", "w") as fh:
        json.dump({"labels": args.labels, "kl": args.kl, "label_seeds": args.label_seeds,
                   "loss": "hard_ce", "rows": rows, "best": best_name, "deploy": deploy_name}, fh, indent=2)
    print(f"\nlowest-loss router: {best_name}")
    print(f"deploy router: {deploy_name} -> {out_dir / f'router{suffix}.pt'}")
    print(f"all trained -> {out_dir / f'routers{suffix}.pt'}")


if __name__ == "__main__":
    main()
