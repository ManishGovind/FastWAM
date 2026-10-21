"""Offline oracle-label check for the inline router (``modality_router_new``).

Question: can the inline router architecture (full R0 + full MoT context, same class
as ``FastWAMJointMultimodalSelectNew.modality_router``) predict the per-window
action-error oracle better than "always Depth" / "per-task majority"?

Differences from ``train_modality_router.py`` (left untouched):

* router class = ``fastwam.models.wan22.modality_router_new.ModalityRouter``;
* features come from the existing ``*_router_feats_nopool.pt`` cache (492 GB),
  compacted once into ``*_router_feats_compact.pt`` (~10 GB): the 128 text tokens are
  identical for every window of a task, so they are stored once per task and only
  the proprio token (last context token) is kept per window;
* everything lives on the GPU in fp16; context is rebuilt per batch;
* ``--norm none`` by default (the inline router has only LayerNorm, no standardize);
* ``--loss ce`` (unweighted) by default; ``ce_weighted`` reproduces the old
  class-balanced CE; ``expected`` minimizes expected action-error regret directly;
* ``--ablate r0|context`` zeroes one input to see what each contributes;
* ``--features pooled`` swaps the full inputs for a small summary, with the same router
  class: R0 ``[C,h,w]`` average-pooled to ``--r0-pool`` x ``--r0-pool`` and the context
  as 2 tokens, ``[mean of the task's real (non-zero) text tokens, proprio token]``;
* early stopping on validation mean action loss of argmax picks (the reported metric).

Usage::

    python notebooks/train_modality_router_new.py \\
        --labels notebooks/outputs/modality_action_labels/labels_epall_stride1_s42-43-44_grip0.2148.pt \\
        --loss ce --tag ce
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_modality_router import (  # noqa: E402
    MODELS,
    _as_np,
    evaluate,
    print_confusion,
    split_by_episode,
)

from fastwam.models.wan22.modality_router_new import ModalityRouter  # noqa: E402

TEXT_LEN = 128  # context = 128 cached T5 tokens + 1 proprio token


# ---------------------------------------------------------------------------
# Feature cache: nopool (per-window full context) -> compact (per-task text)
# ---------------------------------------------------------------------------


def _nopool_path(labels_path: Path) -> Path:
    return labels_path.with_name(labels_path.stem + "_router_feats_nopool.pt")


def _compact_path(labels_path: Path) -> Path:
    return labels_path.with_name(labels_path.stem + "_router_feats_compact.pt")


@torch.no_grad()
def compact_features(nopool_path: Path, out_path: Path, chunk: int = 4096) -> dict:
    """Split the nopool cache into per-task text + per-window r0 / proprio token."""
    src = torch.load(nopool_path, map_location="cpu", weights_only=False, mmap=True)
    r0_src, ctx_src, mask_src = src["r0"], src["context"], src["context_mask"]
    n, length, dim = ctx_src.shape
    if length != TEXT_LEN + 1:
        raise ValueError(f"expected context_len {TEXT_LEN + 1}, got {length}")

    # Task id from the (text-only) mean; identical for all windows of a prompt.
    text_mean = src["text_mean"]
    keys: dict[bytes, int] = {}
    task_id = torch.empty(n, dtype=torch.long)
    first_window: list[int] = []
    for start in range(0, n, chunk):
        block = text_mean[start: start + chunk].numpy()
        for j, row in enumerate(block):
            k = row.tobytes()
            if k not in keys:
                keys[k] = len(keys)
                first_window.append(start + j)
            task_id[start + j] = keys[k]
    num_tasks = len(keys)
    print(f"compact: {n} windows, {num_tasks} tasks", flush=True)

    text_table = torch.stack([ctx_src[i, :TEXT_LEN] for i in first_window]).to(torch.float16)
    text_mask = torch.stack([mask_src[i, :TEXT_LEN] for i in first_window]).bool()

    # Spot-check: text tokens are identical across windows of a task.
    rng = np.random.default_rng(0)
    for t in range(num_tasks):
        members = (task_id == t).nonzero(as_tuple=True)[0].numpy()
        for i in rng.choice(members, size=min(3, len(members)), replace=False):
            if not torch.equal(ctx_src[int(i), :TEXT_LEN], ctx_src[first_window[t], :TEXT_LEN]):
                raise ValueError(f"text tokens differ within task {t} (window {int(i)})")

    r0 = torch.empty((n, r0_src.shape[1]), dtype=torch.float16)
    proprio_tok = torch.empty((n, dim), dtype=torch.float16)
    if not bool(mask_src[:, TEXT_LEN].bool().all()):
        raise ValueError("proprio token (last context position) is masked for some windows")
    t0 = time.time()
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        r0[start:end] = r0_src[start:end].to(torch.float16)
        proprio_tok[start:end] = ctx_src[start:end, TEXT_LEN].to(torch.float16)
        if (start // chunk) % 10 == 0:
            print(f"  compact {end}/{n}  {time.time() - t0:.0f}s", flush=True)

    out = {
        "frame_idx": src["frame_idx"].clone(),
        "r0": r0,
        "proprio_tok": proprio_tok,
        "task_id": task_id,
        "text_table": text_table,
        "text_mask": text_mask,
    }
    torch.save(out, out_path)
    print(f"wrote {out_path}", flush=True)
    return out


def load_compact(labels_path: Path, frame_idxs: np.ndarray) -> dict:
    path = _compact_path(labels_path)
    if path.is_file():
        feats = torch.load(path, map_location="cpu", weights_only=False)
        print(f"loaded {path}", flush=True)
    else:
        src = _nopool_path(labels_path)
        if not src.is_file():
            raise FileNotFoundError(
                f"{src} missing; run train_modality_router.py once to extract features"
            )
        feats = compact_features(src, path)
    if not np.array_equal(_as_np(feats["frame_idx"]).astype(np.int64), frame_idxs):
        raise ValueError("feature frame_idx does not match labels frame_idx")
    return feats


# ---------------------------------------------------------------------------
# Router batches
# ---------------------------------------------------------------------------


class Features:
    """GPU-resident compact features; builds router inputs per batch.

    ``features="full"``: r0 ``[N, C*h*w]``, context = 128 text tokens + proprio token.
    ``features="pooled"``: r0 pooled to ``[N, C*p*p]``, context = [text mean, proprio].
    """

    def __init__(
        self,
        feats: dict,
        device: str,
        norm: str,
        train_idx: np.ndarray,
        ablate: str,
        *,
        features: str = "full",
        r0_pool: int = 2,
        latent_shape: tuple[int, int, int] = (48, 14, 28),
    ):
        self.r0 = feats["r0"].to(device)
        self.proprio_tok = feats["proprio_tok"].to(device)
        self.task_id = feats["task_id"].to(device)
        self.text_table = feats["text_table"].to(device)
        self.text_mask = feats["text_mask"].to(device)
        self.ablate = ablate
        self.features = features
        if features == "pooled":
            self._pool(r0_pool, latent_shape)
        self.text_len = int(self.text_table.shape[1])
        self.norm = None
        if norm == "standardize":
            self.norm = self._train_stats(torch.as_tensor(train_idx, device=device))

    @torch.no_grad()
    def _pool(self, r0_pool: int, latent_shape: tuple[int, int, int], chunk: int = 8192) -> None:
        c, h, w = latent_shape
        if c * h * w != self.r0.shape[1]:
            raise ValueError(f"--latent-shape {latent_shape} does not match r0 dim {self.r0.shape[1]}")
        pooled = [
            F.adaptive_avg_pool2d(self.r0[i: i + chunk].float().view(-1, c, h, w), r0_pool)
            .flatten(1).to(torch.float16)
            for i in range(0, self.r0.shape[0], chunk)
        ]
        self.r0 = torch.cat(pooled, dim=0)
        # Mean over real text tokens only: the cached mask is all-True, padding is zeros.
        text = self.text_table.float()
        real = (text.abs().amax(-1) > 0) & self.text_mask
        mean = (text * real[..., None]).sum(1) / real.sum(1, keepdim=True).clamp_min(1)
        self.text_table = mean[:, None].to(torch.float16)  # [T, 1, D]
        self.text_mask = torch.ones(mean.shape[0], 1, dtype=torch.bool, device=mean.device)

    @torch.no_grad()
    def _train_stats(self, tr: torch.Tensor, chunk: int = 8192) -> dict[str, torch.Tensor]:
        """Per-dim mean/std on the train split (r0 over windows, context over tokens)."""

        def moments(rows_fn, count: int):
            s = s2 = None
            for i in range(0, count, chunk):
                x = rows_fn(i, min(i + chunk, count)).float()
                s = x.sum(0) if s is None else s + x.sum(0)
                s2 = (x * x).sum(0) if s2 is None else s2 + (x * x).sum(0)
            return s, s2

        r0_s, r0_s2 = moments(lambda a, b: self.r0[tr[a:b]], len(tr))
        r0_mu = r0_s / len(tr)
        r0_sd = (r0_s2 / len(tr) - r0_mu ** 2).clamp_min(0).sqrt() + 1e-6

        # Context tokens: text tokens weighted by #train windows per task + proprio tokens.
        counts = torch.bincount(self.task_id[tr], minlength=self.text_table.shape[0]).float()
        text = self.text_table.float()  # [T, L, D]
        ctx_s = (text.sum(1) * counts[:, None]).sum(0)
        ctx_s2 = ((text * text).sum(1) * counts[:, None]).sum(0)
        p_s, p_s2 = moments(lambda a, b: self.proprio_tok[tr[a:b]], len(tr))
        n_tok = float(counts.sum().item()) * self.text_len + len(tr)
        ctx_mu = (ctx_s + p_s) / n_tok
        ctx_sd = ((ctx_s2 + p_s2) / n_tok - ctx_mu ** 2).clamp_min(0).sqrt() + 1e-6
        return {"r0": (r0_mu, r0_sd), "context": (ctx_mu, ctx_sd)}

    def batch(self, idx: torch.Tensor) -> dict[str, torch.Tensor]:
        r0 = self.r0[idx].float()
        tid = self.task_id[idx]
        context = torch.cat(
            [self.text_table[tid].float(), self.proprio_tok[idx].float()[:, None]], dim=1
        )
        mask = torch.cat(
            [self.text_mask[tid], torch.ones_like(tid, dtype=torch.bool)[:, None]], dim=1
        )
        if self.norm is not None:
            r0 = (r0 - self.norm["r0"][0]) / self.norm["r0"][1]
            context = (context - self.norm["context"][0]) / self.norm["context"][1]
        if self.ablate == "r0":
            r0 = torch.zeros_like(r0)
        elif self.ablate == "context":
            context = torch.zeros_like(context)
        return {"r0": r0, "context": context, "context_mask": mask}


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------


def train_router(
    x: Features,
    label: np.ndarray,
    loss_table: np.ndarray,
    masks,
    *,
    loss: str,
    epochs: int,
    patience: int,
    lr: float,
    wd: float,
    batch: int,
    seed: int,
    device: str,
):
    torch.manual_seed(seed)
    tr, va, _ = masks
    model = ModalityRouter(
        r0_dim=int(x.r0.shape[1]),
        context_dim=int(x.text_table.shape[2]),
        context_len=x.text_len + 1,
        num_classes=len(MODELS),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    y = torch.as_tensor(label, dtype=torch.long, device=device)
    lt = torch.as_tensor(loss_table, dtype=torch.float32, device=device)
    # Regret vs the best modality, scaled to unit mean on train (same argmin as loss).
    regret = lt - lt.min(dim=1, keepdim=True).values
    regret = regret / regret[torch.as_tensor(tr, device=device)].mean().clamp_min(1e-12)
    counts = np.bincount(label[tr], minlength=len(MODELS))
    class_w = torch.as_tensor(
        counts.sum() / np.maximum(counts, 1) / len(MODELS), dtype=torch.float32, device=device
    )

    def loss_fn(logits: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if loss == "ce":
            return F.cross_entropy(logits, y[b])
        if loss == "ce_weighted":
            return F.cross_entropy(logits, y[b], weight=class_w)
        return (logits.softmax(-1) * regret[b]).sum(-1).mean()  # expected

    @torch.no_grad()
    def predict(idx: torch.Tensor) -> torch.Tensor:
        model.eval()
        out = [model(x.batch(idx[i: i + batch])) for i in range(0, len(idx), batch)]
        return torch.cat(out, dim=0)

    tr_idx = torch.as_tensor(np.where(tr)[0], device=device)
    va_idx = torch.as_tensor(np.where(va)[0], device=device)
    best, best_state, best_ep, bad = float("inf"), None, 0, 0
    ep = 0
    for ep in range(epochs):
        model.train()
        t0, tot, nb = time.time(), 0.0, 0
        perm = tr_idx[torch.randperm(len(tr_idx), device=device)]
        for i in range(0, len(perm), batch):
            b = perm[i: i + batch]
            loss_b = loss_fn(model(x.batch(b)), b)
            opt.zero_grad(set_to_none=True)
            loss_b.backward()
            opt.step()
            tot, nb = tot + float(loss_b.detach()), nb + 1
        pick = predict(va_idx).argmax(-1)
        val_action = float(lt[va_idx, pick].mean())
        val_acc = float((pick == y[va_idx]).float().mean())
        print(
            f"  ep {ep + 1:3d}  train {loss} {tot / max(nb, 1):.4f}  "
            f"val action-loss {val_action:.6f} acc {val_acc:.3f}  {time.time() - t0:.0f}s",
            flush=True,
        )
        if val_action < best - 1e-7:
            best, best_ep, bad = val_action, ep, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    print(f"  stopped after {ep + 1} epochs; best val action-loss {best:.6f} at epoch {best_ep + 1}")
    model.load_state_dict(best_state)
    all_idx = torch.arange(len(y), device=device)
    logits = predict(all_idx).float().cpu().numpy()
    return model, logits, best


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labels", type=str, required=True)
    p.add_argument("--loss", choices=("ce", "ce_weighted", "expected"), default="ce")
    p.add_argument("--norm", choices=("none", "standardize"), default="none")
    p.add_argument("--ablate", choices=("none", "r0", "context"), default="none")
    p.add_argument("--features", choices=("full", "pooled"), default="full")
    p.add_argument("--r0-pool", type=int, default=2, help="pooled: R0 spatial grid (p x p)")
    p.add_argument("--latent-shape", type=int, nargs=3, default=(48, 14, 28),
                   help="R0 latent [C h w] (224x448 video, VAE /16)")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-2)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", type=str, default=None, help="output suffix (default: loss_norm_ablate)")
    p.add_argument("--compact-only", action="store_true", help="build the compact cache and exit")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()

    labels_path = Path(a.labels)
    d = torch.load(labels_path, map_location="cpu", weights_only=False)
    label = _as_np(d["label"]).astype(np.int64)
    loss_table = _as_np(d["weighted_action_loss"]).astype(np.float32)  # [N, 3]
    frame_idxs = _as_np(d["frame_idx"]).astype(np.int64)
    n = len(label)

    feats = load_compact(labels_path, frame_idxs)
    if a.compact_only:
        return

    masks = split_by_episode(_as_np(d["episode"]), _as_np(d["suite"]), a.seed)
    tr, va, te = masks
    task_id = feats["task_id"].numpy()
    print(
        f"{n} windows, {int(task_id.max()) + 1} tasks | train {tr.sum()} / val {va.sum()} / "
        f"test {te.sum()} | features={a.features} loss={a.loss} norm={a.norm} ablate={a.ablate}",
        flush=True,
    )
    print(f"train label counts R/D/F = {np.bincount(label[tr], minlength=3).tolist()}")

    rows: dict[str, dict] = {}
    best_single = int(loss_table[tr].mean(0).argmin())
    for m in [best_single] + [m for m in range(3) if m != best_single]:
        rows["always " + MODELS[m]] = evaluate(np.full(n, m), loss_table, label, te)
    rows["oracle"] = evaluate(label, loss_table, label, te)
    prior = {
        t: int(np.bincount(label[tr & (task_id == t)], minlength=3).argmax())
        for t in np.unique(task_id[tr])
    }
    rows["per-task majority"] = evaluate(
        np.array([prior.get(t, best_single) for t in task_id]), loss_table, label, te
    )

    x = Features(
        feats, a.device, a.norm, np.where(tr)[0], a.ablate,
        features=a.features, r0_pool=a.r0_pool, latent_shape=tuple(a.latent_shape),
    )
    del feats
    print(f"router inputs: r0 {tuple(x.r0.shape[1:])}, context [{x.text_len + 1}, {x.text_table.shape[2]}]")
    name = f"router_new {a.features} {a.loss}/{a.norm}/ablate={a.ablate}"
    print(name, flush=True)
    model, logits, vbest = train_router(
        x, label, loss_table, masks,
        loss=a.loss, epochs=a.epochs, patience=a.patience, lr=a.lr, wd=a.wd,
        batch=a.batch, seed=a.seed, device=a.device,
    )
    pick = logits.argmax(1)
    rows[name] = evaluate(pick, loss_table, label, te)
    rows[name]["val_action_loss"] = vbest
    print_confusion(name, pick, label, te)

    base = rows["always " + MODELS[best_single]]["loss"]
    print(f"\n{'test set (held-out episodes)':46s} {'loss':>10s} {'vs best':>8s} {'acc':>6s}  picks R/D/F")
    for k, r in rows.items():
        print(f"{k:46s} {r['loss']:10.6f} {100 * (1 - r['loss'] / base):+7.1f}% {r['acc']:6.2f}  {r['picks']}")

    tag = a.tag or (
        f"{a.loss}_{a.norm}_{a.ablate}" if a.features == "full"
        else f"pooled{a.r0_pool}_{a.loss}_{a.norm}_{a.ablate}"
    )
    out_dir = labels_path.parent
    torch.save(
        {
            "state_dict": model.state_dict(),
            "r0_dim": int(x.r0.shape[1]),
            "context_dim": int(x.text_table.shape[2]),
            "context_len": x.text_len + 1,
            "features": a.features,
            "r0_pool": a.r0_pool if a.features == "pooled" else None,
            "latent_shape": list(a.latent_shape),
            "models": list(MODELS),
            "norm": None if x.norm is None else {k: tuple(t.cpu() for t in v) for k, v in x.norm.items()},
            "loss": a.loss,
            "ablate": a.ablate,
            "labels": str(labels_path),
        },
        out_dir / f"routers_new_{tag}.pt",
    )
    with open(out_dir / f"router_results_new_{tag}.json", "w") as fh:
        json.dump({"labels": str(labels_path), "args": vars(a), "rows": rows}, fh, indent=2)
    print(f"\nwrote {out_dir / f'routers_new_{tag}.pt'}")


if __name__ == "__main__":
    main()
