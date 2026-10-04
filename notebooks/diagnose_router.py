"""Is the modality router failing to learn, or is there nothing to learn?

Checks on the action-error labels:
  1. capacity: can the router fit the training set when nothing stops it (no dropout, no early stop)?
  2. ranking: do router probabilities order windows by how much better RGB/Flow is than Depth (AUC),
     even when argmax is always Depth?
  3. regression: predict the per-window error gap directly (continuous target, less noisy than argmin).
  4. noise ceiling: how well do the other seeds' errors predict the gap (best any router could do)?

    .venv/bin/python notebooks/diagnose_router.py --labels notebooks/outputs/modality_action_labels/labels_epall_stride4_s0-1-2-3_grip0.2269.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_modality_router import MODELS, ModalityRouter, split_by_episode  # noqa: E402


def auc(score: np.ndarray, pos: np.ndarray) -> float:
    """Probability a random positive scores above a random negative (ties count half)."""
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score))
    s = score[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    n_pos, n_neg = pos.sum(), (~pos).sum()
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / max(n_pos * n_neg, 1))


def r2(pred: np.ndarray, y: np.ndarray) -> float:
    return float(1 - ((pred - y) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def fit(x, y, groups, masks, *, kind: str, epochs: int, dropout: float, early_stop: bool, device: str):
    """kind: 'ce' (hard labels, 3-way) or 'reg' (regress 2 error gaps). Returns outputs + train curve info."""
    torch.manual_seed(0)
    tr, va, _ = masks
    out_dim = 3 if kind == "ce" else 2
    model = ModalityRouter({g: x[g].shape[1] for g in groups}, dropout=dropout)
    model.head[-1] = nn.Linear(model.head[-1].in_features, out_dim)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4 if not early_stop else 1e-2)
    xt = {g: torch.from_numpy(x[g]).float().to(device) for g in groups}
    yt = torch.from_numpy(y).to(device)
    yt = yt.long() if kind == "ce" else yt.float()
    tr_i = torch.from_numpy(np.where(tr)[0]).to(device)
    va_i = torch.from_numpy(np.where(va)[0]).to(device)

    def loss_fn(o, t):
        return F.cross_entropy(o, t) if kind == "ce" else F.mse_loss(o, t)

    best, best_state, best_ep, bad = float("inf"), None, 0, 0
    for ep in range(epochs):
        model.train()
        perm = tr_i[torch.randperm(len(tr_i), device=device)]
        for i in range(0, len(perm), 512):
            b = perm[i:i + 512]
            loss = loss_fn(model({g: xt[g][b] for g in groups}), yt[b])
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = float(loss_fn(model({g: xt[g][va_i] for g in groups}), yt[va_i]))
        if vl < best - 1e-6:
            best, best_ep, bad = vl, ep, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if early_stop and bad >= 15:
                break
    if early_stop:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out = model(xt).cpu().numpy()
    return out, best_ep, ep


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--labels", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    d = torch.load(a.labels, map_location="cpu", weights_only=False)
    masks = split_by_episode(d["episode"], d["suite"], 0)
    tr, va, te = masks
    loss_all = d["weighted_action_loss"]
    label = d["label"] if "label" in d else loss_all.argmin(1)

    x = {}
    for g in ("r0_pool", "text_mean", "proprio"):
        v = d[g].astype(np.float32)
        x[g] = (v - v[tr].mean(0)) / (v[tr].std(0) + 1e-6)
    x["progress"] = d["progress"].astype(np.float32)[:, None]
    all_feats = ("r0_pool", "text_mean", "proprio")

    prior = np.bincount(label[tr], minlength=3) / tr.sum()
    prior_ce = float(-np.log(prior[label]).mean())

    def ce(logits, sel):
        lp = torch.log_softmax(torch.from_numpy(logits[sel]), -1).numpy()
        return float(-lp[np.arange(sel.sum()), label[sel]].mean())

    print(f"test labels R/D/F = {np.bincount(label[te], minlength=3).tolist()}; "
          f"CE of predicting the train label frequencies = {prior_ce:.4f} (a useful router must go below this)\n")

    # 1. capacity: overfit on train, no dropout, no early stopping.
    logits, _, last = fit(x, label, all_feats, masks, kind="ce", epochs=150, dropout=0.0, early_stop=False,
                          device=a.device)
    print("1. capacity (hard labels, no dropout, no early stop, 150 epochs)")
    for nm, sel in (("train", tr), ("test", te)):
        print(f"   {nm:5s} acc {np.mean(logits[sel].argmax(1) == label[sel]):.3f}  CE {ce(logits, sel):.4f}")

    # 1b. the regular setup (dropout + early stop) as in train_modality_router.py.
    logits_es, best_ep, _ = fit(x, label, all_feats, masks, kind="ce", epochs=200, dropout=0.1, early_stop=True,
                                device=a.device)
    print(f"   with dropout + early stop: best val epoch {best_ep}; "
          f"train CE {ce(logits_es, tr):.4f}, test CE {ce(logits_es, te):.4f}, "
          f"test acc {np.mean(logits_es[te].argmax(1) == label[te]):.3f}")

    # 2. ranking: does P(model) order windows correctly even if argmax is always Depth?
    print("\n2. ranking on test (AUC, 0.5 = no signal): P(RGB) for 'RGB beats Depth', P(Flow) for 'Flow beats Depth'")
    prob = torch.softmax(torch.from_numpy(logits_es), -1).numpy()
    rgb_win = loss_all[:, 0] < loss_all[:, 1]
    flow_win = loss_all[:, 2] < loss_all[:, 1]
    for nm, pr in (("router (all feats)", prob),):
        print(f"   {nm:26s} RGB {auc(pr[te, 0] - pr[te, 1], rgb_win[te]):.3f}   "
              f"Flow {auc(pr[te, 2] - pr[te, 1], flow_win[te]):.3f}")
    print(f"   {'episode progress alone':26s} RGB {auc(d['progress'][te], rgb_win[te]):.3f}   "
          f"Flow {auc(-np.abs(d['progress'][te] - 0.3), flow_win[te]):.3f}")

    # 3. regression on the relative error gap: (Depth - RGB) / Depth and (Depth - Flow) / Depth.
    gap = np.stack([(loss_all[:, 1] - loss_all[:, m]) / loss_all[:, 1] for m in (0, 2)], 1).astype(np.float32)
    gap_clip = np.clip(gap, -3, 1)
    mu, sd = gap_clip[tr].mean(0), gap_clip[tr].std(0)
    print("\n3. regress the relative error gap (Depth - X) / Depth, test R^2 (0 = predicts the mean only)")
    reg_out = {}
    for nm, groups in (("all feats", all_feats), ("all feats + progress", all_feats + ("progress",)),
                       ("R0 only", ("r0_pool",)), ("text only", ("text_mean",)), ("proprio only", ("proprio",))):
        out, _, _ = fit(x, (gap_clip - mu) / sd, groups, masks, kind="reg", epochs=200, dropout=0.1,
                        early_stop=True, device=a.device)
        out = out * sd + mu
        reg_out[nm] = out
        print(f"   {nm:22s} RGB gap R^2 {r2(out[te, 0], gap_clip[te, 0]):+.3f}   "
              f"Flow gap R^2 {r2(out[te, 1], gap_clip[te, 1]):+.3f}")

    # 4. turn the learned signal into a choice: switch away from Depth only when the predicted gain is large.
    print("\n4. switch from Depth to the model with the largest predicted gain when it exceeds a threshold "
          "(test episodes)")
    base = loss_all[te, 1].mean()
    out = reg_out["all feats"][te]
    best_alt = np.where(out[:, 0] >= out[:, 1], 0, 2)
    gain = out.max(1)
    for th in (0.0, 0.05, 0.1, 0.15, 0.2):
        pick = np.where(gain > th, best_alt, 1)
        err = loss_all[te][np.arange(te.sum()), pick].mean()
        print(f"   threshold {th:4.2f}: switches {np.mean(pick != 1) * 100:5.1f}% of windows, "
              f"error {err:.6f} ({100 * (1 - err / base):+.2f}% vs always Depth)")
    ideal = loss_all[te].min(1).mean()
    print(f"   perfect argmin of action error: {100 * (1 - ideal / base):+.1f}% vs always Depth")


if __name__ == "__main__":
    main()
