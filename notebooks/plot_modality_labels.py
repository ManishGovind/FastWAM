"""Plots and a per-window CSV for the full-LIBERO action-error modality labels.

Run after `modality_action_labels.py --analyze` (both `--gripper-weight 0` and the default):
    .venv/bin/python notebooks/plot_modality_labels.py
    .venv/bin/python notebooks/plot_modality_labels.py --task "moka pots"   # every episode of one task
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from modality_action_labels import OUT_DIR, _load, _tag  # noqa: E402
from modality_labels import MODELS, SUITES  # noqa: E402

FRAME_STRIDE = 4
TAG = _tag(0, 8, FRAME_STRIDE, (0, 1, 2, 3))
COLORS = ("#d62728", "#1f77b4", "#2ca02c")
FIG_DIR = OUT_DIR / "figures"


def _load_labels(path: Path) -> dict:
    return torch.load(path, weights_only=False)


def _shares(label: np.ndarray) -> np.ndarray:
    return np.bincount(label, minlength=3) / max(len(label), 1) * 100


def _stacked(ax, names, shares, horizontal=False):
    left = np.zeros(len(names))
    for m in range(3):
        vals = np.array([s[m] for s in shares])
        if horizontal:
            ax.barh(names, vals, left=left, color=COLORS[m], label=MODELS[m])
        else:
            ax.bar(names, vals, bottom=left, color=COLORS[m], label=MODELS[m])
        left += vals


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    grip_path = max(OUT_DIR.glob(f"labels_{TAG}_grip*.pt"), key=lambda p: p.stat().st_mtime)
    variants = {"delta-EEF only": _load_labels(OUT_DIR / f"labels_{TAG}.pt"),
                f"delta-EEF + gripper (λ={grip_path.stem.split('grip')[-1]})": _load_labels(grip_path)}
    d = variants[list(variants)[1]]
    lab, keys = d["label"], d["keys"]

    recs, _ = _load("RGB", TAG)
    tasks = np.array([(recs[k]["prompt"] or "").split("instruction:")[-1].strip().rstrip(".") for k in keys])

    # 1. Overall, per suite, and along the episode, for both label variants.
    fig, axes = plt.subplots(2, 3, figsize=(17, 9), gridspec_kw={"width_ratios": [1, 2, 3]})
    for row, (vname, v) in enumerate(variants.items()):
        vl = v["label"]
        _stacked(axes[row, 0], ["all LIBERO"], [_shares(vl)])
        for m, s in enumerate(_shares(vl)):
            axes[row, 0].text(0, _shares(vl)[:m].sum() + s / 2, f"{MODELS[m]}\n{s:.0f}%", ha="center",
                              va="center", color="white", fontsize=10)
        names = [n for n, _ in SUITES]
        _stacked(axes[row, 1], [n.replace("libero_", "") for n in names],
                 [_shares(vl[v["suite"] == n]) for n in names])
        bins = np.minimum((v["progress"] * 10).astype(int), 9)
        _stacked(axes[row, 2], [f"{10 * b}-{10 * b + 10}%" for b in range(10)],
                 [_shares(vl[bins == b]) for b in range(10)])
        axes[row, 0].set_ylabel(f"{vname}\n% of windows where model is best")
        axes[row, 1].set_title("by suite")
        axes[row, 2].set_title("by position in episode")
        axes[row, 2].tick_params(axis="x", rotation=45)
    axes[0, 2].legend(loc="upper right", bbox_to_anchor=(1.18, 1))
    fig.suptitle(f"Best-modality labels across LIBERO ({len(lab)} windows, {len(set(d['episode']))} episodes)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "distribution.png", dpi=110)
    plt.close(fig)

    # 2. Per task (gripper variant), sorted by Depth share.
    by_task = defaultdict(list)
    for i, t in enumerate(tasks):
        by_task[(d["suite"][i].replace("libero_", ""), t)].append(i)
    order = sorted(by_task, key=lambda k: _shares(lab[by_task[k]])[1])
    fig, ax = plt.subplots(figsize=(13, 13))
    names = [f"[{s}] {t[:60]} (n={len(by_task[(s, t)])})" for s, t in order]
    _stacked(ax, names, [_shares(lab[by_task[k]]) for k in order], horizontal=True)
    ax.axvline(50, color="k", lw=0.5, ls="--")
    ax.set_xlabel("% of windows where model is best (delta-EEF + gripper)")
    ax.set_xlim(0, 100)
    ax.tick_params(axis="y", labelsize=8)
    ax.legend(loc="lower right")
    ax.set_title("Label distribution per task")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "per_task.png", dpi=110)
    plt.close(fig)

    # 3. How decisive the labels are: margin of best over second best, and seed-half agreement.
    err = d["weighted_action_loss"]
    srt = np.sort(err, 1)
    margin = (srt[:, 1] - srt[:, 0]) / srt[:, 1].clip(min=1e-8) * 100
    agree = d["label_half_a"] == np.argmin(d["loss_half_b"], 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    for m in range(3):
        axes[0].hist(margin[lab == m], bins=np.linspace(0, 100, 51), alpha=0.6, color=COLORS[m],
                     label=f"{MODELS[m]} best (median {np.median(margin[lab == m]):.0f}%)")
    axes[0].set_xlabel("how much lower the best model's error is than the runner-up (%)")
    axes[0].set_ylabel("windows")
    axes[0].legend()
    qs = np.quantile(margin, np.linspace(0, 1, 11))
    mb = np.clip(np.searchsorted(qs, margin, side="right") - 1, 0, 9)
    axes[1].plot(range(10), [agree[mb == b].mean() for b in range(10)], "o-")
    axes[1].set_xticks(range(10), [f"{qs[b]:.0f}-{qs[b + 1]:.0f}" for b in range(10)], rotation=45)
    axes[1].set_xlabel("margin decile (%)")
    axes[1].set_ylabel("label agrees between seeds {0,2} and {1,3}")
    axes[1].set_ylim(0, 1)
    fig.suptitle("Label confidence")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "margin.png", dpi=110)
    plt.close(fig)

    # 4. Label timelines for example episodes (6 per suite, evenly spaced).
    episodes = d["episode"]
    fig, axes = plt.subplots(len(SUITES), 1, figsize=(15, 11))
    cmap = matplotlib.colors.ListedColormap(COLORS)
    for ax, (suite, _) in zip(axes, SUITES):
        eps = np.unique(episodes[d["suite"] == suite])
        eps = eps[np.linspace(0, len(eps) - 1, 6).astype(int)]
        for row, ep in enumerate(eps):
            idx = np.where(episodes == ep)[0]
            idx = idx[np.argsort(d["progress"][idx])]
            p = d["progress"][idx]
            edges = np.append(p, 1.0)
            for j in range(len(idx)):
                ax.barh(row, edges[j + 1] - edges[j], left=edges[j], color=COLORS[lab[idx[j]]], height=0.8)
        ax.set_yticks(range(len(eps)), [f"ep {e}: {tasks[np.where(episodes == e)[0][0]][:40]}" for e in eps],
                      fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_title(suite, fontsize=10)
    axes[-1].set_xlabel("episode progress")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in COLORS]
    fig.legend(handles, MODELS, loc="upper right")
    fig.suptitle("Best modality along example episodes (delta-EEF + gripper)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "episode_timelines.png", dpi=110)
    plt.close(fig)

    # 5. Per-window CSV.
    with open(OUT_DIR / f"labels_{TAG}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        v0 = variants["delta-EEF only"]
        w.writerow(["episode", "window", "suite", "task", "progress", "label_delta_only", "label_with_gripper",
                    "margin_pct", "seed_halves_agree"]
                   + [f"delta_mse_{m}" for m in MODELS] + [f"gripper_mismatch_{m}" for m in MODELS])
        for i, k in enumerate(keys):
            w.writerow([k[0], k[1], d["suite"][i], tasks[i], f"{d['progress'][i]:.3f}", MODELS[v0["label"][i]],
                        MODELS[lab[i]], f"{margin[i]:.1f}", int(agree[i])]
                       + [f"{x:.6f}" for x in d["delta_err"][i]] + [f"{x:.4f}" for x in d["gripper_mismatch"][i]])
    print(f"wrote figures to {FIG_DIR} and {OUT_DIR / f'labels_{TAG}.csv'}")
    print(f"median margin {np.median(margin):.1f}%; windows with margin < 10%: {(margin < 10).mean() * 100:.0f}%")
    print(f"labels that change when gripper is added: {(v0['label'] != lab).mean() * 100:.0f}%")


def task_view(query: str) -> None:
    """Label timeline over frames for every episode of the task(s) whose instruction contains `query`."""
    rows = list(csv.DictReader(open(OUT_DIR / f"labels_{TAG}.csv")))
    task_names = sorted({r["task"] for r in rows if query.lower() in r["task"].lower()})
    if not task_names:
        raise SystemExit(f"no task matches {query!r}")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for task in task_names:
        sel = sorted((r for r in rows if r["task"] == task), key=lambda r: (int(r["episode"]), int(r["window"])))
        slug = re.sub(r"[^a-z0-9]+", "_", task.lower()).strip("_")[:60]
        with open(FIG_DIR / f"task_{slug}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["frame"] + list(sel[0]))
            w.writeheader()
            for r in sel:
                w.writerow({"frame": int(r["window"]) * FRAME_STRIDE, **r})

        eps = sorted({int(r["episode"]) for r in sel})
        by_ep = defaultdict(list)
        for r in sel:
            by_ep[int(r["episode"])].append(r)
        max_frame = max(int(r["window"]) for r in sel) * FRAME_STRIDE + FRAME_STRIDE
        fig, axes = plt.subplots(3, 1, figsize=(15, 4 + 0.22 * len(eps)),
                                 gridspec_kw={"height_ratios": [len(eps), len(eps), 6]}, sharex=True)
        for ax, col, title in ((axes[0], "label_delta_only", "delta-EEF only"),
                               (axes[1], "label_with_gripper", "delta-EEF + gripper")):
            grid = np.full((len(eps), max_frame // FRAME_STRIDE), np.nan)
            for i, ep in enumerate(eps):
                for r in by_ep[ep]:
                    grid[i, int(r["window"])] = MODELS.index(r[col])
            ax.imshow(grid, aspect="auto", interpolation="nearest", cmap=matplotlib.colors.ListedColormap(COLORS),
                      vmin=-0.5, vmax=2.5, extent=(0, max_frame, len(eps) - 0.5, -0.5))
            ax.set_yticks(range(len(eps)), [f"ep {e}" for e in eps], fontsize=6)
            ax.set_title(f"{title}: best modality for the 32-action chunk starting at each frame", fontsize=10)
        n_win = grid.shape[1]
        share = np.array([[np.mean(grid[:, j][~np.isnan(grid[:, j])] == m) * 100 if (~np.isnan(grid[:, j])).any()
                           else np.nan for j in range(n_win)] for m in range(3)])
        x = np.arange(n_win) * FRAME_STRIDE
        axes[2].stackplot(x, np.nan_to_num(share), colors=COLORS, labels=MODELS, step="post")
        axes[2].set_ylabel("% of episodes\n(with gripper)")
        axes[2].set_xlabel("frame in episode")
        axes[2].set_ylim(0, 100)
        axes[2].legend(loc="upper right", fontsize=8)
        counts = {m: sum(r["label_with_gripper"] == m for r in sel) for m in MODELS}
        fig.suptitle(f"{task}  ({len(eps)} episodes, {len(sel)} windows; with gripper: "
                     + ", ".join(f"{m} {100 * c / len(sel):.0f}%" for m, c in counts.items()) + ")", fontsize=11)
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"task_{slug}.png", dpi=110)
        plt.close(fig)
        print(f"{task}: {len(eps)} episodes -> {FIG_DIR / f'task_{slug}.png'} and .csv")


def gt_vs_router(pred_path: Path, router_name: str = "router soft CE (all feats)") -> None:
    """Side-by-side GT vs predicted mix, plus confusion, for a saved predictions dump."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pred = torch.load(pred_path, map_location="cpu", weights_only=False)
    lab = pred["label"]
    split = np.asarray(pred["split"])
    r = pred["routers"][router_name]
    pick, prob = r["pick"], r["prob"]
    models = list(pred.get("models", MODELS))
    stem = pred_path.stem.replace("router_predictions", "gt_vs_router")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    groups = [("all", np.ones(len(lab), dtype=bool)),
              ("train", split == "train"), ("val", split == "val"), ("test", split == "test")]
    names = [g[0] for g in groups]
    _stacked(axes[0], names, [_shares(lab[m]) for _, m in groups])
    _stacked(axes[1], names, [_shares(pick[m]) for _, m in groups])
    axes[0].set_title("GT (action-error argmin)")
    axes[1].set_title(f"predicted ({router_name})")
    axes[0].set_ylabel("% of windows")
    axes[1].legend(loc="upper right")
    for ax, src in ((axes[0], lab), (axes[1], pick)):
        for i, (_, m) in enumerate(groups):
            sh = _shares(src[m])
            ax.text(i, 102, "  ".join(f"{models[j][0]}{s:.0f}" for j, s in enumerate(sh)),
                    ha="center", va="bottom", fontsize=8)
        ax.set_ylim(0, 118)
    cm = np.zeros((3, 3), dtype=int)
    for g, p in zip(lab, pick):
        cm[int(g), int(p)] += 1
    im = axes[2].imshow(cm, cmap="Blues")
    axes[2].set_xticks(range(3), models)
    axes[2].set_yticks(range(3), models)
    axes[2].set_xlabel("predicted")
    axes[2].set_ylabel("GT")
    axes[2].set_title(f"confusion  acc={(pick == lab).mean() * 100:.1f}%")
    for i in range(3):
        for j in range(3):
            axes[2].text(j, i, str(cm[i, j]), ha="center", va="center",
                         color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=9)
    fig.colorbar(im, ax=axes[2], fraction=0.046)
    fig.suptitle(f"GT vs router  ({pred_path.name}, n={len(lab)})")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{stem}_distribution.png", dpi=120)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    _stacked(axes[0], names, [_shares(lab[m]) for _, m in groups])
    _stacked(axes[1], names, [_shares(pick[m]) for _, m in groups])
    axes[0].set_title("GT")
    axes[1].set_title("predicted")
    axes[0].set_ylabel("% of windows")
    axes[1].legend()
    fig.suptitle(f"{router_name}: mean p = {prob.mean(0)}")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{stem}_mix.png", dpi=120)
    plt.close(fig)

    csv_path = OUT_DIR / f"{stem}.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["episode", "window", "split", "gt_label", "router_pick", "agree"]
                   + [f"p_{m}" for m in models])
        for i, k in enumerate(pred["keys"]):
            w.writerow([k[0], k[1], split[i], models[int(lab[i])], models[int(pick[i])],
                        int(pick[i] == lab[i])] + [f"{x:.4f}" for x in prob[i]])
    print(f"wrote {FIG_DIR / f'{stem}_distribution.png'} and {csv_path}")
    print("GT    ", {m: float((lab == i).mean()) for i, m in enumerate(models)})
    print("pred  ", {m: float((pick == i).mean()) for i, m in enumerate(models)})
    for name, mask in groups:
        print(f"  {name:5} acc={(pick[mask] == lab[mask]).mean() * 100:.1f}%  "
              f"GT={_shares(lab[mask]).round(1)}  pred={_shares(pick[mask]).round(1)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--task", help="substring of the task instruction; plots every episode of matching tasks")
    p.add_argument("--gt-vs-router", type=Path, help="router_predictions*.pt to plot vs GT labels")
    p.add_argument("--router-name", default="router soft CE (all feats)")
    a = p.parse_args()
    if a.gt_vs_router:
        gt_vs_router(a.gt_vs_router, a.router_name)
    elif a.task:
        task_view(a.task)
    else:
        main()
