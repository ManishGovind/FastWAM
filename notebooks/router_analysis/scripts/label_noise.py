"""How noisy are the action-error oracle labels? Per-seed agreement + held-out-seed ceiling."""
import sys
from itertools import combinations

import numpy as np

sys.path.insert(0, "/data/mgovind/FastWAM/notebooks")
from modality_action_labels import DELTA, GRIPPER, MODELS, load_shard_records  # noqa: E402
from train_modality_router import split_by_episode  # noqa: E402

TAG, GRIP = "epall_stride1_s42-43-44", 0.2148

per_model = [load_shard_records(m, TAG)[0] for m in MODELS]
keys = sorted(set.intersection(*(set(r) for r in per_model)))
pred = np.stack([np.stack([pm[k]["pred"] for k in keys]) for pm in per_model]).astype(np.float32)  # [M,N,S,H,D]
gt = np.stack([per_model[0][k]["gt"] for k in keys])
pad = per_model[0][keys[0]]["action_is_pad"]
valid = (np.ones(gt.shape[:2], np.float32) if pad is None
         else (~np.stack([per_model[0][k]["action_is_pad"] for k in keys])).astype(np.float32))


def masked_mean(x):
    return (x * valid[None, :, None]).sum(-1) / valid.sum(-1).clip(min=1)[None, :, None]


delta = masked_mean(((pred[..., DELTA] - gt[None, :, None, :, DELTA]) ** 2).mean(-1))
grip = masked_mean((np.sign(pred[..., GRIPPER]) != np.sign(gt[None, :, None, :, GRIPPER])).astype(np.float32))
err = delta + GRIP * grip  # [M, N, S]
M, N, S = err.shape
label = err.mean(2).argmin(0)

episodes = np.array([k[0] for k in keys])
from modality_action_labels import suite_of  # noqa: E402
suites = np.array([suite_of(int(e)) for e in episodes])
_, _, te = split_by_episode(episodes, suites, 0)

for name, sel in (("all windows", np.ones(N, bool)), ("test split", te)):
    e = err[:, sel]
    lab_s = e.argmin(0)  # [N, S] per-seed label
    lab = e.mean(2).argmin(0)
    print(f"\n=== {name}: N={sel.sum()}, seeds={S} ===")
    print("per-seed label counts R/D/F:", [np.bincount(lab_s[:, s], minlength=3).tolist() for s in range(S)])
    print(f"all {S} seeds agree:          {np.mean((lab_s == lab_s[:, :1]).all(1)):.3f}")
    print("pairwise seed agreement:      "
          + ", ".join(f"{a}-{b} {np.mean(lab_s[:, a] == lab_s[:, b]):.3f}" for a, b in combinations(range(S), 2)))
    print("chance agreement (same marginals): "
          f"{sum((np.bincount(lab_s[:, 0], minlength=3) / len(lab_s)) * (np.bincount(lab_s[:, 1], minlength=3) / len(lab_s))):.3f}")

    # Held-out-seed ceiling: label from the other seeds, scored on the held-out seed.
    gains, accs = [], []
    for s in range(S):
        others = [o for o in range(S) if o != s]
        lab_o = e[:, :, others].mean(2).argmin(0)
        held = e[:, :, s]  # [M, n]
        base = held[1].mean()  # always Depth on the held-out seed
        orc = held[lab_o, np.arange(held.shape[1])].mean()
        self_orc = held.argmin(0)
        gains.append(1 - orc / base)
        accs.append(np.mean(lab_o == self_orc))
        print(f"held-out seed {s}: always-Depth {base:.6f}  in-sample oracle {held.min(0).mean():.6f} "
              f"({100 * (1 - held.min(0).mean() / base):+.1f}%)  cross-seed oracle {orc:.6f} ({100 * gains[-1]:+.1f}%)  "
              f"label agreement {accs[-1]:.3f}")
    print(f"=> realistic ceiling (cross-seed oracle) mean {100 * np.mean(gains):+.1f}% vs always Depth")

    # Margin between best and second-best modality (relative).
    srt = np.sort(e.mean(2), axis=0)
    rel = (srt[1] - srt[0]) / srt[1].clip(min=1e-12)
    print("relative margin best vs 2nd (quantiles 25/50/75):", np.round(np.quantile(rel, [0.25, 0.5, 0.75]), 3).tolist())

# --- Per-task routing: is the task-level best modality consistent across seeds? ---
import torch  # noqa: E402
c = torch.load("/data/mgovind/FastWAM/notebooks/outputs/modality_action_labels/"
               "labels_epall_stride1_s42-43-44_grip0.2148_router_feats_compact.pt",
               map_location="cpu", weights_only=False)
fi = np.array([int(per_model[0][k]["frame_idx"]) for k in keys])
assert np.array_equal(fi, c["frame_idx"].numpy())
task = c["task_id"].numpy()
T = task.max() + 1
tm = np.stack([[err[:, task == t, s].mean(1) for s in range(S)] for t in range(T)])  # [T, S, M]
best = tm.argmin(2)  # [T, S]
print(f"\n=== per-task: {T} tasks ===")
print("tasks whose best modality agrees across all seeds:", int((best == best[:, :1]).all(1).sum()), "/", T)
print("per-task best (3-seed mean) counts R/D/F:", np.bincount(tm.mean(1).argmin(1), minlength=3).tolist())
g = []
for s in range(S):
    others = [o for o in range(S) if o != s]
    pick = tm[:, others].mean(1).argmin(1)
    w = np.bincount(task, minlength=T)
    held = (tm[np.arange(T), s, pick] * w).sum() / w.sum()
    base = (tm[:, s, 1] * w).sum() / w.sum()
    g.append(1 - held / base)
    print(f"held-out seed {s}: per-task cross-seed pick vs always-Depth {100 * g[-1]:+.1f}%")
print(f"=> per-task routing realistic gain {100 * np.mean(g):+.1f}%")
for t in np.argsort(-(tm.mean(1)[:, 1] - tm.mean(1).min(1)))[:5]:
    print(f"  task {t}: err R/D/F per seed = {np.round(tm[t] * 1e3, 2).tolist()} (x1e-3)")
