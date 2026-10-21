"""Token-level MoT attention masks built by FastWAMJointMultimodal for cross_stream_attention
none (MM) / cascade / full, LIBERO sizes [R0 98 | R_f 196 | D_f 196 | F_f 196 | A 32].

    PYTHONPATH=. python notebooks/router_analysis/scripts/plot_cross_stream_masks.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from fastwam.models.wan22.fastwam_joint_multimodal import FastWAMJointMultimodal

TPF, FUT, A = 98, 196, 32
OUT = "notebooks/router_analysis/outputs/cross_stream_masks"


def mask(mode):
    m = object.__new__(FastWAMJointMultimodal)
    nn.Module.__init__(m)
    m.enabled_video_streams = ("rgb", "depth", "flow")
    m.cross_stream_attention = mode
    return FastWAMJointMultimodal._build_multistream_mot_attention_mask(
        m, stream_seq_lens=[TPF + FUT, FUT, FUT], action_seq_len=A, video_tokens_per_frame=TPF,
        device="cpu", action_attend_mode="all").numpy()


masks = {k: mask(k) for k in ("none", "cascade", "full")}
edges = np.cumsum([0, TPF, FUT, FUT, FUT, A])
labels = ["R$_0$", "R$_f$", "D$_f$", "F$_f$", "A"]
titles = {"none": "MM (none)", "cascade": "MM + cascade (R→D→F)", "full": "MM + full cross-modal"}

fig, axes = plt.subplots(1, 3, figsize=(16, 6.2))
for ax, (k, m) in zip(axes, masks.items()):
    img = np.ones(m.shape + (3,)) * 0.92                    # blocked: light grey
    img[m] = (0.06, 0.43, 0.47)                             # allowed (as MM): teal
    if k != "none":
        img[m & ~masks["none"]] = (0.85, 0.37, 0.01)        # newly allowed vs MM: orange
    ax.imshow(img, interpolation="nearest")
    for e in edges[1:-1]:
        ax.axhline(e - 0.5, color="k", lw=0.6); ax.axvline(e - 0.5, color="k", lw=0.6)
    mids = (edges[:-1] + edges[1:]) / 2
    ax.set_xticks(mids); ax.set_xticklabels(labels); ax.set_yticks(mids); ax.set_yticklabels(labels)
    ax.xaxis.tick_top(); ax.set_xlabel("keys"); ax.set_ylabel("queries")
    new = int((m & ~masks["none"]).sum())
    ax.set_title(f"{titles[k]}\nallowed {int(m.sum()):,} / {m.size:,}" + (f"  (+{new:,} vs MM)" if new else ""), fontsize=10, pad=22)
from matplotlib.patches import Patch
fig.legend(handles=[Patch(color=(0.06, 0.43, 0.47), label="allowed (as in MM)"),
                    Patch(color=(0.85, 0.37, 0.01), label="newly allowed (cross-modal)"),
                    Patch(color=(0.92, 0.92, 0.92), label="blocked")], loc="lower center", ncol=3, frameon=False)
fig.tight_layout(rect=(0, 0.05, 1, 0.97))
fig.savefig(f"{OUT}/mm_vs_cascade_vs_full.png", dpi=110)
print("saved", f"{OUT}/mm_vs_cascade_vs_full.png")
for k, m in masks.items():
    print(f"{k:8s} allowed tokens {int(m.sum()):7d}  new vs MM {int((m & ~masks['none']).sum()):7d}")
