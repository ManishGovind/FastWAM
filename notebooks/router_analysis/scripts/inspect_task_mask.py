"""Inspect the MoT attention mask a task really uses (built from the task config, captured at the
MoT core in one training step and one inference call). Prints the block table, saves a PNG.

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=. python notebooks/router_analysis/scripts/inspect_task_mask.py \
        --task libero_joint_2cam224_1e-4_rgb_depth_flow_crosscascade
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from matplotlib.patches import Patch
from torch.utils.data import default_collate

from fastwam.trainer import Wan22Trainer

REPO = "/data/mgovind/FastWAM"
STATS = "runs/libero_joint_2cam224_1e-4_rgb_depth_flow/2026-08-28_09-13-18/dataset_stats.json"
OUT = "notebooks/router_analysis/outputs/cross_stream_masks"


def block_table(m, sizes, names):
    edges = np.cumsum([0] + sizes)
    rows = []
    for i, r in enumerate(names):
        cells = []
        for j in range(len(names)):
            blk = m[edges[i]:edges[i + 1], edges[j]:edges[j + 1]]
            cells.append("✓" if blk.all() else ("·" if not blk.any() else "partial"))
        rows.append((r, cells))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="libero_joint_2cam224_1e-4_rgb_depth_flow_crosscascade")
    ap.add_argument("--reference", default="libero_joint_2cam224_1e-4_rgb_depth_flow_crossfull",
                    help="only used for the default MM comparison: the MM mask is rebuilt with cross_stream_attention=none")
    args = ap.parse_args()
    os.chdir(REPO); os.makedirs(OUT, exist_ok=True)
    with initialize_config_dir(version_base="1.3", config_dir=f"{REPO}/configs"):
        cfg = compose(config_name="train", overrides=[f"task={args.task}"])
    model = instantiate(cfg.model, model_dtype=torch.float32, device="cpu")
    model._vae_encode_compiled = lambda v, s: model.vae.model.encode(v, s)
    Wan22Trainer._apply_dit_only_train_mode(model)
    print(f"task={args.task}  model={type(model).__name__}  cross_stream_attention={model.cross_stream_attention}")
    ds = instantiate(cfg.data.train, pretrained_norm_stats=STATS)
    s = ds._get(70000)

    cap = []
    core = model.mot.forward_joint_core

    def w(**kw):
        cap.append(kw["attention_mask"].detach().clone())
        return core(**kw)

    model.mot.forward_joint_core = w
    torch.manual_seed(0)
    with torch.no_grad():
        model.training_loss(default_collate([s]))
        train_mask = cap[-1]
        model.eval()
        model.infer_action(prompt=None, input_image=s["video_rgb"][:, 0], action_horizon=32,
                           num_video_frames=int(s["video_rgb"].shape[1]), proprio=s["proprio"][0],
                           context=s["context"], context_mask=s["context_mask"], num_inference_steps=1, seed=0)
        infer_mask = cap[-1]
    model.mot.forward_joint_core = core
    assert torch.equal(train_mask, infer_mask), "training and inference masks differ"

    # MM reference: same model, cross attention off
    mode = model.cross_stream_attention
    model.cross_stream_attention = "none"
    tpf, fut, a = 98, int(s["video_rgb"].shape[1] // 4) * 98, 32
    sizes, names = [tpf, fut, fut, fut, a], ["R0", "R_f", "D_f", "F_f", "A"]
    mm = model._build_multistream_mot_attention_mask(stream_seq_lens=[tpf + fut, fut, fut], action_seq_len=a,
                                                     video_tokens_per_frame=tpf, device="cpu", action_attend_mode="all")
    model.cross_stream_attention = mode
    m = train_mask.bool().numpy()
    assert m.shape == (sum(sizes), sum(sizes)), m.shape
    print(f"mask captured at the MoT core: shape {tuple(m.shape)}, dtype {train_mask.dtype}; training == inference: True")
    print("\nblock table (row = query, column = key; ✓ all allowed, · all blocked)")
    print(f"{'':6s}" + "".join(f"{n:>8s}" for n in names))
    for r, cells in block_table(m, sizes, names):
        print(f"{r:6s}" + "".join(f"{c:>8s}" for c in cells))
    new = m & ~mm.numpy()
    print(f"\nallowed {int(m.sum()):,} / {m.size:,}; MM {int(mm.sum()):,}; new vs MM {int(new.sum()):,}")
    print("new blocks vs MM:", [(names[i], names[j]) for i, (r, cells) in enumerate(block_table(new, sizes, names))
                                for j, c in enumerate(cells) if c == "✓"])

    img = np.ones(m.shape + (3,)) * 0.92
    img[m] = (0.06, 0.43, 0.47)
    img[new] = (0.85, 0.37, 0.01)
    fig, ax = plt.subplots(figsize=(6.5, 6.8))
    ax.imshow(img, interpolation="nearest")
    edges = np.cumsum([0] + sizes)
    for e in edges[1:-1]:
        ax.axhline(e - 0.5, color="k", lw=0.6); ax.axvline(e - 0.5, color="k", lw=0.6)
    mids = (edges[:-1] + edges[1:]) / 2
    ax.set_xticks(mids); ax.set_xticklabels(["R$_0$", "R$_f$", "D$_f$", "F$_f$", "A"])
    ax.set_yticks(mids); ax.set_yticklabels(["R$_0$", "R$_f$", "D$_f$", "F$_f$", "A"])
    ax.xaxis.tick_top(); ax.set_xlabel("keys"); ax.set_ylabel("queries")
    ax.set_title(f"{args.task.split('1e-4_')[-1]}: mask at the MoT core\n(train step == inference)", fontsize=10, pad=20)
    fig.legend(handles=[Patch(color=(0.06, 0.43, 0.47), label="allowed (as MM)"),
                        Patch(color=(0.85, 0.37, 0.01), label="new vs MM"),
                        Patch(color=(0.92, 0.92, 0.92), label="blocked")], loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    p = f"{OUT}/{args.task}_captured.png"
    fig.savefig(p, dpi=110)
    print("saved", p)


if __name__ == "__main__":
    main()
