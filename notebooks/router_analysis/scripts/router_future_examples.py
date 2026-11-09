"""Predicted futures of the router-selected stream (short layout) on a few LIBERO training samples.

For each sample: the router picks a stream (argmax, eval mode, same inputs as closed-loop eval),
infer_joint runs the short layout [R0 | chosen future | A] and decodes the chosen future. For
comparison the two other streams are also run forced (active_future=<stream>). One PNG per sample:
rows = rgb / depth / flow, each row = GT future frames | predicted frames (selected row marked).

    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src:. python notebooks/router_analysis/scripts/router_future_examples.py
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
import router_offline_extract as ex  # noqa: E402  (model + local LIBERO dataset loader)

STREAMS = ("rgb", "depth", "flow")


def to_pil(x):
    """[3, H, W] in [-1, 1] -> PIL."""
    a = ((x.float().clamp(-1, 1) + 1) * 127.5).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(a)


def strip(frames, keep):
    ims = [frames[i] for i in keep]
    w, h = ims[0].size
    out = Image.new("RGB", (w * len(ims), h))
    for i, im in enumerate(ims):
        out.paste(im.resize((w, h)), (i * w, 0))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/data/mgovind/claude_scratch/router_future_examples")
    ap.add_argument("--per-stream", type=int, default=2, help="examples to collect per selected stream")
    ap.add_argument("--max-tries", type=int, default=40)
    ap.add_argument("--steps", type=int, default=10)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    sys.argv = ["x"]
    model, ds = ex.load(ex.parse())
    model.router_layout = "short"
    model.eval()
    rng = np.random.default_rng(0)
    got = {s: 0 for s in STREAMS}
    keep = [0, 2, 4, 6, 8]
    for t, idx in enumerate(rng.choice(len(ds), a.max_tries, replace=False).tolist()):
        if all(v >= a.per_stream for v in got.values()):
            break
        s = ds[idx]
        common = dict(
            prompt=None, input_image=s["video"][:, 0].to("cuda", model.torch_dtype),
            num_video_frames=s["video"].shape[1], action_horizon=s["action"].shape[0],
            proprio=s["proprio"][0:1].to("cuda", model.torch_dtype),
            context=s["context"][None].to("cuda", model.torch_dtype), context_mask=s["context_mask"][None].to("cuda"),
            num_inference_steps=a.steps, seed=0, prompt_text=s["prompt"],
            action_history=s["action_history"], action_history_is_pad=s["action_history_is_pad"].float(),
        )
        routed = model.infer_joint(**common)  # active_future=None -> router argmax
        chosen = routed["active_future"]
        if got[chosen] >= a.per_stream:
            continue
        got[chosen] += 1
        preds = {chosen: routed["video"]}
        for other in STREAMS:
            if other != chosen:
                preds[other] = model.infer_joint(**common, active_future=other)["video"]
        rows = []
        for st in STREAMS:
            gt = [to_pil(s[f"video_{st}"][:, i]) for i in range(s[f"video_{st}"].shape[1])]
            g, p = strip(gt, keep), strip(preds[st], keep)
            row = Image.new("RGB", (g.width + p.width + 10, g.height + 22), "white")
            row.paste(g, (0, 22))
            row.paste(p, (g.width + 10, 22))
            d = ImageDraw.Draw(row)
            tag = "  <== ROUTER SELECTED" if st == chosen else "  (forced, for comparison)"
            d.text((4, 4), f"{st}: GT future", fill="black")
            d.text((g.width + 14, 4), f"{st}: predicted{tag}", fill="red" if st == chosen else "gray")
            rows.append(row)
        img = Image.new("RGB", (rows[0].width, sum(r.height for r in rows) + 26), "white")
        ImageDraw.Draw(img).text((4, 6), f"idx {idx} | {s['prompt'][:90]} | router -> {chosen}", fill="black")
        y = 26
        for r in rows:
            img.paste(r, (0, y))
            y += r.height
        path = out / f"sample{idx:06d}_router_{chosen}.png"
        img.save(path)
        print(f"[{t}] idx {idx}: router -> {chosen}  saved {path}", flush=True)
    print("collected:", got)


if __name__ == "__main__":
    main()
