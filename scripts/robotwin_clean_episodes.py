#!/usr/bin/env python3
"""Label RoboTwin 2.0 episodes as clean vs domain-randomized (the export has no flag).

Clean demos share one plain background; randomized ones have random textures / clutter / lighting.
Each episode's first cam_high frame is compared with the clean background (median of known clean
episodes) on two top wall corners. Writes JSON {"clean": [...], "randomized": [...], "dist": {...}}
and checks the expected layout (each 550-episode task block = 50 clean + 500 randomized).

    python scripts/robotwin_clean_episodes.py [--dataset-dir data/robotwin2.0/robotwin2.0] [--out ...]
"""
import argparse
import json
from pathlib import Path

import av
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--dataset-dir", default="data/robotwin2.0/robotwin2.0")
ap.add_argument("--out", default=None, help="default: <dataset-dir>/meta/clean_episodes.json")
ap.add_argument("--threshold", type=float, default=20.0)
args = ap.parse_args()
root = Path(args.dataset_dir)
chunk = int(json.loads((root / "meta/info.json").read_text())["chunks_size"])
n = sum(1 for l in (root / "meta/episodes.jsonl").read_text().splitlines() if l.strip())
PATCHES = [(slice(0, 40), slice(0, 60)), (slice(0, 40), slice(580, 640))]


def first(e):
    c = av.open(str(root / f"videos/chunk-{e // chunk:03d}/observation.images.cam_high/episode_{e:06d}.mp4"))
    f = next(c.decode(video=0)).to_ndarray(format="rgb24")
    c.close()
    return f.astype(np.float32)


ref = np.median(np.stack([first(e) for e in range(0, 50, 7)]), axis=0)
dist = {}
for e in range(n):
    f = first(e)
    dist[e] = float(max(np.abs(f[p] - ref[p]).mean() for p in PATCHES))
    if e % 1000 == 0:
        print(f"{e}/{n}", flush=True)
clean = [e for e in range(n) if dist[e] < args.threshold]
rand = [e for e in range(n) if dist[e] >= args.threshold]
expected = {e for e in range(n) if e % 550 < 50}
print(f"clean {len(clean)}  randomized {len(rand)}  |  matches 'episode % 550 < 50': "
      f"{len(expected & set(clean))}/{len(expected)} clean, {len(set(clean) - expected)} extra")
print(f"dist clean max {max(dist[e] for e in clean):.1f}  randomized min {min(dist[e] for e in rand):.1f}")
out = Path(args.out) if args.out else root / "meta/clean_episodes.json"
out.write_text(json.dumps({"clean": clean, "randomized": rand, "threshold": args.threshold,
                           "dist": {str(k): round(v, 2) for k, v in dist.items()}}))
print("wrote", out)
