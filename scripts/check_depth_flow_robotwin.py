#!/usr/bin/env python3
"""Count RoboTwin depth / flow videos against the episode list; optionally list missing episodes.

    python scripts/check_depth_flow_robotwin.py [--dataset-dir data/robotwin2.0/robotwin2.0] [--missing out.txt]
"""
import argparse
import json
from pathlib import Path

CAMS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
KEYS = [f"observation.images.{c}_{m}" for c in CAMS for m in ("depth", "flow")]

ap = argparse.ArgumentParser()
ap.add_argument("--dataset-dir", default="data/robotwin2.0/robotwin2.0")
ap.add_argument("--missing", default=None, help="write missing episode indices (one per line)")
args = ap.parse_args()

root = Path(args.dataset_dir)
info = json.loads((root / "meta/info.json").read_text())
chunk = int(info["chunks_size"])
episodes = [json.loads(l)["episode_index"] for l in (root / "meta/episodes.jsonl").read_text().splitlines() if l.strip()]
print(f"episodes: {len(episodes)}   registered keys: {sum(k in info['features'] for k in KEYS)}/{len(KEYS)}")

missing = set()
for key in KEYS:
    miss = [e for e in episodes
            if not (root / f"videos/chunk-{e // chunk:03d}/{key}/episode_{e:06d}.mp4").is_file()]
    missing.update(miss)
    print(f"{key:40s} {len(episodes) - len(miss):6d}/{len(episodes)}  ({100 * (1 - len(miss) / len(episodes)):5.1f}%)")
print(f"episodes complete for all 6 keys: {len(episodes) - len(missing)}/{len(episodes)}")
if args.missing:
    Path(args.missing).write_text("\n".join(map(str, sorted(missing))) + "\n")
    print(f"missing episode indices -> {args.missing}")
