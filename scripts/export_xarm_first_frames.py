#!/usr/bin/env python3
"""Export first-frame reference PNGs from xArm7 collection HDF5 sessions.

Reads experiments/xarm7/data/<task>/<session>/*.hdf5 (obs are already the 480x480
post-crop JPEGs) and writes <task>__episode_<NNNNNN>__<exo|wrist|ego>__frame_<K>.png
into data/xarm7_first_frame_data/frames/ for experiments/xarm7/live_overlay.py.
"""
import argparse
from pathlib import Path

import cv2
import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
VIEWS = {"exo": "agentview_rgb_jpeg", "wrist": "eye_in_hand_rgb_jpeg", "ego": "ego_rgb_jpeg"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=ROOT / "experiments/xarm7/data")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "data/xarm7_first_frame_data/frames")
    ap.add_argument("--frame", type=int, default=0, help="frame index to export (default 0)")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for task_dir in sorted(p for p in args.data_dir.iterdir() if p.is_dir() and p.name != "rollouts"):
        episode = 0
        for path in sorted(task_dir.glob("*/*.hdf5")):
            with h5py.File(path, "r") as f:
                for demo in sorted(f["data"], key=lambda n: int(n.split("_")[1])):
                    obs = f["data"][demo]["obs"]
                    for view, key in VIEWS.items():
                        if key not in obs or args.frame >= len(obs[key]):
                            continue
                        img = cv2.imdecode(np.frombuffer(bytes(obs[key][args.frame]), np.uint8), cv2.IMREAD_COLOR)
                        name = f"{task_dir.name}__episode_{episode:06d}__{view}__frame_{args.frame:06d}.png"
                        cv2.imwrite(str(args.out_dir / name), img)
                        count += 1
                    episode += 1
        print(f"{task_dir.name}: {episode} demos")
    print(f"wrote {count} PNGs to {args.out_dir}")


if __name__ == "__main__":
    main()
