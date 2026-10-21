#!/usr/bin/env python3
"""Build a LeRobot v2.1 subset of RoboTwin 2.0 with only the clean demos (or any episode list).

Clean demos = episode_index % 550 < 50 (50 tasks x [50 clean + 500 randomized]; verified from images
by scripts/robotwin_clean_episodes.py). The subset gets consecutive episode indices 0..N-1:
  data/   parquet copied with episode_index / index (global frame index) rewritten
  videos/ every video key that exists for the source episode (RGB, and depth / flow if present)
  meta/   info.json totals, episodes.jsonl, episodes_stats.jsonl (episode_index / index stats
          updated), tasks.jsonl unchanged (task_index stays valid), source_episodes.json (new -> old)
The source dataset is never modified.

    python scripts/make_robotwin_clean_subset.py \
        --src data/robotwin2.0/robotwin2.0 --dst data/robotwin2.0_clean/robotwin2.0_clean
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="data/robotwin2.0/robotwin2.0")
ap.add_argument("--dst", default="data/robotwin2.0_clean/robotwin2.0_clean")
ap.add_argument("--mod", type=int, default=550)
ap.add_argument("--below", type=int, default=50)
args = ap.parse_args()
src, dst = Path(args.src), Path(args.dst)
if dst.exists() and any(dst.iterdir()):
    raise SystemExit(f"{dst} exists and is not empty; choose another --dst")

info = json.loads((src / "meta/info.json").read_text())
chunk = int(info["chunks_size"])
episodes = [json.loads(l) for l in (src / "meta/episodes.jsonl").read_text().splitlines() if l.strip()]
stats = {json.loads(l)["episode_index"]: json.loads(l)
         for l in (src / "meta/episodes_stats.jsonl").read_text().splitlines() if l.strip()}
video_keys = [k for k, v in info["features"].items() if v.get("dtype") == "video"]
keep = [e["episode_index"] for e in episodes if e["episode_index"] % args.mod < args.below]
print(f"{len(keep)} of {len(episodes)} episodes; video keys {len(video_keys)}", flush=True)


def num_stats(x):
    x = np.asarray(x, dtype=np.float64)
    return {"min": [x.min()], "max": [x.max()], "mean": [x.mean()], "std": [x.std()], "count": [len(x)]}


ep_by_idx = {e["episode_index"]: e for e in episodes}
new_eps, new_stats, n_frames, n_videos, missing = [], [], 0, 0, 0
for new, old in enumerate(keep):
    d = pd.read_parquet(src / f"data/chunk-{old // chunk:03d}/episode_{old:06d}.parquet")
    d["episode_index"] = new
    d["index"] = np.arange(n_frames, n_frames + len(d), dtype=d["index"].dtype)
    p = dst / f"data/chunk-{new // chunk:03d}/episode_{new:06d}.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    d.to_parquet(p, index=False)
    for key in video_keys:
        s = src / f"videos/chunk-{old // chunk:03d}/{key}/episode_{old:06d}.mp4"
        if s.is_file():
            t = dst / f"videos/chunk-{new // chunk:03d}/{key}/episode_{new:06d}.mp4"
            t.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(s, t)
            n_videos += 1
        else:
            missing += 1
    e = dict(ep_by_idx[old]); e["episode_index"] = new; new_eps.append(e)
    st = json.loads(json.dumps(stats[old])); st["episode_index"] = new
    if "episode_index" in st["stats"]:
        st["stats"]["episode_index"] = num_stats(d["episode_index"])
    if "index" in st["stats"]:
        st["stats"]["index"] = num_stats(d["index"])
    new_stats.append(st)
    n_frames += len(d)
    if new % 250 == 0:
        print(f"  {new}/{len(keep)}", flush=True)

meta = dst / "meta"
meta.mkdir(parents=True, exist_ok=True)
(meta / "episodes.jsonl").write_text("".join(json.dumps(e) + "\n" for e in new_eps))
(meta / "episodes_stats.jsonl").write_text("".join(json.dumps(s) + "\n" for s in new_stats))
shutil.copy2(src / "meta/tasks.jsonl", meta / "tasks.jsonl")
info.update(total_episodes=len(keep), total_frames=n_frames, total_videos=len(keep) * len(video_keys),
            total_chunks=(len(keep) - 1) // chunk + 1, splits={"train": f"0:{len(keep)}"})
(meta / "info.json").write_text(json.dumps(info, indent=4))
(meta / "source_episodes.json").write_text(json.dumps(
    {"source": str(src.resolve()), "rule": f"episode_index % {args.mod} < {args.below}",
     "new_to_old": keep}))
print(f"done: {len(keep)} episodes, {n_frames} frames, {n_videos} videos copied "
      f"({missing} source videos not present, e.g. depth/flow not extracted yet) -> {dst}")
