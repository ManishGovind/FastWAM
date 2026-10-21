#!/usr/bin/env python3
"""Verify the clean RoboTwin subset against its source, for EVERY episode.

  1. exactly 2,500 episodes; they map to 2,500 distinct source episodes, exactly 50 per task block,
     and every one satisfies source_index % 550 < 50 (the clean demos);
  2. meta: episodes.jsonl / episodes_stats.jsonl indices 0..N-1, lengths == source, info.json totals;
  3. data: every parquet == its source parquet (all columns except episode_index / index, which are
     re-indexed: consecutive episode_index, continuous global index);
  4. videos: every RGB video exists and is byte-identical (MD5) to its source video.

    python scripts/verify_robotwin_clean_subset.py \
        [--clean data/robotwin2.0_clean/robotwin2.0_clean] [--src data/robotwin2.0/robotwin2.0]
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--clean", default="data/robotwin2.0_clean/robotwin2.0_clean")
ap.add_argument("--src", default="data/robotwin2.0/robotwin2.0")
args = ap.parse_args()
D, S = Path(args.clean), Path(args.src)
info = json.loads((D / "meta/info.json").read_text())
src_info = json.loads((S / "meta/info.json").read_text())
chunk, src_chunk = int(info["chunks_size"]), int(src_info["chunks_size"])
m = json.loads((D / "meta/source_episodes.json").read_text())["new_to_old"]
eps = [json.loads(l) for l in (D / "meta/episodes.jsonl").read_text().splitlines() if l.strip()]
st = [json.loads(l) for l in (D / "meta/episodes_stats.jsonl").read_text().splitlines() if l.strip()]
src_len = {json.loads(l)["episode_index"]: json.loads(l)["length"]
           for l in (S / "meta/episodes.jsonl").read_text().splitlines() if l.strip()}
rgb = [k for k, v in info["features"].items() if v.get("dtype") == "video" and not k.endswith(("_depth", "_flow"))]
ok = lambda c, msg: print(("OK   " if c else "FAIL ") + msg) or c
res = []

# 1. selection
per_block = collections.Counter(o // 550 for o in m)
res.append(ok(len(m) == len(eps) == info["total_episodes"] == 2500, f"2500 episodes (map {len(m)}, episodes.jsonl {len(eps)}, info {info['total_episodes']})"))
res.append(ok(len(set(m)) == 2500, f"2500 distinct source episodes ({len(set(m))})"))
res.append(ok(all(o % 550 < 50 for o in m), "every source episode satisfies index % 550 < 50 (clean)"))
res.append(ok(len(per_block) == 50 and set(per_block.values()) == {50}, f"50 task blocks x 50 episodes ({len(per_block)} blocks, sizes {sorted(set(per_block.values()))})"))
res.append(ok(sorted(m) == m == [o for o in range(len(src_len)) if o % 550 < 50], "same set and order as all clean episodes of the source"))

# 2. meta
res.append(ok([e["episode_index"] for e in eps] == list(range(2500)) and [s["episode_index"] for s in st] == list(range(2500)),
              "episodes.jsonl / episodes_stats.jsonl indexed 0..2499"))
res.append(ok(all(eps[n]["length"] == src_len[m[n]] for n in range(2500)), "episode lengths == source"))
res.append(ok(info["total_frames"] == sum(e["length"] for e in eps), f"info total_frames {info['total_frames']} == sum of lengths"))

# 3. data, 4. videos
bad_data, bad_vid, missing, nf = [], [], [], 0
md5 = lambda p: hashlib.md5(p.read_bytes()).hexdigest()
for n, o in enumerate(m):
    a = pd.read_parquet(D / f"data/chunk-{n // chunk:03d}/episode_{n:06d}.parquet")
    b = pd.read_parquet(S / f"data/chunk-{o // src_chunk:03d}/episode_{o:06d}.parquet")
    same = (list(a.columns) == list(b.columns) and len(a) == len(b)
            and all(np.array_equal(np.stack(a[c].to_numpy()) if a[c].dtype == object else a[c].to_numpy(),
                                   np.stack(b[c].to_numpy()) if b[c].dtype == object else b[c].to_numpy())
                    for c in a.columns if c not in ("episode_index", "index"))
            and (a["episode_index"] == n).all() and a["index"].iloc[0] == nf and (np.diff(a["index"]) == 1).all())
    if not same:
        bad_data.append(n)
    nf += len(a)
    for k in rgb:
        pa = D / f"videos/chunk-{n // chunk:03d}/{k}/episode_{n:06d}.mp4"
        pb = S / f"videos/chunk-{o // src_chunk:03d}/{k}/episode_{o:06d}.mp4"
        if not pa.is_file():
            missing.append((n, k))
        elif pa.stat().st_size != pb.stat().st_size or md5(pa) != md5(pb):
            bad_vid.append((n, k))
    if n % 500 == 0:
        print(f"  checked {n}/2500", flush=True)
res.append(ok(not bad_data, f"all 2500 parquet files == source (re-indexed correctly); mismatches: {bad_data[:5]}"))
res.append(ok(not missing, f"all {2500 * len(rgb)} RGB videos present; missing: {missing[:5]}"))
res.append(ok(not bad_vid, f"all RGB videos byte-identical to source (MD5); mismatches: {bad_vid[:5]}"))
print("\nALL CHECKS PASSED" if all(res) else "\nSOME CHECKS FAILED")
