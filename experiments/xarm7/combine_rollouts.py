#!/usr/bin/env python3
"""Combine per-trial inference episode HDF5s into one HDF5 per task.

Scans --root (default data/rollouts/runs) for episode_data--*.hdf5 files written by
the old xarm_fastwam_inference.py (datasets at the file root), groups them by
their task_description attribute, and writes <out>/<task>.hdf5 with one
data/trial_<N> group per source file, in the layout rollout_viewer.py reads.
N is the experiment start position (from the test_<N> directory). Sources are
never modified or deleted.

Usage: python combine_rollouts.py [--root DIR] [--out DIR] [--hz 15]
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path

import h5py

HERE = Path(__file__).resolve().parent


def _text(v):
    return v.decode() if isinstance(v, bytes) else str(v)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=HERE / "data" / "rollouts" / "runs")
    p.add_argument("--out", type=Path, default=HERE / "data" / "rollouts" / "combined")
    p.add_argument("--hz", type=float, default=15.0,
                   help="control_hz written to trials that lack it (old files didn't store it)")
    args = p.parse_args()

    by_task = defaultdict(list)
    for path in sorted(args.root.rglob("episode_data--*.hdf5")):
        with h5py.File(path, "r", locking=False) as f:
            task = _text(f.attrs.get("task_description", "")) or "unknown_task"
        by_task[task].append(path)
    if not by_task:
        raise SystemExit(f"no episode_data--*.hdf5 under {args.root}")

    args.out.mkdir(parents=True, exist_ok=True)
    for task, paths in by_task.items():
        out_path = args.out / (re.sub(r"[^A-Za-z0-9]+", "_", task).strip("_") + ".hdf5")
        with h5py.File(out_path, "w") as out:
            data = out.create_group("data")
            models = set()
            for src in paths:
                m = re.search(r"--model=(.+)\.hdf5$", src.name)
                model = m.group(1) if m else "unknown"
                models.add(model)
                pos = re.fullmatch(r"test_(\d+)", src.parent.name)
                name = f"trial_{pos.group(1)}" if pos else f"trial_{len(data) + 1}"
                if name in data:  # same position in another run
                    k = 2
                    while f"{name}_run{k}" in data:
                        k += 1
                    name = f"{name}_run{k}"
                g = data.create_group(name)
                with h5py.File(src, "r", locking=False) as f:
                    for key, ds in f.items():
                        f.copy(ds, g, name=key)
                    for key, val in f.attrs.items():
                        g.attrs[key] = val
                g.attrs.setdefault("control_hz", args.hz)
                g.attrs["model_name"] = model
                g.attrs["source_file"] = str(src.relative_to(args.root))
            out.attrs["model_name"] = ",".join(sorted(models))
        print(f"{out_path}: {len(paths)} trials ({task!r})")


if __name__ == "__main__":
    main()
