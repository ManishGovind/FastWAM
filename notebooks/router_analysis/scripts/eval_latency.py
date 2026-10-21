"""Closed-loop latency of LIBERO eval runs, recovered from the existing worker logs.

No eval logs per-inference timings, but every episode prints a tqdm bar over env steps
("Episode k: ... N/M [mm:ss<..."). The final bar state gives (env steps, wall time) per
episode. The model is queried once per ``replan_steps`` env steps (after
``num_steps_wait`` no-op steps), so

    sec / replan = episode wall time / number of replans

That includes the sim steps of one chunk and the per-replan overhead, the same for every
model, so differences between models are inference cost. Runs that overlapped in time with
other eval runs (shared GPUs) are flagged; prefer the non-overlapping ones.

    python notebooks/router_analysis/scripts/eval_latency.py [--root evaluate_results/libero] [--models a,b]
"""

import argparse
import glob
import json
import math
import os
import re
from datetime import datetime

import numpy as np
import yaml

BAR = re.compile(r"Episode (\d+):\s+\d+%\|[^|]*\|\s*(\d+)/(\d+) \[(?:(\d+):)?(\d+):(\d+)")
TS = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)")


def episodes_from_log(path):
    """[(env_steps, seconds)] for every finished episode in one worker log."""
    # One bar is spread over many log lines; walk all bar states in order and keep the
    # last state of each episode (it ends when the episode index changes or steps reset).
    out = []
    prev = None  # (episode, steps, secs)
    for m in BAR.finditer(open(path, errors="ignore").read()):
        ep, steps = int(m.group(1)), int(m.group(2))
        h_or_m, mm, ss = m.group(4), int(m.group(5)), int(m.group(6))
        secs = (int(h_or_m) * 3600 + mm * 60 + ss) if h_or_m is not None else (mm * 60 + ss)
        if prev is not None and (ep != prev[0] or steps < prev[1]):
            if prev[1] > 0 and prev[2] > 0:
                out.append((prev[1], prev[2]))
        prev = (ep, steps, secs)
    if prev is not None and prev[1] > 0 and prev[2] > 0:
        out.append((prev[1], prev[2]))
    return out


def run_span(run_dir):
    """(first, last) timestamp seen in the worker logs."""
    ts = []
    for f in glob.glob(f"{run_dir}/.worker_pool/logs/*.log"):
        for line in open(f, errors="ignore"):
            m = TS.match(line)
            if m:
                ts.append(datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S"))
    return (min(ts), max(ts)) if ts else (None, None)


def load_cfg(run_dir):
    p = f"{run_dir}/manager_config.yaml"
    if not os.path.exists(p):
        return {}
    txt = open(p).read()
    # The saved config holds unresolved ${...} interpolations; plain YAML is enough here.
    return yaml.safe_load(txt) or {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="evaluate_results/libero")
    ap.add_argument("--models", default=None, help="comma list of model folders (default: all)")
    ap.add_argument("--min-episodes", type=int, default=20)
    ap.add_argument("--json", default="notebooks/router_analysis/outputs/eval_latency.json")
    a = ap.parse_args()

    models = sorted(os.listdir(a.root)) if a.models is None else a.models.split(",")
    rows = []
    for model in models:
        for run_dir in sorted(glob.glob(f"{a.root}/{model}/*/")):
            run_dir = run_dir.rstrip("/")
            cfg = load_cfg(run_dir)
            ev = cfg.get("EVALUATION", {}) or {}
            replan = int(ev.get("replan_steps", 5) or 5)
            wait = int(ev.get("num_steps_wait", 0) or 0)
            nis = ev.get("num_inference_steps")
            if isinstance(nis, str):  # "${eval_num_inference_steps}"
                nis = cfg.get("eval_num_inference_steps")
            eps = []
            for f in sorted(glob.glob(f"{run_dir}/.worker_pool/logs/*.log")):
                eps += episodes_from_log(f)
            if len(eps) < a.min_episodes:
                continue
            per_replan = np.array([s / max(1, math.ceil((n - wait) / replan)) for n, s in eps])
            per_step = np.array([s / n for n, s in eps])
            n_replans = sum(max(1, math.ceil((n - wait) / replan)) for n, _ in eps)
            pooled = sum(s for _, s in eps) / n_replans  # 1 s log resolution averages out
            t0, t1 = run_span(run_dir)
            rows.append({
                "model": model,
                "run": os.path.basename(run_dir),
                "episodes": len(eps),
                "sec_per_replan": float(pooled),
                "replans": int(n_replans),
                "sec_per_replan_median": float(np.median(per_replan)),
                "sec_per_replan_mean": float(per_replan.mean()),
                "sec_per_step_median": float(np.median(per_step)),
                "num_inference_steps": nis,
                "replan_steps": replan,
                "compile_action_infer": ev.get("compile_action_infer"),
                "active_future": ev.get("active_future"),
                "action_attend_mode": ev.get("action_attend_mode"),
                "num_gpus": (cfg.get("MULTIRUN", {}) or {}).get("num_gpus"),
                "start": None if t0 is None else t0.isoformat(sep=" "),
                "end": None if t1 is None else t1.isoformat(sep=" "),
            })

    # Overlap: another eval run (any model) active at the same time.
    for r in rows:
        if r["start"] is None:
            r["overlap"] = None
            continue
        s, e = r["start"], r["end"]
        r["overlap"] = [f"{o['model']}/{o['run']}" for o in rows
                        if o is not r and o["start"] is not None and o["start"] < e and s < o["end"]]

    print(f"{'model':58s} {'run':16s} {'eps':>5s} {'s/replan':>9s} {'nis':>4s} {'rp':>3s} {'comp':>5s} {'active':>8s} {'ovl':>4s}")
    for r in rows:
        print(f"{r['model']:58s} {r['run']:16s} {r['episodes']:5d} {r['sec_per_replan']:9.3f} "
              f"{str(r['num_inference_steps']):>4s} {r['replan_steps']:3d} {str(r['compile_action_infer']):>5s} "
              f"{str(r['active_future']):>8s} {len(r['overlap'] or []):4d}")

    print("\n=== per model: median over runs (all runs | runs with no overlapping eval) ===")
    for model in models:
        rs = [r for r in rows if r["model"] == model]
        if not rs:
            continue
        allv = [r["sec_per_replan"] for r in rs]
        clean = [r["sec_per_replan"] for r in rs if r["overlap"] == []]
        cs = f"{np.median(clean):.3f} ({len(clean)} runs)" if clean else "-"
        print(f"{model:58s} {np.median(allv):.3f} ({len(allv)} runs) | {cs}")

    os.makedirs(os.path.dirname(a.json), exist_ok=True)
    json.dump(rows, open(a.json, "w"), indent=1)
    print(f"\nsaved {a.json}")


if __name__ == "__main__":
    main()
