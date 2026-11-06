"""Router pick distribution of every router eval folder on LIBERO and LIBERO-Plus (label-free).

Same matching as libero_router_picks.py: each 'Router selected active_future=X' line in
.worker_pool/logs/worker_<id>.log goes to the task whose [start, start + duration] window (same
worker) contains it. Tasks evaluated with a fixed active_future (not routed) are skipped, and so
are folders without any router line. Picks are counted per replan.

Prints per folder: overall rgb / depth / flow %, then per suite (LIBERO) or per perturbation
category (LIBERO-Plus), the number of tasks dominated by one stream (> 90 % of its replans) and the
success rate of the routed tasks.

    python notebooks/router_analysis/scripts/router_picks_all.py [--bench libero libero_plus]
"""
import argparse, collections, datetime as dt, glob, json, os, re

PAT = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d+\].*Router selected active_future=(\w+)")
STREAMS = ["rgb", "depth", "flow"]
SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
CATS = ["Camera Viewpoints", "Background Textures", "Sensor Noise", "Objects Layout",
        "Robot Initial States", "Language Instructions", "Light Conditions"]


def picks_per_task(d):
    tasks = []
    for f in glob.glob(f"{d}/*/*_results.json"):
        r = json.load(open(f))
        if r.get("active_future") not in (None, "", "none"):
            continue
        t0 = dt.datetime.strptime(r["start_time"], "%Y-%m-%d %H:%M:%S")
        wid = r.get("worker_id", r.get("gpu_id"))   # older evals: one worker per GPU, logged as gpu_id
        tasks.append((str(wid), t0, t0 + dt.timedelta(seconds=float(r["duration"]) + 1), r))
    out = {(r["task_suite"], r["task_id"]): [collections.Counter(), r] for *_, r in tasks}
    for log in glob.glob(f"{d}/.worker_pool/logs/worker_*.log"):
        w = re.search(r"worker_(\d+)", log).group(1)
        for line in open(log, errors="ignore"):
            m = PAT.match(line)
            if not m:
                continue
            ts = dt.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
            for wid, a, b, r in tasks:
                if wid == w and a <= ts <= b:
                    out[(r["task_suite"], r["task_id"])][0][m.group(2)] += 1
                    break
    return {k: v for k, v in out.items() if sum(v[0].values())}


def pct(c):
    n = sum(c.values())
    return " / ".join(f"{100 * c[s] / n:4.1f}" for s in STREAMS) if n else "  -  "


def report(d, bench):
    res = picks_per_task(d)
    if not res:
        return
    tot = collections.Counter()
    for c, _ in res.values():
        tot += c
    succ = 100 * sum(r["successes"] for _, r in res.values()) / sum(r["total_episodes"] for _, r in res.values())
    dom = sum(1 for c, _ in res.values() if max(c.values()) / sum(c.values()) > 0.9)
    name = d.split("/")[-2].replace("libero_joint_2cam224_1e-4_", "") + "/" + d.split("/")[-1]
    print(f"\n{name}\n  ALL  {pct(tot)}   (tasks {len(res)}, replans {sum(tot.values())}, "
          f"one-stream tasks {dom}/{len(res)}, success {succ:.2f})")
    groups = SUITES if bench == "libero" else CATS
    key = (lambda r: r["task_suite"]) if bench == "libero" else (lambda r: r.get("perturbation_category"))
    for g in groups:
        c = collections.Counter()
        for cc, r in res.values():
            if key(r) == g:
                c += cc
        if sum(c.values()):
            print(f"  {g.replace('libero_', ''):22s} {pct(c)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", nargs="+", default=["libero", "libero_plus"])
    args = ap.parse_args()
    for bench in args.bench:
        print(f"\n===== {bench}  (picks rgb / depth / flow %) =====")
        for d in sorted(glob.glob(f"evaluate_results/{bench}/*router*/*")):
            if os.path.isdir(d):
                report(d, bench)


if __name__ == "__main__":
    main()
