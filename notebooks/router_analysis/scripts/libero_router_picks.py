"""Router picks per LIBERO task from an eval folder (label-free).

Maps each 'Router selected active_future=X' line in .worker_pool/logs/worker_<id>.log to the task
whose [start_time, start_time + duration] window (same worker) contains it.

    python notebooks/router_analysis/scripts/libero_router_picks.py <eval_dir> [<eval_dir> ...]
"""
import collections, datetime as dt, glob, json, math, re, sys

PAT = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d+\].*Router selected active_future=(\w+)")
SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
STREAMS = ["rgb", "depth", "flow"]


def picks_per_task(d):
    tasks = []
    for f in glob.glob(f"{d}/*/*_results.json"):
        r = json.load(open(f))
        t0 = dt.datetime.strptime(r["start_time"], "%Y-%m-%d %H:%M:%S")
        tasks.append((str(r["worker_id"]), t0, t0 + dt.timedelta(seconds=float(r["duration"]) + 1),
                      r["task_suite"], int(r["task_id"]), r["successes"] / r["total_episodes"]))
    out = {(s, t): [collections.Counter(), sr] for _, _, _, s, t, sr in tasks}
    for log in glob.glob(f"{d}/.worker_pool/logs/worker_*.log"):
        w = re.search(r"worker_(\d+)", log).group(1)
        for line in open(log, errors="ignore"):
            m = PAT.match(line)
            if not m:
                continue
            ts = dt.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
            for wid, a, b, s, t, _ in tasks:
                if wid == w and a <= ts <= b:
                    out[(s, t)][0][m.group(2)] += 1
                    break
    return out


def ent(c):
    n = sum(c.values())
    return -sum(v / n * math.log(v / n) for v in c.values() if v) if n else float("nan")


for d in sys.argv[1:]:
    res = picks_per_task(d)
    print(f"\n== {d}")
    tot = collections.Counter()
    for s in SUITES:
        c = sum((res[k][0] for k in res if k[0] == s), collections.Counter()); tot += c
        n = sum(c.values()) or 1
        print(f"{s:15s} " + "  ".join(f"{m} {100*c[m]/n:5.1f}%" for m in STREAMS) + f"   (n={n})")
    n = sum(tot.values()) or 1
    print(f"{'all':15s} " + "  ".join(f"{m} {100*tot[m]/n:5.1f}%" for m in STREAMS) + f"   (n={n})")
    dom = [max(v[0].values()) / sum(v[0].values()) for v in res.values() if sum(v[0].values())]
    print(f"within-task: mean share of the task's top stream {100*sum(dom)/len(dom):.1f}%  | "
          f"tasks with one stream >=95%: {sum(x >= .95 for x in dom)}/{len(dom)}  | "
          f"mean per-task pick entropy {sum(ent(v[0]) for v in res.values())/len(res):.3f} (max {math.log(3):.3f})")
