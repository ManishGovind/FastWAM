"""Router picks along the trajectory: does the chosen stream change between early / mid / late replans?

Each 'Router selected active_future=X' line is assigned to (task, episode): the task by the
[start, start + duration] window of the same worker (as in router_picks_all.py), the episode by the
last 'Episode N:' progress-bar counter seen before it in the worker log. Within an episode the
replans are split into --bins equal phases of the episode's own length.

Prints per eval folder: picks r/d/f per phase, switch rate (consecutive replans with a different
stream), share of episodes that use a single stream, and the same split for successful / failed
episodes.

    python notebooks/router_analysis/scripts/router_picks_phase.py DIR [DIR ...] [--bins 5]
"""
import argparse, collections, datetime as dt, glob, json, re

PICK = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d+\].*Router selected active_future=(\w+)")
EP = re.compile(r"Episode (\d+):")
S = ["rgb", "depth", "flow"]


def episodes(d):
    tasks = []
    for f in glob.glob(f"{d}/*/*_results.json"):
        r = json.load(open(f))
        t0 = dt.datetime.strptime(r["start_time"], "%Y-%m-%d %H:%M:%S")
        wid = str(r.get("worker_id", r.get("gpu_id")))
        tasks.append((wid, t0, t0 + dt.timedelta(seconds=float(r["duration"]) + 1), r))
    eps = collections.defaultdict(list)
    for log in glob.glob(f"{d}/.worker_pool/logs/worker_*.log"):
        w = re.search(r"worker_(\d+)", log).group(1)
        ep = None
        for line in open(log, errors="ignore"):
            m = EP.findall(line)
            if m:
                ep = int(m[-1])
            p = PICK.match(line)
            if not p or ep is None:
                continue
            ts = dt.datetime.strptime(p.group(1), "%Y-%m-%d %H:%M:%S")
            for wid, a, b, r in tasks:
                if wid == w and a <= ts <= b:
                    eps[(r["task_suite"], r["task_id"], ep)].append((p.group(2), r))
                    break
    out = []
    for (suite, tid, ep), v in eps.items():
        r = v[0][1]
        if "success_episodes" in r:      # LIBERO: 0-based episode ids of the successes
            ok = (ep - 1) in set(r["success_episodes"])
        else:                            # LIBERO-Plus: one trial per task
            ok = r["successes"] > 0
        out.append(([s for s, _ in v], ok))
    return out


def pct(c):
    n = sum(c.values())
    return " / ".join(f"{100 * c[s] / n:4.1f}" for s in S) if n else "-"


def report(d, bins):
    eps = [(p, ok) for p, ok in episodes(d) if len(p) >= bins]
    print(f"\n{d}  ({len(eps)} episodes with >= {bins} replans)")
    for label, sel in (("all", eps), ("success", [e for e in eps if e[1]]), ("failure", [e for e in eps if not e[1]])):
        if not sel:
            continue
        ph = [collections.Counter() for _ in range(bins)]
        sw = n = single = 0
        for p, _ in sel:
            for i, s in enumerate(p):
                ph[i * bins // len(p)][s] += 1
            sw += sum(a != b for a, b in zip(p, p[1:]))
            n += len(p) - 1
            single += len(set(p)) == 1
        print(f"  {label:8s} n={len(sel):5d}  switch {100 * sw / n:4.1f}%  single-stream eps {100 * single / len(sel):4.1f}%")
        for k, c in enumerate(ph):
            print(f"    phase {k + 1}/{bins}  {pct(c)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--bins", type=int, default=5)
    a = ap.parse_args()
    for d in a.dirs:
        report(d.rstrip("/"), a.bins)


if __name__ == "__main__":
    main()
