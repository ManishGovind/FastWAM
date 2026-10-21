"""LIBERO-Plus (capped 1,495 tasks, seed 42): router full-all + stream embedding + embed router vs baselines.

Success per category / suite, paired McNemar vs the new run, and router picks per category
(from the worker logs, by worker + time window).

    python notebooks/router_analysis/scripts/plus_streamemb_compare.py
"""
import collections, datetime as dt, glob, json, math, re
from scipy.stats import binomtest

B = "evaluate_results/libero_plus"
NEW = "router full-all + streamemb + embed"
runs = {
    NEW: f"{B}/libero_joint_2cam224_1e-4_router_full_all_streamemb_embed/20261017_170855",
    "streamemb (raw router)": f"{B}/libero_joint_2cam224_1e-4_router_full_all_streamemb/20261017_231236",
    "router full-all": f"{B}/libero_joint_2cam224_1e-4_router_full_all/20261014_195653",
    "random full-all": f"{B}/libero_joint_2cam224_1e-4_random_full_all/20261014_000754",
    "random full-all (short)": f"{B}/libero_joint_2cam224_1e-4_random_full_all_short_infer/20261014_100925",
    "MM": f"{B}/libero_joint_2cam224_1e-4_rgb_depth_flow/20260831_164802",
    "unimodal RGB": f"{B}/libero_joint_2cam224_1e-4/20260831_144424",
}
CATS = ["Camera Viewpoints", "Background Textures", "Sensor Noise", "Objects Layout"]
SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
PAT = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d+\].*Router selected active_future=(\w+)")


def load(d):
    return {(r["task_suite"], r["task_id"]): r for r in
            (json.load(open(p)) for p in glob.glob(f"{d}/*/gpu*_task*_results.json"))}


res = {k: load(v) for k, v in runs.items()}
keys = set.intersection(*(set(v) for v in res.values()))
cat = {k: res[NEW][k]["perturbation_category"] for k in keys}
ok = lambda r, k: int(r[k]["successes"] > 0)
print(f"tasks in all runs: {len(keys)}\n")
print(f"{'':38s}" + "".join(f"{c.split()[0]:>11s}" for c in CATS) +
      "".join(f"{s.replace('libero_', ''):>9s}" for s in SUITES) + f"{'TOTAL':>8s}   McNemar vs new")
for name, r in res.items():
    row = f"{name:38s}"
    for c in CATS:
        ks = [k for k in keys if cat[k] == c]; row += f"{100*sum(ok(r, k) for k in ks)/len(ks):11.1f}"
    for s in SUITES:
        ks = [k for k in keys if k[0] == s]; row += f"{100*sum(ok(r, k) for k in ks)/len(ks):9.1f}"
    row += f"{100*sum(ok(r, k) for k in keys)/len(keys):8.2f}"
    if name != NEW:
        a = sum(ok(res[NEW], k) and not ok(r, k) for k in keys); b = sum(ok(r, k) and not ok(res[NEW], k) for k in keys)
        row += f"   +{a}/-{b} p={binomtest(a, a + b).pvalue:.3f}"
    print(row)

# per-category McNemar vs router full-all and MM
for other in ["router full-all", "MM", "random full-all"]:
    out = []
    for c in CATS:
        ks = [k for k in keys if cat[k] == c]
        a = sum(ok(res[NEW], k) and not ok(res[other], k) for k in ks); b = sum(ok(res[other], k) and not ok(res[NEW], k) for k in ks)
        out.append(f"{c.split()[0]} +{a}/-{b} p={binomtest(a, a + b).pvalue:.3f}" if a + b else f"{c.split()[0]} n/a")
    print(f"\nper category, new vs {other}: " + " | ".join(out))


def picks(d):
    tasks = []
    for k, r in load(d).items():
        t0 = dt.datetime.strptime(r["start_time"], "%Y-%m-%d %H:%M:%S")
        tasks.append((str(r["worker_id"]), t0, t0 + dt.timedelta(seconds=float(r["duration"]) + 1), k))
    by_w = collections.defaultdict(list)
    for t in tasks: by_w[t[0]].append(t)
    out = collections.defaultdict(collections.Counter)
    for log in glob.glob(f"{d}/.worker_pool/logs/worker_*.log"):
        w = re.search(r"worker_(\d+)", log).group(1)
        for line in open(log, errors="ignore"):
            m = PAT.match(line)
            if not m: continue
            ts = dt.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
            for _, a, b, k in by_w[w]:
                if a <= ts <= b: out[k][m.group(2)] += 1; break
    return out


for name in [NEW, "streamemb (raw router)", "router full-all"]:
    p = picks(runs[name])
    print(f"\npicks, {name}:")
    for c in CATS + ["all"]:
        cnt = sum((p[k] for k in keys if c == "all" or cat[k] == c), collections.Counter()); n = sum(cnt.values()) or 1
        print(f"  {c:20s} " + "  ".join(f"{m} {100*cnt[m]/n:5.1f}%" for m in ["rgb", "depth", "flow"]) + f"  (n={n})")
    # same base task across perturbations: does the majority pick change?
    base = collections.defaultdict(list)
    for k in keys:
        if sum(p[k].values()):
            desc = re.sub(r"\s*level\d+\s*sample\d+$", "", res[NEW][k].get("task_description_raw", ""))
            base[(k[0], desc)].append(p[k].most_common(1)[0][0])
    multi = [v for v in base.values() if len(v) >= 3]
    changes = sum(len(set(v)) > 1 for v in multi)
    print(f"  base tasks (>=3 variants) whose majority pick differs across variants: {changes}/{len(multi)}")
