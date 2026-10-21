import glob, json, math, re
from collections import defaultdict
from datetime import datetime
B = "evaluate_results/libero_plus"
runs = {
    "ROUTER (REINFORCE, warm-up)": f"{B}/libero_joint_2cam224_1e-4_router_new/20261008_073051",
    "random model, rgb": f"{B}/libero_joint_2cam224_1e-4_rgb_depth_flow_random_future_modality_selection/20261008_072731",
    "gt model, rgb": f"{B}/libero_joint_2cam224_1e-4_rgb_depth_flow_modality_gt/20261005_185443",
    "gt model, depth": f"{B}/libero_joint_2cam224_1e-4_rgb_depth_flow_modality_gt/20261006_112418",
    "gt model, all": f"{B}/libero_joint_2cam224_1e-4_rgb_depth_flow_modality_gt/20261005_220941",
    "unimodal RGB": f"{B}/libero_joint_2cam224_1e-4/20260831_144424",
    "unimodal Depth": f"{B}/libero_joint_2cam224_1e-4_depth/20260831_102459",
    "unimodal Flow": f"{B}/libero_joint_2cam224_1e-4_flow/20260831_123437",
    "MM (all streams)": f"{B}/libero_joint_2cam224_1e-4_rgb_depth_flow/20260831_164802",
}
CATS = ["Camera Viewpoints", "Background Textures", "Sensor Noise", "Objects Layout"]
SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
def load(d):
    out = {}
    for p in glob.glob(f"{d}/*/gpu*_task*_results.json"):
        r = json.load(open(p)); out[(r["task_suite"], r["task_id"])] = r
    return out
res = {k: load(v) for k, v in runs.items()}
keys = set.intersection(*(set(v) for v in res.values()))
print("tasks in all runs:", len(keys))
cat = {k: res["unimodal Depth"][k]["perturbation_category"] for k in keys}
ok = lambda r, k: int(r[k]["successes"] > 0)
print(f"\n{'':30s}" + "".join(f"{c.split()[0]:>11s}" for c in CATS) + "".join(f"{s.replace('libero_',''):>9s}" for s in SUITES) + f"{'TOTAL':>8s}")
for n, r in res.items():
    cells = [100 * sum(ok(r, k) for k in keys if cat[k] == c) / sum(cat[k] == c for k in keys) for c in CATS]
    cells += [100 * sum(ok(r, k) for k in keys if k[0] == s) / sum(k[0] == s for k in keys) for s in SUITES]
    print(f"{n:30s}" + "".join(f"{v:11.1f}" for v in cells[:4]) + "".join(f"{v:9.1f}" for v in cells[4:]) + f"{100*sum(ok(r,k) for k in keys)/len(keys):8.2f}")
R = res["ROUTER (REINFORCE, warm-up)"]
print("\npaired, same task & seed: router wins / other wins / exact McNemar p")
for n, r in res.items():
    if n.startswith("ROUTER"): continue
    w = sum(1 for k in keys if ok(R, k) and not ok(r, k)); l = sum(1 for k in keys if not ok(R, k) and ok(r, k)); m = w + l
    p = min(1.0, 2 * sum(math.comb(m, i) for i in range(min(w, l) + 1)) / 2 ** m) if m else 1.0
    print(f"  vs {n:28s} +{w:3d} / -{l:3d}  net {w-l:+4d}  p={p:.3f}")
# router picks per task from worker logs (by GPU + time window)
d = runs["ROUTER (REINFORCE, warm-up)"]
spans = defaultdict(list)
for k, r in R.items():
    t0 = datetime.strptime(r["start_time"], "%Y-%m-%d %H:%M:%S"); spans[int(r["gpu_id"])].append((t0.timestamp(), t0.timestamp() + float(r["duration"]) + 1, k))
picks = defaultdict(lambda: defaultdict(int))
unassigned = 0
for g in range(4):
    for line in open(f"{d}/.worker_pool/logs/worker_{g}.log"):
        m = re.match(r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d+\].*Router selected active_future=(\w+)", line)
        if not m: continue
        ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").timestamp()
        hit = [k for a, b, k in spans[g] if a - 1 <= ts <= b]
        if hit: picks[hit[0]][m.group(2)] += 1
        else: unassigned += 1
print(f"\nrouter picks assigned to {len(picks)} tasks ({unassigned} unassigned)")
def share(ks):
    tot = defaultdict(int)
    for k in ks:
        for s, c in picks.get(k, {}).items(): tot[s] += c
    n = sum(tot.values()) or 1
    return " / ".join(f"{100*tot[s]/n:4.1f}%" for s in ("rgb", "depth", "flow")), n
print(f"{'':22s} picks rgb / depth / flow     (n replans)")
for c in CATS:
    s, n = share([k for k in keys if cat[k] == c]); print(f"  {c:20s} {s}   ({n})")
for su in SUITES:
    s, n = share([k for k in keys if k[0] == su]); print(f"  {su:20s} {s}   ({n})")
s, n = share(keys); print(f"  {'ALL':20s} {s}   ({n})")
# success split by dominant stream
dom = {k: max(picks[k], key=picks[k].get) for k in keys if k in picks}
print("\nsuccess by router's dominant stream per task (router vs unimodal of that stream vs MM):")
for s, uni in (("rgb", "unimodal RGB"), ("depth", "unimodal Depth"), ("flow", "unimodal Flow")):
    ks = [k for k in dom if dom[k] == s]
    if ks: print(f"  {s:6s} tasks {len(ks):4d}: router {100*sum(ok(R,k) for k in ks)/len(ks):5.1f}%  {uni} {100*sum(ok(res[uni],k) for k in ks)/len(ks):5.1f}%  MM {100*sum(ok(res['MM (all streams)'],k) for k in ks)/len(ks):5.1f}%  random-rgb {100*sum(ok(res['random model, rgb'],k) for k in ks)/len(ks):5.1f}%")
