import glob, json, math, re
from collections import defaultdict
from datetime import datetime
B = "evaluate_results/libero"
runs = {
    "ROUTER (REINFORCE, warm-up)": f"{B}/libero_joint_2cam224_1e-4_router_new/20261008_121313",
    "unimodal RGB": f"{B}/libero_joint_2cam224_1e-4/20260901_100312",
    "unimodal Depth": f"{B}/libero_joint_2cam224_1e-4_depth/20260822_130931",
    "unimodal Flow": f"{B}/libero_joint_2cam224_1e-4_flow/20260826_144704",
    "gt model, rgb": f"{B}/libero_joint_2cam224_1e-4_rgb_depth_flow_modality_gt/20261005_172933",
    "gt model, all": f"{B}/libero_joint_2cam224_1e-4_rgb_depth_flow_modality_gt/20261005_221126",
}
SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
def load(d):
    out = {}
    for p in glob.glob(f"{d}/*/gpu*_task*_results.json"):
        r = json.load(open(p)); key = (r.get("task_suite") or r.get("suite"), int(r["task_id"]))
        n = int(r["total_episodes"]); s = set(r["success_episodes"])
        out[key] = (r, [int(i in s) for i in range(n)])
    return out
res = {k: load(v) for k, v in runs.items()}
keys = sorted(set.intersection(*(set(v) for v in res.values())))
print("tasks:", len(keys), "| episodes per task:", {n: len(next(iter(r.values()))[1]) for n, r in res.items()})
print(f"\n{'':30s}" + "".join(f"{s.replace('libero_',''):>9s}" for s in SUITES) + f"{'overall':>9s}")
for n, r in res.items():
    cells = []
    for s in SUITES:
        ep = [e for k in keys if k[0] == s for e in r[k][1]]; cells.append(100 * sum(ep) / len(ep))
    ep = [e for k in keys for e in r[k][1]]
    print(f"{n:30s}" + "".join(f"{v:9.1f}" for v in cells) + f"{100*sum(ep)/len(ep):9.2f}  ({sum(ep)}/{len(ep)})")
R = res["ROUTER (REINFORCE, warm-up)"]
print("\npaired per episode (same seed, task, init state): router-only / other-only / McNemar p")
for n, r in res.items():
    if n.startswith("ROUTER"): continue
    w = l = 0
    for k in keys:
        for a, b in zip(R[k][1], r[k][1]): w += a and not b; l += b and not a
    m = w + l; p = min(1.0, 2 * sum(math.comb(m, i) for i in range(min(w, l) + 1)) / 2 ** m) if m else 1.0
    print(f"  vs {n:26s} +{w:3d} / -{l:3d}  net {w-l:+4d}  p={p:.3f}")
# picks per suite from worker logs
d = runs["ROUTER (REINFORCE, warm-up)"]
spans = defaultdict(list)
for k, (r, _) in R.items():
    t0 = datetime.strptime(r["start_time"], "%Y-%m-%d %H:%M:%S").timestamp()
    spans[int(r["gpu_id"])].append((t0, t0 + float(r["duration"]) + 1, k))
picks = defaultdict(lambda: defaultdict(int)); un = 0
for f in glob.glob(f"{d}/.worker_pool/logs/worker_*.log"):
    m0 = re.search(r"worker_(\d+)\.log$", f) or re.search(r"gpu(\d+)", f)
    if not m0: continue
    g = int(m0.group(1))
    for line in open(f):
        m = re.match(r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d+\].*Router selected active_future=(\w+)", line)
        if not m: continue
        ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").timestamp()
        hit = [k for a, b, k in spans[g] if a - 1 <= ts <= b]
        if hit: picks[hit[0]][m.group(2)] += 1
        else: un += 1
print(f"\nrouter picks per suite ({sum(sum(v.values()) for v in picks.values())} replans, {un} unassigned): rgb / depth / flow")
for s in SUITES + ["ALL"]:
    tot = defaultdict(int)
    for k in keys:
        if s == "ALL" or k[0] == s:
            for st, c in picks[k].items(): tot[st] += c
    n = sum(tot.values()) or 1
    print(f"  {s:16s} " + " / ".join(f"{100*tot[x]/n:5.1f}%" for x in ("rgb", "depth", "flow")))

print("\nper task: rgb / depth / flow share of replans | router success | task")
for k in keys:
    t = picks[k]; n = sum(t.values()) or 1
    r = R[k][0]
    print(f"  {k[0]:15s} {k[1]:2d}  {100*t['rgb']/n:5.1f} / {100*t['depth']/n:5.1f} / {100*t['flow']/n:5.1f}  | {r['successes']:2d}/{r['total_episodes']}  {r.get('task_description','')[:60]}")
