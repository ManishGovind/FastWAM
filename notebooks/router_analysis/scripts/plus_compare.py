import json, glob, math
from collections import defaultdict
from pathlib import Path
R = "evaluate_results/router_eval_plus_new"
B = "evaluate_results/libero_plus"
routers = {"router FULL (CE, R0+ctx)": f"{R}/routers_new_ce_none_none_seed42_cap5/router",
           "router CTX (pooled, ctx only)": f"{R}/routers_new_pooled2_ce_none_r0_seed42_cap5/router"}
bases = {"RGB": f"{B}/libero_joint_2cam224_1e-4/20260831_144424",
         "Depth": f"{B}/libero_joint_2cam224_1e-4_depth/20260831_102459",
         "Flow": f"{B}/libero_joint_2cam224_1e-4_flow/20260831_123437",
         "MM (all)": f"{B}/libero_joint_2cam224_1e-4_rgb_depth_flow/20260831_164802"}
CATS = ["Camera Viewpoints", "Background Textures", "Sensor Noise", "Objects Layout"]
SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]

def load_router(d):
    out = {}
    for p in glob.glob(f"{d}/*_task*.json"):
        r = json.load(open(p))
        out[(r["suite"], r["task_id"])] = (int(r["successes"] > 0), r["perturbation_category"], r["picks"])
    return out

def load_base(d):
    out = {}
    for p in glob.glob(f"{d}/*/gpu*_task*_results.json"):
        r = json.load(open(p))
        out[(r["task_suite"], r["task_id"])] = (int(r["successes"] > 0), r["perturbation_category"], None)
    return out

res = {k: load_router(v) for k, v in routers.items()}
res.update({k: load_base(v) for k, v in bases.items()})
keys = set.intersection(*(set(v) for v in res.values()))
print("tasks in all runs:", len(keys), {k: len(v) for k, v in res.items()})
cat_of = {k: res["Depth"][k][1] for k in keys}

hdr = f"{'':32s}" + "".join(f"{c.split()[0]:>11s}" for c in CATS) + "".join(f"{s.replace('libero_',''):>9s}" for s in SUITES) + f"{'TOTAL':>8s}"
print("\n" + hdr)
for name, r in res.items():
    cells = []
    for c in CATS:
        ks = [k for k in keys if cat_of[k] == c]; cells.append(f"{100*sum(r[k][0] for k in ks)/len(ks):11.1f}")
    for s in SUITES:
        ks = [k for k in keys if k[0] == s]; cells.append(f"{100*sum(r[k][0] for k in ks)/len(ks):9.1f}")
    print(f"{name:32s}" + "".join(cells) + f"{100*sum(r[k][0] for k in keys)/len(keys):8.2f}")

def mcnemar(a, b):
    n01 = sum(1 for k in keys if a[k][0] == 1 and b[k][0] == 0)
    n10 = sum(1 for k in keys if a[k][0] == 0 and b[k][0] == 1)
    n = n01 + n10
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(0, min(n01, n10) + 1)) / 2 ** n) if n else 1.0
    return n01, n10, p

print("\npaired vs baselines (same task & seed): router-only wins / baseline-only wins / exact McNemar p")
for rn in routers:
    for bn in bases:
        w, l, p = mcnemar(res[rn], res[bn])
        print(f"  {rn:30s} vs {bn:8s}: +{w:3d} / -{l:3d}   net {w-l:+4d}   p={p:.3f}")
w, l, p = mcnemar(res[list(routers)[0]], res[list(routers)[1]])
print(f"  FULL vs CTX: +{w} / -{l}  p={p:.3f}")

for rn in routers:
    r = res[rn]
    tot = defaultdict(int)
    for k in keys:
        for m, c in r[k][2].items(): tot[m] += c
    s = sum(tot.values())
    pure = {m: [k for k in keys if r[k][2][m] == sum(r[k][2].values())] for m in ("RGB", "Depth", "Flow")}
    mixed = [k for k in keys if max(r[k][2].values()) < sum(r[k][2].values())]
    print(f"\n{rn}: replan picks R/D/F = " + " / ".join(f"{100*tot[m]/s:.1f}%" for m in ("RGB", "Depth", "Flow")))
    print(f"  tasks: all-Depth {len(pure['Depth'])}, all-RGB {len(pure['RGB'])}, all-Flow {len(pure['Flow'])}, mixed {len(mixed)}")
    pd = pure["Depth"]
    if pd:
        agree = sum(r[k][0] == res['Depth'][k][0] for k in pd)
        print(f"  determinism: all-Depth tasks, same outcome as Depth baseline {agree}/{len(pd)} "
              f"(router {100*sum(r[k][0] for k in pd)/len(pd):.1f}% vs Depth {100*sum(res['Depth'][k][0] for k in pd)/len(pd):.1f}%)")
    for grp, ks in (("all-RGB", pure["RGB"]), ("mixed", mixed)):
        if ks:
            print(f"  {grp:8s} tasks ({len(ks)}): router {100*sum(r[k][0] for k in ks)/len(ks):.1f}%  "
                  f"Depth {100*sum(res['Depth'][k][0] for k in ks)/len(ks):.1f}%  RGB {100*sum(res['RGB'][k][0] for k in ks)/len(ks):.1f}%")
