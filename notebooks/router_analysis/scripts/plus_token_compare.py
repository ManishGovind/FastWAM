"""LIBERO-Plus (capped): MM + token stream embedding vs baselines, paired on the tasks every run has.

    python notebooks/router_analysis/scripts/plus_token_compare.py
"""
import glob, json
from scipy.stats import binomtest

B = "evaluate_results/libero_plus/libero_joint_2cam224_1e-4_"
R = {
    "MM + stream emb (token)": B + "rgb_depth_flow_streamemb_tok/20261026_111213",
    "MM": B + "rgb_depth_flow/20260831_164802",
    "MM + cascade": B + "rgb_depth_flow_crosscascade/20261020_085544",
    "router + stream emb + embed (main)": B + "router_full_all_streamemb_embed/20261017_170855",
    "router full-all (no stream emb)": B + "router_full_all/20261014_195653",
}
CATS = ["Camera Viewpoints", "Background Textures", "Sensor Noise", "Objects Layout"]


def load(d):
    out = {}
    for p in glob.glob(f"{d}/*/gpu*_task*_results.json"):
        r = json.load(open(p))
        out[(r["task_suite"], r["task_id"])] = r
    return out


L = {k: load(v) for k, v in R.items()}
keys = sorted(set.intersection(*(set(v) for v in L.values())))
ref = L["MM"]
cat = {k: ref[k]["perturbation_category"] for k in keys}
ok = lambda d, k: int(d[k]["successes"] > 0)
A = L["MM + stream emb (token)"]
print(f"tasks in all runs: {len(keys)} (token run has {len(A)}/1495)")
print(f"{'':36s}" + "".join(f"{c.split()[0]:>11s}" for c in CATS) + f"{'TOTAL':>8s}   token better/worse, McNemar p")
for n, d in L.items():
    cells = "".join(f"{100 * sum(ok(d, k) for k in keys if cat[k] == c) / max(1, sum(cat[k] == c for k in keys)):11.1f}" for c in CATS)
    tot = 100 * sum(ok(d, k) for k in keys) / len(keys)
    tail = ""
    if d is not A:
        a = sum(ok(A, k) and not ok(d, k) for k in keys)
        b = sum(ok(d, k) and not ok(A, k) for k in keys)
        tail = f"   +{a}/-{b}  p={binomtest(a, a + b).pvalue:.3f}" if a + b else ""
    print(f"{n:36s}{cells}{tot:8.2f}{tail}")
print("tasks per category:", {c.split()[0]: sum(cat[k] == c for k in keys) for c in CATS})
