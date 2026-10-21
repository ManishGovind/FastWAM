"""Random-select + modality context (target video) vs baselines, LIBERO and LIBERO-Plus.

LIBERO: success per suite (40 tasks x 50 trials). LIBERO-Plus: success per perturbation
category (4 categories, cap 5, 1495 tasks, 1 trial), paired McNemar vs the new runs on the
same task ids (all runs seed 42). Partial runs are reported on the tasks they finished.

    python notebooks/router_analysis/scripts/m_context_compare.py
"""

import glob
import json
import math
import os
from collections import defaultdict

ROOT = "evaluate_results"
NEW = "libero_joint_2cam224_1e-4_random_short_m_context_video"

PLUS_RUNS = {
    # new: random-select + modality condition (target video), one checkpoint
    "ctx random": f"{NEW}/20261011_002059",
    "ctx rgb": f"{NEW}/20261011_032922",
    "ctx depth": f"{NEW}/20261011_063251",
    "ctx flow": f"{NEW}/20261011_084609",
    # baselines (seed 42, same categories / cap)
    "MM (all futures)": "libero_joint_2cam224_1e-4_rgb_depth_flow_modality_specific/20261006_154439",
    "unimodal RGB": "libero_joint_2cam224_1e-4/20260831_144424",
    "unimodal Depth": "libero_joint_2cam224_1e-4_depth/20260831_102459",
    "unimodal Flow": "libero_joint_2cam224_1e-4_flow/20260831_123437",
    "no-ctx random-select rgb": "libero_joint_2cam224_1e-4_rgb_depth_flow_random_future_modality_selection/20261008_072731",
    "no-ctx random-select depth": "libero_joint_2cam224_1e-4_rgb_depth_flow_random_future_modality_selection/20261008_111543",
    "no-ctx random-select flow": "libero_joint_2cam224_1e-4_rgb_depth_flow_random_future_modality_selection/20261008_145846",
    "no-ctx router": "libero_joint_2cam224_1e-4_router_new/20261008_073051",
}
LIBERO_NEW = {
    "ctx random": f"{NEW}/20261010_223621",
    "ctx rgb": f"{NEW}/20261011_002046",
    "ctx depth": f"{NEW}/20261011_023737",
    "ctx flow": f"{NEW}/20261011_045329",
}
CATS = ["Camera Viewpoints", "Background Textures", "Sensor Noise", "Objects Layout"]
SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]


def load(bench, rel):
    out = {}
    for f in glob.glob(f"{ROOT}/{bench}/{rel}/**/*_results.json", recursive=True):
        r = json.load(open(f))
        out[(r["task_suite"], int(r["task_id"]))] = r
    return out


def mcnemar_p(b, c):
    """Exact two-sided binomial McNemar on discordant pairs b (A only) / c (B only)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def libero_table():
    print("=== LIBERO (seed 42, 50 trials x 40 tasks) ===")
    print(f"{'run':14s}" + "".join(f"{s.replace('libero_', ''):>9s}" for s in SUITES) + f"{'overall':>9s}")
    for name, rel in LIBERO_NEW.items():
        res = load("libero", rel)
        row = []
        for s in SUITES:
            rs = [r for (su, _), r in res.items() if su == s]
            row.append(100 * sum(r["successes"] for r in rs) / max(1, sum(r["total_episodes"] for r in rs)))
        tot = 100 * sum(r["successes"] for r in res.values()) / max(1, sum(r["total_episodes"] for r in res.values()))
        print(f"{name:14s}" + "".join(f"{v:9.2f}" for v in row) + f"{tot:9.2f}   ({len(res)} tasks)")
    print("reference (earlier, seed 42): no-ctx router 98.00 | unimodal RGB 97.95, Depth 98.05, Flow 98.40\n")


def plus_table():
    runs = {k: load("libero_plus", v) for k, v in PLUS_RUNS.items()}
    print("=== LIBERO-Plus (seed 42; 4 categories, cap 5; 1 trial per task) ===")
    print(f"{'run':28s}" + "".join(f"{c.split()[0]:>11s}" for c in CATS) + f"{'total':>9s}{'n':>6s}")
    for name, res in runs.items():
        per = defaultdict(list)
        for r in res.values():
            per[r["perturbation_category"]].append(r["successes"] > 0)
        allv = [v for c in CATS for v in per[c]]
        print(f"{name:28s}" + "".join(f"{100 * sum(per[c]) / max(1, len(per[c])):11.1f}" for c in CATS)
              + f"{100 * sum(allv) / max(1, len(allv)):9.2f}{len(allv):6d}")

    print("\n=== paired McNemar on common tasks (A vs B: A-only wins / B-only wins, p) ===")
    for a in ["ctx random", "ctx rgb", "ctx depth", "ctx flow"]:
        for b in ["MM (all futures)", "unimodal RGB", "no-ctx random-select rgb", "no-ctx router", "ctx random"]:
            if a == b:
                continue
            common = sorted(set(runs[a]) & set(runs[b]))
            if not common:
                continue
            sa = [runs[a][k]["successes"] > 0 for k in common]
            sb = [runs[b][k]["successes"] > 0 for k in common]
            only_a = sum(x and not y for x, y in zip(sa, sb))
            only_b = sum(y and not x for x, y in zip(sa, sb))
            print(f"  {a:11s} vs {b:26s} n={len(common):4d}  {100 * sum(sa) / len(common):5.1f} vs "
                  f"{100 * sum(sb) / len(common):5.1f}  ({only_a:3d}/{only_b:3d})  p={mcnemar_p(only_a, only_b):.3g}")


if __name__ == "__main__":
    os.chdir("/data/mgovind/FastWAM")
    libero_table()
    plus_table()
