"""Stream oracles from closed-loop results: random-select + modality context (one model, 3
conditioned streams) vs the 3 unimodal models (one model per stream). All seed 42.

Oracles (each computed the same way for both groups, so they compare like with like):
  single best    : the best fixed stream / model overall
  per-episode OR : success if any of the 3 succeeded on that episode (same init state; LIBERO
                   episode i = init state i). Hindsight upper bound, inflated by 3 tries.
  per-task best  : the stream with the highest success on each task (LIBERO: 40 tasks x 50 ep;
                   Plus: per base task over its variants). In-sample, optimistic.
  held-out       : Plus only. Stream chosen per (suite, base task) on the variants with even
                   classification id, scored on the odd ones, and vice versa (2-fold). What a
                   per-task router could reach without hindsight.

    python notebooks/router_analysis/scripts/oracle_compare.py
"""

import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

E = "evaluate_results"
NEW = "libero_joint_2cam224_1e-4_random_short_m_context_video"
GROUPS = {
    "libero": {
        "ctx": {"rgb": f"{NEW}/20261011_002046", "depth": f"{NEW}/20261011_023737", "flow": f"{NEW}/20261011_045329"},
        "unimodal": {"rgb": "libero_joint_2cam224_1e-4/20260901_100312",
                     "depth": "libero_joint_2cam224_1e-4_depth/20260822_130931",
                     "flow": "libero_joint_2cam224_1e-4_flow/20260826_144704"},
        "ref": {"ctx random": f"{NEW}/20261010_223621"},
    },
    "libero_plus": {
        "ctx": {"rgb": f"{NEW}/20261011_032922", "depth": f"{NEW}/20261011_063251", "flow": f"{NEW}/20261011_084609"},
        "unimodal": {"rgb": "libero_joint_2cam224_1e-4/20260831_144424",
                     "depth": "libero_joint_2cam224_1e-4_depth/20260831_102459",
                     "flow": "libero_joint_2cam224_1e-4_flow/20260831_123437"},
        "ref": {"ctx random": f"{NEW}/20261011_002059",
                "MM": "libero_joint_2cam224_1e-4_rgb_depth_flow_modality_specific/20261006_154439"},
    },
}
STREAMS = ("rgb", "depth", "flow")


def load(bench, rel):
    """{(suite, task_id): (bool success per episode [n_ep], record)}"""
    out = {}
    for f in glob.glob(f"{E}/{bench}/{rel}/**/*_results.json", recursive=True):
        r = json.load(open(f))
        n = int(r["total_episodes"])
        ok = np.zeros(n, dtype=bool)
        ok[[int(i) for i in r.get("success_episodes", []) if int(i) < n]] = True
        if ok.sum() != int(r["successes"]):  # fall back if the episode list is incomplete
            ok[:] = False
            ok[: int(r["successes"])] = True
        out[(r["task_suite"], int(r["task_id"]))] = (ok, r)
    return out


def base_task(r):
    name = r.get("task_name", r.get("task_description", ""))
    # strip the LIBERO-Plus variant suffix (layout / background / camera / noise; goal layout: _moved)
    return re.sub(r"_(level\d|table_\d|view_|light|noise_|add_\d|sample\d|initstate|moved).*$", "", name)


def oracles(runs, bench):
    keys = sorted(set.intersection(*[set(v) for v in runs.values()]))
    succ = {s: np.concatenate([runs[s][k][0] for k in keys]) for s in STREAMS}
    rate = {s: 100 * succ[s].mean() for s in STREAMS}
    out = {f"fixed {s}": rate[s] for s in STREAMS}
    out["single best"] = max(rate.values())
    out["per-episode OR"] = 100 * np.logical_or.reduce([succ[s] for s in STREAMS]).mean()
    if bench == "libero":
        tot = sum(max(runs[s][k][0].sum() for s in STREAMS) for k in keys)
        out["per-task best"] = 100 * tot / sum(len(runs["rgb"][k][0]) for k in keys)
    else:
        groups = defaultdict(list)
        for k in keys:
            groups[(k[0], base_task(runs["rgb"][k][1]))].append(k)
        best = 0
        held = []
        for g, ks in groups.items():
            best += max(sum(runs[s][k][0].sum() for k in ks) for s in STREAMS)
            for fold in (0, 1):
                tr = [k for k in ks if int(runs["rgb"][k][1]["classification_id"]) % 2 == fold]
                te = [k for k in ks if int(runs["rgb"][k][1]["classification_id"]) % 2 != fold]
                if not tr or not te:
                    continue
                pick = max(STREAMS, key=lambda s: (sum(runs[s][k][0].sum() for k in tr), s == "rgb"))
                held += [bool(runs[pick][k][0][0]) for k in te]
        out["per-task best"] = 100 * best / len(keys)
        out["held-out per-task"] = 100 * np.mean(held)
        out["_n_base_tasks"] = len(groups)
    out["_n"] = len(keys)
    return out


def main():
    os.chdir("/data/mgovind/FastWAM")
    for bench, g in GROUPS.items():
        res = {grp: oracles({s: load(bench, g[grp][s]) for s in STREAMS}, bench) for grp in ("ctx", "unimodal")}
        print(f"\n=== {bench} (seed 42; n={res['ctx']['_n']} tasks"
              + (f", {res['ctx']['_n_base_tasks']} base tasks" if "_n_base_tasks" in res["ctx"] else "") + ") ===")
        print(f"{'':22s}{'ctx model (3 streams)':>24s}{'unimodal (3 models)':>22s}")
        for k in res["ctx"]:
            if k.startswith("_"):
                continue
            print(f"{k:22s}{res['ctx'][k]:24.2f}{res['unimodal'][k]:22.2f}")
        for name, rel in g["ref"].items():
            r = load(bench, rel)
            v = 100 * np.concatenate([x[0] for x in r.values()]).mean()
            print(f"{'ref: ' + name:22s}{v:24.2f}")


if __name__ == "__main__":
    main()
