"""RoboTwin clean-trained router vs RGB baseline: per-task success (clean / randomized) and router picks.

Reads the RoboTwin manager logs and the router run's per-task eval logs; writes two JSON row files
for the "RoboTwin Clean-to-Randomized" dashboard.

    python notebooks/router_analysis/scripts/robotwin_router_results.py
"""
import collections, glob, json, os, re, datetime as dt

# Main 50-task run plus later single-task reruns (e.g. open_microwave); a later run's result wins.
ROUTERS = [d for d in sorted(glob.glob("evaluate_results/robotwin/robotwin_joint_3cam_384_1e-4_clean_router_full_all_streamemb_embed_*/*"))
           if d.split("/")[-1] >= "20261027_183313"]
RGB = "evaluate_results/robotwin/robotwin_joint_3cam_384_1e-4_clean_2026-10-20_13-26-23/20261021_211449"
OUT = "notebooks/router_analysis/outputs/robotwin_dashboard"
os.makedirs(OUT, exist_ok=True)


def rates(d):
    r = {}
    for line in open(f"{d}/manager.log"):
        m = re.search(r"done task=(\S+) phase=(\S+) .*success_rate=([0-9.]+)", line)
        if m: r[(m[1], m[2])] = round(100 * float(m[3]), 1)
    return r


R, B = {k: v for d in ROUTERS for k, v in rates(d).items()}, rates(RGB)
tasks = sorted({t for t, _ in B} | {t for t, _ in R})
rows = [{"task": t, "router_clean": R.get((t, "clean")), "rgb_clean": B.get((t, "clean")),
         "router_random": R.get((t, "random")), "rgb_random": B.get((t, "random"))} for t in tasks]

picks = collections.Counter()
ANSI = re.compile(r"\x1b\[[0-9;]*m")
for f in (f for d in ROUTERS for f in glob.glob(f"{d}/eval_*.log")):
    txt = ANSI.sub("", open(f, errors="ignore").read())
    setting = "randomized" if "demo_randomized" in txt else "clean"
    for m in re.finditer(r"Router selected active_future=(\w+)", txt):
        picks[(setting, m[1])] += 1
pick_rows = [{"setting": s, "rgb": picks[(s, "rgb")], "depth": picks[(s, "depth")], "flow": picks[(s, "flow")]}
             for s in ("clean", "randomized")]

now = dt.datetime.now().astimezone().isoformat(timespec="seconds")
json.dump(rows, open(f"{OUT}/robotwin_tasks.json", "w"), indent=0)
json.dump(pick_rows, open(f"{OUT}/robotwin_picks.json", "w"), indent=0)
done_c = sum(r["router_clean"] is not None for r in rows); done_r = sum(r["router_random"] is not None for r in rows)
print(f"{now}  router runs {[d.split('/')[-1] for d in ROUTERS]}: clean {done_c}/50, randomized {done_r}/50 done; picks {dict(picks)}")
