"""Per-window means of router metrics from the latest router_new run log."""
import glob, re, statistics as st, sys
runs = sorted(glob.glob("/data/mgovind/FastWAM/runs/libero_joint_2cam224_1e-4_router_new/*/wandb/run-*/files/output.log"))
path = sys.argv[1] if len(sys.argv) > 1 else runs[-1]
win = int(sys.argv[2]) if len(sys.argv) > 2 else 250
t = re.sub(r"\s+", " ", re.sub(r"\x1b\[[0-9;]*m", "", open(path).read()))
recs = re.findall(r"step=(\d+)/\d+(.*?)(?=step=\d+/\d+|$)", t)
keys = ["loss_action", "router_entropy", "router_collapse_kl", "router_pick_rgb", "router_pick_depth", "router_pick_flow", "router_adv_abs", "router_warmup"]
rows = []
for s, b in recs:
    d = {"step": int(s)}
    for k in keys:
        m = re.search(rf"\b{k}=([-0-9.e]+)", b)
        if m: d[k] = float(m.group(1))
    rows.append(d)
print(path, "| last step", rows[-1]["step"])
print(f"{'steps':>13s} " + " ".join(f"{k.replace('router_', ''):>11s}" for k in keys))
for lo in range(0, rows[-1]["step"] + 1, win):
    w = [r for r in rows if lo < r["step"] <= lo + win]
    if w:
        print(f"{lo + 1:6d}-{lo + win:<6d} " + " ".join(f"{st.mean(r[k] for r in w if k in r):11.4f}" for k in keys))
