"""Per-episode router pick timelines on LIBERO (from the eval worker logs).

Every `Router selected active_future=X` line is tied to the episode and env step of the preceding
tqdm line (`Episode N: ... t/T`). For each method: picks by episode phase (10 bins of t / episode
length), switch rate between consecutive replans, mean run length, and episodes that use 1 / 2 / 3
modalities.

    python notebooks/router_analysis/scripts/router_pick_timelines.py
"""
import collections, glob, json, re, sys
import numpy as np

B = "evaluate_results/libero/libero_joint_2cam224_1e-4_"
METHODS = {
    "+ stream emb (raw router)": "router_full_all_streamemb/20261017_230730",
    "+ embed router (main)": "router_full_all_streamemb_embed/20261017_151848",
    "Transformer (CLS)": "router_full_all_streamemb_embed_transformer/20261023_202845",
    "Query transformer": "router_full_all_streamemb_embed_transformer_query/20261025_040652",
}
S = ["rgb", "depth", "flow"]
TOK = re.compile(r"Episode (\d+):\s+\d+%\|[^|]*\|\s*(\d+)/(\d+)|Router selected active_future=(\w+)")


def episodes(run):
    eps = []                                   # list of (suite?, picks [(t, stream)])
    for log in glob.glob(f"{B}{run}/.worker_pool/logs/worker_*.log"):
        txt = open(log, errors="ignore").read()
        cur, last_ep, t = None, None, 0
        for m in TOK.finditer(txt):
            if m[1]:
                ep = int(m[1]); t = int(m[2])
                if ep != last_ep:
                    if cur: eps.append(cur)
                    cur, last_ep = [], ep
            elif cur is not None:
                cur.append((t, m[4]))
        if cur: eps.append(cur)
    return [e for e in eps if len(e) >= 2]


rows = {}
for name, run in METHODS.items():
    eps = episodes(run)
    phase = np.zeros((10, 3)); switches = steps = 0; runs = []; nmod = collections.Counter()
    for e in eps:
        T = max(t for t, _ in e) or 1
        seq = [s for _, s in e]
        for t, s in e:
            phase[min(9, int(10 * t / (T + 1))), S.index(s)] += 1
        switches += sum(a != b for a, b in zip(seq, seq[1:])); steps += len(seq) - 1
        r = 1
        for a, b in zip(seq, seq[1:]):
            if a == b: r += 1
            else: runs.append(r); r = 1
        runs.append(r); nmod[len(set(seq))] += 1
    phase = 100 * phase / phase.sum(1, keepdims=True)
    rows[name] = dict(eps=len(eps), phase=phase, switch=100 * switches / max(steps, 1), run=np.mean(runs),
                      nmod={k: 100 * nmod[k] / len(eps) for k in (1, 2, 3)})

print(f"{'method':28s} episodes  switch%/replan  mean run (replans)  episodes using 1 / 2 / 3 modalities")
for n, r in rows.items():
    print(f"{n:28s} {r['eps']:8d}  {r['switch']:13.1f}  {r['run']:18.1f}  {r['nmod'][1]:5.1f} / {r['nmod'][2]:5.1f} / {r['nmod'][3]:5.1f}")
print("\npicks by episode phase (rgb/depth/flow %, phase 0 = start, 9 = end)")
for n, r in rows.items():
    print(f"  {n}")
    for i in (0, 2, 4, 6, 9):
        p = r["phase"][i]; print(f"    phase {i}: {p[0]:5.1f} / {p[1]:5.1f} / {p[2]:5.1f}")
json.dump({n: {**r, "phase": r["phase"].round(2).tolist()} for n, r in rows.items()},
          open("notebooks/router_analysis/outputs/router_pick_timelines.json", "w"), indent=1)
