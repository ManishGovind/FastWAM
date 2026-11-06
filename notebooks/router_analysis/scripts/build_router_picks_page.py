"""Build outputs/router_picks.html: router pick distributions during training and at eval (LIBERO, LIBERO-Plus).

Training: router_pick_{rgb,depth,flow} from each run's local wandb file (logged every 10 steps by
rank 0: the fraction of that step's batch routed to each stream). Eval: the per-replan picks from
the worker logs (router_picks_all.picks_per_task). The data is embedded into
scripts/router_picks_template.html at the /*__DATA__*/ marker.

    .venv/bin/python notebooks/router_analysis/scripts/build_router_picks_page.py
"""
import collections, glob, json, os, sys

sys.path.insert(0, os.path.dirname(__file__))
from router_picks_all import picks_per_task, SUITES, STREAMS  # noqa: E402

from wandb.proto import wandb_internal_pb2 as pb  # noqa: E402
from wandb.sdk.internal import datastore  # noqa: E402

P = "libero_joint_2cam224_1e-4_"
# key, label, short description, training run, LIBERO eval, LIBERO-Plus eval (capped, 4 categories)
VARIANTS = [
    ("full_all", "Full layout", "all futures trained, action reads the routed one",
     "router_full_all/2026-10-12_22-29-10", "router_full_all/20261014_184947", "router_full_all/20261014_195653"),
    ("streamemb", "+ stream embedding", "full layout + stream identity, raw router input",
     "router_full_all_streamemb/2026-10-16_04-46-14", "router_full_all_streamemb/20261017_230730",
     "router_full_all_streamemb/20261017_231236"),
    ("streamemb_embed", "+ embed router (main)", "stream identity + router on the expert's embeddings (MLP)",
     "router_full_all_streamemb_embed/2026-10-16_06-33-56", "router_full_all_streamemb_embed/20261017_151848",
     "router_full_all_streamemb_embed/20261017_170855"),
    ("hist", "+ history", "main + RGB observation history for the router",
     "router_full_all_streamemb_embed_hist/2026-10-18_16-38-33", "router_full_all_streamemb_embed_hist/20261020_035607",
     "router_full_all_streamemb_embed_hist/20261020_082231"),
    ("transformer", "Transformer router", "main with a 2-layer transformer router (CLS token)",
     "router_full_all_streamemb_embed_transformer/2026-10-22_17-05-44",
     "router_full_all_streamemb_embed_transformer/20261023_202845",
     "router_full_all_streamemb_embed_transformer/20261023_203354"),
    ("query", "Query router", "main with one learned query per modality",
     "router_full_all_streamemb_embed_transformer_query/2026-10-24_00-00-47",
     "router_full_all_streamemb_embed_transformer_query/20261025_040652",
     "router_full_all_streamemb_embed_transformer_query/20261025_040824"),
]
# What each picked slot actually decodes: rows = slot (rgb, depth, flow), columns = content (rgb, depth, flow).
ID = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
DECODE = {
    "full_all": ([[0, 1, 0], [0, 1, 0], [0.70, 0.19, 0.11]], "measured",
                 "no stream identity. router_picks_decoding.ipynb (40 LIBERO tasks, 10-step rollout): rgb slot -> depth 40/40, "
                 "depth slot -> depth 40/40, flow slot -> RGB 19 / depth 5 / flow 3 of 27. A 3-trial LIBERO-10 eval with "
                 "saved videos gave flow 46 / depth 16 / RGB 11 for the flow slot: its content drifts with the task phase"),
    "streamemb": (ID, "measured", "sigma=1 slot check: every slot closest to its own stream; 3-trial LIBERO-10 eval videos: "
                  "flow picks decoded flow 71/71"),
    "streamemb_embed": (ID, "measured", "full 10-step evaluate(): rgb / depth / flow slots decode RGB / depth / flow "
                        "(rgb slot vs GT 24.9 dB)"),
    "hist": (ID, "measured", "sigma=1 slot check: every slot closest to its own stream; 10-step decode of 3 LIBERO windows: every slot matches its own GT stream (rgb 23-28 dB, depth 29-34 dB, flow 28-38 dB)"),
    "transformer": (ID, "measured", "sigma=1 slot check: every slot closest to its own stream; 10-step decode of 3 LIBERO windows: every slot matches its own GT stream (rgb 23-29 dB, depth 29-34 dB, flow 28-37 dB)"),
    "query": (ID, "measured", "sigma=1 slot check: every slot closest to its own stream; 10-step decode of 3 LIBERO windows: every slot matches its own GT stream (rgb 24-29 dB, depth 30-34 dB, flow 26-38 dB)"),
}


def decoded(picks, m):
    if picks is None or m is None:
        return None
    return [round(sum(picks[i] * m[i][j] for i in range(3)), 1) for j in range(3)]


CATS = ["Camera Viewpoints", "Background Textures", "Sensor Noise", "Objects Layout"]
WARMUP = 2000
SMOOTH = 50          # logged points (= 500 steps) in the rolling mean
EVERY = 5            # keep every 5th smoothed point


def wandb_history(run_dir):
    f = glob.glob(f"{run_dir}/wandb/run-*/run-*.wandb")[0]
    ds = datastore.DataStore()
    ds.open_for_scan(f)
    rows = []
    while True:
        d = ds.scan_data()
        if d is None:
            break
        r = pb.Record()
        r.ParseFromString(d)
        if r.WhichOneof("record_type") != "history":
            continue
        row = {(it.key or "/".join(it.nested_key)): json.loads(it.value_json) for it in r.history.item}
        if "train/router_pick_rgb" in row:
            rows.append((int(row["_step"]), [float(row[f"train/router_pick_{s}"]) for s in STREAMS],
                         [float(row.get(f"train/router_p_{s}", "nan")) for s in STREAMS]))
    rows.sort()
    return rows


def train_block(run_dir):
    rows = wandb_history(run_dir)
    steps = [r[0] for r in rows]
    picks = [r[1] for r in rows]
    curve = []
    for i in range(0, len(rows), EVERY):
        win = picks[max(0, i - SMOOTH + 1): i + 1]
        curve.append([steps[i]] + [round(100 * sum(p[k] for p in win) / len(win), 2) for k in range(3)])
    last = max(steps)

    def mean(sel):
        sel = [p for s, p in zip(steps, picks) if sel(s)]
        return [round(100 * sum(p[k] for p in sel) / len(sel), 1) for k in range(3)] if sel else None

    return {"curve": curve, "last_step": last, "warmup": WARMUP,
            "post_warmup": mean(lambda s: s > WARMUP), "final": mean(lambda s: s >= 0.9 * last),
            "n_logged": len(rows)}


def counts(c):
    n = sum(c.values())
    return [round(100 * c[s] / n, 1) for s in STREAMS] if n else None


def eval_block(d, bench):
    res = picks_per_task(f"evaluate_results/{bench}/{P}{d}")
    tot, groups = collections.Counter(), collections.defaultdict(collections.Counter)
    gsucc = collections.defaultdict(lambda: [0, 0])
    tasks = []
    for (suite, tid), (c, r) in sorted(res.items(), key=lambda kv: (SUITES.index(kv[0][0]), kv[0][1])):
        tot += c
        g = suite if bench == "libero" else r.get("perturbation_category")
        groups[g] += c
        gsucc[g][0] += r["successes"]
        gsucc[g][1] += r["total_episodes"]
        if bench == "libero":
            tasks.append({"suite": suite.replace("libero_", ""), "id": tid,
                          "name": r.get("task_description") or r.get("task_name"),
                          "picks": counts(c), "n": sum(c.values()),
                          "success": round(100 * r["successes"] / r["total_episodes"], 1)})
    order = SUITES if bench == "libero" else CATS
    out = {
        "all": counts(tot), "replans": sum(tot.values()), "tasks": len(res),
        "success": round(100 * sum(g[0] for g in gsucc.values()) / sum(g[1] for g in gsucc.values()), 2),
        "one_stream": sum(1 for c, _ in res.values() if max(c.values()) / sum(c.values()) > 0.9),
        "groups": [{"name": g.replace("libero_", ""), "picks": counts(groups[g]),
                    "success": round(100 * gsucc[g][0] / gsucc[g][1], 1)} for g in order if groups[g]],
    }
    if tasks:
        out["per_task"] = tasks
    return out


def main():
    os.chdir("/data/mgovind/FastWAM")
    data = []
    for key, label, desc, run, lib, plus in VARIANTS:
        print("building", key, flush=True)
        m, status, source = DECODE[key]
        libero, plus_ = eval_block(lib, "libero"), eval_block(plus, "libero_plus")
        data.append({"key": key, "label": label, "desc": desc, "run": run,
                     "train": train_block(f"runs/{P}{run}"), "libero": libero, "plus": plus_,
                     "decode": {"status": status, "source": source, "slot_map": m,
                                "libero": decoded(libero["all"], m), "plus": decoded(plus_["all"], m)},
                     "eval_dirs": {"libero": lib, "plus": plus}})
    here = os.path.dirname(os.path.abspath(__file__))
    tpl = open(f"{here}/router_picks_template.html").read()
    html = tpl.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
    out = os.path.join(here, "..", "outputs", "router_picks.html")
    open(out, "w").write(html)
    print("wrote", os.path.normpath(out), f"({len(html) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
