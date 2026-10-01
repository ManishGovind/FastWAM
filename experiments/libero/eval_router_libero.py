"""Closed-loop LIBERO eval where a per-replan modality choice picks which unimodal joint model acts.

Drivers (all three unimodal models are loaded; only the chosen one is run at each replan):
  router  the offline-trained router (`notebooks/train_modality_router.py`, soft CE, all features)
          applied to the live observation: pooled R0 VAE latent, normalized proprio, instruction.
  gt_knn  the action-error label of the nearest demo window of the same task, nearest by the
          same R0 latent + proprio features. This is the closest available stand-in for the
          ground-truth label once the rollout leaves the demo states.

Once an expert is chosen, everything follows `eval_libero_single`: each expert is loaded with its
own task config, checkpoint and dataset stats; the episode loop, seed, initial states, replan
cadence and gripper handling are the same. A model name as driver (`--drivers Depth`) runs that
expert alone, which checks this harness against the baseline eval episode by episode.

Usage::

    # one worker per GPU; workers claim (driver, suite, task) items from a shared queue in <dir>
    CUDA_VISIBLE_DEVICES=0 python experiments/libero/eval_router_libero.py --worker 0 --out <dir>
    python experiments/libero/eval_router_libero.py --summarize --out <dir>
    # pipeline check: Depth alone on a few tasks, compared with the Depth baseline (which used seed 44)
    python experiments/libero/eval_router_libero.py --drivers Depth --seed 44 --suites libero_goal --tasks 3 9 --out <dir>
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

project_root = Path(__file__).resolve().parents[2]
for p in (project_root, project_root / "experiments" / "libero", project_root / "notebooks"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from experiments.libero.eval_action_agreement import compose_eval_cfg  # noqa: E402
from experiments.libero.eval_infer import validate_eval_infer_model  # noqa: E402
from experiments.libero.eval_libero_single import (  # noqa: E402
    _get_max_steps,
    _load_model_checkpoint,
    _mixed_precision_to_model_dtype,
    _obs_to_model_input,
    _predict_action_chunk,
)
from experiments.libero.libero_utils import (  # noqa: E402
    LIBERO_ENV_RESOLUTION,
    get_libero_dummy_action,
    get_libero_env,
)
from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor  # noqa: E402
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json  # noqa: E402
from fastwam.utils.pytorch_utils import set_global_seed  # noqa: E402
from hydra.utils import instantiate  # noqa: E402
from libero.libero import benchmark, get_libero_path  # noqa: E402
from repr_analysis import JOINT_UNIMODAL  # noqa: E402
from train_modality_router import MODELS, ModalityRouter  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

LABEL_DIR = project_root / "notebooks" / "outputs" / "modality_action_labels"
LABELS = LABEL_DIR / "labels_epall_stride4_s0-1-2-3_grip0.2269.pt"
ROUTERS = LABEL_DIR / "routers.pt"
ASSETS = LABEL_DIR / "router_eval_assets.pt"
ROUTER_NAME = "router soft CE (all feats)"
SELECTORS = ("router", "gt_knn")
DRIVERS = SELECTORS + MODELS  # a model name drives with that expert alone (pipeline check / control)
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
BASELINES = {
    "MM (RGB+Depth+Flow)": "libero_joint_2cam224_1e-4_rgb_depth_flow",
    "RGB": "libero_joint_2cam224_1e-4",
    "Depth": "libero_joint_2cam224_1e-4_depth",
    "Flow": "libero_joint_2cam224_1e-4_flow",
}


def norm_task(text: str) -> str:
    return " ".join(text.lower().strip().rstrip(".").split())


def build_assets() -> dict[str, Any]:
    """Per-task instruction features and kNN index, plus the router, in one small file."""
    from modality_action_labels import _load, _tag

    d = torch.load(LABELS, map_location="cpu", weights_only=False)
    recs, _ = _load("RGB", _tag(0, 8, 4, (0, 1, 2, 3)))
    tasks = np.array([norm_task((recs[k]["prompt"] or "").split("instruction:")[-1]) for k in d["keys"]])
    r0, pr = d["r0_pool"].astype(np.float32), d["proprio"].astype(np.float32)
    knn_norm = {"r0_pool": (r0.mean(0), r0.std(0) + 1e-6), "proprio": (pr.mean(0), pr.std(0) + 1e-6)}
    per_task = {}
    for t in np.unique(tasks):
        sel = tasks == t
        per_task[t] = {
            "text_mean": d["text_mean"][sel][0].astype(np.float32),
            "r0_pool": r0[sel], "proprio": pr[sel], "label": d["label"][sel].astype(np.int64),
        }
    routers = torch.load(ROUTERS, map_location="cpu", weights_only=False)
    assets = {"tasks": per_task, "knn_norm": knn_norm, "router": routers["routers"][ROUTER_NAME],
              "router_name": ROUTER_NAME, "feature_norm": routers["feature_norm"], "labels": str(LABELS)}
    torch.save(assets, ASSETS)
    logger.info("built %s (%d tasks)", ASSETS, len(per_task))
    return assets


def load_eval_assets(router_name: str | None = None, routers_path: str | Path | None = None) -> dict[str, Any]:
    """kNN index from the cached assets file; weights from ``routers_path`` (default routers.pt)."""
    name = router_name or ROUTER_NAME
    path = Path(routers_path) if routers_path else ROUTERS
    assets = torch.load(ASSETS, map_location="cpu", weights_only=False) if ASSETS.exists() else build_assets()
    routers = torch.load(path, map_location="cpu", weights_only=False)
    if name not in routers["routers"]:
        raise KeyError(f"unknown router {name!r} in {path}; have {sorted(routers['routers'])}")
    assets = dict(assets)
    assets["router"] = routers["routers"][name]
    assets["router_name"] = name
    assets["feature_norm"] = routers["feature_norm"]
    logger.info("using %s from %s (not rewriting %s)", name, path, ASSETS)
    return assets


class Selector:
    def __init__(self, assets: dict[str, Any], rgb_model, device: str):
        self.assets, self.rgb_model, self.device = assets, rgb_model, device
        spec = assets["router"]
        self.router = ModalityRouter(spec["dims"]).to(device).eval()
        self.router.load_state_dict(spec["state_dict"])
        self.groups = spec["groups"]
        self.norm = {g: tuple(torch.as_tensor(np.asarray(v), dtype=torch.float32, device=device) for v in mv)
                     for g, mv in assets["feature_norm"].items()}
        kn = assets["knn_norm"]
        self.knn = {}
        for t, v in assets["tasks"].items():
            self.knn[t] = (((v["r0_pool"] - kn["r0_pool"][0]) / kn["r0_pool"][1]) / np.sqrt(v["r0_pool"].shape[1]),
                           ((v["proprio"] - kn["proprio"][0]) / kn["proprio"][1]) / np.sqrt(v["proprio"].shape[1]),
                           v["label"])

    @torch.no_grad()
    def features(self, obs, cfg, processor, input_w: int, input_h: int) -> dict[str, np.ndarray]:
        image, proprio, _ = _obs_to_model_input(obs, cfg=cfg, processor=processor, width=input_w, height=input_h,
                                                device=self.device, dtype=self.rgb_model.torch_dtype)
        z = self.rgb_model._encode_input_image_latents_tensor(input_image=image)  # [1, C, 1, h, w]
        r0 = F.adaptive_avg_pool2d(z[:, :, 0].float(), 2).flatten(1)[0]
        return {"r0_pool": r0.cpu().numpy(), "proprio": proprio.reshape(-1).float().cpu().numpy()}

    @torch.no_grad()
    def choose(self, driver: str, feats: dict[str, np.ndarray], task: str) -> tuple[int, np.ndarray]:
        if driver == "router":
            x = dict(feats, text_mean=self.assets["tasks"][task]["text_mean"])
            inp = {g: (torch.as_tensor(x[g], dtype=torch.float32, device=self.device)[None] - self.norm[g][0])
                   / self.norm[g][1] for g in self.groups}
            prob = torch.softmax(self.router(inp), -1)[0].cpu().numpy()
            return int(prob.argmax()), prob
        if driver == "gt_knn":
            kn = self.assets["knn_norm"]
            q_r0 = ((feats["r0_pool"] - kn["r0_pool"][0]) / kn["r0_pool"][1]) / np.sqrt(feats["r0_pool"].shape[0])
            q_pr = ((feats["proprio"] - kn["proprio"][0]) / kn["proprio"][1]) / np.sqrt(feats["proprio"].shape[0])
            r0, pr, lab = self.knn[task]
            dist = ((r0 - q_r0) ** 2).sum(1) + ((pr - q_pr) ** 2).sum(1)
            j = int(dist.argmin())
            return int(lab[j]), np.array([float(dist[j])])
        raise ValueError(driver)


def load_expert(name: str, seed: int, device: str) -> dict[str, Any]:
    """One unimodal expert exactly as `eval_libero_single.eval_single_process` loads it: its own task
    config, checkpoint, precision, and its own run's dataset_stats.json for the processor."""
    spec = JOINT_UNIMODAL[name]
    cfg = compose_eval_cfg(spec["task"])
    cfg.seed = int(seed)
    cfg.ckpt = str(spec["ckpt"])
    model = instantiate(cfg.model, model_dtype=_mixed_precision_to_model_dtype(cfg.get("mixed_precision", "bf16")),
                        device=device)
    _load_model_checkpoint(model, cfg.ckpt)
    model = model.to(device).eval()
    validate_eval_infer_model(model, action_attend_mode=cfg.EVALUATION.get("action_attend_mode", None))
    stats_path = Path(spec["run_dir"]) / "dataset_stats.json"
    processor: FastWAMProcessor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(load_dataset_stats_from_json(str(stats_path)))
    horizon_cfg = cfg.EVALUATION.get("action_horizon", None)
    action_horizon = int(cfg.data.train.num_frames) - 1 if horizon_cfg is None else int(horizon_cfg)
    input_h, input_w = (int(v) for v in cfg.data.train.get("video_size", [224, 224]))
    logger.info("loaded %s: %s (stats %s)", name, Path(cfg.ckpt).name, stats_path)
    return {"model": model, "cfg": cfg, "processor": processor, "action_horizon": action_horizon,
            "input_w": input_w, "input_h": input_h}


def run_episode(env, initial_state, task_description: str, *, driver: str, selector: Selector | None,
                experts: dict[str, dict[str, Any]], device: str, max_steps: int,
                selector_task: str | None = None) -> dict[str, Any]:
    """`eval_libero_single.run_single_episode`, except the expert is chosen at each replan.

    ``selector_task`` is the assets key for the router / GT lookup (original LIBERO
    instruction). The expert still conditions on ``task_description`` (plus language
    when evaluating LIBERO-plus).
    """
    base = experts["RGB"]
    replan_steps = int(base["cfg"].EVALUATION.get("replan_steps", 5))
    num_steps_wait = int(base["cfg"].EVALUATION.get("num_steps_wait", 5))
    lookup = selector_task if selector_task is not None else norm_task(task_description)
    env.reset()
    obs = env.set_init_state(initial_state)
    picks, extra, pending = [], [], []
    t, done = 0, False
    while t < max_steps + num_steps_wait:
        if t < num_steps_wait:
            obs, _, done, _ = env.step(get_libero_dummy_action())
            t += 1
            continue
        if not pending:
            if driver in MODELS:
                m, info = MODELS.index(driver), np.zeros(0)
            else:
                feats = selector.features(obs, base["cfg"], base["processor"], base["input_w"], base["input_h"])
                m, info = selector.choose(driver, feats, lookup)
            picks.append(m)
            extra.append(info.tolist())
            ex = experts[MODELS[m]]
            chunk, _imgs, _ = _predict_action_chunk(
                obs=obs, task_description=task_description, model=ex["model"], processor=ex["processor"],
                cfg=ex["cfg"], action_horizon=ex["action_horizon"], input_w=ex["input_w"], input_h=ex["input_h"],
                model_device=device,
            )
            pending = chunk[:replan_steps].tolist()
        obs, _, done, _ = env.step(pending.pop(0))
        if done:
            break
        t += 1
    return {"success": bool(done), "steps": t, "picks": picks, "info": extra}


def work_items(drivers: tuple[str, ...], suites: tuple[str, ...]) -> list[tuple[str, str, int]]:
    """(driver, suite, task) items, libero_10 first: its episodes are ~2x longer, so starting them
    early keeps the workers finishing together."""
    order = tuple(s for s in ("libero_10",) + SUITES if s in suites)
    order = tuple(dict.fromkeys(order))
    items = []
    for suite in order:
        n_tasks = benchmark.get_benchmark_dict()[suite]().n_tasks
        items += [(driver, suite, tid) for tid in range(n_tasks) for driver in drivers]
    return items


def _claim(out_dir: Path, item: tuple[str, str, int]) -> bool:
    """Atomically claim an item so concurrent workers never run the same task twice."""
    driver, suite, task_id = item
    if (out_dir / driver / f"{suite}_task{task_id}.json").exists():
        return False
    claim = out_dir / "claims" / f"{driver}_{suite}_task{task_id}"
    claim.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.close(os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        return True
    except FileExistsError:
        return False


def run_worker(args) -> None:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_global_seed(int(args.seed), get_worker_init_fn=False)
    drivers = tuple(args.drivers)
    items = work_items(drivers, tuple(args.suites))
    if args.tasks:
        items = [it for it in items if it[2] in args.tasks]
    if args.max_items:
        items = items[:args.max_items]
    todo = [it for it in items if not (out_dir / it[0] / f"{it[1]}_task{it[2]}.json").exists()]
    logger.info("worker %d: %d of %d items not done yet (seed %d)", args.worker, len(todo), len(items), args.seed)
    if not todo:
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    experts = {name: load_expert(name, args.seed, device) for name in MODELS}
    selector = None
    if any(d in SELECTORS for d in drivers):
        assets = load_eval_assets(getattr(args, "router_name", None), getattr(args, "routers", None))
        logger.info("selector router=%s", assets.get("router_name"))
        selector = Selector(assets, experts["RGB"]["model"], device)
    base_cfg = experts["RGB"]["cfg"]
    num_trials = int(args.num_trials or base_cfg.EVALUATION.num_trials)

    for item in todo:
        if not _claim(out_dir, item):
            continue
        driver, suite, task_id = item
        t0 = time.time()
        task = benchmark.get_benchmark_dict()[suite]().get_task(task_id)
        init_states = torch.load(Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file,
                                 weights_only=False)
        while len(init_states) < num_trials:
            init_states.extend(init_states[: num_trials - len(init_states)])
        env, task_description = get_libero_env(task, LIBERO_ENV_RESOLUTION, base_cfg.get("seed"))
        if selector is not None and norm_task(task_description) not in selector.assets["tasks"]:
            raise KeyError(f"no demo labels for task {task_description!r}")
        episodes = []
        for trial in range(num_trials):
            episodes.append(run_episode(env, init_states[trial], task_description, driver=driver, selector=selector,
                                        experts=experts, device=device, max_steps=_get_max_steps(suite)))
        env.close()
        n_ok = sum(e["success"] for e in episodes)
        picks = np.bincount(np.concatenate([e["picks"] for e in episodes]).astype(int), minlength=3)
        result = {"driver": driver, "suite": suite, "task_id": task_id, "task_description": task_description,
                  "seed": int(args.seed), "successes": n_ok, "total_episodes": len(episodes),
                  "success_rate": 100.0 * n_ok / len(episodes),
                  "success_episodes": [i for i, e in enumerate(episodes) if e["success"]],
                  "failure_episodes": [i for i, e in enumerate(episodes) if not e["success"]],
                  "picks": dict(zip(MODELS, picks.tolist())), "duration": time.time() - t0, "episodes": episodes,
                  "router_name": None if selector is None else selector.assets.get("router_name")}
        (out_dir / driver).mkdir(exist_ok=True)
        with open(out_dir / driver / f"{suite}_task{task_id}.json", "w") as fh:
            json.dump(result, fh)
        logger.info("%s %s task%d: %d/%d, picks R/D/F %s, %.0fs", driver, suite, task_id, n_ok, len(episodes),
                    picks.tolist(), time.time() - t0)


def _baseline(task_name: str) -> tuple[dict[str, Any], Path] | None:
    """Latest full-protocol (2000-episode) summary for a baseline eval, and its run dir."""
    runs = sorted((project_root / "evaluate_results" / "libero" / task_name).glob("*/summary.json"), reverse=True)
    for path in runs:
        s = json.load(open(path))
        if sum(v["total_trials"] for v in s["suite_stats"].values()) >= 2000:
            return s, path.parent
    return None


def _baseline_seed(run_dir: Path) -> int | None:
    for line in open(run_dir / "manager_config.yaml"):
        if line.startswith("seed:"):
            return int(line.split(":")[1])
    return None


def pipeline_check(out_dir: Path) -> None:
    """Fixed-model control runs vs that model's baseline eval, episode by episode."""
    names = {"RGB": BASELINES["RGB"], "Depth": BASELINES["Depth"], "Flow": BASELINES["Flow"]}
    for model, task_name in names.items():
        res = _driver_task_results(out_dir, model)
        found = _baseline(task_name)
        if not res or found is None:
            continue
        s, run_dir = found
        print(f"\npipeline check: {model} driver here (seed {res[0].get('seed')}) vs baseline "
              f"{run_dir.name} (seed {_baseline_seed(run_dir)})")
        same = total = 0
        for r in res:
            b = s["task_results"].get(f"{r['suite']}_{r['task_id']}")
            if b is None:
                continue
            files = list((run_dir / r["suite"]).glob(f"gpu*_task{r['task_id']}_results.json"))
            b_ok = set(json.load(open(files[0]))["success_episodes"]) if files else None
            ours = set(r["success_episodes"])
            match = "" if b_ok is None else f", same outcome on {sum((i in ours) == (i in b_ok) for i in range(r['total_episodes']))}/{r['total_episodes']} episodes"
            if b_ok is not None:
                same += sum((i in ours) == (i in b_ok) for i in range(r["total_episodes"]))
                total += r["total_episodes"]
            print(f"  {r['suite']} task{r['task_id']}: here {r['successes']}/{r['total_episodes']}, "
                  f"baseline {b['successes']}/{b['total_episodes']}{match}")
        if total:
            print(f"  episode-level agreement: {same}/{total}")


def _driver_task_results(out_dir: Path, driver: str) -> list[dict[str, Any]]:
    return [json.load(open(p)) for p in sorted((out_dir / driver).glob("*_task*.json"))]


def write_libero_style_csvs(out_dir: Path) -> None:
    """Write summary.csv / task_success_rates.csv / summary.json in the same layout as
    experiments/libero/summarize_results.py, one set per driver plus a combined comparison."""
    import csv as csv_mod

    combined_rows = []
    for driver in DRIVERS:
        res = _driver_task_results(out_dir, driver)
        if not res:
            continue
        dest = out_dir / driver
        dest.mkdir(exist_ok=True)
        task_results = {}
        suite_stats = {s: {"total_tasks": 0, "total_trials": 0, "total_successes": 0,
                           "total_time": 0.0, "max_time": 0.0} for s in SUITES}
        for r in res:
            key = f"{r['suite']}_{r['task_id']}"
            n = int(r["total_episodes"])
            ok = int(r["successes"])
            dur = float(r.get("duration") or 0.0)
            task_results[key] = {
                "success_rate": 100.0 * ok / max(n, 1),
                "duration": dur,
                "total_episodes": n,
                "successes": ok,
                "task_description": r.get("task_description", ""),
                "success_episodes": r.get("success_episodes", []),
                "failure_episodes": r.get("failure_episodes", []),
                "picks": r.get("picks", {}),
            }
            st = suite_stats[r["suite"]]
            st["total_tasks"] += 1
            st["total_trials"] += n
            st["total_successes"] += ok
            st["total_time"] += dur
            st["max_time"] = max(st["max_time"], dur)
            combined_rows.append({
                "Driver": driver, "Task": key, "Description": r.get("task_description", ""),
                "Success Rate (%)": f"{100.0 * ok / max(n, 1):.2f}",
                "Successes": ok, "Episodes": n,
            })

        with open(dest / "task_success_rates.csv", "w", newline="") as fh:
            w = csv_mod.writer(fh)
            w.writerow(["Task", "Description", "Success Rate (%)"])
            for suite in SUITES:
                for tid in range(10):
                    key = f"{suite}_{tid}"
                    if key not in task_results:
                        continue
                    t = task_results[key]
                    w.writerow([key, t["task_description"], f"{t['success_rate']:.2f}"])

        overall_ok = sum(v["total_successes"] for v in suite_stats.values())
        overall_n = sum(v["total_trials"] for v in suite_stats.values())
        overall_time = sum(v["total_time"] for v in suite_stats.values())
        overall_tasks = sum(v["total_tasks"] for v in suite_stats.values())
        title = f"{driver} seed={res[0].get('seed', '')}"
        cols = list(SUITES) + ["Overall"]
        rates = [f"{100 * suite_stats[s]['total_successes'] / max(suite_stats[s]['total_trials'], 1):.2f}"
                 for s in SUITES] + [f"{100 * overall_ok / max(overall_n, 1):.2f}"]
        avgs = [f"{suite_stats[s]['total_time'] / max(suite_stats[s]['total_tasks'], 1):.2f}" for s in SUITES]
        avgs.append(f"{overall_time / max(overall_tasks, 1):.2f}")
        maxs = [f"{suite_stats[s]['max_time']:.2f}" for s in SUITES] + [
            f"{max(v['max_time'] for v in suite_stats.values()):.2f}"]
        with open(dest / "summary.csv", "w", newline="") as fh:
            w = csv_mod.writer(fh)
            w.writerow([title])
            w.writerow([""] + cols)
            w.writerow(["Success Rate (%)"] + rates)
            w.writerow(["Average Time (s)"] + avgs)
            w.writerow(["Max Time (s)"] + maxs)

        summary = {
            "run_id": out_dir.name, "driver": driver, "seed": res[0].get("seed"),
            "config": "eval_router_libero",
            "suite_stats": suite_stats, "task_results": task_results,
            "overall": {
                "average_success_rate": 100.0 * overall_ok / max(overall_n, 1),
                "total_time": overall_time,
                "average_task_time": overall_time / max(overall_tasks, 1),
            },
        }
        with open(dest / "summary.json", "w") as fh:
            json.dump(summary, fh, indent=2)
        logger.info("wrote %s", dest / "task_success_rates.csv")

    if combined_rows:
        with open(out_dir / "task_success_rates.csv", "w", newline="") as fh:
            w = csv_mod.DictWriter(fh, fieldnames=list(combined_rows[0]))
            w.writeheader()
            w.writerows(combined_rows)
        # wide comparison: one row per task, one success column per driver
        by_task = {}
        for row in combined_rows:
            by_task.setdefault(row["Task"], {"Task": row["Task"], "Description": row["Description"]})
            by_task[row["Task"]][row["Driver"]] = row["Success Rate (%)"]
        drivers_present = [d for d in DRIVERS if any(r["Driver"] == d for r in combined_rows)]
        with open(out_dir / "task_success_comparison.csv", "w", newline="") as fh:
            w = csv_mod.DictWriter(fh, fieldnames=["Task", "Description"] + drivers_present)
            w.writeheader()
            for key in sorted(by_task, key=lambda k: (SUITES.index(k.rsplit("_", 1)[0])
                                                      if k.rsplit("_", 1)[0] in SUITES else 99,
                                                      int(k.rsplit("_", 1)[1]))):
                w.writerow(by_task[key])
        logger.info("wrote %s and %s", out_dir / "task_success_rates.csv",
                    out_dir / "task_success_comparison.csv")


def summarize(out_dir: Path) -> None:
    write_libero_style_csvs(out_dir)
    rows = {}
    for name, task_name in BASELINES.items():
        found = _baseline(task_name)
        if found:
            s, run_dir = found
            rows[f"{name} [seed {_baseline_seed(run_dir)}]"] = {
                suite: (v["total_successes"], v["total_trials"]) for suite, v in s["suite_stats"].items()}
    picks = {}
    for driver in DRIVERS:
        res = _driver_task_results(out_dir, driver)
        if not res:
            continue
        label = {"router": "router", "gt_knn": "GT label (nearest demo)"}.get(driver, f"{driver} (this harness)")
        name = f"{label} [seed {res[0].get('seed', '?')}]"
        rows[name] = {s: (sum(r["successes"] for r in res if r["suite"] == s),
                          sum(r["total_episodes"] for r in res if r["suite"] == s)) for s in SUITES}
        picks[name] = np.sum([[r["picks"][m] for m in MODELS] for r in res], 0)

    print(f"\n{'success rate %':34s}" + "".join(f"{s.replace('libero_', ''):>10s}" for s in SUITES)
          + f"{'overall':>10s}{'episodes':>10s}   picks R/D/F")
    for name, r in rows.items():
        ok = sum(v[0] for v in r.values())
        n = sum(v[1] for v in r.values())
        cells = "".join(f"{100 * r[s][0] / r[s][1]:10.1f}" if r.get(s, (0, 0))[1] else f"{'-':>10s}" for s in SUITES)
        pk = picks.get(name)
        pk_s = "" if pk is None else "   " + "/".join(f"{100 * x / pk.sum():.0f}%" for x in pk)
        print(f"{name:34s}{cells}{100 * ok / max(n, 1):10.2f}{n:10d}{pk_s}")
    print("\nstd. error of an overall rate near 98% on 2000 episodes: ~0.3 points")
    pipeline_check(out_dir)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--worker", type=int, default=0, help="only used in log lines")
    p.add_argument("--max-items", type=int, default=0, help="only the first N (driver, suite, task) items; smoke")
    p.add_argument("--num-trials", type=int, default=0, help="0 = EVALUATION.num_trials from the config (50)")
    p.add_argument("--seed", type=int, default=42,
                   help="the RGB baseline eval used 42, MM 43, Depth and Flow 44")
    p.add_argument("--drivers", nargs="+", default=list(SELECTORS), choices=DRIVERS)
    p.add_argument("--suites", nargs="+", default=list(SUITES), choices=SUITES)
    p.add_argument("--tasks", type=int, nargs="*", default=None, help="only these task ids (all suites)")
    p.add_argument("--router-name", type=str, default=ROUTER_NAME,
                   help="key in the --routers file")
    p.add_argument("--routers", type=str, default=str(ROUTERS),
                   help="e.g. notebooks/outputs/modality_action_labels/routers_kl1.pt")
    p.add_argument("--build-assets", action="store_true")
    p.add_argument("--summarize", action="store_true")
    args = p.parse_args()
    os.chdir(project_root)
    if args.build_assets:
        build_assets()
        return
    if args.out is None:
        args.out = str(project_root / "evaluate_results" / "router_eval" / datetime.now().strftime("%Y%m%d_%H%M%S"))
    if args.summarize:
        summarize(Path(args.out))
    else:
        run_worker(args)


if __name__ == "__main__":
    main()
