"""Closed-loop LIBERO-plus eval with the same router / GT-knn drivers as eval_router_libero.py.

After a modality is chosen, inference matches ``eval_libero_plus_single``: plus BDDL
instruction, plus init-state remapping, 1 trial per perturbed task, no videos.

Router and GT lookup use the *original* LIBERO skill (``plus_base_task_name``), because
labels were collected on unperturbed demos. Experts still see the plus instruction.

Usage::

    CUDA_VISIBLE_DEVICES=0 python experiments/libero_plus/eval_router_libero_plus.py --worker 0 --out <dir>
    python experiments/libero_plus/eval_router_libero_plus.py --summarize --out <dir>
    # smoke: 5 variants per (suite, category, base task)
    python experiments/libero_plus/eval_router_libero_plus.py --max-tasks-per-category 5 --out <dir>
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import sys
import time
from collections import defaultdict
from contextlib import redirect_stdout
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from experiments.libero_plus.libero_plus_utils import (  # noqa: E402
    LEADERBOARD_CATEGORIES,
    bootstrap_libero_plus,
    load_task_classification,
    load_task_init_states,
    lookup_task_classification,
    plus_base_task_name,
    plus_policy_instruction,
    resolve_perturbation_categories,
)

bootstrap_libero_plus(project_root)

for p in (project_root / "experiments" / "libero", project_root / "notebooks"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from experiments.libero.eval_libero_single import _get_max_steps  # noqa: E402
from experiments.libero.eval_router_libero import (  # noqa: E402
    ASSETS,
    DRIVERS,
    MODELS,
    SELECTORS,
    SUITES,
    Selector,
    _claim,
    build_assets,
    load_expert,
    load_eval_assets,
    norm_task,
    run_episode,
)
from experiments.libero.libero_utils import LIBERO_ENV_RESOLUTION, get_libero_env  # noqa: E402
from fastwam.utils.pytorch_utils import set_global_seed  # noqa: E402
from libero.libero import benchmark  # noqa: E402
import torch  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def plus_work_items(
    drivers: tuple[str, ...],
    suites: tuple[str, ...],
    perturbation_categories: frozenset[str] | None,
    max_tasks_per_category: int | None,
) -> list[tuple[str, str, int]]:
    order = tuple(s for s in ("libero_10",) + SUITES if s in suites)
    order = tuple(dict.fromkeys(order))
    items: list[tuple[str, str, int]] = []
    cap = max_tasks_per_category
    counts: dict[tuple, int] = {}
    bench = benchmark.get_benchmark_dict()
    for suite in order:
        with redirect_stdout(StringIO()):
            task_suite = bench[suite]()
        for task_id in range(int(task_suite.n_tasks)):
            task = task_suite.get_task(task_id)
            classification = lookup_task_classification(suite, task.name)
            category = None if classification is None else classification.get("category")
            if perturbation_categories is not None and category not in perturbation_categories:
                continue
            key = (suite, category, plus_base_task_name(task.name))
            if cap is not None and counts.get(key, 0) >= cap:
                continue
            counts[key] = counts.get(key, 0) + 1
            items += [(driver, suite, task_id) for driver in drivers]
    return items


_LEVEL_SAMPLE_RE = re.compile(r"_level\d+_sample\d+$")


def original_instruction(base_name: str, known: set[str], slugs: list[tuple[str, str]] | None = None) -> str:
    """Map a plus skill filename onto the original LIBERO instruction in router assets.

    Plus BDDL language is often not the vanilla instruction (especially Language
    perturbations). Filenames are ``[SCENE_]instruction_with_underscores[_levelN_sampleM]``.
    """
    slug = _LEVEL_SAMPLE_RE.sub("", base_name)
    cand = norm_task(slug.replace("_", " "))
    if cand in known:
        return cand
    if slugs is None:
        slugs = sorted(((inst, inst.replace(" ", "_")) for inst in known), key=lambda kv: -len(kv[1]))
    for inst, inst_slug in slugs:
        if slug == inst_slug or slug.startswith(inst_slug + "_") or slug.endswith("_" + inst_slug):
            return inst
    raise KeyError(f"no router labels for plus skill {base_name!r} (tried {cand!r})")


def base_instruction_map(assets: dict[str, Any]) -> dict[str, str]:
    """plus_base_task_name -> original instruction key in router assets."""
    known = set(assets["tasks"])
    slugs = sorted(((inst, inst.replace(" ", "_")) for inst in known), key=lambda kv: -len(kv[1]))
    mapping: dict[str, str] = {}
    index = load_task_classification(str(project_root))
    for suite in SUITES:
        for name in index.get(suite, {}):
            base = plus_base_task_name(name)
            if base in mapping:
                continue
            mapping[base] = original_instruction(base, known, slugs)
    unused = known - set(mapping.values())
    if unused:
        raise KeyError(f"{len(unused)} original tasks never appear in plus: {sorted(unused)[:5]}")
    logger.info("mapped %d plus base skills onto original router tasks", len(mapping))
    return mapping


def write_plus_csvs(out_dir: Path) -> None:
    for driver in DRIVERS:
        res = [json.load(open(p)) for p in sorted((out_dir / driver).glob("*_task*.json"))]
        if not res:
            continue
        dest = out_dir / driver
        dest.mkdir(exist_ok=True)
        cat_stats: dict[str, dict[str, int]] = defaultdict(
            lambda: {"total_tasks": 0, "total_trials": 0, "total_successes": 0}
        )
        suite_stats: dict[str, dict[str, float]] = defaultdict(
            lambda: {"total_tasks": 0, "total_trials": 0, "total_successes": 0, "total_time": 0.0}
        )
        with open(dest / "task_success_rates.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["Task", "Description", "Category", "Success Rate (%)"])
            for r in sorted(res, key=lambda x: (SUITES.index(x["suite"]) if x["suite"] in SUITES else 99,
                                                int(x["task_id"]))):
                n = int(r["total_episodes"])
                ok = int(r["successes"])
                cat = r.get("perturbation_category") or ""
                w.writerow([f"{r['suite']}_{r['task_id']}", r.get("task_description", ""), cat,
                            f"{100.0 * ok / max(n, 1):.2f}"])
                st = suite_stats[r["suite"]]
                st["total_tasks"] += 1
                st["total_trials"] += n
                st["total_successes"] += ok
                st["total_time"] += float(r.get("duration") or 0.0)
                if cat:
                    c = cat_stats[cat]
                    c["total_tasks"] += 1
                    c["total_trials"] += n
                    c["total_successes"] += ok

        overall_ok = sum(int(v["total_successes"]) for v in suite_stats.values())
        overall_n = sum(int(v["total_trials"]) for v in suite_stats.values())
        cols = [s for s in SUITES if s in suite_stats] + ["Overall"]
        rates = [f"{100 * suite_stats[s]['total_successes'] / max(suite_stats[s]['total_trials'], 1):.2f}"
                 for s in SUITES if s in suite_stats]
        rates.append(f"{100 * overall_ok / max(overall_n, 1):.2f}")
        with open(dest / "summary.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow([f"{driver} seed={res[0].get('seed', '')}"])
            w.writerow([""] + cols)
            w.writerow(["Success Rate (%)"] + rates)

        header = [short for _, short in LEADERBOARD_CATEGORIES] + ["Total"]
        values = []
        tot_ok = tot_n = 0
        for full, short in LEADERBOARD_CATEGORIES:
            st = cat_stats.get(full, {"total_trials": 0, "total_successes": 0})
            if st["total_trials"] > 0:
                values.append(f"{100 * st['total_successes'] / st['total_trials']:.1f}")
                tot_ok += st["total_successes"]
                tot_n += st["total_trials"]
            else:
                values.append("N/A")
        values.append(f"{100 * tot_ok / tot_n:.1f}" if tot_n else "N/A")
        with open(dest / "plus_leaderboard.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(header)
            w.writerow(values)
        logger.info("wrote %s (%d tasks, overall %s)", dest / "plus_leaderboard.csv", len(res), values[-1])


def run_worker(args) -> None:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_global_seed(int(args.seed), get_worker_init_fn=False)
    cats = resolve_perturbation_categories(args.perturbation_categories)
    drivers = tuple(args.drivers)
    items = plus_work_items(drivers, tuple(args.suites), cats, args.max_tasks_per_category or None)
    if args.max_items:
        items = items[: args.max_items]
    todo = [it for it in items if not (out_dir / it[0] / f"{it[1]}_task{it[2]}.json").exists()]
    logger.info("worker %d: %d of %d plus items not done (seed %d, cats=%s, cap=%s)",
                args.worker, len(todo), len(items), args.seed,
                sorted(cats) if cats else "all", args.max_tasks_per_category)
    if not todo:
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    selector = None
    base_map: dict[str, str] = {}
    assets = None
    if any(d in SELECTORS for d in drivers):
        assets = load_eval_assets(getattr(args, "router_name", None), getattr(args, "routers", None))
        logger.info("selector router=%s", assets.get("router_name"))
        base_map = base_instruction_map(assets)
    experts = {name: load_expert(name, args.seed, device) for name in MODELS}
    if assets is not None:
        selector = Selector(assets, experts["RGB"]["model"], device)
    num_trials = int(args.num_trials)
    bench = benchmark.get_benchmark_dict()

    for item in todo:
        if not _claim(out_dir, item):
            continue
        driver, suite, task_id = item
        t0 = time.time()
        with redirect_stdout(StringIO()):
            task_suite = bench[suite]()
        task = task_suite.get_task(task_id)
        init_states = load_task_init_states(task_suite, task_id)
        while len(init_states) < num_trials:
            init_states.extend(init_states[: num_trials - len(init_states)])
        env, raw = get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
        task_description = plus_policy_instruction(task, raw)
        classification = lookup_task_classification(suite, task.name) or {}
        selector_task = None
        if selector is not None:
            selector_task = base_map[plus_base_task_name(task.name)]
        episodes = []
        for trial in range(num_trials):
            episodes.append(run_episode(
                env, init_states[trial], task_description, driver=driver, selector=selector,
                experts=experts, device=device, max_steps=_get_max_steps(suite),
                selector_task=selector_task,
            ))
        env.close()
        n_ok = sum(e["success"] for e in episodes)
        picks = np.bincount(np.concatenate([e["picks"] for e in episodes]).astype(int), minlength=3)
        result = {
            "driver": driver, "suite": suite, "task_id": task_id, "task_name": task.name,
            "task_description": task_description, "task_description_raw": raw,
            "perturbation_category": classification.get("category"),
            "difficulty_level": classification.get("difficulty_level"),
            "seed": int(args.seed), "successes": n_ok, "total_episodes": len(episodes),
            "success_rate": 100.0 * n_ok / len(episodes),
            "success_episodes": [i for i, e in enumerate(episodes) if e["success"]],
            "failure_episodes": [i for i, e in enumerate(episodes) if not e["success"]],
            "picks": dict(zip(MODELS, picks.tolist())), "duration": time.time() - t0,
            "episodes": episodes,
        }
        (out_dir / driver).mkdir(exist_ok=True)
        with open(out_dir / driver / f"{suite}_task{task_id}.json", "w") as fh:
            json.dump(result, fh)
        logger.info("%s %s task%d %s: %d/%d, picks R/D/F %s, %.0fs",
                    driver, suite, task_id, classification.get("category"), n_ok, len(episodes),
                    picks.tolist(), time.time() - t0)


def summarize(out_dir: Path) -> None:
    write_plus_csvs(out_dir)
    for driver in DRIVERS:
        path = out_dir / driver / "plus_leaderboard.csv"
        if path.exists():
            print(f"\n{driver}:")
            print(path.read_text())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--worker", type=int, default=0)
    p.add_argument("--max-items", type=int, default=0)
    p.add_argument("--num-trials", type=int, default=1, help="official plus protocol is 1")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--drivers", nargs="+", default=list(SELECTORS), choices=DRIVERS)
    p.add_argument("--suites", nargs="+", default=list(SUITES), choices=SUITES)
    p.add_argument("--perturbation-categories", nargs="*", default=None,
                   help="Camera Robot Language Light Background Noise Layout; default = all")
    p.add_argument("--max-tasks-per-category", type=int, default=0,
                   help="cap variants per (suite, category, base task); 0 = all")
    p.add_argument("--router-name", type=str, default=None,
                   help="key in --routers; default is the soft-CE all-feats router")
    p.add_argument("--routers", type=str, default=None,
                   help="e.g. notebooks/outputs/modality_action_labels/routers_kl1.pt")
    p.add_argument("--summarize", action="store_true")
    args = p.parse_args()
    os.chdir(project_root)
    if args.out is None:
        args.out = str(project_root / "evaluate_results" / "router_eval_plus" / datetime.now().strftime("%Y%m%d_%H%M%S"))
    if args.summarize:
        summarize(Path(args.out))
    else:
        run_worker(args)


if __name__ == "__main__":
    main()
