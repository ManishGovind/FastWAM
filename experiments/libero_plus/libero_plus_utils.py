"""LIBERO-plus bootstrap and evaluation helpers.

LIBERO-plus is a drop-in replacement for the original ``libero`` package, but
it cannot be pip-installed alongside vanilla LIBERO because both export the
same top-level module name. Evaluation therefore prepends
``third_party/LIBERO-plus`` to ``sys.path`` and points ``LIBERO_CONFIG_PATH``
at a project-local config so original LIBERO eval keeps using ``~/.libero``.
"""

from __future__ import annotations

import json
import os
import re
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

LIBERO_PLUS_ENV = "LIBERO_PLUS"
LIBERO_PLUS_ROOT_NAME = "LIBERO-plus"

# Matches LIBERO-plus perturbation suffixes so we can load the base
# ``.pruned_init`` file (the on-disk name for a variant often does not exist).
# Same mapping as LeRobot's LIBERO-plus init-state loader.
_LIBERO_PERTURBATION_SUFFIX_RE = re.compile(
    r"_(?:language|view|light)_[^.]*|_(?:table|tb)_\d+"
)

LEADERBOARD_CATEGORIES = [
    ("Camera Viewpoints", "Camera"),
    ("Robot Initial States", "Robot"),
    ("Language Instructions", "Language"),
    ("Light Conditions", "Light"),
    ("Background Textures", "Background"),
    ("Sensor Noise", "Noise"),
    ("Objects Layout", "Layout"),
]

_CATEGORY_ALIAS = {
    short.lower(): full for full, short in LEADERBOARD_CATEGORIES
}
_CATEGORY_ALIAS.update({full.lower(): full for full, _ in LEADERBOARD_CATEGORIES})
_CATEGORY_ALIAS.update(
    {
        "view": "Camera Viewpoints",
        "viewpoint": "Camera Viewpoints",
        "cameras": "Camera Viewpoints",
        "sensor": "Sensor Noise",
        "object": "Objects Layout",
        "objects": "Objects Layout",
    }
)


def resolve_perturbation_categories(requested: Any) -> frozenset[str] | None:
    """Map user aliases (Noise, camera, ...) to official LIBERO-plus names.

    Returns None when no filter is requested (evaluate all ~10k tasks).
    """
    if requested is None:
        return None
    if isinstance(requested, str):
        items = [requested]
    else:
        items = list(requested)
    if len(items) == 0:
        return None

    resolved: set[str] = set()
    unknown: list[str] = []
    for raw in items:
        key = str(raw).strip()
        if not key:
            continue
        mapped = _CATEGORY_ALIAS.get(key.lower())
        if mapped is None:
            unknown.append(key)
            continue
        resolved.add(mapped)
    if unknown:
        allowed = ", ".join(short for _, short in LEADERBOARD_CATEGORIES)
        raise ValueError(
            f"Unknown LIBERO-plus perturbation categor(y/ies): {unknown}. "
            f"Use one of: {allowed} (or the full names in task_classification.json)."
        )
    return frozenset(resolved) if resolved else None


def project_root_from_here() -> Path:
    return Path(__file__).resolve().parents[2]


def libero_plus_root(project_root: Path | None = None) -> Path:
    root = project_root or project_root_from_here()
    override = os.environ.get("LIBERO_PLUS_ROOT")
    if override:
        return Path(os.path.expanduser(os.path.expandvars(override))).resolve()
    return (root / "third_party" / LIBERO_PLUS_ROOT_NAME).resolve()


def libero_plus_config_dir(project_root: Path | None = None) -> Path:
    root = project_root or project_root_from_here()
    return (root / ".libero_plus").resolve()


def detect_libero_plus_from_argv(argv: list[str] | None = None) -> bool:
    if os.environ.get(LIBERO_PLUS_ENV) == "1":
        return True
    args = argv if argv is not None else sys.argv
    for i, raw in enumerate(args):
        key = raw.split("=", 1)[0]
        if raw in {"--config-name", "-cn"} and i + 1 < len(args):
            if "libero_plus" in args[i + 1]:
                return True
        if key in {"--config-name", "-cn"} and "=" in raw and "libero_plus" in raw:
            return True
        if "sim_libero_plus" in raw:
            return True
        if raw.lower().lstrip("+") == "evaluation.libero_plus=true":
            return True
    return False


def write_libero_plus_config(project_root: Path | None = None) -> Path:
    """Write a non-interactive LIBERO config that points at the local plus checkout."""
    root = project_root or project_root_from_here()
    plus_root = libero_plus_root(root)
    benchmark_root = plus_root / "libero" / "libero"
    config_dir = libero_plus_config_dir(root)
    config_dir.mkdir(parents=True, exist_ok=True)
    config_file = config_dir / "config.yaml"
    payload = (
        f"assets: {benchmark_root / 'assets'}\n"
        f"bddl_files: {benchmark_root / 'bddl_files'}\n"
        f"benchmark_root: {benchmark_root}\n"
        f"datasets: {plus_root / 'libero' / 'datasets'}\n"
        f"init_states: {benchmark_root / 'init_files'}\n"
    )
    config_file.write_text(payload, encoding="utf-8")
    return config_file


def magick_home(project_root: Path | None = None) -> Path | None:
    """Project-local ImageMagick prefix used by uv-installed ``wand``."""
    override = os.environ.get("MAGICK_HOME")
    if override:
        path = Path(os.path.expanduser(os.path.expandvars(override))).resolve()
        if (path / "lib").exists():
            return path
    root = project_root or project_root_from_here()
    candidate = (root / ".deps" / "imagemagick").resolve()
    if (candidate / "lib").exists():
        return candidate
    return None


def configure_magickwand(project_root: Path | None = None) -> Path | None:
    """Point ``wand`` at MagickWand without leaving the uv virtualenv."""
    home = magick_home(project_root)
    if home is None:
        return None
    os.environ["MAGICK_HOME"] = str(home)
    lib = str(home / "lib")
    current = os.environ.get("LD_LIBRARY_PATH", "")
    parts = [p for p in current.split(os.pathsep) if p]
    if lib not in parts:
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join([lib, *parts])
    return home


def _require_libero_plus_checkout(plus_root: Path) -> None:
    marker = plus_root / "libero" / "libero" / "benchmark" / "__init__.py"
    if not marker.exists():
        raise FileNotFoundError(
            "LIBERO-plus checkout is missing. Run: bash scripts/setup_libero_plus.sh"
        )
    assets_marker = plus_root / "libero" / "libero" / "assets" / "new_objects"
    if not assets_marker.exists():
        raise FileNotFoundError(
            "LIBERO-plus assets are missing (expected "
            f"{assets_marker}). Download them with: "
            "bash scripts/setup_libero_plus.sh --download-assets"
        )


def bootstrap_libero_plus(project_root: Path | None = None) -> Path:
    """Make ``import libero`` resolve to LIBERO-plus for this process."""
    root = (project_root or project_root_from_here()).resolve()
    plus_root = libero_plus_root(root)
    _require_libero_plus_checkout(plus_root)
    if configure_magickwand(root) is None:
        raise FileNotFoundError(
            "MagickWand shared library not found (needed by LIBERO-plus "
            "sensor-noise via the uv-installed wand package). Either:\n"
            "  source scripts/env.sh   # exports MAGICK_HOME=.deps/imagemagick\n"
            "or install the system library: sudo apt-get install -y libmagickwand-dev"
        )

    config_dir = libero_plus_config_dir(root)
    config_file = config_dir / "config.yaml"
    if not config_file.exists():
        write_libero_plus_config(root)

    os.environ[LIBERO_PLUS_ENV] = "1"
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)

    plus_str = str(plus_root)
    if plus_str in sys.path:
        sys.path.remove(plus_str)
    sys.path.insert(0, plus_str)

    stale = [name for name in sys.modules if name == "libero" or name.startswith("libero.")]
    for name in stale:
        del sys.modules[name]
    return plus_root


def maybe_bootstrap_libero_plus(project_root: Path | None = None) -> bool:
    if not detect_libero_plus_from_argv():
        return False
    bootstrap_libero_plus(project_root)
    return True


def is_libero_plus_enabled(cfg: Any | None = None) -> bool:
    if os.environ.get(LIBERO_PLUS_ENV) == "1":
        return True
    if cfg is not None:
        evaluation = cfg.get("EVALUATION") if hasattr(cfg, "get") else None
        if evaluation is not None:
            return bool(evaluation.get("libero_plus", False))
    return False


def worker_env_updates(project_root: Path | None = None) -> dict[str, str]:
    """Environment variables that worker subprocesses need for LIBERO-plus."""
    root = (project_root or project_root_from_here()).resolve()
    plus_root = libero_plus_root(root)
    config_dir = libero_plus_config_dir(root)
    pythonpath = str(plus_root)
    existing = os.environ.get("PYTHONPATH", "")
    if existing:
        pythonpath = pythonpath + os.pathsep + existing
    env = {
        LIBERO_PLUS_ENV: "1",
        "LIBERO_PLUS_ROOT": str(plus_root),
        "LIBERO_CONFIG_PATH": str(config_dir),
        "PYTHONPATH": pythonpath,
    }
    home = configure_magickwand(root)
    if home is not None:
        env["MAGICK_HOME"] = str(home)
        env["LD_LIBRARY_PATH"] = os.environ.get("LD_LIBRARY_PATH", "")
    return env


def classification_json_path(project_root: Path | None = None) -> Path:
    return (
        libero_plus_root(project_root)
        / "libero"
        / "libero"
        / "benchmark"
        / "task_classification.json"
    )


@lru_cache(maxsize=1)
def load_task_classification(project_root: str | None = None) -> dict[str, dict[str, dict[str, Any]]]:
    path = classification_json_path(Path(project_root) if project_root else None)
    if not path.exists():
        raise FileNotFoundError(f"LIBERO-plus task_classification.json not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    index: dict[str, dict[str, dict[str, Any]]] = {}
    for suite_name, tasks in raw.items():
        by_name: dict[str, dict[str, Any]] = {}
        for entry in tasks:
            by_name[str(entry["name"])] = entry
        index[suite_name] = by_name
    return index


def lookup_task_classification(
    suite_name: str,
    task_name: str,
    project_root: Path | None = None,
) -> Optional[dict[str, Any]]:
    try:
        index = load_task_classification(str(project_root) if project_root else None)
    except FileNotFoundError:
        return None
    return index.get(suite_name, {}).get(task_name)


def load_task_init_states(task_suite: Any, task_id: int) -> list[Any]:
    """Load init states with LIBERO-plus suffix remapping and PyTorch 2.6+ safety.

    LIBERO-plus's own ``Benchmark.get_task_init_states`` calls ``torch.load``
    without ``weights_only=False``, which fails on PyTorch 2.6+. The on-disk
    ``.pruned_init`` files also omit perturbation suffixes, so we strip them
    the same way LeRobot does.
    """
    from libero.libero import get_libero_path

    task = task_suite.get_task(task_id)
    filename = Path(task.init_states_file)
    root = Path(get_libero_path("init_states"))

    if "_add_" in filename.name or "_level" in filename.name:
        init_states_path = root / "libero_newobj" / task.problem_folder / filename.name
        init_states = torch.load(init_states_path, weights_only=False)
        if hasattr(init_states, "reshape"):
            init_states = init_states.reshape(1, -1)
    else:
        stripped = _LIBERO_PERTURBATION_SUFFIX_RE.sub("", filename.stem) + filename.suffix
        init_states_path = root / task.problem_folder / stripped
        if not init_states_path.exists():
            init_states_path = root / task.problem_folder / filename.name
        init_states = torch.load(init_states_path, weights_only=False)

    return _init_states_to_list(init_states)


def _init_states_to_list(init_states: Any) -> list[np.ndarray]:
    """Split torch/numpy init dumps into a list of 1D MuJoCo flat states.

    LIBERO-plus ``.init`` / ``.pruned_init`` files are often a 2D ndarray
    ``(N, D)``. Passing that 2D array to ``set_state_from_flattened`` makes
    robosuite treat the first row as ``time``.
    """
    if torch.is_tensor(init_states):
        init_states = init_states.detach().cpu().numpy()
    if isinstance(init_states, np.ndarray):
        if init_states.ndim == 1:
            return [np.asarray(init_states, dtype=np.float64)]
        if init_states.ndim == 2:
            return [np.asarray(row, dtype=np.float64) for row in init_states]
        raise ValueError(
            f"Unexpected init-state ndarray shape {init_states.shape}; expected 1D or 2D."
        )
    if isinstance(init_states, (list, tuple)):
        rows: list[np.ndarray] = []
        for item in init_states:
            if torch.is_tensor(item):
                item = item.detach().cpu().numpy()
            rows.append(np.asarray(item, dtype=np.float64).reshape(-1))
        return rows
    return [np.asarray(init_states, dtype=np.float64).reshape(-1)]
