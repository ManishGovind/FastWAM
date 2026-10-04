"""Helpers shared with paired-agreement evals.

``eval_router_libero`` only needs ``compose_eval_cfg``. Kept separate from
``repr_analysis.compose_task_cfg``, which disables the text encoder for offline
probing; production eval passes a raw ``prompt`` and needs T5 loaded.
"""

from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

project_root = Path(__file__).resolve().parents[2]


def compose_eval_cfg(task: str):
    """Compose the real eval config for a task (text encoder left on)."""
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(version_base="1.3", config_dir=str(project_root / "configs")):
        return compose(
            config_name="sim_libero.yaml",
            overrides=[f"task={task}"],
        )
