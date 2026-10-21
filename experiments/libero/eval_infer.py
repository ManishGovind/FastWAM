"""LIBERO eval inference routing (training-matched MoT masks).

Branches:
  * ``FastWAMJointMultimodalSelect`` (+ onepass subclass): Select ``infer_action``
    with optional ``EVALUATION.active_future`` / router resolution.
  * ``FastWAMJointMultimodal``: multistream ``infer_action`` (all futures by default).
  * else: the model's own ``infer_action`` (unimodal joint / IDM / …).
"""

from __future__ import annotations

import functools
import inspect
import json
import logging
import os
from typing import Any, Sequence

import torch.nn as nn

logger = logging.getLogger(__name__)


def _is_multimodal_select(model: nn.Module) -> bool:
    from fastwam.models.wan22.fastwam_joint_multimodal_select import FastWAMJointMultimodalSelect

    return isinstance(model, FastWAMJointMultimodalSelect) or _is_multimodal_select_new(model)


def _is_multimodal_select_new(model: nn.Module) -> bool:
    """``_new`` inline-router models (and the Gumbel subclass). Not subclasses of
    ``FastWAMJointMultimodalSelect``, so they need their own check."""
    from fastwam.models.wan22.fastwam_joint_multimodal_select_new import FastWAMJointMultimodalSelectNew

    return isinstance(model, FastWAMJointMultimodalSelectNew)


def _is_multimodal(model: nn.Module) -> bool:
    from fastwam.models.wan22.fastwam_joint_multimodal import FastWAMJointMultimodal

    return isinstance(model, FastWAMJointMultimodal)


def describe_eval_mot_mask(
    model: nn.Module,
    action_attend_mode: str | None = None,
    active_future: str | None = None,
) -> str:
    from fastwam.models.wan22.fastwam_joint import FastWAMJoint

    mode = action_attend_mode or "all"

    if _is_multimodal_select(model):
        streams = getattr(model, "enabled_video_streams", ("rgb", "depth", "flow"))
        select = getattr(model, "modality_select", "all")
        if active_future is not None and str(active_future).lower() not in ("", "all", "none", "null"):
            af = str(active_future).lower()
            if af == "rgb":
                parts = ["R0", "R_fut", "A"]
            elif af == "random":
                parts = ["R0", "X_fut (random X per replan)", "A"]
            elif af.startswith("random:"):
                parts = ["R0", f"X_fut (random X in {af.split(':', 1)[1]} per replan)", "A"]
            else:
                parts = ["R0", f"{af[0].upper()}_fut", "A"]
            return (
                f"Select MoT [{' | '.join(parts)}] "
                f"(modality_select={select}; active_future={af}; "
                f"action_attend_mode={mode}; RGB input only)"
            )
        parts = ["R0", "R_fut"]
        if "depth" in streams:
            parts.append("D_fut")
        if "flow" in streams:
            parts.append("F_fut")
        parts.append("A")
        return (
            f"Select MoT [{' | '.join(parts)}] "
            f"(modality_select={select}; active_future=all; "
            f"action_attend_mode={mode}; RGB input only)"
        )

    if _is_multimodal(model):
        streams = getattr(model, "enabled_video_streams", ("rgb", "depth", "flow"))
        parts = ["R0", "R_fut"]
        if "depth" in streams:
            parts.append("D_fut")
        if "flow" in streams:
            parts.append("F_fut")
        parts.append("A")
        return (
            f"multistream joint MoT [{' | '.join(parts)}] "
            f"(streams={list(streams)}; action_attend_mode={mode}; RGB input only)"
        )
    if isinstance(model, FastWAMJoint):
        return "joint MoT [R0|R_fut|A] (action attends all video)"
    return "uncond MoT [R0|A] (action attends condition frame only)"


def validate_eval_infer_model(
    model: nn.Module,
    action_attend_mode: str | None = None,
    active_future: str | None = None,
) -> None:
    from fastwam.models.wan22.fastwam_joint_multimodal import FastWAMJointMultimodal
    from fastwam.models.wan22.fastwam_joint_multimodal_select import FastWAMJointMultimodalSelect
    from fastwam.models.wan22.fastwam_joint import FastWAMJoint

    if _is_multimodal_select(model):
        if model.infer_action.__func__ is FastWAMJointMultimodal.infer_action:
            raise RuntimeError(
                "FastWAMJointMultimodalSelect must override infer_action "
                "(active_future / router path)."
            )
        if model.infer_action.__func__ is FastWAMJoint.infer_action:
            raise RuntimeError(
                "FastWAMJointMultimodalSelect must not use FastWAMJoint.infer_action."
            )
        if action_attend_mode is not None:
            FastWAMJointMultimodalSelect._validate_action_attend_mode(action_attend_mode)
    elif _is_multimodal(model):
        if model.infer_action.__func__ is FastWAMJoint.infer_action:
            raise RuntimeError(
                "FastWAMJointMultimodal must override infer_action with the "
                "multistream training-matched path."
            )
        if action_attend_mode is not None:
            FastWAMJointMultimodal._validate_action_attend_mode(action_attend_mode)

    logger.info(
        "LIBERO eval MoT mask: %s",
        describe_eval_mot_mask(
            model,
            action_attend_mode=action_attend_mode,
            active_future=active_future,
        ),
    )


def resolve_action_infer_method(model: nn.Module):
    """Bind the correct ``infer_action`` for this model class."""
    from fastwam.models.wan22.fastwam_joint_multimodal import FastWAMJointMultimodal
    from fastwam.models.wan22.fastwam_joint_multimodal_select import FastWAMJointMultimodalSelect

    if _is_multimodal_select_new(model):
        # _new / Gumbel: their own infer_action (router argmax -> active_future).
        return model.infer_action
    if _is_multimodal_select(model):
        # Explicit Select instance path (includes onepass subclass).
        return FastWAMJointMultimodalSelect.infer_action.__get__(
            model, FastWAMJointMultimodalSelect
        )
    if _is_multimodal(model):
        return FastWAMJointMultimodal.infer_action.__get__(model, FastWAMJointMultimodal)
    validate_eval_infer_model(model)
    return model.infer_action


@functools.lru_cache(maxsize=8)
def _load_stream_map(path: str) -> dict[str, dict[str, str]]:
    with open(os.path.expanduser(os.path.expandvars(path)), encoding="utf-8") as f:
        raw = json.load(f)
    def stream(v) -> str:
        # A list means "random per replan among these streams" -> "random:a,b".
        if isinstance(v, (list, tuple)):
            return "random:" + ",".join(str(x).lower() for x in v)
        return str(v).lower()

    return {str(suite): {str(k): stream(v) for k, v in tasks.items()} for suite, tasks in raw.items()}


def plus_base_task(task_name: str) -> str:
    """LIBERO-Plus variant name -> base task name (strip the perturbation suffix)."""
    import re

    return re.sub(r"_(level\d|table_\d|view_|light|noise_|add_\d|sample\d|initstate|moved).*$", "", str(task_name))


def apply_per_task_active_future(task_cfg, suite_name: str, task_keys: Sequence[str]) -> str | None:
    """``EVALUATION.active_future_per_task``: JSON ``{suite: {task_key: stream | [streams]}}``.

    A list (e.g. ``["depth", "rgb"]``) draws one of those streams at random per replan.

    Sets ``task_cfg.EVALUATION.active_future`` for this task (overrides a global
    ``active_future``). ``task_keys`` are tried in order (LIBERO: the task id; LIBERO-Plus:
    the base task name). A task missing from the map is an error, so a partial map never
    silently falls back to another stream. Returns the stream, or None when unset.
    """
    path = task_cfg.EVALUATION.get("active_future_per_task", None)
    if path is None or str(path).strip() in ("", "null", "none"):
        return None
    suite_map = _load_stream_map(str(path)).get(str(suite_name))
    if suite_map is None:
        raise KeyError(f"active_future_per_task {path}: no entry for suite {suite_name!r}")
    for key in task_keys:
        if str(key) in suite_map:
            stream = suite_map[str(key)]
            task_cfg.EVALUATION.active_future = stream
            logger.info("Per-task active_future: %s / %s -> %s", suite_name, key, stream)
            return stream
    raise KeyError(f"active_future_per_task {path}: no entry for {suite_name} task keys {list(task_keys)}")


def build_action_infer_kwargs(
    model: nn.Module,
    *,
    cfg,
    compile_action_infer: bool,
) -> dict[str, Any]:
    if (_is_multimodal(model) or _is_multimodal_select(model)) and compile_action_infer:
        logger.warning(
            "Disabling compile_action_infer for multistream MoT "
            "(does not support the single-stream video-cache path)."
        )
        compile_action_infer = False

    extra: dict[str, Any] = {"compile_action_infer": compile_action_infer}
    if cfg is None:
        return extra

    if _is_multimodal(model) or _is_multimodal_select(model):
        mode = cfg.EVALUATION.get("action_attend_mode", None)
        if mode is not None:
            extra["action_attend_mode"] = str(mode)

    # active_future is Select-only (GT/random fixed stream, or router override).
    if _is_multimodal_select(model):
        active = cfg.EVALUATION.get("active_future", None)
        if active is not None and str(active).strip() != "":
            # "all" → all enabled futures (same as omitting / None inside Select).
            extra["active_future"] = str(active).lower()

    return extra


def call_action_infer(
    model: nn.Module,
    infer_kwargs: dict[str, Any],
    *,
    compile_action_infer: bool,
    cfg=None,
) -> dict[str, Any]:
    extra = build_action_infer_kwargs(
        model, cfg=cfg, compile_action_infer=compile_action_infer
    )
    action_attend_mode = extra.get("action_attend_mode")
    active_future = extra.get("active_future")
    validate_eval_infer_model(
        model,
        action_attend_mode=action_attend_mode,
        active_future=active_future,
    )
    infer_method = resolve_action_infer_method(model)
    sig = inspect.signature(infer_method)
    filtered = {k: v for k, v in infer_kwargs.items() if k in sig.parameters}
    for key in ("compile_action_infer", "action_attend_mode", "active_future"):
        if key in sig.parameters and key in extra:
            filtered[key] = extra[key]
    return infer_method(**filtered)


class RouterHistoryBuffer:
    """Per-episode RGB history for ``router_history`` models (no-op for every other model).

    Matches training (data.train.image_history_frames = H): frames t-(H+1)*s (VAE lead-in), t-H*s,
    .., t-s -- R0 (frame t) is not included -- with s = action_video_freq_ratio * global_sample_stride
    env steps; before the episode has that many steps the first policy frame is repeated (LeRobot
    clamps to the episode start the same way).
    ``to_image(obs)`` must build the model input exactly like the R0 image ([3, H, W] in [-1, 1]).
    """

    def __init__(self, model, cfg, to_image):
        data = cfg.data.train
        self.n = int(data.get("image_history_frames", 0) or 0)
        self.s = int(data.get("action_video_freq_ratio", 1)) * int(data.get("global_sample_stride", 1))
        self.on = bool(getattr(model, "router_history", False)) and self.n > 0
        self.to_image = to_image
        self.frames = []

    def push(self, obs) -> None:
        if self.on:
            self.frames.append(self.to_image(obs))

    def clip(self):
        """[3, H+1, h, w] or None."""
        if not self.on or not self.frames:
            return None
        import torch
        last = len(self.frames) - 1
        return torch.stack([self.frames[max(0, last - k * self.s)] for k in range(self.n + 1, 0, -1)], dim=1)

