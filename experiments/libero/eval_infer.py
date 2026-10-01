"""LIBERO eval inference routing (training-matched MoT masks)."""

from __future__ import annotations

import inspect
import logging
from typing import Any

import torch.nn as nn

logger = logging.getLogger(__name__)


def describe_eval_mot_mask(model: nn.Module, action_attend_mode: str | None = None) -> str:
    from fastwam.models.wan22.fastwam_joint_multimodal import FastWAMJointMultimodal
    from fastwam.models.wan22.fastwam_joint import FastWAMJoint

    if isinstance(model, FastWAMJointMultimodal):
        mode = action_attend_mode or "all"
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


def validate_eval_infer_model(model: nn.Module, action_attend_mode: str | None = None) -> None:
    from fastwam.models.wan22.fastwam_joint_multimodal import FastWAMJointMultimodal
    from fastwam.models.wan22.fastwam_joint import FastWAMJoint

    if isinstance(model, FastWAMJointMultimodal):
        if model.infer_action.__func__ is FastWAMJoint.infer_action:
            raise RuntimeError(
                "FastWAMJointMultimodal must override infer_action with the "
                "multistream training-matched path."
            )
        if action_attend_mode is not None:
            FastWAMJointMultimodal._validate_action_attend_mode(action_attend_mode)
    logger.info(
        "LIBERO eval MoT mask: %s",
        describe_eval_mot_mask(model, action_attend_mode=action_attend_mode),
    )


def resolve_action_infer_method(model: nn.Module):
    """Return infer_action bound method that matches training attention."""
    from fastwam.models.wan22.fastwam_joint_multimodal import FastWAMJointMultimodal

    if isinstance(model, FastWAMJointMultimodal):
        return FastWAMJointMultimodal.infer_action.__get__(model, FastWAMJointMultimodal)
    validate_eval_infer_model(model)
    return model.infer_action


def build_action_infer_kwargs(
    model: nn.Module,
    *,
    cfg,
    compile_action_infer: bool,
) -> dict[str, Any]:
    from fastwam.models.wan22.fastwam_joint_multimodal import FastWAMJointMultimodal

    if isinstance(model, FastWAMJointMultimodal) and compile_action_infer:
        logger.warning(
            "Disabling compile_action_infer for FastWAMJointMultimodal "
            "(multistream MoT does not support the single-stream video-cache path)."
        )
        compile_action_infer = False
    extra: dict[str, Any] = {"compile_action_infer": compile_action_infer}
    if isinstance(model, FastWAMJointMultimodal) and cfg is not None:
        mode = cfg.EVALUATION.get("action_attend_mode", None)
        if mode is not None:
            extra["action_attend_mode"] = str(mode)
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
    validate_eval_infer_model(model, action_attend_mode=action_attend_mode)
    infer_method = resolve_action_infer_method(model)
    sig = inspect.signature(infer_method)
    filtered = {k: v for k, v in infer_kwargs.items() if k in sig.parameters}
    for key in ("compile_action_infer", "action_attend_mode"):
        if key in sig.parameters and key in extra:
            filtered[key] = extra[key]
    return infer_method(**filtered)
