"""Hydra factory for the ``_new`` inline-router model.

Kept separate from ``runtime.py`` so the original factory is untouched.
Target: ``fastwam.runtime_new.create_fastwam_joint_multimodal_select_new``.
"""

from __future__ import annotations

import torch
from omegaconf import DictConfig, ListConfig, OmegaConf


def create_fastwam_joint_multimodal_select_new(
    model_id: str,
    tokenizer_model_id: str,
    video_dit_config,
    tokenizer_max_len: int = 512,
    load_text_encoder: bool = True,
    proprio_dim: int | None = None,
    action_dit_config=None,
    action_dit_pretrained_path: str | None = None,
    skip_dit_load_from_pretrain: bool = False,
    video_scheduler=None,
    action_scheduler=None,
    loss=None,
    enabled_video_streams=None,
    modality_select: str = "all",
    stream_router_temperature: float = 1.0,
    router_entropy_coef: float = 0.01,
    router_collapse_kl: float = 0.1,
    router_hidden: int = 256,
    router_dropout: float = 0.1,
    router_video_size=None,
    router_reward: str = "action",
    router_baseline: str = "batch",
    router_baseline_bins: int = 10,
    router_baseline_momentum: float = 0.99,
    router_adv_normalize: bool = True,
    router_warmup_steps: int = 0,
    router_grad_accum: int = 1,
    router_layout: str = "short",
    router_video_loss: str = "chosen",
    modality_prompt: bool = False,
    modality_prompt_sentences=None,
    modality_prompt_cache_dir: str = "./data/text_embeds_cache/libero_modality",
    modality_prompt_context_len: int = 128,
    modality_prompt_enc_id: str = "wan22ti2v5b",
    modality_prompt_mode: str = "joint",
    mot_checkpoint_mixed_attn: bool = False,
    redirect_common_files: bool = True,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    _model_cls=None,
    _extra_kwargs: dict | None = None,
):
    """Factory for FastWAMJointMultimodalSelectNew (inline-router ``_new`` variant)."""
    from .models.wan22.fastwam_joint_multimodal_select_new import FastWAMJointMultimodalSelectNew

    model_cls = FastWAMJointMultimodalSelectNew if _model_cls is None else _model_cls

    if isinstance(video_dit_config, DictConfig):
        video_dit_config = OmegaConf.to_container(video_dit_config, resolve=True)
    if not isinstance(video_dit_config, dict):
        raise ValueError(f"`video_dit_config` must resolve to a dict, got {type(video_dit_config)}")

    if isinstance(action_dit_config, DictConfig):
        action_dit_config = OmegaConf.to_container(action_dit_config, resolve=True)
    if action_dit_config is None:
        action_dit_config = {}
    if not isinstance(action_dit_config, dict):
        raise ValueError(f"`action_dit_config` must resolve to a dict, got {type(action_dit_config)}")

    if isinstance(video_scheduler, DictConfig):
        video_scheduler = OmegaConf.to_container(video_scheduler, resolve=True)
    if video_scheduler is None:
        video_scheduler = {}
    if not isinstance(video_scheduler, dict):
        raise ValueError(f"`video_scheduler` must be dict-like, got {type(video_scheduler)}")

    if isinstance(action_scheduler, DictConfig):
        action_scheduler = OmegaConf.to_container(action_scheduler, resolve=True)
    if action_scheduler is None:
        raise ValueError("`action_scheduler` is required for FastWAM.")
    if not isinstance(action_scheduler, dict):
        raise ValueError(f"`action_scheduler` must be dict-like, got {type(action_scheduler)}")
    required_action_scheduler_keys = {"train_shift", "infer_shift", "num_train_timesteps"}
    missing_keys = required_action_scheduler_keys - set(action_scheduler.keys())
    if missing_keys:
        raise ValueError(
            f"`action_scheduler` missing required keys: {sorted(missing_keys)}. "
            "Expected keys: train_shift, infer_shift, num_train_timesteps."
        )

    if isinstance(loss, DictConfig):
        loss = OmegaConf.to_container(loss, resolve=True)
    if loss is None:
        loss = {}
    if not isinstance(loss, dict):
        raise ValueError(f"`loss` must be dict-like, got {type(loss)}")

    if isinstance(enabled_video_streams, DictConfig):
        enabled_video_streams = OmegaConf.to_container(enabled_video_streams, resolve=True)
    if isinstance(modality_prompt_sentences, (DictConfig, ListConfig)):
        modality_prompt_sentences = OmegaConf.to_container(modality_prompt_sentences, resolve=True)
    if isinstance(router_video_size, (DictConfig, ListConfig)):
        router_video_size = OmegaConf.to_container(router_video_size, resolve=True)

    return model_cls.from_wan22_pretrained(
        device=device,
        torch_dtype=model_dtype,
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        tokenizer_max_len=int(tokenizer_max_len),
        load_text_encoder=bool(load_text_encoder),
        proprio_dim=(None if proprio_dim is None else int(proprio_dim)),
        redirect_common_files=bool(redirect_common_files),
        video_dit_config=video_dit_config,
        action_dit_config=action_dit_config,
        action_dit_pretrained_path=action_dit_pretrained_path,
        skip_dit_load_from_pretrain=bool(skip_dit_load_from_pretrain),
        mot_checkpoint_mixed_attn=bool(mot_checkpoint_mixed_attn),
        video_train_shift=float(video_scheduler.get("train_shift", 5.0)),
        video_infer_shift=float(video_scheduler.get("infer_shift", 5.0)),
        video_num_train_timesteps=int(video_scheduler.get("num_train_timesteps", 1000)),
        action_train_shift=float(action_scheduler["train_shift"]),
        action_infer_shift=float(action_scheduler["infer_shift"]),
        action_num_train_timesteps=int(action_scheduler["num_train_timesteps"]),
        loss_lambda_video=float(loss.get("lambda_video", 1.0)),
        loss_lambda_action=float(loss.get("lambda_action", 1.0)),
        loss_lambda_rgb=float(loss.get("lambda_rgb", 1.0)),
        loss_lambda_depth=float(loss.get("lambda_depth", 1.0)),
        loss_lambda_flow=float(loss.get("lambda_flow", 1.0)),
        enabled_video_streams=enabled_video_streams,
        modality_select=str(modality_select),
        stream_router_temperature=float(stream_router_temperature),
        router_entropy_coef=float(router_entropy_coef),
        router_collapse_kl=float(router_collapse_kl),
        router_hidden=int(router_hidden),
        router_dropout=float(router_dropout),
        router_video_size=router_video_size,
        router_reward=str(router_reward),
        router_baseline=str(router_baseline),
        router_baseline_bins=int(router_baseline_bins),
        router_baseline_momentum=float(router_baseline_momentum),
        router_adv_normalize=bool(router_adv_normalize),
        router_warmup_steps=int(router_warmup_steps),
        router_grad_accum=int(router_grad_accum),
        router_layout=str(router_layout),
        router_video_loss=str(router_video_loss),
        modality_prompt=bool(modality_prompt),
        modality_prompt_sentences=modality_prompt_sentences,
        modality_prompt_cache_dir=str(modality_prompt_cache_dir),
        modality_prompt_context_len=int(modality_prompt_context_len),
        modality_prompt_enc_id=str(modality_prompt_enc_id),
        modality_prompt_mode=str(modality_prompt_mode),
        **(_extra_kwargs or {}),
    )


def create_fastwam_joint_multimodal_select_gumbel(
    router_gumbel_tau_start: float = 1.0,
    router_gumbel_tau_end: float = 0.5,
    router_gumbel_tau_anneal_steps: int = 20000,
    router_gumbel_hard: bool = True,
    router_gumbel_video_loss: str = "all",
    **kwargs,
):
    """Factory for FastWAMJointMultimodalSelectGumbel (straight-through Gumbel inline router).

    All other arguments are those of ``create_fastwam_joint_multimodal_select_new``;
    the REINFORCE-only keys (router_reward / router_baseline*) are accepted and unused.
    """
    from .models.wan22.fastwam_joint_multimodal_select_gumbel import (
        FastWAMJointMultimodalSelectGumbel,
    )

    return create_fastwam_joint_multimodal_select_new(
        _model_cls=FastWAMJointMultimodalSelectGumbel,
        _extra_kwargs={
            "router_gumbel_tau_start": float(router_gumbel_tau_start),
            "router_gumbel_tau_end": float(router_gumbel_tau_end),
            "router_gumbel_tau_anneal_steps": int(router_gumbel_tau_anneal_steps),
            "router_gumbel_hard": bool(router_gumbel_hard),
            "router_gumbel_video_loss": str(router_gumbel_video_loss),
        },
        **kwargs,
    )
