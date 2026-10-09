"""Inline modality router trained with straight-through Gumbel-softmax (full R0 + context).

Subclass of ``FastWAMJointMultimodalSelectNew`` (same router class, same detached full
inputs: flattened R0 latent + full MoT context). Only training changes; inference is
the parent's ``modality_select=router`` path (router argmax -> ``active_future``).

How the router gets a gradient
------------------------------
REINFORCE (``_new``) treats the task loss as a scalar reward. Here the routing
weights enter the forward pass instead:

* every enabled future is in the MoT sequence, ``[R0 | R_fut | D_fut | F_fut | A]``,
  with the same structural mask as ``modality_select=all`` (futures attend R0 + self,
  video never attends action);
* per sample, the router draws ``y ~ GumbelSoftmax(logits, tau)`` and the action rows
  of the attention mask get an additive bias ``log y_k`` on stream ``k``'s columns
  (``R_fut`` for rgb, ``D_fut`` for depth, ``F_fut`` for flow; R0 frame-0 tokens and
  action tokens are always visible);
* straight-through (``router_gumbel_hard=true``, default): the forward bias is the
  hard one-hot (0 for the sampled stream, ``-1e4`` for the others), so the action
  tokens see exactly the select-mode topology ``R0 + chosen future``; the backward pass
  uses ``log y_soft``. The action loss therefore trains the router directly.
  ``router_gumbel_hard=false`` uses the soft mixture in the forward pass as well
  (lower-bias gradients, but train/test mismatch until ``tau`` is small).

Because video tokens never attend action tokens and futures never attend each other,
the gate changes only the action prediction. Video losses are computed for every
enabled stream (``router_gumbel_video_loss=all``, default; compute is already paid) or
only for the sampled stream (``chosen``, same as select-mode training).

Cost: one MoT pass over all futures per sample (same as ``modality_select=all``),
instead of R0 + one future.

``router_warmup_steps`` (inherited from ``_new``): during warm-up each sample gets a
uniformly random hard pick (no gradient to the router), so every branch trains first;
the tau schedule starts counting after warm-up.

``tau`` anneals exponentially from ``router_gumbel_tau_start`` to
``router_gumbel_tau_end`` over ``router_gumbel_tau_anneal_steps`` router-training
steps. The step counter is not checkpointed; on resume set ``tau_start`` to the
logged ``router_tau``.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F

from fastwam.utils.logging_config import get_logger

from .fastwam_joint_multimodal import _STREAM_KEYS
from .fastwam_joint_multimodal_select_new import FastWAMJointMultimodalSelectNew
from .modality_router_new import collapse_kl

logger = get_logger(__name__)

GUMBEL_VIDEO_LOSS = ("all", "chosen")
# Additive bias for "not selected" in the hard forward: masks the stream in bf16
# without producing -inf (whose gradient through the straight-through path is NaN).
_GATE_OFF = -1.0e4


class FastWAMJointMultimodalSelectGumbel(FastWAMJointMultimodalSelectNew):
    """``modality_select=router`` trained end-to-end with straight-through Gumbel-softmax."""

    @classmethod
    def from_wan22_pretrained(cls, **kwargs):
        tau_start = float(kwargs.pop("router_gumbel_tau_start", 1.0))
        tau_end = float(kwargs.pop("router_gumbel_tau_end", 0.5))
        tau_steps = int(kwargs.pop("router_gumbel_tau_anneal_steps", 20000))
        hard = bool(kwargs.pop("router_gumbel_hard", True))
        video_loss = str(kwargs.pop("router_gumbel_video_loss", "all")).lower()
        if tau_start <= 0 or tau_end <= 0:
            raise ValueError(f"Gumbel tau must be positive, got start={tau_start} end={tau_end}")
        if tau_steps < 0:
            raise ValueError(f"`router_gumbel_tau_anneal_steps` must be >= 0, got {tau_steps}")
        if video_loss not in GUMBEL_VIDEO_LOSS:
            raise ValueError(
                f"`router_gumbel_video_loss` must be one of {GUMBEL_VIDEO_LOSS}, got {video_loss!r}"
            )
        if bool(kwargs.get("modality_prompt", False)):
            # Gumbel training_loss does not insert the sentence; inference would -> mismatch.
            raise ValueError("modality_prompt is not supported by FastWAMJointMultimodalSelectGumbel yet.")
        if str(kwargs.get("modality_select", "router")).lower() != "router":
            raise ValueError("FastWAMJointMultimodalSelectGumbel requires modality_select=router.")
        kwargs["modality_select"] = "router"

        model = super().from_wan22_pretrained(**kwargs)
        model.router_gumbel_tau_start = tau_start
        model.router_gumbel_tau_end = tau_end
        model.router_gumbel_tau_anneal_steps = tau_steps
        model.router_gumbel_hard = hard
        model.router_gumbel_video_loss = video_loss
        model._gumbel_step = 0
        # Per-sample [B, K] log-gate consumed by _build_multistream_mot_attention_mask.
        model._action_stream_log_gate = None
        logger.info(
            "FastWAMJointMultimodalSelectGumbel: hard(ST)=%s tau %.3f -> %.3f over %d steps, "
            "video_loss=%s entropy=%.4f collapse_kl=%.4f",
            hard, tau_start, tau_end, tau_steps, video_loss,
            model.router_entropy_coef, model.router_collapse_kl,
        )
        return model

    # ------------------------------------------------------------------
    # Gumbel gate
    # ------------------------------------------------------------------

    def _gumbel_tau(self) -> float:
        steps = int(self.router_gumbel_tau_anneal_steps)
        if steps <= 0:
            return float(self.router_gumbel_tau_end)
        frac = min(float(self._gumbel_step) / steps, 1.0)
        start, end = float(self.router_gumbel_tau_start), float(self.router_gumbel_tau_end)
        return start * math.exp(frac * math.log(end / start))

    def _router_log_gate(
        self, logits: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        """Return (log_gate [B,K], choices [B], tau).

        Train (router in train mode): Gumbel sample at temperature tau; hard ST or soft.
        Eval: deterministic argmax, hard gate, no gradient path needed.
        """
        if not self._router_is_training():
            choices = logits.argmax(dim=-1)
            hard = F.one_hot(choices, logits.shape[-1]).bool()
            log_gate = torch.where(hard, 0.0, _GATE_OFF).to(logits.dtype)
            return log_gate, choices, 0.0

        tau = self._gumbel_tau()
        if self._router_in_warmup():
            # Warm-up: uniform random hard pick, constant gate (no gradient to the router).
            choices = torch.randint(logits.shape[-1], (logits.shape[0],), device=logits.device)
            hard = F.one_hot(choices, logits.shape[-1]).bool()
            return torch.where(hard, 0.0, _GATE_OFF).to(logits.dtype), choices, tau
        u = torch.rand_like(logits).clamp_(1e-10, 1.0 - 1e-10)
        gumbel = -torch.log(-torch.log(u))
        log_y_soft = F.log_softmax((logits + gumbel) / tau, dim=-1)
        choices = log_y_soft.argmax(dim=-1)
        if self.router_gumbel_hard:
            hard = F.one_hot(choices, logits.shape[-1]).bool()
            log_hard = torch.where(hard, 0.0, _GATE_OFF).to(log_y_soft.dtype)
            # Straight-through in log space: forward = hard mask, backward = d/d log_y_soft.
            log_gate = log_y_soft + (log_hard - log_y_soft).detach()
        else:
            log_gate = log_y_soft.clamp_min(_GATE_OFF)
        return log_gate, choices, tau

    # _action_stream_gate / _build_multistream_mot_attention_mask (per-sample gate on the
    # action rows) are inherited from FastWAMJointMultimodalSelectNew.

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_loss(self, sample, tiled: bool = False):
        enabled = self._enabled_streams()
        missing = [f"video_{k}" for k in enabled if f"video_{k}" not in sample]
        if missing:
            raise ValueError(
                f"FastWAMJointMultimodalSelectGumbel.training_loss requires enabled video keys "
                f"{list(enabled)}; missing {missing}."
            )

        # --- Same as FastWAMJointMultimodalSelectNew.training_loss up to the targets. ---
        stream_latents: dict[str, torch.Tensor] = {}
        first_frames: dict[str, Optional[torch.Tensor]] = {}
        fuse_flag = False
        for name in enabled:
            latents, first_frame, fuse = self._encode_stream_latents(
                sample[f"video_{name}"], tiled=tiled
            )
            stream_latents[name] = latents
            first_frames[name] = first_frame
            fuse_flag = fuse_flag or fuse

        batch_size = stream_latents["rgb"].shape[0]
        num_frames = int(sample["video_rgb"].shape[2])
        for name in enabled:
            if name != "rgb" and stream_latents[name].shape != stream_latents["rgb"].shape:
                raise ValueError(
                    f"Latent shape mismatch for '{name}': "
                    f"{tuple(stream_latents[name].shape)} vs rgb {tuple(stream_latents['rgb'].shape)}"
                )

        shared = self._build_shared_context_action(
            sample=sample, batch_size=batch_size, num_frames=num_frames
        )
        context = shared["context"]
        context_mask = shared["context_mask"]
        action = shared["action"]
        action_is_pad = shared["action_is_pad"]
        image_is_pad = shared["image_is_pad"]

        timestep_video = self.train_video_scheduler.sample_training_t(
            batch_size=batch_size, device=self.device, dtype=stream_latents["rgb"].dtype
        )
        noisy_latents: dict[str, torch.Tensor] = {}
        target_video: dict[str, torch.Tensor] = {}
        for name in enabled:
            noise = torch.randn_like(stream_latents[name])
            noisy = self.train_video_scheduler.add_noise(stream_latents[name], noise, timestep_video)
            target_video[name] = self.train_video_scheduler.training_target(
                stream_latents[name], noise, timestep_video
            )
            if first_frames[name] is not None:
                noisy[:, :, 0:1] = first_frames[name]
            noisy_latents[name] = noisy

        noise_action = torch.randn_like(action)
        timestep_action = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size, device=self.device, dtype=action.dtype
        )
        noisy_action = self.train_action_scheduler.add_noise(action, noise_action, timestep_action)
        target_action = self.train_action_scheduler.training_target(
            action, noise_action, timestep_action
        )
        # --- end of shared part ---

        # Router over detached R0 + MoT context (same inputs as the REINFORCE router).
        feats = self._build_router_features(
            r0_latent=stream_latents["rgb"][:, :, 0], context=context, context_mask=context_mask
        )
        logits = self.modality_router(feats)  # [B, K] float32
        log_gate, choices, tau = self._router_log_gate(logits)

        # One MoT pass with every future; the gate decides what the action tokens see.
        with self._action_stream_gate(log_gate):
            pred_video, pred_action = self._joint_denoise_multistream(
                noisy_latents=noisy_latents,
                latents_action=noisy_action,
                timestep_video=timestep_video,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                fuse_vae_embedding_in_latents=fuse_flag,
                action_condition=action,
                action_attend_mode="all",
                active_future=None,
            )

        # Video losses (independent of the gate).
        include_initial_video_step = first_frames["rgb"] is None
        lambdas = {
            "rgb": float(self.loss_lambda_rgb),
            "depth": float(self.loss_lambda_depth),
            "flow": float(self.loss_lambda_flow),
        }
        video_weight = self.train_video_scheduler.training_weight(timestep_video).float()
        loss_video_per_sample = torch.zeros(batch_size, device=self.device, dtype=torch.float32)
        loss_dict: dict[str, float] = {f"loss_video_{name}": 0.0 for name in _STREAM_KEYS}
        for k, name in enumerate(enabled):
            pred, tgt = pred_video[name], target_video[name]
            if first_frames[name] is not None:
                pred, tgt = pred[:, :, 1:], tgt[:, :, 1:]
            loss_ps = lambdas[name] * self._compute_video_loss_per_sample(
                pred_video=pred,
                target_video=tgt,
                image_is_pad=image_is_pad,
                include_initial_video_step=include_initial_video_step,
            ).float()
            if self.router_gumbel_video_loss == "chosen":
                loss_ps = loss_ps * (choices == k).float()
            loss_video_per_sample = loss_video_per_sample + loss_ps
            loss_dict[f"loss_video_{name}"] = float((loss_ps * video_weight).mean().detach())
        loss_video_total = (loss_video_per_sample * video_weight).mean()

        # Action loss (gradient reaches the router through the gate).
        action_loss_token = F.mse_loss(pred_action.float(), target_action.float(), reduction="none").mean(dim=2)
        if action_is_pad is not None:
            valid = (~action_is_pad).to(device=action_loss_token.device, dtype=action_loss_token.dtype)
            action_loss_per_sample = (action_loss_token * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        else:
            action_loss_per_sample = action_loss_token.mean(dim=1)
        action_weight = self.train_action_scheduler.training_weight(timestep_action).to(
            action_loss_per_sample.device, dtype=action_loss_per_sample.dtype
        )
        loss_action = (action_loss_per_sample * action_weight.float()).mean()
        loss_action_weighted = self.loss_lambda_action * loss_action

        loss_total = loss_video_total + loss_action_weighted
        loss_dict["loss_action"] = float(loss_action_weighted.detach())
        loss_dict["loss_video"] = float(loss_video_total.detach())

        # Router regularizers (on the un-noised policy).
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -(log_probs.exp() * log_probs).sum(dim=-1).mean()
        collapse = collapse_kl(logits)
        router_reg = (
            -float(self.router_entropy_coef) * entropy + float(self.router_collapse_kl) * collapse
        )
        in_warmup = self._router_in_warmup()
        if self._router_is_training():
            # x0 in warm-up: zero router grads (keeps DeepSpeed happy); tau clock starts after.
            loss_total = loss_total + (0.0 if in_warmup else 1.0) * router_reg
            if not in_warmup:
                self._gumbel_step += 1
        self._router_tick()
        loss_dict["router_warmup"] = float(in_warmup)
        loss_dict["loss_router_reg"] = float(router_reg.detach())
        loss_dict["router_entropy"] = float(entropy.detach())
        loss_dict["router_collapse_kl"] = float(collapse.detach())
        loss_dict["router_tau"] = float(tau)
        probs = log_probs.exp().detach().mean(0)
        for k, name in enumerate(enabled):
            loss_dict[f"router_p_{name}"] = float(probs[k])
            loss_dict[f"router_pick_{name}"] = float((choices == k).float().mean())
        canon = torch.tensor(
            [_STREAM_KEYS.index(enabled[int(i)]) for i in choices.tolist()], dtype=torch.float32
        )
        loss_dict["modality_select_idx"] = float(canon.mean())
        return loss_total, loss_dict
