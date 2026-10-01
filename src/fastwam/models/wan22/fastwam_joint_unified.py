"""Joint FastWAM: fused future as the noised state; three modality losses on one pred.

  - Clean video latents: RGB frame 0 + mean of enabled futures (rgb/depth/flow).
  - Flow-match that fused volume (joint MoT, same as FastWAMJoint).
  - Supervise the same ``pred`` with per-modality targets (shared ``noise_video``),
    same as the previous multi-loss unified setup.

Inference: unchanged ``FastWAMJoint`` (RGB R0 + noise futures).
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn.functional as F

from fastwam.utils.logging_config import get_logger

from .fastwam_joint import FastWAMJoint

logger = get_logger(__name__)

_STREAM_KEYS = ("rgb", "depth", "flow")


class FastWAMJointUnified(FastWAMJoint):
    """Noise fused futures; keep three modality MSEs on the same prediction."""

    @classmethod
    def from_wan22_pretrained(cls, **kwargs):
        loss_lambda_rgb = float(kwargs.pop("loss_lambda_rgb", 1.0))
        loss_lambda_depth = float(kwargs.pop("loss_lambda_depth", 1.0))
        loss_lambda_flow = float(kwargs.pop("loss_lambda_flow", 1.0))
        enabled_video_streams = kwargs.pop("enabled_video_streams", None)
        model = super().from_wan22_pretrained(**kwargs)
        model.loss_lambda_rgb = loss_lambda_rgb
        model.loss_lambda_depth = loss_lambda_depth
        model.loss_lambda_flow = loss_lambda_flow
        model.enabled_video_streams = cls._normalize_enabled_streams(enabled_video_streams)
        logger.info(
            "FastWAMJointUnified future-fusion=mean streams=%s loss lambdas: "
            "rgb=%.4f depth=%.4f flow=%.4f action=%.4f",
            list(model.enabled_video_streams),
            model.loss_lambda_rgb,
            model.loss_lambda_depth,
            model.loss_lambda_flow,
            model.loss_lambda_action,
        )
        return model

    @staticmethod
    def _normalize_enabled_streams(streams: Optional[Sequence[str]]) -> tuple[str, ...]:
        if streams is None:
            return _STREAM_KEYS
        seen: list[str] = []
        for raw in streams:
            name = str(raw).lower()
            if name not in _STREAM_KEYS:
                raise ValueError(
                    f"`enabled_video_streams` must be a subset of {_STREAM_KEYS}, got {streams!r}"
                )
            if name not in seen:
                seen.append(name)
        if "rgb" not in seen:
            raise ValueError("`enabled_video_streams` must include 'rgb' (uses sample['video']).")
        return tuple(name for name in _STREAM_KEYS if name in seen)

    def _enabled_streams(self) -> tuple[str, ...]:
        return getattr(self, "enabled_video_streams", _STREAM_KEYS)

    def _fuse_future_latents(
        self,
        stream_latents: dict[str, torch.Tensor],
        enabled: tuple[str, ...],
    ) -> torch.Tensor:
        """Mean-fuse futures; keep RGB latent frame 0. Shape stays [B,C,T,H,W]."""
        z_rgb = stream_latents["rgb"]
        futures = [stream_latents[name][:, :, 1:] for name in enabled]
        for name, fut in zip(enabled, futures):
            if fut.shape != futures[0].shape:
                raise ValueError(
                    f"Future latent shape mismatch for '{name}': "
                    f"{tuple(fut.shape)} vs {tuple(futures[0].shape)}"
                )
        z_fut = torch.stack(futures, dim=0).mean(dim=0)
        return torch.cat([z_rgb[:, :, 0:1], z_fut], dim=2)

    def training_loss(self, sample, tiled: bool = False):
        enabled = self._enabled_streams()
        if "video" not in sample:
            if "video_rgb" not in sample:
                raise ValueError(
                    "FastWAMJointUnified requires `sample['video']` or `sample['video_rgb']`."
                )
            sample = dict(sample)
            sample["video"] = sample["video_rgb"]

        missing = [
            f"video_{name}"
            for name in enabled
            if name != "rgb" and f"video_{name}" not in sample
        ]
        if missing:
            raise ValueError(
                f"FastWAMJointUnified.training_loss missing modality videos: {missing}."
            )

        # RGB path for context / action / pads / R0 encode.
        inputs = self.build_inputs(sample, tiled=tiled)
        z_rgb = inputs["input_latents"]

        # Encode all enabled streams (rgb already done).
        stream_latents: dict[str, torch.Tensor] = {"rgb": z_rgb}
        for name in enabled:
            if name == "rgb":
                continue
            other_video = sample[f"video_{name}"].to(
                device=self.device, dtype=self.torch_dtype, non_blocking=True
            )
            z_other = self._encode_video_latents(other_video, tiled=tiled)
            if z_other.shape != z_rgb.shape:
                raise ValueError(
                    f"Latent shape mismatch for '{name}': "
                    f"{tuple(z_other.shape)} vs rgb {tuple(z_rgb.shape)}"
                )
            stream_latents[name] = z_other

        # *** only change vs prior unified: noise the fused future (R0 stays RGB) ***
        input_latents = self._fuse_future_latents(stream_latents, enabled)
        first_frame_latents = z_rgb[:, :, 0:1]
        if inputs["first_frame_latents"] is not None:
            inputs["first_frame_latents"] = first_frame_latents

        batch_size = input_latents.shape[0]
        context = inputs["context"]
        context_mask = inputs["context_mask"]
        action = inputs["action"]
        action_is_pad = inputs["action_is_pad"]
        image_is_pad = inputs["image_is_pad"]

        noise_video = torch.randn_like(input_latents)
        timestep_video = self.train_video_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=input_latents.dtype,
        )
        latents = self.train_video_scheduler.add_noise(input_latents, noise_video, timestep_video)

        if inputs["first_frame_latents"] is not None:
            latents[:, :, 0:1] = inputs["first_frame_latents"]

        # Per-modality FM targets from each clean stream (shared ε / σ), same as before.
        targets = {
            name: self.train_video_scheduler.training_target(
                stream_latents[name], noise_video, timestep_video
            )
            for name in enabled
        }

        noise_action = torch.randn_like(action)
        timestep_action = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=action.dtype,
        )
        noisy_action = self.train_action_scheduler.add_noise(action, noise_action, timestep_action)
        target_action = self.train_action_scheduler.training_target(action, noise_action, timestep_action)

        patch_t, patch_h, patch_w = (int(size) for size in self.video_expert.patch_size)
        latent_t, latent_h, latent_w = latents.shape[-3:]
        tokens_per_frame = (latent_h // patch_h) * (latent_w // patch_w)
        attention_mask = self._build_mot_attention_mask(
            video_seq_len=(latent_t // patch_t) * tokens_per_frame,
            action_seq_len=noisy_action.shape[1],
            video_tokens_per_frame=tokens_per_frame,
            device=latents.device,
        )
        pred_video, pred_action = self._joint_denoise_core(
            latents_video=latents,
            latents_action=noisy_action,
            timestep_video=timestep_video,
            timestep_action=timestep_action,
            context=context,
            context_mask=context_mask,
            attention_mask=attention_mask,
            fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
            action_condition=action,
        )

        include_initial_video_step = inputs["first_frame_latents"] is None
        pred_for_loss = pred_video
        if inputs["first_frame_latents"] is not None:
            pred_for_loss = pred_video[:, :, 1:]

        lambdas = {
            "rgb": float(self.loss_lambda_rgb),
            "depth": float(self.loss_lambda_depth),
            "flow": float(self.loss_lambda_flow),
        }
        loss_video_total = None
        loss_dict: dict[str, float] = {f"loss_video_{name}": 0.0 for name in _STREAM_KEYS}
        video_weight = self.train_video_scheduler.training_weight(timestep_video)

        for name in enabled:
            tgt = targets[name]
            if inputs["first_frame_latents"] is not None:
                tgt = tgt[:, :, 1:]
            loss_per_sample = self._compute_video_loss_per_sample(
                pred_video=pred_for_loss,
                target_video=tgt,
                image_is_pad=image_is_pad,
                include_initial_video_step=include_initial_video_step,
            )
            w = video_weight.to(device=loss_per_sample.device, dtype=loss_per_sample.dtype)
            loss_name = (loss_per_sample * w).mean()
            weighted = lambdas[name] * loss_name
            loss_video_total = weighted if loss_video_total is None else loss_video_total + weighted
            loss_dict[f"loss_video_{name}"] = float(weighted.detach().item())

        if loss_video_total is None:
            raise RuntimeError("No unified video losses were computed.")

        # --- identical action loss to FastWAM.training_loss ---
        action_loss_token = F.mse_loss(pred_action.float(), target_action.float(), reduction="none").mean(dim=2)
        if action_is_pad is not None:
            valid = (~action_is_pad).to(device=action_loss_token.device, dtype=action_loss_token.dtype)
            valid_sum = valid.sum(dim=1).clamp(min=1.0)
            action_loss_per_sample = (action_loss_token * valid).sum(dim=1) / valid_sum
        else:
            action_loss_per_sample = action_loss_token.mean(dim=1)

        action_weight = self.train_action_scheduler.training_weight(timestep_action).to(
            action_loss_per_sample.device, dtype=action_loss_per_sample.dtype
        )
        loss_action = (action_loss_per_sample * action_weight).mean()
        loss_action_weighted = self.loss_lambda_action * loss_action

        loss_dict["loss_action"] = float(loss_action_weighted.detach().item())
        loss_dict["loss_video"] = float(sum(loss_dict[f"loss_video_{n}"] for n in enabled))
        loss_total = loss_video_total + loss_action_weighted
        return loss_total, loss_dict
