"""Joint FastWAM with multi-stream RGB + depth + flow video training.

Extends FastWAMJoint. Shared RGB condition R0; depth/flow futures attend R0.
D0/F0 exist only for VAE fuse (RGB pasted at t=0) and are dropped from the
MoT token sequence. Infer stays RGB-only via inherited FastWAMJoint.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn.functional as F

from fastwam.utils.logging_config import get_logger

from .fastwam_joint import FastWAMJoint

logger = get_logger(__name__)

_STREAM_KEYS = ("rgb", "depth", "flow")


class FastWAMJointMultimodal(FastWAMJoint):
    """Train on RGB/depth/flow futures; infer with RGB only (inherited)."""

    @classmethod
    def from_wan22_pretrained(cls, **kwargs):
        loss_lambda_rgb = float(kwargs.pop("loss_lambda_rgb", 1.0))
        loss_lambda_depth = float(kwargs.pop("loss_lambda_depth", 1.0))
        loss_lambda_flow = float(kwargs.pop("loss_lambda_flow", 1.0))
        model = super().from_wan22_pretrained(**kwargs)
        model.loss_lambda_rgb = loss_lambda_rgb
        model.loss_lambda_depth = loss_lambda_depth
        model.loss_lambda_flow = loss_lambda_flow
        logger.info(
            "FastWAMJointMultimodal loss lambdas: rgb=%.4f depth=%.4f flow=%.4f action=%.4f",
            model.loss_lambda_rgb,
            model.loss_lambda_depth,
            model.loss_lambda_flow,
            model.loss_lambda_action,
        )
        return model

    @torch.no_grad()
    def _build_multistream_mot_attention_mask(
        self,
        stream_seq_lens: list[int],
        action_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Attention mask for [R0 | R_fut | D_fut | F_fut | A].

        D0/F0 tokens are omitted from the MoT sequence (RGB condition is only R0).
        Depth/flow still VAE-fuse RGB at latent t=0 during prepare, but those
        condition tokens are not passed into attention.

          - R0 → R0 ; R_fut → R0 + R_fut
          - D_fut → R0 + D_fut
          - F_fut → R0 + F_fut
          - Action → Action + R0 + R_fut + D_fut + F_fut
          - No cross-modality between futures; Video ↛ Action
        """
        if len(stream_seq_lens) != 3:
            raise ValueError(
                f"Expected 3 stream_seq_lens (rgb_full, depth_fut, flow_fut), got {stream_seq_lens}"
            )
        if any(s <= 0 for s in stream_seq_lens):
            raise ValueError(f"All stream_seq_lens must be positive, got {stream_seq_lens}")
        if video_tokens_per_frame <= 0:
            raise ValueError(
                f"`video_tokens_per_frame` must be positive, got {video_tokens_per_frame}"
            )

        rgb_len, depth_fut_len, flow_fut_len = (int(s) for s in stream_seq_lens)
        tpf = int(video_tokens_per_frame)
        if rgb_len % tpf != 0:
            raise ValueError(
                f"rgb stream seq_len must be divisible by tokens_per_frame, got {rgb_len}, {tpf}"
            )
        if depth_fut_len % tpf != 0 or flow_fut_len % tpf != 0:
            raise ValueError(
                "depth/flow future seq_lens must be divisible by tokens_per_frame, "
                f"got depth={depth_fut_len}, flow={flow_fut_len}, tpf={tpf}"
            )

        rgb_sl = slice(0, rgb_len)
        rgb_cond_sl = slice(0, tpf)
        rgb_fut_sl = slice(tpf, rgb_len)
        depth_fut_sl = slice(rgb_len, rgb_len + depth_fut_len)
        flow_fut_sl = slice(
            rgb_len + depth_fut_len, rgb_len + depth_fut_len + flow_fut_len
        )

        video_seq_len = rgb_len + depth_fut_len + flow_fut_len
        total_seq_len = video_seq_len + int(action_seq_len)
        action_sl = slice(video_seq_len, total_seq_len)
        mask = torch.zeros((total_seq_len, total_seq_len), dtype=torch.bool, device=device)

        mask[rgb_cond_sl, rgb_cond_sl] = True
        mask[rgb_fut_sl, rgb_sl] = True
        mask[depth_fut_sl, rgb_cond_sl] = True
        mask[depth_fut_sl, depth_fut_sl] = True
        mask[flow_fut_sl, rgb_cond_sl] = True
        mask[flow_fut_sl, flow_fut_sl] = True

        mask[action_sl, action_sl] = True
        mask[action_sl, rgb_sl] = True
        mask[action_sl, depth_fut_sl] = True
        mask[action_sl, flow_fut_sl] = True

        return mask

    def _encode_stream_latents(
        self,
        video: torch.Tensor,
        tiled: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], bool]:
        if video.ndim != 5:
            raise ValueError(f"Stream video must be [B,3,T,H,W], got {tuple(video.shape)}")
        if video.shape[1] != 3:
            raise ValueError(f"Stream video channel dim must be 3, got {tuple(video.shape)}")

        _, _, num_frames, height, width = video.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(f"Video spatial dims must be multiples of 16, got H={height}, W={width}")
        if num_frames % 4 != 1:
            raise ValueError(f"Video T must satisfy T % 4 == 1, got T={num_frames}")
        if num_frames <= 1:
            raise ValueError(f"Video T must be > 1, got T={num_frames}")

        input_video = video.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        input_latents = self._encode_video_latents(input_video, tiled=tiled)
        first_frame_latents = None
        fuse_flag = False
        if getattr(self.video_expert, "fuse_vae_embedding_in_latents", False):
            first_frame_latents = input_latents[:, :, 0:1]
            fuse_flag = True
        return input_latents, first_frame_latents, fuse_flag

    def _build_shared_context_action(
        self,
        sample: dict[str, Any],
        batch_size: int,
        num_frames: int,
    ) -> dict[str, Any]:
        if "action" not in sample:
            raise ValueError("`sample['action']` is required for multimodal training.")
        action = sample["action"]
        if action.ndim != 3:
            raise ValueError(f"`sample['action']` must be [B,T,a_dim], got {tuple(action.shape)}")
        action_horizon = int(action.shape[1])
        if action_horizon % (num_frames - 1) != 0:
            raise ValueError(
                f"`sample['action']` temporal dim must be divisible by video transitions "
                f"({num_frames - 1}), got {action_horizon}"
            )

        action_is_pad = sample.get("action_is_pad", None)
        if action_is_pad is not None:
            if action_is_pad.ndim != 2 or action_is_pad.shape != (batch_size, action_horizon):
                raise ValueError(
                    f"`action_is_pad` shape mismatch: got {tuple(action_is_pad.shape)} "
                    f"vs expected ({batch_size}, {action_horizon})"
                )

        image_is_pad = sample.get("image_is_pad", None)
        if image_is_pad is not None:
            if image_is_pad.ndim != 2 or image_is_pad.shape != (batch_size, num_frames):
                raise ValueError(
                    f"`image_is_pad` shape mismatch: got {tuple(image_is_pad.shape)} "
                    f"vs expected ({batch_size}, {num_frames})"
                )

        context = sample.get("context")
        context_mask = sample.get("context_mask")
        if context is None and context_mask is None:
            prompt = sample.get("prompt")
            if prompt is None:
                raise ValueError("Multimodal training requires `context/context_mask` or `prompt`.")
            context, context_mask = self.encode_prompt(prompt)
        elif context is None or context_mask is None:
            raise ValueError("`context` and `context_mask` must both exist when either is provided.")

        if context.ndim != 3 or context_mask.ndim != 2:
            raise ValueError(
                f"`context/context_mask` must be [B,L,D]/[B,L], got "
                f"{tuple(context.shape)} and {tuple(context_mask.shape)}"
            )
        context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)

        proprio = sample.get("proprio", None)
        if self.proprio_encoder is not None:
            if proprio is None:
                raise ValueError("`sample['proprio']` is required when `proprio_dim` is enabled.")
            if proprio.ndim != 3:
                raise ValueError(f"`sample['proprio']` must be [B,T,d], got {tuple(proprio.shape)}")
            if proprio.shape[2] != self.proprio_dim:
                raise ValueError(
                    f"`sample['proprio']` last dim must be {self.proprio_dim}, got {proprio.shape[2]}"
                )
            proprio = proprio[:, 0, :]
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio.to(device=self.device, dtype=self.torch_dtype),
            )

        action = action.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        if action_is_pad is not None:
            action_is_pad = action_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if image_is_pad is not None:
            image_is_pad = image_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)

        return {
            "context": context,
            "context_mask": context_mask,
            "action": action,
            "action_is_pad": action_is_pad,
            "image_is_pad": image_is_pad,
        }

    def _joint_denoise_multistream(
        self,
        noisy_latents: dict[str, torch.Tensor],
        latents_action: torch.Tensor,
        timestep_video: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
        action_condition: Optional[torch.Tensor] = None,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        prepared = {}
        for name in _STREAM_KEYS:
            (
                video_tokens,
                t_video,
                t_mod_video,
                context_video,
                context_mask_video,
                freqs_video,
                f_video,
                h_video,
                w_video,
                tokens_per_frame,
            ) = self.video_expert.prepare(
                x=noisy_latents[name],
                timestep=timestep_video,
                context=context,
                context_mask=context_mask,
                action=action_condition,
                fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
            )
            prepared[name] = {
                "tokens": video_tokens,
                "t": t_video,
                "t_mod": t_mod_video,
                "context": context_video,
                "context_mask": context_mask_video,
                "freqs": freqs_video,
                "f": f_video,
                "h": h_video,
                "w": w_video,
                "tokens_per_frame": tokens_per_frame,
            }

        # Geometry must match across streams so RoPE / post unpatchify stay aligned.
        ref = prepared["rgb"]
        tpf = int(ref["tokens_per_frame"])
        for name in ("depth", "flow"):
            cur = prepared[name]
            if (cur["f"], cur["h"], cur["w"], cur["tokens_per_frame"]) != (
                ref["f"],
                ref["h"],
                ref["w"],
                ref["tokens_per_frame"],
            ):
                raise ValueError(
                    f"Stream '{name}' grid mismatch vs rgb: "
                    f"got fhw/tpf={(cur['f'], cur['h'], cur['w'], cur['tokens_per_frame'])} "
                    f"vs {(ref['f'], ref['h'], ref['w'], ref['tokens_per_frame'])}"
                )
            if cur["tokens"].shape[1] <= tpf:
                raise ValueError(
                    f"Stream '{name}' must have future tokens beyond frame0, "
                    f"got seq_len={cur['tokens'].shape[1]}, tpf={tpf}"
                )

        # Drop D0/F0 tokens from MoT (RGB condition lives only in R0).
        # Keep full RGB tokens. Depth/flow keep latent frame0 for VAE fuse in prepare.
        def _futures_only(p: dict[str, Any]) -> dict[str, Any]:
            return {
                "tokens": p["tokens"][:, tpf:],
                "t": p["t"][:, tpf:],
                "t_mod": p["t_mod"][:, tpf:],
                "context": p["context"],
                "context_mask": p["context_mask"][:, tpf:],
                "freqs": p["freqs"][tpf:],
                "f": int(p["f"]) - 1,
                "h": p["h"],
                "w": p["w"],
                "tokens_per_frame": tpf,
                "drop_condition_tokens": True,
            }

        mot_parts = {
            "rgb": {**prepared["rgb"], "drop_condition_tokens": False},
            "depth": _futures_only(prepared["depth"]),
            "flow": _futures_only(prepared["flow"]),
        }

        video_tokens = torch.cat([mot_parts[n]["tokens"] for n in _STREAM_KEYS], dim=1)
        t_mod_video = torch.cat([mot_parts[n]["t_mod"] for n in _STREAM_KEYS], dim=1)
        freqs_video = torch.cat([mot_parts[n]["freqs"] for n in _STREAM_KEYS], dim=0)
        context_mask_video = torch.cat(
            [mot_parts[n]["context_mask"] for n in _STREAM_KEYS], dim=1
        )
        context_video = mot_parts["rgb"]["context"]

        (
            action_tokens,
            _t_action,
            t_mod_action,
            context_action,
            context_mask_action,
            freqs_action,
        ) = self.action_expert.prepare(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )

        stream_seq_lens = [int(mot_parts[n]["tokens"].shape[1]) for n in _STREAM_KEYS]
        attention_mask = self._build_multistream_mot_attention_mask(
            stream_seq_lens=stream_seq_lens,
            action_seq_len=action_tokens.shape[1],
            video_tokens_per_frame=tpf,
            device=video_tokens.device,
        )

        video_tokens_out, action_tokens_out = self.mot.forward_joint_core(
            video_tokens=video_tokens,
            action_tokens=action_tokens,
            video_freqs=freqs_video,
            action_freqs=freqs_action,
            video_t_mod=t_mod_video,
            action_t_mod=t_mod_action,
            video_context=context_video,
            video_context_mask=context_mask_video,
            action_context=context_action,
            action_context_mask=context_mask_action,
            attention_mask=attention_mask,
        )

        pred_video = {}
        offset = 0
        for name, seq_len in zip(_STREAM_KEYS, stream_seq_lens):
            chunk = video_tokens_out[:, offset : offset + seq_len]
            part = mot_parts[name]
            pred = self.video_expert.post(
                chunk,
                part["t"],
                part["f"],
                part["h"],
                part["w"],
            )
            if part["drop_condition_tokens"]:
                # Pad dummy frame0 so training_loss can uniformly drop[:,:,1:].
                cond = noisy_latents[name][:, :, 0:1]
                pred = torch.cat([cond, pred], dim=2)
            pred_video[name] = pred
            offset += seq_len

        pred_action = self.action_expert.post(action_tokens_out)
        return pred_video, pred_action

    def training_loss(self, sample, tiled: bool = False):
        missing = [f"video_{k}" for k in _STREAM_KEYS if f"video_{k}" not in sample]
        if missing:
            raise ValueError(
                f"FastWAMJointMultimodal.training_loss requires {list(_STREAM_KEYS)} video keys; "
                f"missing {missing}."
            )

        stream_latents: dict[str, torch.Tensor] = {}
        first_frames: dict[str, Optional[torch.Tensor]] = {}
        fuse_flag = False
        for name in _STREAM_KEYS:
            latents, first_frame, fuse = self._encode_stream_latents(
                sample[f"video_{name}"], tiled=tiled
            )
            stream_latents[name] = latents
            first_frames[name] = first_frame
            fuse_flag = fuse_flag or fuse

        batch_size = stream_latents["rgb"].shape[0]
        num_frames = int(sample["video_rgb"].shape[2])
        for name in ("depth", "flow"):
            if stream_latents[name].shape != stream_latents["rgb"].shape:
                raise ValueError(
                    f"Latent shape mismatch for '{name}': "
                    f"{tuple(stream_latents[name].shape)} vs rgb {tuple(stream_latents['rgb'].shape)}"
                )

        shared = self._build_shared_context_action(
            sample=sample,
            batch_size=batch_size,
            num_frames=num_frames,
        )
        context = shared["context"]
        context_mask = shared["context_mask"]
        action = shared["action"]
        action_is_pad = shared["action_is_pad"]
        image_is_pad = shared["image_is_pad"]

        timestep_video = self.train_video_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=stream_latents["rgb"].dtype,
        )
        noisy_latents: dict[str, torch.Tensor] = {}
        target_video: dict[str, torch.Tensor] = {}
        for name in _STREAM_KEYS:
            noise = torch.randn_like(stream_latents[name])
            noisy = self.train_video_scheduler.add_noise(
                stream_latents[name], noise, timestep_video
            )
            target_video[name] = self.train_video_scheduler.training_target(
                stream_latents[name], noise, timestep_video
            )
            if first_frames[name] is not None:
                noisy[:, :, 0:1] = first_frames[name]
            noisy_latents[name] = noisy

        noise_action = torch.randn_like(action)
        timestep_action = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=action.dtype,
        )
        noisy_action = self.train_action_scheduler.add_noise(action, noise_action, timestep_action)
        target_action = self.train_action_scheduler.training_target(
            action, noise_action, timestep_action
        )

        pred_video, pred_action = self._joint_denoise_multistream(
            noisy_latents=noisy_latents,
            latents_action=noisy_action,
            timestep_video=timestep_video,
            timestep_action=timestep_action,
            context=context,
            context_mask=context_mask,
            fuse_vae_embedding_in_latents=fuse_flag,
            action_condition=action,
        )

        include_initial_video_step = first_frames["rgb"] is None
        lambdas = {
            "rgb": float(self.loss_lambda_rgb),
            "depth": float(self.loss_lambda_depth),
            "flow": float(self.loss_lambda_flow),
        }
        loss_video_total = None
        loss_dict: dict[str, float] = {}
        video_weight = self.train_video_scheduler.training_weight(timestep_video)

        for name in _STREAM_KEYS:
            pred = pred_video[name]
            tgt = target_video[name]
            if first_frames[name] is not None:
                pred = pred[:, :, 1:]
                tgt = tgt[:, :, 1:]
            loss_per_sample = self._compute_video_loss_per_sample(
                pred_video=pred,
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
            raise RuntimeError("No multimodal video losses were computed.")

        action_loss_token = F.mse_loss(
            pred_action.float(), target_action.float(), reduction="none"
        ).mean(dim=2)
        if action_is_pad is not None:
            valid = (~action_is_pad).to(
                device=action_loss_token.device, dtype=action_loss_token.dtype
            )
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
        loss_dict["loss_video"] = float(
            sum(loss_dict[f"loss_video_{n}"] for n in _STREAM_KEYS)
        )

        loss_total = loss_video_total + loss_action_weighted
        return loss_total, loss_dict
