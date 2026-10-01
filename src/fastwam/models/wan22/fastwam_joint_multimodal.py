"""Joint FastWAM with multi-stream RGB + depth + flow video training.

Extends FastWAMJoint. Shared RGB condition R0; depth/flow futures attend R0.
D0/F0 exist only for VAE fuse (RGB pasted at t=0) and are dropped from the
MoT token sequence.

Inference:
  - ``infer_action`` / ``infer_joint`` / ``infer``: denoise all three streams
    with the same multistream MoT graph as training (RGB frame-0 conditions
    depth/flow). LIBERO deploy can still call ``infer_action``; only RGB input
    is required at runtime.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import torch
import torch.nn.functional as F

from fastwam.utils.logging_config import get_logger

from .fastwam_joint import FastWAMJoint

logger = get_logger(__name__)

_STREAM_KEYS = ("rgb", "depth", "flow")
# Infer-only ablation: which video tokens action may attend (video self-attn unchanged).
ACTION_ATTEND_MODES = (
    "all",
    "rgb",
    "depth",
    "flow",
    "rgb_depth",
    "rgb_flow",
    "depth_flow",
    "cond_only",
)


class FastWAMJointMultimodal(FastWAMJoint):
    """Train and optionally infer on RGB/depth/flow futures with a shared MoT core."""

    @classmethod
    def from_wan22_pretrained(cls, **kwargs):
        loss_lambda_rgb = float(kwargs.pop("loss_lambda_rgb", 1.0))
        loss_lambda_depth = float(kwargs.pop("loss_lambda_depth", 1.0))
        loss_lambda_flow = float(kwargs.pop("loss_lambda_flow", 1.0))
        enabled_video_streams = kwargs.pop("enabled_video_streams", None)
        stream_router_mode = str(kwargs.pop("stream_router_mode", "off")).lower()
        stream_router_temperature = float(kwargs.pop("stream_router_temperature", 1.0))
        model = super().from_wan22_pretrained(**kwargs)
        model.loss_lambda_rgb = loss_lambda_rgb
        model.loss_lambda_depth = loss_lambda_depth
        model.loss_lambda_flow = loss_lambda_flow
        model.enabled_video_streams = cls._normalize_enabled_streams(enabled_video_streams)

        # Stream router is currently unused; keep kwargs accepted so Hydra configs
        # with stream_router_* still instantiate cleanly.
        del stream_router_mode, stream_router_temperature

        logger.info(
            "FastWAMJointMultimodal streams=%s loss lambdas: rgb=%.4f depth=%.4f flow=%.4f action=%.4f",
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
            raise ValueError("`enabled_video_streams` must include 'rgb' (R0 condition).")
        return tuple(name for name in _STREAM_KEYS if name in seen)

    def _enabled_streams(self) -> tuple[str, ...]:
        return getattr(self, "enabled_video_streams", _STREAM_KEYS)

    @staticmethod
    def _validate_action_attend_mode(action_attend_mode: str) -> str:
        mode = str(action_attend_mode).lower()
        if mode not in ACTION_ATTEND_MODES:
            raise ValueError(
                f"`action_attend_mode` must be one of {ACTION_ATTEND_MODES}, got {action_attend_mode!r}"
            )
        return mode

    @torch.no_grad()
    def _build_multistream_mot_attention_mask(
        self,
        stream_seq_lens: list[int],
        action_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
        action_attend_mode: str = "all",
        enabled_streams: Optional[Sequence[str]] = None,
    ) -> torch.Tensor:
        """Attention mask for enabled streams, e.g. [R0 | R_fut | D_fut | F_fut | A].

        Disabled streams are omitted. D0/F0 tokens are never in the MoT sequence.

          - R0 → R0 ; R_fut → R0 + R_fut
          - D_fut → R0 + D_fut (if depth enabled)
          - F_fut → R0 + F_fut (if flow enabled)
          - Action → enabled video keys allowed by ``action_attend_mode``
          - No cross-modality between futures; Video ↛ Action
        """
        action_attend_mode = self._validate_action_attend_mode(action_attend_mode)
        enabled = tuple(enabled_streams) if enabled_streams is not None else self._enabled_streams()
        if list(enabled) != [n for n in _STREAM_KEYS if n in enabled]:
            raise ValueError(f"`enabled_streams` must be in canonical order, got {enabled}")
        if "rgb" not in enabled:
            raise ValueError("`enabled_streams` must include 'rgb'.")
        if len(stream_seq_lens) != len(enabled):
            raise ValueError(
                f"stream_seq_lens length {len(stream_seq_lens)} != enabled streams {enabled}"
            )
        if any(s <= 0 for s in stream_seq_lens):
            raise ValueError(f"All stream_seq_lens must be positive, got {stream_seq_lens}")
        if video_tokens_per_frame <= 0:
            raise ValueError(
                f"`video_tokens_per_frame` must be positive, got {video_tokens_per_frame}"
            )

        tpf = int(video_tokens_per_frame)
        seq_by_name = {name: int(length) for name, length in zip(enabled, stream_seq_lens)}
        rgb_len = seq_by_name["rgb"]
        if rgb_len % tpf != 0:
            raise ValueError(
                f"rgb stream seq_len must be divisible by tokens_per_frame, got {rgb_len}, {tpf}"
            )
        for name in enabled:
            if name == "rgb":
                continue
            if seq_by_name[name] % tpf != 0:
                raise ValueError(
                    f"{name} future seq_len must be divisible by tokens_per_frame, "
                    f"got {seq_by_name[name]}, {tpf}"
                )

        offset = 0
        slices: dict[str, slice] = {}
        for name in enabled:
            length = seq_by_name[name]
            slices[name] = slice(offset, offset + length)
            offset += length

        rgb_sl = slices["rgb"]
        rgb_cond_sl = slice(0, tpf)
        rgb_fut_sl = slice(tpf, rgb_len)
        depth_fut_sl = slices.get("depth")
        flow_fut_sl = slices.get("flow")

        video_seq_len = offset
        total_seq_len = video_seq_len + int(action_seq_len)
        action_sl = slice(video_seq_len, total_seq_len)
        mask = torch.zeros((total_seq_len, total_seq_len), dtype=torch.bool, device=device)

        mask[rgb_cond_sl, rgb_cond_sl] = True
        mask[rgb_fut_sl, rgb_sl] = True
        if depth_fut_sl is not None:
            mask[depth_fut_sl, rgb_cond_sl] = True
            mask[depth_fut_sl, depth_fut_sl] = True
        if flow_fut_sl is not None:
            mask[flow_fut_sl, rgb_cond_sl] = True
            mask[flow_fut_sl, flow_fut_sl] = True

        mask[action_sl, action_sl] = True

        def _attend(*maybe_slices: Optional[slice]) -> None:
            for sl in maybe_slices:
                if sl is not None:
                    mask[action_sl, sl] = True

        if action_attend_mode == "all":
            _attend(rgb_sl, depth_fut_sl, flow_fut_sl)
        elif action_attend_mode == "rgb":
            _attend(rgb_sl)
        elif action_attend_mode == "depth":
            _attend(rgb_cond_sl, depth_fut_sl)
        elif action_attend_mode == "flow":
            _attend(rgb_cond_sl, flow_fut_sl)
        elif action_attend_mode == "rgb_depth":
            _attend(rgb_sl, depth_fut_sl)
        elif action_attend_mode == "rgb_flow":
            _attend(rgb_sl, flow_fut_sl)
        elif action_attend_mode == "depth_flow":
            _attend(rgb_cond_sl, depth_fut_sl, flow_fut_sl)
        elif action_attend_mode == "cond_only":
            _attend(rgb_cond_sl)

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
        action_attend_mode: str = "all",
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        action_attend_mode = self._validate_action_attend_mode(action_attend_mode)
        enabled = self._enabled_streams()
        missing_latents = [name for name in enabled if name not in noisy_latents]
        if missing_latents:
            raise ValueError(f"Missing noisy latents for enabled streams: {missing_latents}")

        prepared = {}
        for name in enabled:
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
        for name in enabled:
            if name == "rgb":
                continue
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

        mot_parts = {"rgb": {**prepared["rgb"], "drop_condition_tokens": False}}
        for name in enabled:
            if name != "rgb":
                mot_parts[name] = _futures_only(prepared[name])

        video_tokens = torch.cat([mot_parts[n]["tokens"] for n in enabled], dim=1)
        t_mod_video = torch.cat([mot_parts[n]["t_mod"] for n in enabled], dim=1)
        freqs_video = torch.cat([mot_parts[n]["freqs"] for n in enabled], dim=0)
        context_mask_video = torch.cat(
            [mot_parts[n]["context_mask"] for n in enabled], dim=1
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

        stream_seq_lens = [int(mot_parts[n]["tokens"].shape[1]) for n in enabled]
        attention_mask = self._build_multistream_mot_attention_mask(
            stream_seq_lens=stream_seq_lens,
            action_seq_len=action_tokens.shape[1],
            video_tokens_per_frame=tpf,
            device=video_tokens.device,
            action_attend_mode=action_attend_mode,
            enabled_streams=enabled,
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
        for name, seq_len in zip(enabled, stream_seq_lens):
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
        enabled = self._enabled_streams()
        missing = [f"video_{k}" for k in enabled if f"video_{k}" not in sample]
        if missing:
            raise ValueError(
                f"FastWAMJointMultimodal.training_loss requires enabled video keys {list(enabled)}; "
                f"missing {missing}."
            )

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
            if name == "rgb":
                continue
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
        for name in enabled:
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
        loss_dict: dict[str, float] = {f"loss_video_{name}": 0.0 for name in _STREAM_KEYS}
        video_weight = self.train_video_scheduler.training_weight(timestep_video)

        for name in enabled:
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
            sum(loss_dict[f"loss_video_{n}"] for n in enabled)
        )

        loss_total = loss_video_total + loss_action_weighted
        return loss_total, loss_dict

    @staticmethod
    def _validate_infer_input_image(
        input_image: torch.Tensor,
        num_video_frames: int,
        *,
        check_fn,
    ) -> tuple[torch.Tensor, int, int]:
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        _, _, height, width = input_image.shape
        checked_h, checked_w, checked_t = check_fn(height, width, num_video_frames)
        if (checked_h, checked_w) != (height, width):
            raise ValueError(
                f"`input_image` must be resized before infer, expected multiples of 16 "
                f"but got HxW=({height},{width})"
            )
        if checked_t != num_video_frames:
            raise ValueError(
                f"`num_video_frames` must satisfy T % 4 == 1, got {num_video_frames}"
            )
        return input_image, height, width

    def _resolve_infer_context(
        self,
        *,
        prompt: Optional[str],
        proprio: Optional[torch.Tensor],
        context: Optional[torch.Tensor],
        context_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")

        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must be both provided together.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got "
                    f"{tuple(context.shape)} and {tuple(context_mask.shape)}"
                )
            context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)

        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError("`proprio` was provided but `proprio_dim=None` so `proprio_encoder` is disabled.")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            elif proprio.ndim == 2 and proprio.shape[0] == 1:
                pass
            else:
                raise ValueError(f"`proprio` must be [D] or [1,D], got shape {tuple(proprio.shape)}")
            if proprio.shape[1] != self.proprio_dim:
                raise ValueError(f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}")
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio,
            )
        return context, context_mask

    def _init_multistream_infer_latents(
        self,
        *,
        num_video_frames: int,
        height: int,
        width: int,
        action_horizon: int,
        input_image: torch.Tensor,
        seed: Optional[int],
        rand_device: str,
        tiled: bool,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict[str, torch.Tensor], bool]:
        latent_t = (num_video_frames - 1) // self.vae.temporal_downsample_factor + 1
        latent_h = height // self.vae.upsampling_factor
        latent_w = width // self.vae.upsampling_factor
        latent_shape = (1, self.vae.model.z_dim, latent_t, latent_h, latent_w)

        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_input_image_latents_tensor(
            input_image=input_image, tiled=tiled
        )
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))

        latents_video: dict[str, torch.Tensor] = {}
        first_frames: dict[str, torch.Tensor] = {}
        enabled = self._enabled_streams()
        for stream_idx, name in enumerate(enabled):
            generator = None
            if seed is not None:
                generator = torch.Generator(device=rand_device).manual_seed(int(seed) + stream_idx)
            noisy = torch.randn(
                latent_shape,
                generator=generator,
                device=rand_device,
                dtype=torch.float32,
            ).to(device=self.device, dtype=self.torch_dtype)
            noisy[:, :, 0:1] = first_frame_latents.clone()
            latents_video[name] = noisy
            first_frames[name] = first_frame_latents.clone()

        action_generator = None
        if seed is not None:
            action_generator = torch.Generator(device=rand_device).manual_seed(int(seed) + len(enabled))
        latents_action = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=action_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        return latents_video, latents_action, first_frames, fuse_flag

    def _decode_stream_videos(
        self,
        latents_video: dict[str, torch.Tensor],
        *,
        tiled: bool,
    ) -> dict[str, Any]:
        decoded = {
            name: self._decode_latents(latents_video[name], tiled=tiled)
            for name in self._enabled_streams()
        }
        decoded["video"] = decoded["rgb"]
        return decoded

    def _run_multistream_infer_loop(
        self,
        *,
        latents_video: dict[str, torch.Tensor],
        latents_action: torch.Tensor,
        first_frames: dict[str, torch.Tensor],
        fuse_flag: bool,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        num_inference_steps: int,
        sigma_shift: Optional[float],
        action_condition: Optional[torch.Tensor],
        action_attend_mode: str = "all",
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        action_attend_mode = self._validate_action_attend_mode(action_attend_mode)
        infer_timesteps_video, infer_deltas_video = self.infer_video_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_video["rgb"].dtype,
            shift_override=sigma_shift,
        )
        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        for step_t_video, step_delta_video, step_t_action, step_delta_action in zip(
            infer_timesteps_video,
            infer_deltas_video,
            infer_timesteps_action,
            infer_deltas_action,
        ):
            timestep_video = step_t_video.unsqueeze(0).to(
                dtype=latents_video["rgb"].dtype, device=self.device
            )
            timestep_action = step_t_action.unsqueeze(0).to(
                dtype=latents_action.dtype, device=self.device
            )
            pred_video, pred_action = self._joint_denoise_multistream(
                noisy_latents=latents_video,
                latents_action=latents_action,
                timestep_video=timestep_video,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                fuse_vae_embedding_in_latents=fuse_flag,
                action_condition=action_condition,
                action_attend_mode=action_attend_mode,
            )
            for name in self._enabled_streams():
                latents_video[name] = self.infer_video_scheduler.step(
                    pred_video[name], step_delta_video, latents_video[name]
                )
                latents_video[name][:, :, 0:1] = first_frames[name].clone()
            latents_action = self.infer_action_scheduler.step(
                pred_action, step_delta_action, latents_action
            )
        return latents_video, latents_action

    @torch.no_grad()
    def infer_action(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        num_video_frames: int,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        compile_action_infer: bool = False,
        action_attend_mode: str = "all",
    ) -> dict[str, Any]:
        del compile_action_infer, negative_prompt, text_cfg_scale
        self.eval()
        action_attend_mode = self._validate_action_attend_mode(action_attend_mode)
        if action_attend_mode != "all":
            logger.info(
                "FastWAMJointMultimodal.infer_action action_attend_mode=%s (infer ablation)",
                action_attend_mode,
            )

        input_image, height, width = self._validate_infer_input_image(
            input_image,
            num_video_frames,
            check_fn=self._check_resize_height_width,
        )
        context, context_mask = self._resolve_infer_context(
            prompt=prompt,
            proprio=proprio,
            context=context,
            context_mask=context_mask,
        )
        latents_video, latents_action, first_frames, fuse_flag = self._init_multistream_infer_latents(
            num_video_frames=num_video_frames,
            height=height,
            width=width,
            action_horizon=action_horizon,
            input_image=input_image,
            seed=seed,
            rand_device=rand_device,
            tiled=tiled,
        )
        _, latents_action = self._run_multistream_infer_loop(
            latents_video=latents_video,
            latents_action=latents_action,
            first_frames=first_frames,
            fuse_flag=fuse_flag,
            context=context,
            context_mask=context_mask,
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
            action_condition=None,
            action_attend_mode=action_attend_mode,
        )
        return {
            "action": latents_action[0].detach().to(device="cpu", dtype=torch.float32),
            "action_attend_mode": action_attend_mode,
        }

    @torch.no_grad()
    def infer_joint(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_video_frames: int,
        action_horizon: int,
        action: Optional[torch.Tensor] = None,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        test_action_with_infer_action: bool = False,
        compile_action_infer: bool = False,
        action_attend_mode: str = "all",
    ) -> dict[str, Any]:
        del compile_action_infer, negative_prompt, text_cfg_scale, test_action_with_infer_action
        self.eval()
        action_attend_mode = self._validate_action_attend_mode(action_attend_mode)
        if action_attend_mode != "all":
            logger.info(
                "FastWAMJointMultimodal.infer_joint action_attend_mode=%s (infer ablation)",
                action_attend_mode,
            )

        input_image, height, width = self._validate_infer_input_image(
            input_image,
            num_video_frames,
            check_fn=self._check_resize_height_width,
        )
        if action is not None:
            if action.ndim == 2:
                action = action.unsqueeze(0)
            if action.ndim != 3 or action.shape[0] != 1 or action.shape[1] != action_horizon:
                raise ValueError(
                    f"`action` must have shape [1, T, a_dim] or [T, a_dim], got {tuple(action.shape)} "
                    f"with action_horizon={action_horizon}"
                )
            action = action.to(device=self.device, dtype=self.torch_dtype)

        context, context_mask = self._resolve_infer_context(
            prompt=prompt,
            proprio=proprio,
            context=context,
            context_mask=context_mask,
        )
        latents_video, latents_action, first_frames, fuse_flag = self._init_multistream_infer_latents(
            num_video_frames=num_video_frames,
            height=height,
            width=width,
            action_horizon=action_horizon,
            input_image=input_image,
            seed=seed,
            rand_device=rand_device,
            tiled=tiled,
        )
        latents_video, latents_action = self._run_multistream_infer_loop(
            latents_video=latents_video,
            latents_action=latents_action,
            first_frames=first_frames,
            fuse_flag=fuse_flag,
            context=context,
            context_mask=context_mask,
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
            action_condition=action,
            action_attend_mode=action_attend_mode,
        )
        decoded = self._decode_stream_videos(latents_video, tiled=tiled)
        return {
            "video": decoded["rgb"],
            "video_rgb": decoded.get("rgb"),
            "video_depth": decoded.get("depth"),
            "video_flow": decoded.get("flow"),
            "action": latents_action[0].detach().to(device="cpu", dtype=torch.float32),
            "action_attend_mode": action_attend_mode,
        }

    @torch.no_grad()
    def infer(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_frames: int,
        action: Optional[torch.Tensor] = None,
        action_horizon: Optional[int] = None,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 5.0,
        action_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        action_attend_mode: str = "all",
    ) -> dict[str, Any]:
        del action_cfg_scale
        if action_horizon is None:
            raise ValueError("`action_horizon` is required for FastWAMJointMultimodal.infer.")
        return self.infer_joint(
            prompt=prompt,
            input_image=input_image,
            num_video_frames=num_frames,
            action_horizon=action_horizon,
            action=action,
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            negative_prompt=negative_prompt,
            text_cfg_scale=text_cfg_scale,
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
            seed=seed,
            rand_device=rand_device,
            tiled=tiled,
            action_attend_mode=action_attend_mode,
        )
