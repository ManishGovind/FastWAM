"""One-pass modality selection: full multistream MoT + per-sample masks.

Same selection semantics as ``FastWAMJointMultimodalSelect`` (random / gt / router),
but keeps the full ``R0 | Rf | Df | Ff | A`` layout in **one** forward and applies
a per-sample attention mask so inactive futures are isolated. Video loss is
applied only on the chosen stream.

Compare against the grouped short-MoT path in ``fastwam_joint_multimodal_select.py``.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import torch
import torch.nn.functional as F

from fastwam.utils.logging_config import get_logger

from .fastwam_joint_multimodal import _STREAM_KEYS
from .fastwam_joint_multimodal_select import FastWAMJointMultimodalSelect
from .modality_router import collapse_kl

logger = get_logger(__name__)


class FastWAMJointMultimodalSelectOnePass(FastWAMJointMultimodalSelect):
    """Modality selection with one full MoT forward and batched attention masks."""

    @classmethod
    def from_wan22_pretrained(cls, **kwargs):
        model = super().from_wan22_pretrained(**kwargs)
        logger.info(
            "FastWAMJointMultimodalSelectOnePass: full-layout MoT + per-sample "
            "active-future masks; video loss on chosen stream only."
        )
        return model

    @torch.no_grad()
    def _build_selection_mot_attention_mask_batch(
        self,
        *,
        stream_seq_lens: list[int],
        action_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
        enabled_streams: Sequence[str],
        modality_choices: torch.Tensor,
    ) -> torch.Tensor:
        """Per-sample masks [B,S,S] for hard active-future selection in full layout.

        Active topology matches the short-MoT select path; inactive future blocks
        keep self-attention only (isolated) so SDPA rows are never all-False.
        """
        enabled = tuple(enabled_streams)
        if "rgb" not in enabled:
            raise ValueError("`enabled_streams` must include 'rgb'.")
        if modality_choices.ndim != 1:
            raise ValueError(
                f"`modality_choices` must be [B], got {tuple(modality_choices.shape)}"
            )
        batch_size = int(modality_choices.shape[0])
        tpf = int(video_tokens_per_frame)

        seq_by_name = {name: int(length) for name, length in zip(enabled, stream_seq_lens)}
        offset = 0
        slices: dict[str, slice] = {}
        for name in enabled:
            length = seq_by_name[name]
            slices[name] = slice(offset, offset + length)
            offset += length
        video_seq_len = offset
        total_seq_len = video_seq_len + int(action_seq_len)
        action_sl = slice(video_seq_len, total_seq_len)

        rgb_sl = slices["rgb"]
        rgb_len = seq_by_name["rgb"]
        rgb_cond_sl = slice(0, tpf)
        rgb_fut_sl = slice(tpf, rgb_len)
        depth_fut_sl = slices.get("depth")
        flow_fut_sl = slices.get("flow")

        masks = torch.zeros(
            (batch_size, total_seq_len, total_seq_len),
            dtype=torch.bool,
            device=device,
        )
        # R0 and action self-attn always.
        masks[:, rgb_cond_sl, rgb_cond_sl] = True
        masks[:, action_sl, action_sl] = True

        choices = modality_choices.tolist()
        for b, choice_i in enumerate(choices):
            name = enabled[int(choice_i)]
            m = masks[b]
            if name == "rgb":
                m[rgb_fut_sl, rgb_sl] = True
                m[action_sl, rgb_sl] = True
                # Isolate unused futures.
                if depth_fut_sl is not None:
                    m[depth_fut_sl, depth_fut_sl] = True
                if flow_fut_sl is not None:
                    m[flow_fut_sl, flow_fut_sl] = True
            elif name == "depth":
                if depth_fut_sl is None:
                    raise ValueError("modality choice 'depth' but depth stream disabled.")
                m[depth_fut_sl, rgb_cond_sl] = True
                m[depth_fut_sl, depth_fut_sl] = True
                m[action_sl, rgb_cond_sl] = True
                m[action_sl, depth_fut_sl] = True
                # Isolate Rf / Ff.
                m[rgb_fut_sl, rgb_fut_sl] = True
                if flow_fut_sl is not None:
                    m[flow_fut_sl, flow_fut_sl] = True
            elif name == "flow":
                if flow_fut_sl is None:
                    raise ValueError("modality choice 'flow' but flow stream disabled.")
                m[flow_fut_sl, rgb_cond_sl] = True
                m[flow_fut_sl, flow_fut_sl] = True
                m[action_sl, rgb_cond_sl] = True
                m[action_sl, flow_fut_sl] = True
                m[rgb_fut_sl, rgb_fut_sl] = True
                if depth_fut_sl is not None:
                    m[depth_fut_sl, depth_fut_sl] = True
            else:
                raise ValueError(f"Unhandled modality choice {name!r}")
        return masks

    def _joint_denoise_multistream_onepass(
        self,
        noisy_latents: dict[str, torch.Tensor],
        latents_action: torch.Tensor,
        timestep_video: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
        action_condition: Optional[torch.Tensor] = None,
        modality_choices: Optional[torch.Tensor] = None,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Full-layout joint denoise; optional per-sample selection masks."""
        enabled_all = self._enabled_streams()
        mot_streams = enabled_all
        missing_latents = [name for name in mot_streams if name not in noisy_latents]
        if missing_latents:
            raise ValueError(f"Missing noisy latents for MoT streams: {missing_latents}")

        prepared = {}
        for name in mot_streams:
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

        ref = prepared["rgb"]
        tpf = int(ref["tokens_per_frame"])
        for name in mot_streams:
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

        mot_parts: dict[str, dict[str, Any]] = {
            "rgb": {**prepared["rgb"], "drop_condition_tokens": False},
        }
        for name in mot_streams:
            if name == "rgb":
                continue
            mot_parts[name] = _futures_only(prepared[name])

        video_tokens = torch.cat([mot_parts[n]["tokens"] for n in mot_streams], dim=1)
        t_mod_video = torch.cat([mot_parts[n]["t_mod"] for n in mot_streams], dim=1)
        freqs_video = torch.cat([mot_parts[n]["freqs"] for n in mot_streams], dim=0)
        context_mask_video = torch.cat(
            [mot_parts[n]["context_mask"] for n in mot_streams], dim=1
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

        stream_seq_lens = [int(mot_parts[n]["tokens"].shape[1]) for n in mot_streams]
        if modality_choices is None:
            attention_mask = self._build_multistream_mot_attention_mask(
                stream_seq_lens=stream_seq_lens,
                action_seq_len=action_tokens.shape[1],
                video_tokens_per_frame=tpf,
                device=video_tokens.device,
                action_attend_mode="all",
                enabled_streams=mot_streams,
            )
        else:
            if int(modality_choices.shape[0]) != int(video_tokens.shape[0]):
                raise ValueError(
                    f"modality_choices batch {tuple(modality_choices.shape)} != "
                    f"video batch {video_tokens.shape[0]}"
                )
            attention_mask = self._build_selection_mot_attention_mask_batch(
                stream_seq_lens=stream_seq_lens,
                action_seq_len=int(action_tokens.shape[1]),
                video_tokens_per_frame=tpf,
                device=video_tokens.device,
                enabled_streams=mot_streams,
                modality_choices=modality_choices,
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
        for name, seq_len in zip(mot_streams, stream_seq_lens):
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
                f"FastWAMJointMultimodalSelectOnePass.training_loss requires "
                f"enabled video keys {list(enabled)}; missing {missing}."
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

        selected_modality = sample.get("selected_modality", None)
        if selected_modality is not None and not torch.is_tensor(selected_modality):
            selected_modality = torch.as_tensor(selected_modality)
        modality_choices, router_logits, router_log_probs = self._sample_modality_select(
            enabled=enabled,
            batch_size=batch_size,
            device=stream_latents["rgb"].device,
            r0_latent=stream_latents["rgb"][:, :, 0],
            context=context,
            context_mask=context_mask,
            selected_modality=selected_modality,
        )

        pred_video, pred_action = self._joint_denoise_multistream_onepass(
            noisy_latents=noisy_latents,
            latents_action=noisy_action,
            timestep_video=timestep_video,
            timestep_action=timestep_action,
            context=context,
            context_mask=context_mask,
            fuse_vae_embedding_in_latents=fuse_flag,
            action_condition=action,
            modality_choices=modality_choices,
        )

        include_initial_video_step = first_frames["rgb"] is None
        lambdas = {
            "rgb": float(self.loss_lambda_rgb),
            "depth": float(self.loss_lambda_depth),
            "flow": float(self.loss_lambda_flow),
        }
        loss_dict: dict[str, float] = {f"loss_video_{name}": 0.0 for name in _STREAM_KEYS}
        video_weight = self.train_video_scheduler.training_weight(timestep_video).to(
            device=stream_latents["rgb"].device,
            dtype=stream_latents["rgb"].dtype,
        )
        loss_video_per_sample = torch.zeros(
            batch_size, device=stream_latents["rgb"].device, dtype=torch.float32
        )
        stream_loss_acc = {name: 0.0 for name in enabled}
        stream_loss_count = {name: 0 for name in enabled}

        # Per-stream video loss; keep only the chosen modality per sample.
        for stream_idx, name in enumerate(enabled):
            pred = pred_video[name]
            tgt = target_video[name]
            if first_frames[name] is not None:
                pred = pred[:, :, 1:]
                tgt = tgt[:, :, 1:]
            loss_ps = self._compute_video_loss_per_sample(
                pred_video=pred,
                target_video=tgt,
                image_is_pad=image_is_pad,
                include_initial_video_step=include_initial_video_step,
            )
            weighted_ps = lambdas[name] * loss_ps.float()
            if modality_choices is None:
                keep = torch.ones(batch_size, device=weighted_ps.device, dtype=torch.bool)
            else:
                keep = modality_choices == stream_idx
            if keep.any():
                loss_video_per_sample = loss_video_per_sample + weighted_ps * keep.float()
                w = video_weight.float()
                stream_loss_acc[name] += float((weighted_ps * w * keep.float()).sum().detach().item())
                stream_loss_count[name] += int(keep.sum().item())

        loss_video_total = (loss_video_per_sample * video_weight.float()).mean()
        for name in enabled:
            if stream_loss_count[name] > 0:
                loss_dict[f"loss_video_{name}"] = (
                    stream_loss_acc[name] / float(stream_loss_count[name])
                )

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
        loss_action = (action_loss_per_sample.float() * action_weight.float()).mean()
        loss_action_weighted = self.loss_lambda_action * loss_action
        loss_dict["loss_action"] = float(loss_action_weighted.detach().item())
        loss_dict["loss_video"] = float(loss_video_total.detach().item())
        if modality_choices is not None:
            canon = torch.tensor(
                [_STREAM_KEYS.index(enabled[int(i)]) for i in modality_choices.tolist()],
                device=modality_choices.device,
                dtype=torch.float32,
            )
            loss_dict["modality_select_idx"] = float(canon.mean().item())

        loss_total = loss_video_total + loss_action_weighted

        if (
            getattr(self, "modality_select", "all") == "router"
            and router_logits is not None
            and router_log_probs is not None
            and modality_choices is not None
        ):
            task_ps = loss_video_per_sample + self.loss_lambda_action * (
                action_loss_per_sample.float() * action_weight.float()
            )
            baseline = task_ps.detach().mean()
            advantage = task_ps.detach() - baseline
            chosen_log_prob = router_log_probs.gather(
                1, modality_choices.unsqueeze(1)
            ).squeeze(1)
            pg_loss = (chosen_log_prob * advantage).mean()
            entropy = -(router_log_probs.exp() * router_log_probs).sum(dim=-1).mean()
            collapse = collapse_kl(router_logits)
            router_loss = (
                pg_loss
                - float(self.router_entropy_coef) * entropy
                + float(self.router_collapse_kl) * collapse
            )
            loss_total = loss_total + router_loss
            loss_dict["loss_router"] = float(router_loss.detach().item())
            loss_dict["router_entropy"] = float(entropy.detach().item())
            loss_dict["router_collapse_kl"] = float(collapse.detach().item())
            probs = router_log_probs.exp().detach().mean(0)
            for i, name in enumerate(enabled):
                loss_dict[f"router_p_{name}"] = float(probs[i].item())

        return loss_total, loss_dict
