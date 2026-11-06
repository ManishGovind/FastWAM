"""Privileged MM teacher: every stream keeps its clean frame 0 (R0 / D0 / F0).

Sequence [R0 | R_fut | D0 | D_fut | F0 | F_fut | A], per-stream mask:
  R0 -> R0            R_fut -> R0 + R_fut
  D0 -> R0 + D0       D_fut -> R0 + D0 + D_fut     (flow the same)
  A  -> everything    video never attends A; futures never attend each other.
Token stream embedding (if on) goes on every depth/flow token incl. D0/F0 (R0 untouched, as in MM),
otherwise D0/F0 share R0's RoPE positions and are indistinguishable.

The data must keep the real D0/F0: data.train.rgb_condition_modalities: [].
Training only (teacher for distillation, never evaluated): inference raises.
Loss / noise / timesteps: FastWAMJointMultimodal.training_loss (frame 0 is never a target).
"""

from __future__ import annotations

from typing import Optional

import torch

from .fastwam_joint_multimodal import _STREAM_KEYS, FastWAMJointMultimodal


class FastWAMJointMultimodalPriv(FastWAMJointMultimodal):
    @torch.no_grad()
    def _priv_mask(self, lens: list[int], action_len: int, tpf: int, device) -> torch.Tensor:
        video_len = sum(lens)
        mask = torch.zeros(video_len + action_len, video_len + action_len, dtype=torch.bool, device=device)
        r0 = slice(0, tpf)
        off = 0
        for length in lens:
            cond, fut = slice(off, off + tpf), slice(off + tpf, off + length)
            mask[cond, r0] = mask[cond, cond] = True
            mask[fut, r0] = mask[fut, cond] = mask[fut, fut] = True
            off += length
        mask[video_len:, :] = True  # action -> all video + action
        return mask

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
        if action_attend_mode != "all" or self.cross_stream_attention != "none":
            raise ValueError("Priv teacher supports only action_attend_mode=all, cross_stream_attention=none")
        names = self._enabled_streams()
        parts = {}
        for name in names:
            tok, t, t_mod, ctx_v, ctx_mask_v, freqs, f, h, w, tpf = self.video_expert.prepare(
                x=noisy_latents[name],
                timestep=timestep_video,
                context=context,
                context_mask=context_mask,
                action=action_condition,
                fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
            )
            if self.use_stream_embedding:
                s = self.video_expert.stream_embedding[_STREAM_KEYS.index(name)]
                start = tpf if name == "rgb" else 0  # R0 untouched; D0/F0 get their stream identity
                if self.stream_embedding_mode == "token":
                    tok = torch.cat([tok[:, :start], tok[:, start:] + s.to(tok.dtype)], dim=1)
                else:
                    t = torch.cat([t[:, :start], t[:, start:] + s.to(t.dtype)], dim=1)
                    t_mod = self.video_expert.time_projection(t).unflatten(2, (6, self.video_expert.hidden_dim))
            parts[name] = dict(tok=tok, t=t, t_mod=t_mod, ctx=ctx_v, ctx_mask=ctx_mask_v, freqs=freqs, f=f, h=h, w=w)

        lens = [p["tok"].shape[1] for p in parts.values()]
        if len(set(lens)) != 1:
            raise ValueError(f"Stream token lengths differ: {dict(zip(names, lens))}")
        action_tokens, _, t_mod_action, context_action, context_mask_action, freqs_action = self.action_expert.prepare(
            action_tokens=latents_action, timestep=timestep_action, context=context, context_mask=context_mask,
        )
        video_out, action_out = self.mot.forward_joint_core(
            video_tokens=torch.cat([p["tok"] for p in parts.values()], dim=1),
            action_tokens=action_tokens,
            video_freqs=torch.cat([p["freqs"] for p in parts.values()], dim=0),
            action_freqs=freqs_action,
            video_t_mod=torch.cat([p["t_mod"] for p in parts.values()], dim=1),
            action_t_mod=t_mod_action,
            video_context=parts["rgb"]["ctx"],
            video_context_mask=torch.cat([p["ctx_mask"] for p in parts.values()], dim=1),
            action_context=context_action,
            action_context_mask=context_mask_action,
            attention_mask=self._priv_mask(lens, action_tokens.shape[1], int(tpf), action_tokens.device),
        )
        pred_video, off = {}, 0
        for name, length in zip(names, lens):
            p = parts[name]
            pred_video[name] = self.video_expert.post(video_out[:, off:off + length], p["t"], p["f"], p["h"], p["w"])
            off += length
        return pred_video, self.action_expert.post(action_out)

    def infer(self, *args, **kwargs):
        raise NotImplementedError("Privileged teacher (needs real D0/F0) is training-only.")

    infer_action = infer_joint = infer
