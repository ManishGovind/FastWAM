"""Learned router over modality-specific futures (RGB / depth / flow).

Features match FastWAM training MoT inputs (no extra pooling):

* ``r0`` — full first latent frame flattened ``[B, C*H*W]``
* ``context`` — full text+proprio token sequence ``[B, L, D]``
  (optional ``context_mask`` zeros pad tokens *after* the per-token projection)

Inline (in-graph) router used by ``fastwam_joint_multimodal_select_new``.
Differences from ``modality_router.py``:

* Pad tokens are masked after ``context_proj``. Masking before it does not remove
  them: LayerNorm maps a zero vector to its bias, so every pad token still
  contributed ``Linear(beta)`` to the head input.
"""

from __future__ import annotations

from typing import Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ModalityRouter(nn.Module):
    """R0 vector + per-token context projection (no mean pool), then MLP head."""

    def __init__(
        self,
        *,
        r0_dim: int,
        context_dim: int,
        context_len: int,
        num_classes: int,
        d: int = 128,
        hidden: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        if num_classes <= 0:
            raise ValueError(f"`num_classes` must be positive, got {num_classes}")
        if r0_dim <= 0 or context_dim <= 0 or context_len <= 0:
            raise ValueError(
                f"Router dims must be positive, got r0_dim={r0_dim}, "
                f"context_dim={context_dim}, context_len={context_len}"
            )
        self.r0_dim = int(r0_dim)
        self.context_dim = int(context_dim)
        self.context_len = int(context_len)
        self.num_classes = int(num_classes)
        self.r0_proj = nn.Sequential(nn.LayerNorm(self.r0_dim), nn.Linear(self.r0_dim, d))
        # Shared over tokens — preserves the sequence; does not mean-pool.
        self.context_proj = nn.Sequential(
            nn.LayerNorm(self.context_dim), nn.Linear(self.context_dim, d)
        )
        self.head = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d + d * self.context_len, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, self.num_classes),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for linear in (self.r0_proj[1], self.context_proj[1], self.head[2], self.head[5]):
            nn.init.kaiming_normal_(linear.weight, nonlinearity="relu")
            if linear.bias is not None:
                nn.init.zeros_(linear.bias)

    def forward(self, feats: Mapping[str, torch.Tensor]) -> torch.Tensor:
        if "r0" not in feats or "context" not in feats:
            raise KeyError(
                f"ModalityRouter requires 'r0' and 'context'; have {list(feats)}"
            )
        r0 = feats["r0"]
        context = feats["context"]
        if r0.ndim != 2 or r0.shape[-1] != self.r0_dim:
            raise ValueError(
                f"`r0` must be [B, {self.r0_dim}], got {tuple(r0.shape)}"
            )
        if context.ndim != 3:
            raise ValueError(f"`context` must be [B,L,D], got {tuple(context.shape)}")
        if context.shape[1] != self.context_len or context.shape[2] != self.context_dim:
            raise ValueError(
                f"`context` must be [B, {self.context_len}, {self.context_dim}], "
                f"got {tuple(context.shape)}"
            )
        # Run in the router's parameter dtype: DeepSpeed bf16 casts the whole model
        # (router included) to bf16, while callers pass float32 features.
        param_dtype = self.r0_proj[1].weight.dtype
        r0 = r0.to(param_dtype)
        context = context.to(param_dtype)
        r0_h = self.r0_proj(r0)
        ctx_h = self.context_proj(context)  # [B, L, d]
        # Zero pad positions after the projection so they carry no signal
        # (before it, LayerNorm would turn them into a constant nonzero bias).
        mask: Optional[torch.Tensor] = feats.get("context_mask")
        if mask is not None:
            if mask.shape != context.shape[:2]:
                raise ValueError(
                    f"`context_mask` must be {tuple(context.shape[:2])}, got {tuple(mask.shape)}"
                )
            ctx_h = ctx_h * mask.to(dtype=ctx_h.dtype).unsqueeze(-1)
        ctx_h = ctx_h.flatten(1)  # [B, L*d]
        # float32 logits so softmax / sampling / router losses stay in full precision.
        return self.head(torch.cat([r0_h, ctx_h], dim=-1)).float()

    @torch.no_grad()
    def route_offline(self, feats: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Hard argmax routing (eval / deploy)."""
        was_training = self.training
        self.eval()
        out = self.forward(feats).argmax(dim=-1)
        if was_training:
            self.train()
        return out

    def route_online(self, feats: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample a modality; returns (choices, log_probs) for REINFORCE-style losses."""
        logits = self.forward(feats)
        dist = torch.distributions.Categorical(logits=logits)
        chosen = dist.sample()
        return chosen, dist.log_prob(chosen)


def collapse_kl(logits: torch.Tensor) -> torch.Tensor:
    """KL(Uniform || batch-mean softmax). Penalizes collapse to a single modality."""
    p_bar = F.softmax(logits, dim=-1).mean(0).clamp_min(1e-8)
    uniform = torch.full_like(p_bar, 1.0 / p_bar.numel())
    return F.kl_div(p_bar.log(), uniform, reduction="sum")
