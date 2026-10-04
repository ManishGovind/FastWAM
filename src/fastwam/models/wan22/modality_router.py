"""Learned router over modality-specific futures (RGB / depth / flow).

Mirrors the offline ``notebooks/train_modality_router.ModalityRouter``: each feature
group (pooled R0, text mean, proprio) gets its own LayerNorm + projection so the
large text vector cannot drown out image / proprio.
"""

from __future__ import annotations

from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


class ModalityRouter(nn.Module):
    def __init__(
        self,
        dims: Mapping[str, int],
        num_classes: int,
        d: int = 128,
        hidden: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        if num_classes <= 0:
            raise ValueError(f"`num_classes` must be positive, got {num_classes}")
        if not dims:
            raise ValueError("`dims` must contain at least one feature group.")
        self.names = list(dims)
        self.num_classes = int(num_classes)
        self.proj = nn.ModuleDict(
            {k: nn.Sequential(nn.LayerNorm(int(v)), nn.Linear(int(v), d)) for k, v in dims.items()}
        )
        self.head = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d * len(dims), hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, self.num_classes),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, feats: Mapping[str, torch.Tensor]) -> torch.Tensor:
        missing = [k for k in self.names if k not in feats]
        if missing:
            raise KeyError(f"ModalityRouter missing features {missing}; have {list(feats)}")
        x = torch.cat([self.proj[k](feats[k]) for k in self.names], dim=-1)
        return self.head(x)


def collapse_kl(logits: torch.Tensor) -> torch.Tensor:
    """KL(Uniform || batch-mean softmax). Penalizes collapse to a single modality."""
    p_bar = F.softmax(logits, dim=-1).mean(0).clamp_min(1e-8)
    uniform = torch.full_like(p_bar, 1.0 / p_bar.numel())
    return F.kl_div(p_bar.log(), uniform, reduction="sum")


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Masked mean over sequence dim. ``x`` [B,L,D], ``mask`` [B,L] bool/float."""
    if x.ndim != 3:
        raise ValueError(f"`x` must be [B,L,D], got {tuple(x.shape)}")
    if mask.ndim != 2 or mask.shape[:2] != x.shape[:2]:
        raise ValueError(
            f"`mask` must be [B,L] matching x, got {tuple(mask.shape)} vs {tuple(x.shape)}"
        )
    w = mask.to(dtype=x.dtype).unsqueeze(-1)
    return (x * w).sum(dim=1) / w.sum(dim=1).clamp(min=1.0)
