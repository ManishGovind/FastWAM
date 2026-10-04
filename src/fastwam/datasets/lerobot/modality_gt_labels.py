"""Load offline action-error modality GT labels for training/eval selection.

Expects ``labels_*.pt`` from ``notebooks/modality_action_labels.py --analyze``:
``keys`` = list of ``(episode, window)``, ``label`` = int array in {0,1,2}
(RGB/Depth/Flow). Frame indices are reconstructed with the same stride used
when the labels were collected (default 4).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from fastwam.utils.logging_config import get_logger

logger = get_logger(__name__)

_STREAM_ORDER = ("rgb", "depth", "flow")


def parse_frame_stride_from_path(path: str | Path, default: int = 4) -> int:
    name = Path(path).name
    m = re.search(r"stride(\d+)", name)
    if m:
        return int(m.group(1))
    return int(default)


def _episode_windows(epi_from: int, epi_to: int, obs_size: int, frame_stride: int) -> list[int]:
    last = epi_to - int(obs_size)
    if last < epi_from:
        return []
    if frame_stride <= 0:
        raise ValueError("frame_stride must be positive to reconstruct GT label windows.")
    return list(range(epi_from, last + 1, frame_stride))


def load_modality_gt_label_map(
    path: str | Path,
    *,
    episode_from: np.ndarray | torch.Tensor,
    episode_to: np.ndarray | torch.Tensor,
    obs_size: int,
    frame_stride: Optional[int] = None,
) -> dict[str, Any]:
    """Build dataset-index → modality class maps from a labels ``.pt`` file.

    Returns a dict with:
      - ``by_index``: dict[int, int] exact labeled frames
      - ``episode_labeled_idx``: dict[int, np.ndarray] sorted labeled indices per episode
      - ``episode_labeled_lab``: dict[int, np.ndarray] labels aligned with those indices
      - ``num_classes``: 3
      - ``counts``: per-class counts
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Modality GT label file not found: {path}")

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if "label" not in payload or "keys" not in payload:
        raise KeyError(f"{path} must contain `keys` and `label` (from modality_action_labels --analyze).")

    labels = np.asarray(payload["label"]).astype(np.int64).reshape(-1)
    keys = list(payload["keys"])
    if len(keys) != len(labels):
        raise ValueError(f"keys/label length mismatch: {len(keys)} vs {len(labels)}")

    if "frame_idx" in payload:
        frame_idxs = np.asarray(payload["frame_idx"]).astype(np.int64).reshape(-1)
        if len(frame_idxs) != len(labels):
            raise ValueError("frame_idx/label length mismatch.")
        logger.info("Using exported frame_idx from %s (n=%d)", path, len(frame_idxs))
    else:
        stride = int(frame_stride) if frame_stride is not None else parse_frame_stride_from_path(path)
        if "frame_stride" in payload:
            stride = int(payload["frame_stride"])
        logger.info("Reconstructing GT frame_idx from keys with frame_stride=%d", stride)
        epi_from = np.asarray(
            episode_from.cpu().numpy() if torch.is_tensor(episode_from) else episode_from
        ).astype(np.int64)
        epi_to = np.asarray(
            episode_to.cpu().numpy() if torch.is_tensor(episode_to) else episode_to
        ).astype(np.int64)
        frame_idxs = np.empty(len(keys), dtype=np.int64)
        for i, key in enumerate(keys):
            ep, window = int(key[0]), int(key[1])
            if ep < 0 or ep >= len(epi_from):
                raise IndexError(f"Episode {ep} out of range for episode_data_index (n={len(epi_from)}).")
            wins = _episode_windows(int(epi_from[ep]), int(epi_to[ep]), int(obs_size), stride)
            if window < 0 or window >= len(wins):
                raise IndexError(
                    f"Window {window} out of range for episode {ep} (n_windows={len(wins)}, stride={stride})."
                )
            frame_idxs[i] = wins[window]

    by_index: dict[int, int] = {}
    ep_to_idxs: dict[int, list[int]] = {}
    ep_to_labs: dict[int, list[int]] = {}
    # Infer episode for each frame via from/to.
    epi_from = np.asarray(
        episode_from.cpu().numpy() if torch.is_tensor(episode_from) else episode_from
    ).astype(np.int64)
    epi_to = np.asarray(
        episode_to.cpu().numpy() if torch.is_tensor(episode_to) else episode_to
    ).astype(np.int64)

    def _episode_of(idx: int) -> int:
        # episode_data_index is contiguous; binary search on `from`.
        ep = int(np.searchsorted(epi_from, idx, side="right") - 1)
        if ep < 0 or ep >= len(epi_from) or not (epi_from[ep] <= idx < epi_to[ep]):
            raise IndexError(f"Dataset index {idx} not inside any episode.")
        return ep

    for fi, lab in zip(frame_idxs.tolist(), labels.tolist()):
        lab_i = int(lab)
        if lab_i < 0 or lab_i >= len(_STREAM_ORDER):
            raise ValueError(f"Invalid modality label {lab_i} at frame {fi}.")
        by_index[int(fi)] = lab_i
        ep = _episode_of(int(fi))
        ep_to_idxs.setdefault(ep, []).append(int(fi))
        ep_to_labs.setdefault(ep, []).append(lab_i)

    episode_labeled_idx = {}
    episode_labeled_lab = {}
    for ep, idxs in ep_to_idxs.items():
        order = np.argsort(np.asarray(idxs, dtype=np.int64))
        episode_labeled_idx[ep] = np.asarray(idxs, dtype=np.int64)[order]
        episode_labeled_lab[ep] = np.asarray(ep_to_labs[ep], dtype=np.int64)[order]

    counts = np.bincount(labels, minlength=3)
    logger.info(
        "Loaded modality GT labels from %s: n=%d counts rgb/depth/flow=%s",
        path,
        len(by_index),
        counts.tolist(),
    )
    return {
        "by_index": by_index,
        "episode_labeled_idx": episode_labeled_idx,
        "episode_labeled_lab": episode_labeled_lab,
        "episode_from": epi_from,
        "episode_to": epi_to,
        "num_classes": 3,
        "counts": counts,
        "path": str(path),
    }


def lookup_modality_gt_label(
    label_maps: dict[str, Any],
    dataset_index: int,
    *,
    nearest: bool = True,
) -> Optional[int]:
    """Return modality class for a dataset index, or None if unavailable."""
    by_index: dict[int, int] = label_maps["by_index"]
    if dataset_index in by_index:
        return int(by_index[dataset_index])
    if not nearest:
        return None

    epi_from = label_maps["episode_from"]
    epi_to = label_maps["episode_to"]
    ep = int(np.searchsorted(epi_from, dataset_index, side="right") - 1)
    if ep < 0 or ep >= len(epi_from) or not (epi_from[ep] <= dataset_index < epi_to[ep]):
        return None
    idxs = label_maps["episode_labeled_idx"].get(ep)
    labs = label_maps["episode_labeled_lab"].get(ep)
    if idxs is None or len(idxs) == 0:
        return None
    pos = int(np.searchsorted(idxs, dataset_index))
    if pos <= 0:
        return int(labs[0])
    if pos >= len(idxs):
        return int(labs[-1])
    # Closer of left/right labeled frames.
    left, right = int(idxs[pos - 1]), int(idxs[pos])
    if abs(dataset_index - left) <= abs(right - dataset_index):
        return int(labs[pos - 1])
    return int(labs[pos])
