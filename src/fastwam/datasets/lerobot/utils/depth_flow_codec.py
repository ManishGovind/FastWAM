"""Pack / unpack depth and optical flow into LeRobot RGB mp4 videos.

Depth (`observation.images.depth`):
  - Per-video percentile normalize to uint8, then replicate to 3 channels.
  - Decode: take channel 0 and (optionally) invert the saved scale metadata.

Flow (`observation.images.flow`) — FlowWAM-style RAFT postprocess + HSV encoding:
  - Noise threshold 0.5 px: displacements with |f| < 0.5 are zeroed.
  - Magnitude cap 25 px before normalizing saturation.
  - White-background HSV color-wheel (H=direction, S=mag/cap, V=1; zero -> white).
  - Last frame is zeros (no next frame). Flow at t is motion from t -> t+1.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


# FlowWAM defaults (arxiv:2607.13017 appendix A.3 / Table)
DEFAULT_FLOW_CLIP_PX = 25.0
DEFAULT_FLOW_NOISE_THRESH_PX = 0.5
DEFAULT_DEPTH_PERCENTILES = (2.0, 98.0)


def depth_to_uint8_rgb(
    depth: np.ndarray | torch.Tensor,
    percentiles: tuple[float, float] = DEFAULT_DEPTH_PERCENTILES,
) -> tuple[np.ndarray, dict[str, float]]:
    """Convert [T,H,W] depth to [T,H,W,3] uint8 RGB + scale metadata."""
    if isinstance(depth, torch.Tensor):
        depth_np = depth.detach().float().cpu().numpy()
    else:
        depth_np = np.asarray(depth, dtype=np.float32)

    if depth_np.ndim != 3:
        raise ValueError(f"Expected depth [T,H,W], got shape {depth_np.shape}")

    lo_p, hi_p = percentiles
    finite = np.isfinite(depth_np)
    if not finite.any():
        raise ValueError("Depth map has no finite values.")

    vals = depth_np[finite]
    lo = float(np.percentile(vals, lo_p))
    hi = float(np.percentile(vals, hi_p))
    if hi <= lo:
        hi = lo + 1e-6

    norm = (np.clip(depth_np, lo, hi) - lo) / (hi - lo)
    depth_u8 = np.round(norm * 255.0).astype(np.uint8)
    rgb = np.stack([depth_u8, depth_u8, depth_u8], axis=-1)
    meta = {"depth_min": lo, "depth_max": hi, "percentile_lo": lo_p, "percentile_hi": hi_p}
    return rgb, meta


def uint8_rgb_to_depth(
    rgb: np.ndarray | torch.Tensor,
    depth_min: float,
    depth_max: float,
) -> np.ndarray:
    """Inverse of `depth_to_uint8_rgb` (approximate due to uint8 quantization)."""
    if isinstance(rgb, torch.Tensor):
        rgb_np = rgb.detach().cpu().numpy()
    else:
        rgb_np = np.asarray(rgb)

    if rgb_np.ndim == 4 and rgb_np.shape[-1] == 3:
        gray = rgb_np[..., 0].astype(np.float32) / 255.0
    elif rgb_np.ndim == 3:
        gray = rgb_np.astype(np.float32) / 255.0
    else:
        raise ValueError(f"Unexpected depth rgb shape {rgb_np.shape}")

    return gray * (float(depth_max) - float(depth_min)) + float(depth_min)


def _split_flow_uv(flow_np: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if flow_np.ndim != 4:
        raise ValueError(f"Expected flow 4D, got shape {flow_np.shape}")
    if flow_np.shape[1] == 2:
        return flow_np[:, 0], flow_np[:, 1]
    if flow_np.shape[-1] == 2:
        return flow_np[..., 0], flow_np[..., 1]
    raise ValueError(f"Cannot find 2 flow channels in shape {flow_np.shape}")


def apply_flowwam_postprocess(
    flow: np.ndarray | torch.Tensor,
    magnitude_cap_px: float = DEFAULT_FLOW_CLIP_PX,
    noise_thresh_px: float = DEFAULT_FLOW_NOISE_THRESH_PX,
) -> np.ndarray:
    """FlowWAM RAFT postprocess: zero tiny motion, then clip magnitude to cap.

    Returns flow as [T,2,H,W] float32.
    """
    if isinstance(flow, torch.Tensor):
        flow_np = flow.detach().float().cpu().numpy()
    else:
        flow_np = np.asarray(flow, dtype=np.float32).copy()

    u, v = _split_flow_uv(flow_np)
    # Ensure channel-first layout for output.
    u = np.asarray(u, dtype=np.float32).copy()
    v = np.asarray(v, dtype=np.float32).copy()

    mag = np.sqrt(u * u + v * v)
    noise_mask = mag < float(noise_thresh_px)
    u[noise_mask] = 0.0
    v[noise_mask] = 0.0

    mag = np.sqrt(u * u + v * v)
    cap = float(magnitude_cap_px)
    if cap <= 0:
        raise ValueError(f"magnitude_cap_px must be > 0, got {cap}")
    over = mag > cap
    if np.any(over):
        scale = np.ones_like(mag)
        scale[over] = cap / np.maximum(mag[over], 1e-8)
        u *= scale
        v *= scale

    return np.stack([u, v], axis=1)


def _hsv_to_rgb(h: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Vectorized HSV->[0,1] RGB. h,s,v in [0,1], same shape."""
    h6 = h * 6.0
    i = np.floor(h6).astype(np.int32)
    f = h6 - i
    i = np.mod(i, 6)
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)

    r = np.empty_like(v)
    g = np.empty_like(v)
    b = np.empty_like(v)
    for idx, (rr, gg, bb) in enumerate(
        (
            (v, t, p),
            (q, v, p),
            (p, v, t),
            (p, q, v),
            (t, p, v),
            (v, p, q),
        )
    ):
        mask = i == idx
        r[mask], g[mask], b[mask] = rr[mask], gg[mask], bb[mask]
    return np.stack([r, g, b], axis=-1)


def flow_to_uint8_rgb(
    flow: np.ndarray | torch.Tensor,
    flow_clip_px: float = DEFAULT_FLOW_CLIP_PX,
    noise_thresh_px: float = DEFAULT_FLOW_NOISE_THRESH_PX,
) -> np.ndarray:
    """Convert flow to FlowWAM-style white-background HSV RGB [T,H,W,3] uint8.

    Applies noise threshold then magnitude cap, then HSV color-wheel encoding.
    """
    flow_np = apply_flowwam_postprocess(
        flow,
        magnitude_cap_px=flow_clip_px,
        noise_thresh_px=noise_thresh_px,
    )
    u, v = flow_np[:, 0], flow_np[:, 1]
    clip = float(flow_clip_px)
    if clip <= 0:
        raise ValueError(f"flow_clip_px must be > 0, got {clip}")

    mag = np.sqrt(u * u + v * v)
    sat = np.clip(mag / clip, 0.0, 1.0)
    # Angle in [0, 1); white when sat=0.
    ang = (np.arctan2(v, u) + np.pi) / (2.0 * np.pi)
    val = np.ones_like(sat)
    rgb = _hsv_to_rgb(ang.astype(np.float32), sat.astype(np.float32), val.astype(np.float32))
    return np.round(np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)


def uint8_rgb_to_flow(
    rgb: np.ndarray | torch.Tensor,
    flow_clip_px: float = DEFAULT_FLOW_CLIP_PX,
) -> np.ndarray:
    """Legacy inverse for older rg_signed_clip flow videos only.

    White-background color-wheel videos are not invertible to exact (u, v).
    """
    if isinstance(rgb, torch.Tensor):
        rgb_np = rgb.detach().cpu().numpy()
    else:
        rgb_np = np.asarray(rgb)

    if rgb_np.ndim != 4 or rgb_np.shape[-1] != 3:
        raise ValueError(f"Expected flow rgb [T,H,W,3], got {rgb_np.shape}")

    # Heuristic: white-bg frames have many near-white pixels; refuse decode.
    white_frac = float(np.mean(np.all(rgb_np.astype(np.float32) > 250.0, axis=-1)))
    if white_frac > 0.05:
        raise ValueError(
            "Flow video appears to use white-background color-wheel encoding, "
            "which cannot be inverted to exact (u, v)."
        )

    clip = float(flow_clip_px)
    u = (rgb_np[..., 0].astype(np.float32) - 127.5) / 127.5 * clip
    v = (rgb_np[..., 1].astype(np.float32) - 127.5) / 127.5 * clip
    return np.stack([u, v], axis=1)


def feature_dict_for_depth(height: int, width: int, fps: int, extra_info: dict[str, Any] | None = None) -> dict:
    info = {
        "video.height": int(height),
        "video.width": int(width),
        "video.codec": "h264",
        "video.pix_fmt": "yuv420p",
        "video.is_depth_map": True,
        "video.fps": int(fps),
        "video.channels": 3,
        "has_audio": False,
        "depth_encoding": "uint8_percentile_rgb_replicate",
        "depth_model": "Depth-Anything-V2",
        "depth_repo": "https://github.com/DepthAnything/Depth-Anything-V2",
    }
    if extra_info:
        info.update(extra_info)
    return {
        "dtype": "video",
        "shape": [int(height), int(width), 3],
        "names": ["height", "width", "rgb"],
        "info": info,
    }


def feature_dict_for_flow(
    height: int,
    width: int,
    fps: int,
    flow_clip_px: float = DEFAULT_FLOW_CLIP_PX,
    extra_info: dict[str, Any] | None = None,
) -> dict:
    info = {
        "video.height": int(height),
        "video.width": int(width),
        "video.codec": "h264",
        "video.pix_fmt": "yuv420p",
        "video.is_depth_map": False,
        "video.fps": int(fps),
        "video.channels": 3,
        "has_audio": False,
        "flow_encoding": "flowwam_hsv_white_background",
        "flow_clip_px": float(flow_clip_px),
        "flow_noise_thresh_px": float(DEFAULT_FLOW_NOISE_THRESH_PX),
        "flow_channels": "HSV colorwheel (H=dir,S=mag/25,V=1; |f|<0.5->0/white)",
        "flow_alignment": "flow[t]=frame[t]->frame[t+1]; flow[-1]=0",
        "flow_estimator": "RAFT",
    }
    if extra_info:
        info.update(extra_info)
    return {
        "dtype": "video",
        "shape": [int(height), int(width), 3],
        "names": ["height", "width", "rgb"],
        "info": info,
    }
