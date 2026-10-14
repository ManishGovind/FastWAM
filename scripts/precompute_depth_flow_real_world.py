#!/usr/bin/env python3
"""Precompute Depth Anything V2 depth and RAFT flow for real-world LeRobot datasets (e.g. R1-Lite).

Same outputs / encodings as scripts/precompute_depth_flow.py, with flow cleanup for real cameras:

  1. Duplicate-frame aware pairing. Real-world recordings are often stamped at a higher fps than the
     camera delivers (R1-Lite: 20 fps videos, every other frame an exact duplicate). Pairing t -> t+1
     then alternates between zero flow (duplicate) and a full sensor step plus fresh sensor/codec noise.
     Here flow[t] is computed from frame t to the next *distinct* frame, so every frame carries one
     real sensor step and the flow video no longer flickers.
  2. Forward-backward consistency. Pixels whose forward flow is not undone by the backward flow
     (typical on textureless walls / reflections) are zeroed.
  3. Optional de-shake (per camera). The median flow vector is subtracted to remove global camera
     vibration. Enable only for fixed cameras (head); on wrist cameras global motion is real ego-motion.

Depth is computed once per run of duplicate frames and repeated.

Example (1 episode smoke test):
  python scripts/precompute_depth_flow_real_world.py \\
    --dataset-dir ./data/real-world-r1-lite --depth-encoder vitl --max-episodes 1 --device cuda

Example (re-do flow only, keep existing depth):
  python scripts/precompute_depth_flow_real_world.py \\
    --dataset-dir ./data/real-world-r1-lite --no-depth --overwrite --device cuda
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import precompute_depth_flow as base  # noqa: E402

from fastwam.datasets.lerobot.utils.depth_flow_codec import (  # noqa: E402
    DEFAULT_FLOW_CLIP_PX,
    DEFAULT_FLOW_NOISE_THRESH_PX,
    depth_to_uint8_rgb,
    feature_dict_for_depth,
    feature_dict_for_flow,
    flow_to_uint8_rgb,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("precompute_depth_flow_real_world")

# R1-Lite: RGB source -> depth / flow output keys. Only the fixed head camera is de-shaken.
DEFAULT_CAMERA_STREAMS = (
    {
        "source_key": "observation.images.image",
        "depth_key": "observation.images.depth",
        "flow_key": "observation.images.flow",
        "deshake": True,
    },
    {
        "source_key": "observation.images.wrist_left_image",
        "depth_key": "observation.images.wrist_left_depth",
        "flow_key": "observation.images.wrist_left_flow",
        "deshake": False,
    },
    {
        "source_key": "observation.images.wrist_right_image",
        "depth_key": "observation.images.wrist_right_depth",
        "flow_key": "observation.images.wrist_right_flow",
        "deshake": False,
    },
)

DEFAULT_DUP_THRESH = 0.5  # mean |frame diff| (uint8 units) below which a frame is a duplicate
DEFAULT_FB_ABS = 1.0  # fwd-bwd check: |f + b_warped| < fb_rel * (|f|^2 + |b_warped|^2) + fb_abs
DEFAULT_FB_REL = 0.05


def duplicate_runs(frames_thwc_u8: np.ndarray, dup_thresh: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (run_start[t], next_distinct[t]) for every frame.

    run_start[t]: first frame of the duplicate run containing t.
    next_distinct[t]: first frame of the following run, or -1 if t is in the last run.
    """
    t = frames_thwc_u8.shape[0]
    diff = np.array(
        [
            np.abs(frames_thwc_u8[i + 1].astype(np.int16) - frames_thwc_u8[i].astype(np.int16)).mean()
            for i in range(t - 1)
        ],
        dtype=np.float32,
    )
    is_new = np.concatenate([[True], diff >= dup_thresh])  # frame starts a new run
    starts = np.flatnonzero(is_new)
    run_id = np.cumsum(is_new) - 1
    run_start = starts[run_id]
    next_distinct = np.array(
        [starts[r + 1] if r + 1 < len(starts) else -1 for r in run_id], dtype=np.int64
    )
    return run_start, next_distinct


def _backward_warp(x: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Sample x [B,C,H,W] at p + flow(p)."""
    _, _, h, w = flow.shape
    yy, xx = torch.meshgrid(
        torch.arange(h, device=flow.device, dtype=flow.dtype),
        torch.arange(w, device=flow.device, dtype=flow.dtype),
        indexing="ij",
    )
    gx = (xx + flow[:, 0]) / max(w - 1, 1) * 2 - 1
    gy = (yy + flow[:, 1]) / max(h - 1, 1) * 2 - 1
    return F.grid_sample(x, torch.stack([gx, gy], dim=-1), align_corners=True)


class RealWorldRaftFlowEstimator(base.RaftFlowEstimator):
    def __init__(
        self,
        device: torch.device,
        model_size: str = "large",
        dup_thresh: float = DEFAULT_DUP_THRESH,
        fb_check: bool = True,
        fb_abs: float = DEFAULT_FB_ABS,
        fb_rel: float = DEFAULT_FB_REL,
    ):
        super().__init__(device=device, model_size=model_size)
        self.dup_thresh = float(dup_thresh)
        self.fb_check = bool(fb_check)
        self.fb_abs = float(fb_abs)
        self.fb_rel = float(fb_rel)

    def _raft(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """a, b: [B,3,H,W] float in [0,1] on device -> flow [B,2,H,W] in original pixels."""
        h, w = a.shape[-2:]
        a_t, b_t = self.transforms(a, b)
        flow = self.model(a_t, b_t)[-1].float()
        ht, wt = int(flow.shape[-2]), int(flow.shape[-1])
        if (ht, wt) != (h, w):
            flow = F.interpolate(flow, size=(h, w), mode="bilinear", align_corners=False)
            flow[:, 0] *= w / float(wt)
            flow[:, 1] *= h / float(ht)
        return flow

    @torch.inference_mode()
    def __call__(
        self, frames_thwc_u8: np.ndarray, batch_size: int = 4, deshake: bool = False
    ) -> np.ndarray:
        """frames [T,H,W,3] -> flow [T,2,H,W]; flow[t] = t -> next distinct frame (0 in last run)."""
        t, h, w, _ = frames_thwc_u8.shape
        out = np.zeros((t, 2, h, w), dtype=np.float32)
        if t == 1:
            return out

        run_start, next_distinct = duplicate_runs(frames_thwc_u8, self.dup_thresh)
        # One RAFT pair per duplicate run; frames in the same run share the result.
        pairs = sorted({(int(s), int(n)) for s, n in zip(run_start, next_distinct) if n >= 0})
        frames = torch.from_numpy(frames_thwc_u8).permute(0, 3, 1, 2)

        pair_flow: dict[tuple[int, int], np.ndarray] = {}
        for start in range(0, len(pairs), batch_size):
            chunk = pairs[start : start + batch_size]
            a = frames[[p[0] for p in chunk]].to(self.device).float() / 255.0
            b = frames[[p[1] for p in chunk]].to(self.device).float() / 255.0
            fw = self._raft(a, b)
            if self.fb_check:
                bw_warped = _backward_warp(self._raft(b, a), fw)
                err = (fw + bw_warped).square().sum(dim=1)
                bound = self.fb_rel * (fw.square().sum(dim=1) + bw_warped.square().sum(dim=1)) + self.fb_abs
                valid = err < bound
            else:
                valid = torch.ones_like(fw[:, 0], dtype=torch.bool)
            if deshake:
                # Median over consistent pixels approximates the static background's motion.
                for i in range(fw.shape[0]):
                    sel = fw[i][:, valid[i]] if valid[i].any() else fw[i].flatten(1)
                    fw[i] -= sel.median(dim=1).values[:, None, None]
            fw = fw * valid[:, None]
            for i, p in enumerate(chunk):
                pair_flow[p] = fw[i].cpu().numpy()

        for i in range(t):
            if next_distinct[i] >= 0:
                out[i] = pair_flow[(int(run_start[i]), int(next_distinct[i]))]
        return out


def estimate_depth_dedup(
    depth_estimator: base.DepthAnythingV2Estimator,
    frames_thwc_u8: np.ndarray,
    dup_thresh: float,
    batch_size: int,
) -> np.ndarray:
    """Run depth once per duplicate run and repeat it to all frames of the run."""
    run_start, _ = duplicate_runs(frames_thwc_u8, dup_thresh)
    starts = np.unique(run_start)
    depth_unique = depth_estimator(frames_thwc_u8[starts], batch_size=batch_size)
    return depth_unique[np.searchsorted(starts, run_start)]


def _register_features(
    info: dict,
    camera_streams: list[dict],
    sizes: dict[str, tuple[int, int]],
    fps: int,
    args: argparse.Namespace,
    compute_depth: bool,
    compute_flow: bool,
) -> None:
    """Like base._register_features, but with each camera's own resolution."""
    info.setdefault("features", {})
    for stream in camera_streams:
        h, w = sizes[stream["source_key"]]
        if compute_depth:
            info["features"][stream["depth_key"]] = feature_dict_for_depth(
                h,
                w,
                fps,
                extra_info={"depth_encoder": args.depth_encoder, "depth_dedup_frames": True},
            )
        if compute_flow:
            info["features"][stream["flow_key"]] = feature_dict_for_flow(
                h,
                w,
                fps,
                flow_clip_px=args.flow_clip_px,
                extra_info={
                    "flow_noise_thresh_px": float(args.flow_noise_thresh_px),
                    "flow_alignment": "flow[t]=frame[t]->next distinct frame; 0 in last run",
                    "flow_dup_thresh": float(args.dup_thresh),
                    "flow_fb_check": not args.no_fb_check,
                    "flow_fb_abs": float(args.fb_abs),
                    "flow_fb_rel": float(args.fb_rel),
                    "flow_deshake": bool(stream["deshake"]),
                },
            )
    n_keys = sum(1 for f in info["features"].values() if f.get("dtype") == "video")
    info["total_videos"] = int(info["total_episodes"]) * int(n_keys)


def process_episode(
    dataset_dir: Path,
    info: dict,
    episode_index: int,
    stream: dict,
    depth_estimator: base.DepthAnythingV2Estimator | None,
    flow_estimator: RealWorldRaftFlowEstimator | None,
    args: argparse.Namespace,
    vcodec: str,
) -> dict[str, float]:
    src = base._video_path(dataset_dir, info, episode_index, stream["source_key"])
    if not src.exists():
        raise FileNotFoundError(src)

    depth_out = base._video_path(dataset_dir, info, episode_index, stream["depth_key"])
    flow_out = base._video_path(dataset_dir, info, episode_index, stream["flow_key"])

    need_depth = depth_estimator is not None and (
        args.overwrite or not (args.skip_existing and depth_out.exists())
    )
    need_flow = flow_estimator is not None and (
        args.overwrite or not (args.skip_existing and flow_out.exists())
    )
    if not need_depth and not need_flow:
        return {"skipped": 1.0}

    frames = base.read_rgb_video(src)
    fps = int(info["fps"])
    stats: dict[str, float] = {"frames": float(frames.shape[0])}

    if need_depth:
        depth = estimate_depth_dedup(
            depth_estimator, frames, dup_thresh=args.dup_thresh, batch_size=args.depth_batch_size
        )
        depth_rgb, meta = depth_to_uint8_rgb(depth)
        base.write_rgb_video(depth_out, depth_rgb, fps=fps, vcodec=vcodec, overwrite=True)
        stats["depth_min"] = meta["depth_min"]
        stats["depth_max"] = meta["depth_max"]

    if need_flow:
        flow = flow_estimator(frames, batch_size=args.flow_batch_size, deshake=stream["deshake"])
        flow_rgb = flow_to_uint8_rgb(
            flow,
            flow_clip_px=args.flow_clip_px,
            noise_thresh_px=args.flow_noise_thresh_px,
        )
        base.write_rgb_video(flow_out, flow_rgb, fps=fps, vcodec=vcodec, overwrite=True)
        stats["flow_abs_p95"] = float(np.percentile(np.abs(flow), 95))

    return stats


def _resolve_camera_streams(source_keys: list[str] | None, deshake_keys: list[str] | None) -> list[dict]:
    catalog = {s["source_key"]: s for s in DEFAULT_CAMERA_STREAMS}
    keys = source_keys or list(catalog)
    streams = []
    for key in keys:
        if key not in catalog:
            raise ValueError(
                f"Unknown source key '{key}'. Known: {sorted(catalog)}. "
                "Add a mapping in DEFAULT_CAMERA_STREAMS if needed."
            )
        stream = dict(catalog[key])
        if deshake_keys is not None:
            stream["deshake"] = key in deshake_keys
        streams.append(stream)
    return streams


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-dir", nargs="+", required=True, help="One or more LeRobot dataset roots (glob ok).")
    p.add_argument("--source-key", nargs="+", default=None, help="RGB video key(s). Default: all R1-Lite cameras.")
    p.add_argument(
        "--deshake-keys",
        nargs="*",
        default=None,
        help="Source keys to de-shake (overrides defaults; head camera only by default). Pass with no "
        "values to disable de-shake everywhere.",
    )
    p.add_argument(
        "--depth-encoder",
        default=base.DEFAULT_DEPTH_ENCODER,
        choices=sorted(base.DEPTH_ANYTHING_V2_CONFIGS),
    )
    p.add_argument("--depth-checkpoint", default=None)
    p.add_argument("--depth-input-size", type=int, default=518)
    p.add_argument("--raft-size", choices=["large", "small"], default="large")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--flow-clip-px", type=float, default=DEFAULT_FLOW_CLIP_PX)
    p.add_argument("--flow-noise-thresh-px", type=float, default=DEFAULT_FLOW_NOISE_THRESH_PX)
    p.add_argument(
        "--dup-thresh",
        type=float,
        default=DEFAULT_DUP_THRESH,
        help="Mean |frame diff| (uint8) below which consecutive frames count as duplicates.",
    )
    p.add_argument("--no-fb-check", action="store_true", help="Disable forward-backward consistency masking.")
    p.add_argument("--fb-abs", type=float, default=DEFAULT_FB_ABS)
    p.add_argument("--fb-rel", type=float, default=DEFAULT_FB_REL)
    p.add_argument("--depth-batch-size", type=int, default=8)
    p.add_argument("--flow-batch-size", type=int, default=4)
    p.add_argument("--vcodec", default="libx264", choices=["libx264", "libsvtav1", "h264"])
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--max-episodes", type=int, default=None)
    p.add_argument("--start-episode", type=int, default=0)
    p.add_argument("--shard-id", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--no-depth", action="store_true")
    p.add_argument("--no-flow", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.no_depth and args.no_flow:
        raise SystemExit("Nothing to do: both --no-depth and --no-flow set.")
    if args.overwrite and args.skip_existing:
        raise SystemExit("Use only one of --overwrite / --skip-existing.")

    dataset_dirs = base._expand_dataset_dirs(args.dataset_dir)
    device = torch.device(args.device)
    compute_depth = not args.no_depth
    compute_flow = not args.no_flow

    depth_estimator = (
        base.DepthAnythingV2Estimator(
            encoder=args.depth_encoder,
            device=device,
            checkpoint=args.depth_checkpoint,
            input_size=args.depth_input_size,
        )
        if compute_depth
        else None
    )
    flow_estimator = (
        RealWorldRaftFlowEstimator(
            device=device,
            model_size=args.raft_size,
            dup_thresh=args.dup_thresh,
            fb_check=not args.no_fb_check,
            fb_abs=args.fb_abs,
            fb_rel=args.fb_rel,
        )
        if compute_flow
        else None
    )
    vcodec = "libx264" if args.vcodec == "h264" else args.vcodec

    camera_streams = _resolve_camera_streams(args.source_key, args.deshake_keys)
    logger.info(
        "Camera streams: %s",
        [(s["source_key"], s["depth_key"], s["flow_key"], f"deshake={s['deshake']}") for s in camera_streams],
    )

    for dataset_dir in dataset_dirs:
        info = base._load_info(dataset_dir)
        for stream in camera_streams:
            if stream["source_key"] not in info.get("features", {}):
                raise KeyError(f"{stream['source_key']} not in features of {dataset_dir}")

        episodes = base._episode_indices(dataset_dir, info)
        episodes = [e for e in episodes if e >= args.start_episode]
        episodes = [e for i, e in enumerate(episodes) if i % args.num_shards == args.shard_id]
        if args.max_episodes is not None:
            episodes = episodes[: args.max_episodes]
        if not episodes:
            logger.warning("No episodes selected for %s", dataset_dir)
            continue

        # Cameras can differ in resolution (R1-Lite head 640x640, wrists 360x640).
        sizes = {}
        for stream in camera_streams:
            probe = base.read_rgb_video(base._video_path(dataset_dir, info, episodes[0], stream["source_key"]))
            sizes[stream["source_key"]] = (int(probe.shape[1]), int(probe.shape[2]))
            del probe
        _register_features(
            info,
            camera_streams=camera_streams,
            sizes=sizes,
            fps=int(info["fps"]),
            args=args,
            compute_depth=compute_depth,
            compute_flow=compute_flow,
        )
        base._save_info(dataset_dir, info)
        logger.info(
            "Dataset %s: %d episodes (shard %d/%d) fps=%s sizes=%s",
            dataset_dir,
            len(episodes),
            args.shard_id,
            args.num_shards,
            info["fps"],
            sizes,
        )

        for ep in tqdm(episodes, desc=dataset_dir.name):
            try:
                for stream in camera_streams:
                    process_episode(
                        dataset_dir,
                        info,
                        ep,
                        stream,
                        depth_estimator=depth_estimator,
                        flow_estimator=flow_estimator,
                        args=args,
                        vcodec=vcodec,
                    )
            except Exception:
                logger.exception("Failed episode %s in %s", ep, dataset_dir)
                raise

        logger.info("Finished %s", dataset_dir)


if __name__ == "__main__":
    main()
