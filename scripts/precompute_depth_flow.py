#!/usr/bin/env python3
"""Offline precompute Depth Anything V2 depth and RAFT optical flow for LeRobot datasets.

Depth uses the official Depth-Anything-V2 implementation:
  third_party/Depth-Anything-V2  (https://github.com/DepthAnything/Depth-Anything-V2)
with checkpoints depth_anything_v2_{vits|vitb|vitl}.pth (auto-downloaded to checkpoints/).

Writes RGB-compatible mp4 videos under (per camera):
  videos/chunk-XXX/observation.images.depth/episode_YYYYYY.mp4
  videos/chunk-XXX/observation.images.flow/episode_YYYYYY.mp4
  videos/chunk-XXX/observation.images.wrist_depth/episode_YYYYYY.mp4
  videos/chunk-XXX/observation.images.wrist_flow/episode_YYYYYY.mp4

and registers the features in meta/info.json.

Encoding conventions are documented in:
  fastwam.datasets.lerobot.utils.depth_flow_codec

Example (single suite, 1 episode smoke test):
  python scripts/precompute_depth_flow.py \\
    --dataset-dir data/libero_mujoco3.3.2/libero_goal_no_noops_lerobot \\
    --depth-encoder vits --max-episodes 1 --device cuda

Example (all LIBERO suites, Large encoder):
  python scripts/precompute_depth_flow.py \\
    --dataset-dir data/libero_mujoco3.3.2/libero_*_lerobot \\
    --depth-encoder vitl --device cuda --skip-existing
"""

from __future__ import annotations

import argparse
import glob as globmod
import json
import logging
import shutil
import sys
from pathlib import Path

import av
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from fastwam.datasets.lerobot.utils.depth_flow_codec import (
    DEFAULT_FLOW_CLIP_PX,
    DEFAULT_FLOW_NOISE_THRESH_PX,
    depth_to_uint8_rgb,
    feature_dict_for_depth,
    feature_dict_for_flow,
    flow_to_uint8_rgb,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("precompute_depth_flow")

# RGB source -> depth / flow output keys
DEFAULT_CAMERA_STREAMS = (
    {
        "source_key": "observation.images.image",
        "depth_key": "observation.images.depth",
        "flow_key": "observation.images.flow",
    },
    {
        "source_key": "observation.images.wrist_image",
        "depth_key": "observation.images.wrist_depth",
        "flow_key": "observation.images.wrist_flow",
    },
)

# Official Depth Anything V2 (https://github.com/DepthAnything/Depth-Anything-V2)
DEPTH_ANYTHING_V2_ROOT = Path(__file__).resolve().parents[1] / "third_party" / "Depth-Anything-V2"
DEPTH_ANYTHING_V2_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}
DEPTH_ANYTHING_V2_CKPT_URLS = {
    "vits": "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth",
    "vitb": "https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth",
    "vitl": "https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth",
}
DEFAULT_DEPTH_ENCODER = "vits"


def _expand_dataset_dirs(patterns: list[str]) -> list[Path]:
    dirs: list[Path] = []
    for pattern in patterns:
        p = Path(pattern).expanduser()
        if any(ch in pattern for ch in "*?["):
            matches = [Path(m) for m in sorted(globmod.glob(pattern))]
            dirs.extend([m.resolve() for m in matches if m.is_dir()])
        elif p.is_dir():
            dirs.append(p.resolve())
        else:
            raise FileNotFoundError(f"Dataset dir not found: {pattern}")
    seen = set()
    out = []
    for d in dirs:
        if d not in seen:
            seen.add(d)
            out.append(d)
    if not out:
        raise FileNotFoundError(f"No dataset dirs matched: {patterns}")
    return out


def _load_info(dataset_dir: Path) -> dict:
    info_path = dataset_dir / "meta" / "info.json"
    with open(info_path) as f:
        return json.load(f)


def _save_info(dataset_dir: Path, info: dict) -> None:
    info_path = dataset_dir / "meta" / "info.json"
    backup = info_path.with_suffix(".json.bak_depth_flow")
    if not backup.exists():
        shutil.copy2(info_path, backup)
    with open(info_path, "w") as f:
        json.dump(info, f, indent=4)
        f.write("\n")


def _episode_indices(dataset_dir: Path, info: dict) -> list[int]:
    episodes_path = dataset_dir / "meta" / "episodes.jsonl"
    indices: list[int] = []
    if episodes_path.exists():
        with open(episodes_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                indices.append(int(json.loads(line)["episode_index"]))
        return indices
    return list(range(int(info["total_episodes"])))


def _video_path(dataset_dir: Path, info: dict, episode_index: int, video_key: str) -> Path:
    chunks_size = int(info["chunks_size"])
    episode_chunk = episode_index // chunks_size
    rel = info["video_path"].format(
        episode_chunk=episode_chunk,
        video_key=video_key,
        episode_index=episode_index,
    )
    return dataset_dir / rel


def read_rgb_video(path: Path) -> np.ndarray:
    """Return uint8 RGB frames [T,H,W,3]."""
    frames: list[np.ndarray] = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for frame in container.decode(stream):
            frames.append(frame.to_ndarray(format="rgb24"))
    if not frames:
        raise RuntimeError(f"No frames decoded from {path}")
    return np.stack(frames, axis=0)


def write_rgb_video(
    path: Path,
    frames_thwc_u8: np.ndarray,
    fps: int,
    vcodec: str = "libx264",
    crf: int = 20,
    overwrite: bool = False,
) -> None:
    if frames_thwc_u8.ndim != 4 or frames_thwc_u8.shape[-1] != 3:
        raise ValueError(f"Expected [T,H,W,3] uint8, got {frames_thwc_u8.shape}")
    if frames_thwc_u8.dtype != np.uint8:
        raise ValueError(f"Expected uint8 frames, got {frames_thwc_u8.dtype}")

    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    t, h, w, _ = frames_thwc_u8.shape
    # yuv420p requires even spatial dims
    if h % 2 != 0 or w % 2 != 0:
        raise ValueError(f"Frame size must be even for yuv420p, got HxW={h}x{w}")

    options = {"crf": str(crf)}
    if vcodec == "libx264":
        options["preset"] = "veryfast"

    with av.open(str(path), mode="w") as output:
        stream = output.add_stream(vcodec, rate=fps, options=options)
        stream.width = w
        stream.height = h
        stream.pix_fmt = "yuv420p"
        for i in range(t):
            frame = av.VideoFrame.from_ndarray(frames_thwc_u8[i], format="rgb24")
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)


def _ensure_depth_anything_v2_on_path() -> None:
    if not DEPTH_ANYTHING_V2_ROOT.is_dir():
        raise FileNotFoundError(
            f"Official Depth-Anything-V2 repo not found at {DEPTH_ANYTHING_V2_ROOT}. "
            "Clone it with: git clone https://github.com/DepthAnything/Depth-Anything-V2.git "
            "third_party/Depth-Anything-V2"
        )
    root_str = str(DEPTH_ANYTHING_V2_ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def _resolve_depth_anything_v2_checkpoint(encoder: str, checkpoint: str | None) -> Path:
    if checkpoint is not None:
        ckpt_path = Path(checkpoint).expanduser().resolve()
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Depth Anything V2 checkpoint not found: {ckpt_path}")
        return ckpt_path

    ckpt_dir = Path(__file__).resolve().parents[1] / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"depth_anything_v2_{encoder}.pth"
    if ckpt_path.is_file():
        return ckpt_path

    if encoder not in DEPTH_ANYTHING_V2_CKPT_URLS:
        raise ValueError(
            f"No auto-download URL for encoder={encoder}. Pass --depth-checkpoint explicitly."
        )
    url = DEPTH_ANYTHING_V2_CKPT_URLS[encoder]
    logger.info("Downloading Depth Anything V2 checkpoint (%s) -> %s", encoder, ckpt_path)
    try:
        from huggingface_hub import hf_hub_download

        repo_map = {
            "vits": "depth-anything/Depth-Anything-V2-Small",
            "vitb": "depth-anything/Depth-Anything-V2-Base",
            "vitl": "depth-anything/Depth-Anything-V2-Large",
        }
        downloaded = hf_hub_download(
            repo_id=repo_map[encoder],
            filename=f"depth_anything_v2_{encoder}.pth",
            local_dir=str(ckpt_dir),
            local_dir_use_symlinks=False,
        )
        downloaded_path = Path(downloaded)
        if downloaded_path.resolve() != ckpt_path.resolve():
            shutil.copy2(downloaded_path, ckpt_path)
    except Exception as hub_err:
        logger.warning("huggingface_hub download failed (%s); falling back to urllib.", hub_err)
        import urllib.request

        tmp_path = ckpt_path.with_suffix(".pth.partial")
        urllib.request.urlretrieve(url, tmp_path)
        tmp_path.replace(ckpt_path)
    return ckpt_path


class DepthAnythingV2Estimator:
    """Official Depth Anything V2 estimator (DepthAnything/Depth-Anything-V2)."""

    def __init__(
        self,
        encoder: str = DEFAULT_DEPTH_ENCODER,
        device: torch.device | None = None,
        checkpoint: str | None = None,
        input_size: int = 518,
    ):
        if encoder not in DEPTH_ANYTHING_V2_CONFIGS:
            raise ValueError(
                f"Unknown Depth Anything V2 encoder '{encoder}'. "
                f"Choose from {sorted(DEPTH_ANYTHING_V2_CONFIGS)}"
            )

        _ensure_depth_anything_v2_on_path()
        import cv2  # noqa: F401  # required by depth_anything_v2 transforms
        from depth_anything_v2.dpt import DepthAnythingV2

        self.encoder = encoder
        self.input_size = int(input_size)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt_path = _resolve_depth_anything_v2_checkpoint(encoder, checkpoint)

        logger.info(
            "Loading official Depth Anything V2 encoder=%s checkpoint=%s device=%s",
            encoder,
            ckpt_path,
            self.device,
        )
        self.model = DepthAnythingV2(**DEPTH_ANYTHING_V2_CONFIGS[encoder])
        state = torch.load(ckpt_path, map_location="cpu")
        self.model.load_state_dict(state)
        self.model = self.model.to(self.device).eval()

    @torch.inference_mode()
    def __call__(self, frames_thwc_u8: np.ndarray, batch_size: int = 8) -> np.ndarray:
        """frames RGB [T,H,W,3] uint8 -> depth [T,H,W] float32 (DA-V2 disparity-like)."""
        t, h, w, _ = frames_thwc_u8.shape
        depths: list[np.ndarray] = []
        for start in range(0, t, batch_size):
            batch = frames_thwc_u8[start : start + batch_size]
            tensors = []
            for fr_rgb in batch:
                # Official API expects OpenCV BGR uint8.
                fr_bgr = fr_rgb[:, :, ::-1].copy()
                image, (_h, _w) = self.model.image2tensor(fr_bgr, self.input_size)
                tensors.append(image.to(self.device))
            x = torch.cat(tensors, dim=0)
            pred = self.model.forward(x)  # [B,H',W']
            pred = F.interpolate(
                pred[:, None].float(),
                size=(h, w),
                mode="bilinear",
                align_corners=True,
            ).squeeze(1)
            depths.append(pred.cpu().numpy())
        return np.concatenate(depths, axis=0).astype(np.float32)


class RaftFlowEstimator:
    def __init__(self, device: torch.device, model_size: str = "large"):
        from torchvision.models.optical_flow import (
            Raft_Large_Weights,
            Raft_Small_Weights,
            raft_large,
            raft_small,
        )

        self.device = device
        logger.info("Loading torchvision RAFT (%s)", model_size)
        if model_size == "large":
            weights = Raft_Large_Weights.DEFAULT
            self.model = raft_large(weights=weights, progress=True)
        elif model_size == "small":
            weights = Raft_Small_Weights.DEFAULT
            self.model = raft_small(weights=weights, progress=True)
        else:
            raise ValueError(f"Unknown RAFT size: {model_size}")
        self.transforms = weights.transforms()
        self.model.to(device)
        self.model.eval()

    @torch.inference_mode()
    def __call__(self, frames_thwc_u8: np.ndarray, batch_size: int = 4) -> np.ndarray:
        """frames [T,H,W,3] -> flow [T,2,H,W]; flow[t]=t->t+1, flow[-1]=0."""
        t, h, w, _ = frames_thwc_u8.shape
        if t == 1:
            return np.zeros((1, 2, h, w), dtype=np.float32)

        # torchvision RAFT expects float tensors in [0,1], CHW
        frames = torch.from_numpy(frames_thwc_u8).permute(0, 3, 1, 2).float() / 255.0
        img1 = frames[:-1]
        img2 = frames[1:]
        flows = []
        for start in range(0, img1.shape[0], batch_size):
            a = img1[start : start + batch_size].to(self.device)
            b = img2[start : start + batch_size].to(self.device)
            a_t, b_t = self.transforms(a, b)
            pred_list = self.model(a_t, b_t)
            flow = pred_list[-1]  # [B,2,H',W']
            ht, wt = int(flow.shape[-2]), int(flow.shape[-1])
            if (ht, wt) != (h, w):
                flow = F.interpolate(flow, size=(h, w), mode="bilinear", align_corners=False)
                # rescale flow vectors from transformed resolution to original pixels
                flow = flow.clone()
                flow[:, 0] *= w / float(wt)
                flow[:, 1] *= h / float(ht)
            flows.append(flow.float().cpu())
        flow_tm1 = torch.cat(flows, dim=0).numpy().astype(np.float32)  # [T-1,2,H,W]
        last = np.zeros((1, 2, h, w), dtype=np.float32)
        return np.concatenate([flow_tm1, last], axis=0)


def _register_features(
    info: dict,
    camera_streams: list[dict[str, str]],
    height: int,
    width: int,
    fps: int,
    flow_clip_px: float,
    flow_noise_thresh_px: float = DEFAULT_FLOW_NOISE_THRESH_PX,
    depth_encoder: str | None = None,
) -> None:
    info.setdefault("features", {})
    depth_extra = None
    if depth_encoder is not None:
        depth_extra = {"depth_encoder": depth_encoder}
    for stream in camera_streams:
        info["features"][stream["depth_key"]] = feature_dict_for_depth(
            height, width, fps, extra_info=depth_extra
        )
        info["features"][stream["flow_key"]] = feature_dict_for_flow(
            height,
            width,
            fps,
            flow_clip_px=flow_clip_px,
            extra_info={"flow_noise_thresh_px": float(flow_noise_thresh_px)},
        )
    n_keys = sum(1 for f in info["features"].values() if f.get("dtype") == "video")
    info["total_videos"] = int(info["total_episodes"]) * int(n_keys)


def process_episode(
    dataset_dir: Path,
    info: dict,
    episode_index: int,
    source_key: str,
    depth_key: str,
    flow_key: str,
    depth_estimator: DepthAnythingV2Estimator | None,
    flow_estimator: RaftFlowEstimator | None,
    *,
    compute_depth: bool,
    compute_flow: bool,
    flow_clip_px: float,
    flow_noise_thresh_px: float,
    depth_batch_size: int,
    flow_batch_size: int,
    vcodec: str,
    skip_existing: bool,
    overwrite: bool,
) -> dict[str, float]:
    src = _video_path(dataset_dir, info, episode_index, source_key)
    if not src.exists():
        raise FileNotFoundError(src)

    depth_out = _video_path(dataset_dir, info, episode_index, depth_key)
    flow_out = _video_path(dataset_dir, info, episode_index, flow_key)

    need_depth = compute_depth and (overwrite or not (skip_existing and depth_out.exists()))
    need_flow = compute_flow and (overwrite or not (skip_existing and flow_out.exists()))
    if not need_depth and not need_flow:
        return {"skipped": 1.0}

    frames = read_rgb_video(src)
    fps = int(info["fps"])
    stats: dict[str, float] = {"frames": float(frames.shape[0])}

    if need_depth:
        assert depth_estimator is not None
        depth = depth_estimator(frames, batch_size=depth_batch_size)
        depth_rgb, meta = depth_to_uint8_rgb(depth)
        write_rgb_video(depth_out, depth_rgb, fps=fps, vcodec=vcodec, overwrite=True)
        stats["depth_min"] = meta["depth_min"]
        stats["depth_max"] = meta["depth_max"]

    if need_flow:
        assert flow_estimator is not None
        flow = flow_estimator(frames, batch_size=flow_batch_size)
        flow_rgb = flow_to_uint8_rgb(
            flow,
            flow_clip_px=flow_clip_px,
            noise_thresh_px=flow_noise_thresh_px,
        )
        write_rgb_video(flow_out, flow_rgb, fps=fps, vcodec=vcodec, overwrite=True)
        stats["flow_abs_p95"] = float(np.percentile(np.abs(flow[:, :2]), 95))

    return stats


def _resolve_camera_streams(source_keys: list[str] | None) -> list[dict[str, str]]:
    catalog = {s["source_key"]: s for s in DEFAULT_CAMERA_STREAMS}
    if not source_keys:
        return [dict(s) for s in DEFAULT_CAMERA_STREAMS]
    streams = []
    for key in source_keys:
        if key not in catalog:
            raise ValueError(
                f"Unknown source key '{key}'. Known: {sorted(catalog)}. "
                "Add a mapping in DEFAULT_CAMERA_STREAMS if needed."
            )
        streams.append(dict(catalog[key]))
    return streams


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--dataset-dir",
        nargs="+",
        required=True,
        help="One or more LeRobot dataset roots (glob ok).",
    )
    p.add_argument(
        "--source-key",
        nargs="+",
        default=None,
        help=(
            "RGB video key(s) to process. Default: both agentview and wrist "
            "(observation.images.image, observation.images.wrist_image)."
        ),
    )
    p.add_argument(
        "--depth-encoder",
        default=DEFAULT_DEPTH_ENCODER,
        choices=sorted(DEPTH_ANYTHING_V2_CONFIGS),
        help="Official Depth Anything V2 encoder (vits/vitb/vitl/vitg).",
    )
    p.add_argument(
        "--depth-checkpoint",
        default=None,
        help="Path to depth_anything_v2_{encoder}.pth (auto-download to checkpoints/ if omitted).",
    )
    p.add_argument("--depth-input-size", type=int, default=518, help="DA-V2 inference size.")
    p.add_argument("--raft-size", choices=["large", "small"], default="large")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--flow-clip-px",
        type=float,
        default=DEFAULT_FLOW_CLIP_PX,
        help="FlowWAM magnitude cap in pixels before HSV saturation normalize (default 25).",
    )
    p.add_argument(
        "--flow-noise-thresh-px",
        type=float,
        default=DEFAULT_FLOW_NOISE_THRESH_PX,
        help="FlowWAM noise threshold: zero displacements with |f| below this (default 0.5).",
    )
    p.add_argument("--depth-batch-size", type=int, default=8)
    p.add_argument("--flow-batch-size", type=int, default=4)
    p.add_argument("--vcodec", default="libx264", choices=["libx264", "libsvtav1", "h264"])
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--max-episodes", type=int, default=None)
    p.add_argument("--start-episode", type=int, default=0)
    p.add_argument("--shard-id", type=int, default=0, help="For multi-job sharding.")
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

    dataset_dirs = _expand_dataset_dirs(args.dataset_dir)
    device = torch.device(args.device)
    compute_depth = not args.no_depth
    compute_flow = not args.no_flow

    depth_estimator = (
        DepthAnythingV2Estimator(
            encoder=args.depth_encoder,
            device=device,
            checkpoint=args.depth_checkpoint,
            input_size=args.depth_input_size,
        )
        if compute_depth
        else None
    )
    flow_estimator = RaftFlowEstimator(device=device, model_size=args.raft_size) if compute_flow else None

    vcodec = "libx264" if args.vcodec == "h264" else args.vcodec

    camera_streams = _resolve_camera_streams(args.source_key)
    logger.info(
        "Camera streams: %s",
        [(s["source_key"], s["depth_key"], s["flow_key"]) for s in camera_streams],
    )

    for dataset_dir in dataset_dirs:
        info = _load_info(dataset_dir)
        for stream in camera_streams:
            if stream["source_key"] not in info.get("features", {}):
                raise KeyError(f"{stream['source_key']} not in features of {dataset_dir}")

        # probe resolution from first existing source video
        episodes = _episode_indices(dataset_dir, info)
        episodes = [e for e in episodes if e >= args.start_episode]
        episodes = [e for i, e in enumerate(episodes) if i % args.num_shards == args.shard_id]
        if args.max_episodes is not None:
            episodes = episodes[: args.max_episodes]

        if not episodes:
            logger.warning("No episodes selected for %s", dataset_dir)
            continue

        probe = read_rgb_video(
            _video_path(dataset_dir, info, episodes[0], camera_streams[0]["source_key"])
        )
        _, h, w, _ = probe.shape
        del probe
        _register_features(
            info,
            camera_streams=camera_streams,
            height=h,
            width=w,
            fps=int(info["fps"]),
            flow_clip_px=args.flow_clip_px,
            flow_noise_thresh_px=args.flow_noise_thresh_px,
            depth_encoder=args.depth_encoder if compute_depth else None,
        )
        _save_info(dataset_dir, info)
        logger.info(
            "Dataset %s: %d episodes (shard %d/%d) HxW=%dx%d fps=%s cameras=%d",
            dataset_dir,
            len(episodes),
            args.shard_id,
            args.num_shards,
            h,
            w,
            info["fps"],
            len(camera_streams),
        )

        for ep in tqdm(episodes, desc=dataset_dir.name):
            try:
                for stream in camera_streams:
                    process_episode(
                        dataset_dir,
                        info,
                        ep,
                        source_key=stream["source_key"],
                        depth_key=stream["depth_key"],
                        flow_key=stream["flow_key"],
                        depth_estimator=depth_estimator,
                        flow_estimator=flow_estimator,
                        compute_depth=compute_depth,
                        compute_flow=compute_flow,
                        flow_clip_px=args.flow_clip_px,
                        flow_noise_thresh_px=args.flow_noise_thresh_px,
                        depth_batch_size=args.depth_batch_size,
                        flow_batch_size=args.flow_batch_size,
                        vcodec=vcodec,
                        skip_existing=args.skip_existing,
                        overwrite=args.overwrite,
                    )
            except Exception:
                logger.exception("Failed episode %s in %s", ep, dataset_dir)
                raise

        logger.info("Finished %s", dataset_dir)


if __name__ == "__main__":
    main()
