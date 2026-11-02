#!/usr/bin/env python3
"""Replay one training episode to the xArm7 + XHand1 robot client over the FastWAM HTTP protocol.

Stand-in for serve_fastwam_xhand1.py that answers /infer with the recorded
actions of one LeRobot episode instead of model predictions. The unchanged
robot client (inference_xhand1_fastwam.py) then executes the dataset
trajectory through the same interpolation and safety path as the policy, which
separates data problems from model problems.

/metadata is derived from the training config the way the policy server
derives it (cameras, horizon, policy_hz = fps / global_sample_stride, absolute
action semantics). Request chunk_id=k returns the action window a training
sample starting at frame f holds:
    action[f + i * stride], i = 0..horizon-1, f = (k - 1) * execute_steps * stride
with the last action repeated past the episode end, as training pads it.
--execute-steps must equal the client's --execute-steps.

At startup, frame 0 of every camera is written to
<observations-dir>/reference_episode_<index>/ for object placement. Each chunk
saves [dataset frame f | live frame] per model camera plus chunk.json under
<observations-dir>/run_N/demo_1/chunks/chunk_K/.

Usage (FastWAM environment, same --config as the policy server):
  python experiments/xhand1/serve_replay_xhand1.py --config runs/<RUN>/config.yaml --list-episodes
  python experiments/xhand1/serve_replay_xhand1.py --config runs/<RUN>/config.yaml --episode-index 0
then run inference_xhand1_fastwam.py exactly as for the policy, with the prompt
of that episode.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import logging
from pathlib import Path
import time
from typing import Any

import av
import numpy as np
from omegaconf import DictConfig, OmegaConf
import pandas as pd
from fastapi import FastAPI, HTTPException
from PIL import Image
import uvicorn

from serve_fastwam_xhand1 import (
    ACTION_DIM,
    ACTION_NAMES,
    PROJECT_ROOT,
    STATE_DIM,
    InferenceRequest,
    ObservationRecorder,
    StartRunRequest,
    _decode_rgb,
    _load_config,
)

LOGGER = logging.getLogger("fastwam.xhand1.replay")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _lerobot_key(prefix: str, key: str) -> str:
    """BaseLerobotDataset's mapping from shape_meta keys to LeRobot feature names."""
    return prefix if key == "default" else f"{prefix}.{key}"


def _dataset_dir(cfg: DictConfig, override: Path | None) -> Path:
    if override is not None:
        return override.expanduser().resolve()
    dataset_dirs = list(cfg.data.train.dataset_dirs)
    if len(dataset_dirs) != 1:
        raise ValueError(f"Config lists {len(dataset_dirs)} dataset_dirs; choose one with --dataset-dir")
    return Path(str(dataset_dirs[0])).expanduser().resolve()


def _decode_video_frames(path: Path, keep: set[int]) -> tuple[dict[int, np.ndarray], int]:
    """Decode in order (frame i <-> timestamp i / fps); returns the kept frames and the stream's frame count."""
    frames = {}
    last = max(keep)
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        for index, frame in enumerate(container.decode(stream)):
            if index in keep:
                frames[index] = frame.to_ndarray(format="rgb24")
            if index >= last:
                break
        return frames, int(stream.frames)


def _rotvec_to_matrix(rotvec: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(rotvec))
    if angle < 1e-12:
        return np.eye(3)
    x, y, z = rotvec / angle
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * skew @ skew


def _state_error(measured: np.ndarray, recorded: np.ndarray) -> dict[str, float]:
    """Measured robot state vs the dataset state at the same frame."""
    relative = _rotvec_to_matrix(measured[3:6]).T @ _rotvec_to_matrix(recorded[3:6])
    cos_angle = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return {
        "position_mm": float(np.linalg.norm(measured[:3] - recorded[:3]) * 1000.0),
        "rotation_deg": float(np.degrees(np.arccos(cos_angle))),
        "hand_max_rad": float(np.abs(measured[6:] - recorded[6:]).max()),
    }


class DatasetReplay:
    def __init__(self, args: argparse.Namespace) -> None:
        self.recorder = ObservationRecorder(args.observations_dir)
        self.cfg = _load_config(args.config, args.override)
        data_cfg = self.cfg.data.train
        shape_meta = OmegaConf.to_container(data_cfg.shape_meta, resolve=True)
        self.dataset_dir = _dataset_dir(self.cfg, args.dataset_dir)
        info = json.loads((self.dataset_dir / "meta" / "info.json").read_text(encoding="utf-8"))

        # Contract fields, derived as FastWAMXHand1Policy derives them.
        state_meta, action_meta = shape_meta["state"], shape_meta["action"]
        if (
            len(state_meta) != 1 or len(action_meta) != 1
            or state_meta[0]["raw_shape"] != STATE_DIM or action_meta[0]["raw_shape"] != ACTION_DIM
        ):
            raise ValueError("xhand1 replay requires one 18-D state key and one 18-D action key")
        image_meta = shape_meta["images"]
        self.image_keys = [meta["key"] for meta in image_meta]
        self.image_shapes = {meta["key"]: tuple(meta["raw_shape"][1:]) for meta in image_meta}
        self.video_size = tuple(int(x) for x in data_cfg.video_size)
        self.concat_multi_camera = data_cfg.get("concat_multi_camera")
        self.stride = int(data_cfg.global_sample_stride)
        self.policy_hz = float(info["fps"]) / self.stride
        evaluation_cfg = self.cfg.get("EVALUATION", {})
        self.action_semantics = str(evaluation_cfg.get("action_semantics", "absolute"))
        if self.action_semantics != "absolute" or data_cfg.processor.get("delta_action_dim_mask") is not None:
            raise ValueError(
                "Replay sends the dataset's absolute targets; the config must use absolute "
                "action_semantics with processor.delta_action_dim_mask null"
            )
        self.relative_dims = [False] * ACTION_DIM
        self.action_horizon = int(
            args.action_horizon
            if args.action_horizon is not None
            else evaluation_cfg.get("action_horizon") or int(data_cfg.num_frames) - 1
        )
        self.execute_steps = int(args.execute_steps)
        if not 1 <= self.execute_steps <= self.action_horizon:
            raise ValueError(f"--execute-steps must be in [1, {self.action_horizon}]")

        # The episode's recorded actions/states, read from the columns training reads.
        episodes = {int(item["episode_index"]): item for item in _read_jsonl(self.dataset_dir / "meta" / "episodes.jsonl")}
        self.episode_index = int(args.episode_index)
        if self.episode_index not in episodes:
            raise ValueError(f"Episode {self.episode_index} is not in {self.dataset_dir}; see --list-episodes")
        episode = episodes[self.episode_index]
        self.tasks = [str(task) for task in episode["tasks"]]
        episode_chunk = self.episode_index // int(info["chunks_size"])
        data_path = self.dataset_dir / info["data_path"].format(
            episode_chunk=episode_chunk, episode_index=self.episode_index
        )
        action_column = _lerobot_key("action", action_meta[0]["key"])
        state_column = _lerobot_key("observation.state", state_meta[0]["key"])
        table = pd.read_parquet(data_path, columns=[action_column, state_column, "frame_index"])
        self.num_frames = len(table)
        if self.num_frames != int(episode["length"]) or not np.array_equal(
            table["frame_index"].to_numpy(), np.arange(self.num_frames)
        ):
            raise ValueError(f"{data_path}: frame_index is not 0..{int(episode['length']) - 1}")
        self.actions = np.stack(table[action_column].to_numpy()).astype(np.float64)
        self.states = np.stack(table[state_column].to_numpy()).astype(np.float64)
        if (
            self.actions.shape != (self.num_frames, ACTION_DIM) or self.states.shape != (self.num_frames, STATE_DIM)
            or not np.isfinite(self.actions).all() or not np.isfinite(self.states).all()
        ):
            raise ValueError(f"{data_path}: expected finite ({self.num_frames}, 18) {action_column} and {state_column}")
        # Chunks the client needs (--max-chunks) for its last executed target to be the last action.
        chunk_frames = self.execute_steps * self.stride
        self.full_chunks = max(
            1, -(-(self.num_frames - 1 - (self.execute_steps - 1) * self.stride) // chunk_frames) + 1
        )

        # Frame 0 of every camera for object placement; chunk-start frames of the model cameras.
        reference_dir = self.recorder.root / f"reference_episode_{self.episode_index:06d}"
        reference_dir.mkdir(parents=True, exist_ok=True)
        model_videos = {_lerobot_key("observation.images", key): key for key in self.image_keys}
        videos = [key for key, feature in info["features"].items() if feature.get("dtype") == "video"]
        if missing := sorted(set(model_videos) - set(videos)):
            raise ValueError(f"Model cameras {missing} are not videos of {self.dataset_dir}")
        self.dataset_frames: dict[str, dict[int, np.ndarray]] = {}
        for video_key in videos:
            camera = model_videos.get(video_key)
            keep = set(range(0, self.num_frames, chunk_frames)) if camera else {0}
            video_path = self.dataset_dir / info["video_path"].format(
                episode_chunk=episode_chunk, video_key=video_key, episode_index=self.episode_index
            )
            frames, stream_frames = _decode_video_frames(video_path, keep)
            if len(frames) != len(keep):
                raise ValueError(f"{video_path}: decoded {len(frames)} of the {len(keep)} needed frames")
            if stream_frames not in (0, self.num_frames):
                LOGGER.warning("%s has %d frames but the episode has %d rows", video_path, stream_frames, self.num_frames)
            Image.fromarray(frames[0]).save(reference_dir / f"{video_key.rsplit('.', 1)[-1]}_frame0.jpg", quality=95)
            if camera is not None:
                if frames[0].shape[:2] != self.image_shapes[camera]:
                    raise ValueError(f"{video_path}: frames are {frames[0].shape[:2]}, config raw_shape {self.image_shapes[camera]}")
                self.dataset_frames[camera] = frames
        (reference_dir / "episode.json").write_text(
            json.dumps(
                {
                    "dataset_dir": str(self.dataset_dir),
                    "episode_index": self.episode_index,
                    "tasks": self.tasks,
                    "num_frames": self.num_frames,
                    "fps": float(info["fps"]),
                    "state_names": ACTION_NAMES,
                    "start_state": self.states[0].tolist(),
                    "first_action": self.actions[0].tolist(),
                },
                indent=2,
            ) + "\n",
            encoding="utf-8",
        )
        self.writer = ThreadPoolExecutor(max_workers=1)
        start = self.states[0]
        LOGGER.info(
            "Episode %d %s: %d frames (%.1f s), cameras %s, horizon %d @ %g Hz, %d actions per chunk",
            self.episode_index, self.tasks, self.num_frames, self.num_frames / float(info["fps"]),
            self.image_keys, self.action_horizon, self.policy_hz, self.execute_steps,
        )
        LOGGER.info(
            "Start state: xyz %s mm, rotvec %s rad, hand %s rad",
            np.round(start[:3] * 1000.0, 1).tolist(), np.round(start[3:6], 4).tolist(), np.round(start[6:], 3).tolist(),
        )
        LOGGER.info("Reference frames for object placement: %s", reference_dir)

    def start_run(self, request: StartRunRequest) -> str:
        if request.prompt.strip() not in self.tasks:
            raise ValueError(f"Prompt {request.prompt!r} is not a task of episode {self.episode_index}: {self.tasks}")
        run_id = self.recorder.start_run(request)
        LOGGER.info(
            "%s: replaying episode %d; the client must use --execute-steps %d, and --max-chunks %d plays it once",
            run_id, self.episode_index, self.execute_steps, self.full_chunks,
        )
        return run_id

    def infer(self, request: InferenceRequest) -> tuple[np.ndarray, dict[str, Any]]:
        if set(request.images) != set(self.image_keys):
            raise ValueError(f"Expected images {self.image_keys}, got {sorted(request.images)}")
        live = {}
        for key in self.image_keys:
            rgb = _decode_rgb(request.images[key], f"images[{key}]")
            if rgb.shape[:2] != self.image_shapes[key]:
                raise ValueError(f"images[{key}] is {rgb.shape[1]}x{rgb.shape[0]}, expected post-ROI {self.image_shapes[key]}")
            live[key] = rgb
        state = np.asarray(request.state, dtype=np.float64)
        if not np.isfinite(state).all():
            raise ValueError("state contains non-finite values")

        start = (request.chunk_id - 1) * self.execute_steps * self.stride
        indices = np.minimum(start + self.stride * np.arange(self.action_horizon), self.num_frames - 1)
        actions = self.actions[indices]
        finished = start >= self.num_frames - 1
        compared_frame = min(start, self.num_frames - 1)
        error = _state_error(state, self.states[compared_frame])
        LOGGER.info(
            "%s chunk %d: frames %d-%d of %d%s | measured vs dataset frame %d: %.1f mm, %.2f deg, hand max %.3f rad",
            request.run_id, request.chunk_id, indices[0], indices[self.execute_steps - 1], self.num_frames,
            " (end reached, holding the last action)" if finished else "",
            compared_frame, error["position_mm"], error["rotation_deg"], error["hand_max_rad"],
        )
        record = {
            "chunk_id": request.chunk_id,
            "episode_index": self.episode_index,
            "frame_indices": indices.tolist(),
            "finished": finished,
            "measured_state": state.tolist(),
            "dataset_state": self.states[compared_frame].tolist(),
            "dataset_state_frame": compared_frame,
            "state_error": error,
            "actions": actions.tolist(),
        }
        chunk_dir = self.recorder.chunk_dir(request.run_id, request.demo_id, request.chunk_id)
        self.writer.submit(self._save_chunk, chunk_dir, record, live, start)
        replay = {"episode_index": self.episode_index, "frame": start, "num_frames": self.num_frames, "finished": finished}
        return actions, replay

    def _save_chunk(self, chunk_dir: Path, record: dict[str, Any], live: dict[str, np.ndarray], start: int) -> None:
        try:
            for key, rgb in live.items():
                recorded = self.dataset_frames[key].get(start)
                image = rgb if recorded is None else np.concatenate([recorded, rgb], axis=1)
                Image.fromarray(image).save(chunk_dir / f"{key}_dataset_vs_live.jpg", quality=90)
            (chunk_dir / "chunk.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        except Exception:  # noqa: BLE001
            LOGGER.exception("Saving %s failed", chunk_dir)


def build_app(replay: DatasetReplay) -> FastAPI:
    app = FastAPI(title="FastWAM xArm7 + XHand1 dataset replay")

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/metadata")
    def metadata() -> dict[str, object]:
        height, width = replay.video_size
        return {
            "state_dim": STATE_DIM,
            "action_dim": ACTION_DIM,
            "action_names": ACTION_NAMES,
            "action_horizon": replay.action_horizon,
            "policy_hz": replay.policy_hz,
            "image_keys": replay.image_keys,
            "image_shapes": {key: list(shape) for key, shape in replay.image_shapes.items()},
            "camera_layout": (
                f"post-ROI inputs {replay.image_keys}; concat_multi_camera={replay.concat_multi_camera}; "
                f"model image {height}x{width}"
            ),
            "action_semantics": replay.action_semantics,
            "relative_dims": replay.relative_dims,
            "observations_dir": str(replay.recorder.root),
            "replay": {
                "dataset_dir": str(replay.dataset_dir),
                "episode_index": replay.episode_index,
                "tasks": replay.tasks,
                "num_frames": replay.num_frames,
                "execute_steps": replay.execute_steps,
                "full_chunks": replay.full_chunks,
            },
        }

    @app.post("/runs/start")
    def start_run(request: StartRunRequest) -> dict[str, object]:
        try:
            run_id = replay.start_run(request)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"run_id": run_id, "num_demos": request.num_demos}

    @app.post("/infer")
    def infer(request: InferenceRequest) -> dict[str, object]:
        started = time.perf_counter()
        try:
            actions, info = replay.infer(request)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {
            "actions": actions.tolist(),
            "action_horizon": replay.action_horizon,
            "action_semantics": replay.action_semantics,
            "policy_hz": replay.policy_hz,
            "active_future": None,
            "server_timing": {"infer_ms": (time.perf_counter() - started) * 1000.0},
            "replay": info,
        }

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="The policy server's --config: runs/<run>/config.yaml saved by training, or configs/task/<task>.yaml",
    )
    parser.add_argument("--override", action="append", default=[], help="Repeat for each Hydra override")
    parser.add_argument(
        "--dataset-dir", type=Path, help="LeRobot dataset to replay; default: the config's single data.train.dataset_dirs entry"
    )
    parser.add_argument("--episode-index", type=int, help="episode_index in the dataset's meta/episodes.jsonl")
    parser.add_argument("--list-episodes", action="store_true", help="Print episode index, length, and tasks, then exit")
    parser.add_argument("--execute-steps", type=int, default=8, help="Must equal the client's --execute-steps")
    parser.add_argument("--action-horizon", type=int)
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Loopback by default (no auth); reach it through an SSH tunnel. 0.0.0.0 exposes it to the network.",
    )
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--observations-dir",
        type=Path,
        default=PROJECT_ROOT / "server_observations" / "replay",
        help="Root for reference frames and run_N/demo_M chunk logs",
    )
    args = parser.parse_args()
    if not args.list_episodes and args.episode_index is None:
        parser.error("--episode-index is required (see --list-episodes)")
    return args


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    if args.list_episodes:
        dataset_dir = _dataset_dir(_load_config(args.config, args.override), args.dataset_dir)
        for item in _read_jsonl(dataset_dir / "meta" / "episodes.jsonl"):
            print(f"{item['episode_index']:6d}  {item['length']:6d} frames  {', '.join(item['tasks'])}")
        return
    replay = DatasetReplay(args)
    uvicorn.run(build_app(replay), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
