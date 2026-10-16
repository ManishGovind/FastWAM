#!/usr/bin/env python3
"""Serve FastWAM for the Galaxea R1 Lite over HTTP.

The server owns FastWAM preprocessing, state normalization, inference, and
action de-normalization. Returned actions are:
    [d_left_6, d_right_6, left_gripper_target, right_gripper_target]
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime
import hashlib
import inspect
import io
import json
import logging
from pathlib import Path
import queue
import re
import sys
import threading
import time
from typing import Annotated, Any

import numpy as np
from fastapi import FastAPI, HTTPException
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from pydantic import BaseModel, Field
import torch
import uvicorn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor
from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json

LOGGER = logging.getLogger("fastwam.r1lite.server")
STATE_DIM = ACTION_DIM = 14
# Saving run purposes
RUN_ID_PATTERN = r"^run_[1-9][0-9]*$"
DEMO_ID_PATTERN = r"^demo_[1-9][0-9]*$"

# HTTP payload schemes
class InferenceRequest(BaseModel):
    """Current feedback plus base64 JPEG/PNG RGB images."""

    prompt: Annotated[str, Field(min_length=1)]
    state: Annotated[list[float], Field(min_length=STATE_DIM, max_length=STATE_DIM)]
    head_image: Annotated[str, Field(min_length=1)]
    left_wrist_image: Annotated[str, Field(min_length=1)]
    right_wrist_image: Annotated[str, Field(min_length=1)]
    run_id: Annotated[str, Field(pattern=RUN_ID_PATTERN)]
    demo_id: Annotated[str, Field(pattern=DEMO_ID_PATTERN)]
    chunk_id: Annotated[int, Field(ge=1)]


class StartRunRequest(BaseModel):
    prompt: Annotated[str, Field(min_length=1)]
    num_demos: Annotated[int, Field(ge=1)]
    execute: bool
    control_hz: Annotated[float, Field(gt=0)]


def _wall_time() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


class ObservationRecorder:
    """Persist server model inputs and robot execution traces by run/demo."""

    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.pending: queue.Queue[tuple[str, tuple[Any, ...]]] = queue.Queue()
        self.worker = threading.Thread(target=self._writer_loop, daemon=True)
        self.worker.start()

    def _writer_loop(self) -> None:
        while True:
            operation, arguments = self.pending.get()
            try:
                if operation == "model_input":
                    self._save_model_input_sync(*arguments)
                elif operation == "model_output":
                    self._save_model_output_sync(*arguments)
            except Exception:  # noqa: BLE001
                LOGGER.exception("Asynchronous observation write failed (%s)", operation)
            finally:
                self.pending.task_done()

    def _demo_dir(self, run_id: str, demo_id: str) -> Path:
        if re.fullmatch(RUN_ID_PATTERN, run_id) is None or re.fullmatch(DEMO_ID_PATTERN, demo_id) is None:
            raise ValueError("Invalid run/demo identifier")
        path = self.root / run_id / demo_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)

    def start_run(self, request: StartRunRequest) -> str:
        with self.lock:
            indices = [
                int(match.group(1))
                for path in self.root.iterdir()
                if path.is_dir() and (match := re.fullmatch(r"run_([1-9][0-9]*)", path.name))
            ]
            run_id = f"run_{max(indices, default=0) + 1}"
            run_dir = self.root / run_id
            run_dir.mkdir()
            self._write_json(
                run_dir / "run.json",
                {
                    "run_id": run_id,
                    "created_at": _wall_time(),
                    **request.model_dump(),
                },
            )
        return run_id


    def chunk_dir(self, run_id: str, demo_id: str, chunk_id: int) -> Path:
        with self.lock:
            path = self._demo_dir(run_id, demo_id) / "chunks" / f"chunk_{chunk_id}"
            path.mkdir(parents=True, exist_ok=True)
        return path

    def save_model_input(
        self,
        request: InferenceRequest,
        composed_image: np.ndarray,
        model_kwargs: dict[str, Any],
    ) -> Path:
        """Save the exact tensor inputs immediately before model inference."""
        chunk_dir = self.chunk_dir(request.run_id, request.demo_id, request.chunk_id)
        tensor_payload = {
            key: value.detach().cpu().clone() if isinstance(value, torch.Tensor) else value
            for key, value in model_kwargs.items()
        }
        metadata = {
            "saved_at": _wall_time(),
            "run_id": request.run_id,
            "demo_id": request.demo_id,
            "chunk_id": request.chunk_id,
            "prompt": request.prompt,
            "raw_state": request.state,
            "tensor_shapes": {
                key: list(value.shape)
                for key, value in tensor_payload.items()
                if isinstance(value, torch.Tensor)
            },
            "tensor_dtypes": {
                key: str(value.dtype)
                for key, value in tensor_payload.items()
                if isinstance(value, torch.Tensor)
            },
        }
        self.pending.put_nowait(
            ("model_input", (chunk_dir, composed_image.copy(), tensor_payload, metadata))
        )
        return chunk_dir

    def _save_model_input_sync(
        self,
        chunk_dir: Path,
        composed_image: np.ndarray,
        tensor_payload: dict[str, Any],
        metadata: dict[str, Any],
    ) -> None:
        with self.lock:
            Image.fromarray(composed_image).save(chunk_dir / "model_input.png")
            tensor_tmp = chunk_dir / "model_input.pt.tmp"
            torch.save(tensor_payload, tensor_tmp)
            tensor_tmp.replace(chunk_dir / "model_input.pt")
            self._write_json(chunk_dir / "request.json", metadata)

    def save_model_output(
        self,
        chunk_dir: Path,
        normalized_action: torch.Tensor,
        actions: np.ndarray,
        decoded_images: dict[str, np.ndarray],
    ) -> None:
        payload = {
            "saved_at": _wall_time(),
            "action_semantics": [
                "left_arm_delta_6",
                "right_arm_delta_6",
                "left_gripper_absolute",
                "right_gripper_absolute",
            ],
            "actions": actions.tolist(),
        }
        self.pending.put_nowait(
            (
                "model_output",
                (
                    chunk_dir,
                    normalized_action.detach().cpu().clone(),
                    payload,
                    {name: image.copy() for name, image in decoded_images.items()},
                ),
            )
        )

    def _save_model_output_sync(
        self,
        chunk_dir: Path,
        normalized_action: torch.Tensor,
        payload: dict[str, Any],
        decoded_images: dict[str, np.ndarray],
    ) -> None:
        with self.lock:
            normalized_tmp = chunk_dir / "normalized_model_output.pt.tmp"
            torch.save(normalized_action, normalized_tmp)
            normalized_tmp.replace(chunk_dir / "normalized_model_output.pt")
            self._write_json(chunk_dir / "model_actions.json", payload)
            for name, image in decoded_images.items():
                Image.fromarray(image).save(chunk_dir / f"decoded_{name}.png")


def _model_dtype(name: str) -> torch.dtype:
    return {"no": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[name]

# Decoding/Converting from base64 to RGB
def _decode_rgb(encoded: str, name: str) -> np.ndarray:
    encoded = encoded.split(",", 1)[-1] if encoded.startswith("data:") else encoded
    try:
        with Image.open(io.BytesIO(base64.b64decode(encoded, validate=True))) as image:
            return np.asarray(image.convert("RGB"), dtype=np.uint8)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Invalid {name}: {exc}") from exc


def _center_crop(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Match the R1 Lite collection framing before FastWAM resizing.

    The collector records a 640x640 center crop for agentview.
    """
    source_h, source_w = image.shape[:2]
    if source_w < width or source_h < height:
        raise ValueError(
            f"Image is smaller than the required collection crop {width}x{height}: "
            f"received {source_w}x{source_h}"
        )
    left = (source_w - width) // 2
    top = (source_h - height) // 2
    return np.ascontiguousarray(image[top : top + height, left : left + width])


def _resize(image: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    """Match the existing FastWAM RobotWin deployment's PIL bilinear resize."""
    return np.asarray(Image.fromarray(image).resize(size_wh, Image.BILINEAR), dtype=np.uint8)

# Visual input assembled into the 384 * 320 mosaic
def compose_robotwin_image(head: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Match collection framing, then FastWAM's 3-camera 320x384 layout."""
    head = _center_crop(head, 640, 640)
    # Wrist streams are native 640x360 and must not be cropped.
    if left.shape[:2] != (360, 640) or right.shape[:2] != (360, 640):
        raise ValueError(
            "R1 Lite wrist images must remain native 640x360; "
            f"received left={left.shape[1]}x{left.shape[0]}, right={right.shape[1]}x{right.shape[0]}"
        )
    # Resize to the required layout
    top = _resize(head, (320, 256))
    bottom = np.concatenate((_resize(left, (160, 128)), _resize(right, (160, 128))), axis=1)
    return np.concatenate((top, bottom), axis=0)  # HWC: 384x320x3


def _compose_config(config_path: Path, overrides: list[str]) -> DictConfig:
    config_path = config_path.resolve()
    configs_root = (PROJECT_ROOT / "configs").resolve()
    try:
        relative = config_path.relative_to(configs_root)
    except ValueError as exc:
        raise ValueError(f"--config must be under {configs_root}: {config_path}") from exc
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(version_base="1.3", config_dir=str(configs_root)):
        return compose(config_name=relative.as_posix(), overrides=overrides)


class FastWAMR1LitePolicy:
    def __init__(self, args: argparse.Namespace) -> None:
        self.recorder = ObservationRecorder(args.observations_dir)
        self.cfg = _compose_config(args.config, args.override)
        if args.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        if not args.checkpoint.is_file() or not args.dataset_stats.is_file():
            raise FileNotFoundError("Checkpoint and dataset stats must both exist")

        # Model + normalizer loading
        model_cfg = OmegaConf.create(OmegaConf.to_container(self.cfg.model, resolve=True))
        # Use precomputed context embeddings so T5 is not loaded on the
        # inference GPU. The cache is generated by scripts/precompute_text_embeds.py.
        model_cfg.load_text_encoder = False
        self.model = instantiate(model_cfg, model_dtype=_model_dtype(args.mixed_precision), device=args.device)
        self.model.load_checkpoint(str(args.checkpoint))
        self.model = self.model.to(args.device).eval()
        self.processor: FastWAMProcessor = instantiate(self.cfg.data.train.processor).eval()
        self.processor.set_normalizer_from_stats(load_dataset_stats_from_json(str(args.dataset_stats)))
        self._validate_contract()
        action_key = self.processor.shape_meta["action"][0]["key"]
        action_stats = self.processor.normalizer.normalizers["action"][action_key].get_stats()
        self.action_min = action_stats["min"].detach().cpu().numpy().astype(np.float32)
        self.action_max = action_stats["max"].detach().cpu().numpy().astype(np.float32)
        if self.action_min.shape != (ACTION_DIM,) or self.action_max.shape != (ACTION_DIM,):
            raise ValueError("Expected 14-D global action min/max in dataset statistics")
        if not np.isfinite(self.action_min).all() or not np.isfinite(self.action_max).all():
            raise ValueError("Dataset action min/max contains non-finite values")
        if np.any(self.action_min > self.action_max):
            raise ValueError("Dataset action min exceeds action max")

        # Training configs do not necessarily define the optional uppercase
        # EVALUATION section used by some evaluation configs.
        evaluation_cfg = self.cfg.get("EVALUATION", {})
        self.action_semantics = str(evaluation_cfg.get("action_semantics", "relative"))
        if self.action_semantics not in {"relative", "absolute"}:
            raise ValueError(
                "EVALUATION.action_semantics must be 'relative' or 'absolute', "
                f"got {self.action_semantics!r}"
            )
        self.action_horizon = int(
            args.action_horizon
            if args.action_horizon is not None
            else evaluation_cfg.get("action_horizon") or int(self.cfg.data.train.num_frames) - 1
        )
        self.num_inference_steps = int(
            args.num_inference_steps
            if args.num_inference_steps is not None
            else evaluation_cfg.get("num_inference_steps", self.cfg.get("eval_num_inference_steps", 20))
        )
        self.sigma_shift = args.sigma_shift if args.sigma_shift is not None else evaluation_cfg.get("sigma_shift")
        self.seed = args.seed
        self.text_cfg_scale = args.text_cfg_scale if args.text_cfg_scale is not None else evaluation_cfg.get("text_cfg_scale", 1.0)
        self.negative_prompt = args.negative_prompt if args.negative_prompt is not None else evaluation_cfg.get("negative_prompt", "")
        self.rand_device = args.rand_device or evaluation_cfg.get("rand_device", "cpu")
        self.tiled = args.tiled if args.tiled is not None else evaluation_cfg.get("tiled", False)
        configured_cache = self.cfg.data.train.get("text_embedding_cache_dir")
        cache_dir = args.text_embedding_cache_dir or configured_cache
        if cache_dir is None:
            raise ValueError("Set --text-embedding-cache-dir or data.train.text_embedding_cache_dir")
        self.text_embedding_cache_dir = Path(str(cache_dir)).expanduser()
        if not self.text_embedding_cache_dir.is_absolute():
            self.text_embedding_cache_dir = (PROJECT_ROOT / self.text_embedding_cache_dir).resolve()
        self.context_len = int(self.cfg.data.train.get("context_len", 128))
        model_id = str(self.cfg.model.get("model_id", "Wan-AI/Wan2.2-TI2V-5B"))
        self.embedding_model_tag = "".join(ch for ch in model_id.split("/")[-1].lower() if ch.isalnum()) or "textenc"
        self.num_video_frames = (int(self.cfg.data.train.num_frames) - 1) // int(self.cfg.data.train.action_video_freq_ratio) + 1
        self.lock = threading.Lock()

    # Validating shape of states and actions
    def _validate_contract(self) -> None:
        state_meta, action_meta = self.processor.shape_meta["state"], self.processor.shape_meta["action"]
        if len(state_meta) != 1 or len(action_meta) != 1:
            raise ValueError("r1lite serving requires one merged state and action key")
        for meta, name in ((state_meta[0], "state"), (action_meta[0], "action")):
            if meta["raw_shape"] != 14 or meta["shape"] != 14:
                raise ValueError(f"Expected 14-D {name} config, got {meta}")
        if self.processor.proprio_output_dim != 14 or self.processor.action_output_dim != 14:
            raise ValueError("r1lite processor dimensions must both be 14")

    # Proprio normalization
    def _normalize_state(self, state: np.ndarray) -> torch.Tensor:
        key = self.processor.shape_meta["state"][0]["key"]
        batch = {"state": {key: torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)}}
        batch = self.processor.action_state_transform(batch)
        return self.processor.normalizer.forward(batch)["state"][key]

    # Action denormalization
    def _denormalize_action(self, action: torch.Tensor) -> np.ndarray:
        if action.ndim == 2:
            action = action.unsqueeze(0)
        if action.ndim != 3:
            raise ValueError(f"Expected [B,T,D] action, got {tuple(action.shape)}")
        key = self.processor.shape_meta["action"][0]["key"]
        normalizer = self.processor.normalizer.normalizers["action"][key]
        denormalized = normalizer.backward(
            action.detach().to(device="cpu", dtype=torch.float32)
        ).numpy()
        return np.clip(denormalized, self.action_min, self.action_max)

    # Load the precomputed T5 embedding cache
    def _load_context(self, task: str) -> tuple[torch.Tensor, torch.Tensor]:
        prompt = DEFAULT_PROMPT.format(task=task)
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cache_path = self.text_embedding_cache_dir / f"{digest}.t5_len{self.context_len}.{self.embedding_model_tag}.pt"
        if not cache_path.is_file():
            raise FileNotFoundError(
                f"Missing cached embedding for prompt {task!r}: {cache_path}. "
                "Run scripts/precompute_text_embeds.py with +override_instruction=<task>."
            )
        payload = torch.load(str(cache_path), map_location="cpu", weights_only=True)
        context, context_mask = payload["context"], payload["mask"].bool()
        if context.ndim != 2 or context.shape[0] != self.context_len or context_mask.shape != (self.context_len,):
            raise ValueError(f"Invalid cached embedding shape in {cache_path}")

        # Match RobotVideoDataset._get exactly. Wan's text-conditioning path
        # receives zero vectors at padding locations and an all-valid mask.
        # Passing the tokenizer mask from the cache changes cross-attention
        # relative to training.
        context = context.clone()
        context[~context_mask] = 0.0
        return context, torch.ones_like(context_mask)

    def infer(self, request: InferenceRequest) -> np.ndarray:
        state = np.asarray(request.state, dtype=np.float32)
        if not np.isfinite(state).all():
            raise ValueError("state contains non-finite values")
        decoded_images = {
            "head": _decode_rgb(request.head_image, "head_image"),
            "left_wrist": _decode_rgb(request.left_wrist_image, "left_wrist_image"),
            "right_wrist": _decode_rgb(request.right_wrist_image, "right_wrist_image"),
        }
        image = compose_robotwin_image(
            decoded_images["head"], decoded_images["left_wrist"], decoded_images["right_wrist"]
        )
        image_tensor_cpu = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(
            dtype=self.model.torch_dtype
        )
        image_tensor_cpu = image_tensor_cpu * (2.0 / 255.0) - 1.0
        image_tensor = image_tensor_cpu.to(device=self.model.device)
        context, context_mask = self._load_context(request.prompt)
        normalized_state = self._normalize_state(state)
        kwargs: dict[str, Any] = {
            "prompt": None,
            "context": context,
            "context_mask": context_mask,
            "input_image": image_tensor,
            "action_horizon": self.action_horizon,
            "proprio": normalized_state,
            "negative_prompt": self.negative_prompt,
            "text_cfg_scale": self.text_cfg_scale,
            "num_inference_steps": self.num_inference_steps,
            "sigma_shift": self.sigma_shift,
            "seed": self.seed,
            "rand_device": self.rand_device,
            "tiled": self.tiled,
        }
        if "num_video_frames" in inspect.signature(self.model.infer_action).parameters:
            kwargs["num_video_frames"] = self.num_video_frames
        # This is intentionally the final operation before infer_action: the
        # .pt contains the exact normalized tensors passed to the model.
        recorded_kwargs = dict(kwargs)
        recorded_kwargs["input_image"] = image_tensor_cpu
        # Chunk saving
        chunk_dir = self.recorder.save_model_input(request, image, recorded_kwargs)

        with self.lock, torch.no_grad():
            action = self.model.infer_action(**kwargs)["action"]
        actions = self._denormalize_action(action)[0]
        if actions.shape != (self.action_horizon, ACTION_DIM) or not np.isfinite(actions).all():
            raise RuntimeError(f"Invalid model output: {actions.shape}")
        # Enqueued only after infer_action returns and the output passes validation.
        self.recorder.save_model_output(chunk_dir, action, actions, decoded_images)
        return actions


def build_app(policy: FastWAMR1LitePolicy) -> FastAPI:
    app = FastAPI(title="FastWAM R1 Lite Policy")

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/metadata")
    def metadata() -> dict[str, object]:
        return {
            "state_dim": STATE_DIM,
            "action_dim": ACTION_DIM,
            "action_horizon": policy.action_horizon,
            "camera_layout": "center-crop head 640x640; native wrists 640x360; head 320x256 above wrists 160x128",
            "action_semantics": policy.action_semantics,
            "observations_dir": str(policy.recorder.root),
        }

    @app.post("/runs/start")
    def start_run(request: StartRunRequest) -> dict[str, object]:
        run_id = policy.recorder.start_run(request)
        return {"run_id": run_id, "num_demos": request.num_demos}
    @app.post("/infer")
    def infer(request: InferenceRequest) -> dict[str, object]:
        started = time.perf_counter()
        try:
            actions = policy.infer(request)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Inference failed")
            raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc
        return {
            "actions": actions.tolist(),
            "action_horizon": policy.action_horizon,
            "action_semantics": policy.action_semantics,
            "server_timing": {"infer_ms": (time.perf_counter() - started) * 1000.0},
        }

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "sim_robotwin.yaml")
    parser.add_argument("--checkpoint", type=lambda x: Path(x).expanduser().resolve(), required=True)
    parser.add_argument("--dataset-stats", type=lambda x: Path(x).expanduser().resolve(), required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mixed-precision", choices=("no", "fp16", "bf16"), default="bf16")
    parser.add_argument("--action-horizon", type=int)
    parser.add_argument("--num-inference-steps", type=int)
    parser.add_argument("--sigma-shift", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--text-cfg-scale", type=float)
    parser.add_argument("--negative-prompt")
    parser.add_argument("--text-embedding-cache-dir", type=Path, help="Directory written by precompute_text_embeds.py")
    parser.add_argument(
        "--observations-dir",
        type=Path,
        default=PROJECT_ROOT / "server_observations",
        help="Root for run_N/demo_M model-input and execution logs",
    )
    parser.add_argument("--rand-device")
    parser.add_argument("--tiled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--override", action="append", default=[], help="Repeat for each Hydra override")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    policy = FastWAMR1LitePolicy(args)
    uvicorn.run(build_app(policy), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
