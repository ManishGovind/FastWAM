#!/usr/bin/env python3
"""Serve the xArm7 3-task FastWAM checkpoint over HTTP for xarm_fastwam_inference.py.

Adapted from experiments/r1lite/serve_fastwam_r1lite.py. The server owns image
tiling, state normalization, inference, and action de-normalization. Everything
is read from the checkpoint bundle, so no base Wan weights, T5 encoder, or
training YAML are needed. Returned actions are ABSOLUTE targets at 15 Hz:
    [x, y, z (m), roll, pitch, yaw (rad, roll wrapped to [0, 2*pi)), gripper (-1 open, +1 closed)]
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import logging
import os
from pathlib import Path
import sys
import threading
import time
from typing import Annotated

import numpy as np
from fastapi import FastAPI, HTTPException
from hydra.utils import instantiate
from omegaconf import OmegaConf
from PIL import Image
from pydantic import BaseModel, Field
import torch
import torchvision.transforms.functional as transforms_F
import uvicorn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json

LOGGER = logging.getLogger("fastwam.xarm7.server")
STATE_DIM = ACTION_DIM = 7
EXPECTED_IMAGE_SIZE = 480
DEFAULT_BUNDLE = Path(__file__).resolve().parent / "models"


class InferenceRequest(BaseModel):
    """Current state plus post-crop 480x480 base64 JPEG/PNG RGB images."""

    prompt: Annotated[str, Field(min_length=1)]
    state: Annotated[list[float], Field(min_length=STATE_DIM, max_length=STATE_DIM)]
    primary_image: Annotated[str, Field(min_length=1)]
    wrist_image: Annotated[str, Field(min_length=1)]
    ego_image: Annotated[str, Field(min_length=1)]


def _decode_rgb(encoded: str, name: str) -> torch.Tensor:
    """Decode to a [C,H,W] float tensor in [0,1], as the training ToTensor does."""
    try:
        with Image.open(io.BytesIO(base64.b64decode(encoded, validate=True))) as image:
            array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Invalid {name}: {exc}") from exc
    if array.shape[:2] != (EXPECTED_IMAGE_SIZE, EXPECTED_IMAGE_SIZE):
        raise ValueError(
            f"{name} must be {EXPECTED_IMAGE_SIZE}x{EXPECTED_IMAGE_SIZE} (post-crop), got {array.shape[1]}x{array.shape[0]}"
        )
    return torch.from_numpy(array.copy()).permute(2, 0, 1).to(torch.float32) / 255.0


def _resize(image: torch.Tensor, size_hw: list[int]) -> torch.Tensor:
    return transforms_F.resize(
        image,
        size=size_hw,
        interpolation=transforms_F.InterpolationMode.BILINEAR,
        antialias=True,
    )


def compose_robotwin_image(primary: torch.Tensor, wrist: torch.Tensor, ego: torch.Tensor) -> torch.Tensor:
    """Same ops as training: transform to 240x320, then RobotVideoDataset's robotwin tiling."""
    primary, wrist, ego = (_resize(image, [240, 320]) for image in (primary, wrist, ego))
    bottom = torch.cat([_resize(wrist, [128, 160]), _resize(ego, [128, 160])], dim=-1)
    return torch.cat([_resize(primary, [256, 320]), bottom], dim=-2)  # [C, 384, 320]


class FastWAMXArm7Policy:
    def __init__(self, args: argparse.Namespace) -> None:
        bundle = args.bundle
        self.cfg = OmegaConf.load(bundle / "config.yaml")
        self.text_embedding_cache_dir = bundle / "xarm7_3task"
        self.context_len = int(self.cfg.data.train.context_len)
        self.action_horizon = int(self.cfg.data.train.num_frames) - 1
        self.num_inference_steps = args.num_inference_steps
        self.seed = args.seed
        self.lock = threading.Lock()
        self.num_cached_prompts = len(list(self.text_embedding_cache_dir.glob("*.pt")))
        if self.num_cached_prompts == 0:
            raise FileNotFoundError(f"No cached text embeddings in {self.text_embedding_cache_dir}")

        # The loader finds the VAE at <base>/<model_id>/Wan2.2_VAE.pth; link the
        # bundled copy there so nothing is downloaded.
        vae_link = bundle / str(self.cfg.model.model_id) / "Wan2.2_VAE.pth"
        vae_link.parent.mkdir(parents=True, exist_ok=True)
        if vae_link.is_symlink() and not vae_link.exists():  # dangling, e.g. bundle dir was moved
            vae_link.unlink()
        if not vae_link.exists():
            vae_link.symlink_to(bundle / "Wan2.2_VAE.pth")
        os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = str(bundle)
        os.environ["DIFFSYNTH_SKIP_DOWNLOAD"] = "true"

        model_cfg = OmegaConf.create(OmegaConf.to_container(self.cfg.model, resolve=True))
        model_cfg.load_text_encoder = False
        model_cfg.skip_dit_load_from_pretrain = True
        # The loader builds the DiT as WanVideoDiT(...).to(device, dtype): built on the
        # CPU in fp32 that is ~20 GB of host RAM (it OOM-killed the process and froze the
        # machine). Allocating on the GPU directly, in bf16, keeps host RAM small and
        # puts the model on the requested GPU (cuda:0 by default).
        torch.cuda.set_device(torch.device(args.device))
        prev_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            with torch.device(args.device):
                self.model = instantiate(model_cfg, model_dtype=torch.bfloat16, device=args.device)
        finally:
            torch.set_default_dtype(prev_dtype)
        self._load_checkpoint(bundle / args.checkpoint)
        self.model = self.model.to(args.device).eval()

        self.processor = instantiate(self.cfg.data.train.processor).eval()
        self.processor.set_normalizer_from_stats(load_dataset_stats_from_json(str(bundle / "dataset_stats.json")))
        action_stats = self.processor.normalizer.normalizers["action"]["default"].get_stats()
        self.action_min = action_stats["min"].detach().cpu().numpy().astype(np.float32)
        self.action_max = action_stats["max"].detach().cpu().numpy().astype(np.float32)
        if self.action_min.shape != (ACTION_DIM,) or self.action_max.shape != (ACTION_DIM,):
            raise ValueError("Expected 7-D global action min/max in dataset statistics")

    def _load_checkpoint(self, path: Path) -> None:
        # model.load_checkpoint() reads all 12 GB into RAM and loads with
        # strict=False. mmap avoids the RAM spike, and the bundle was trained
        # from this exact config, so any key mismatch is a real error.
        payload = torch.load(str(path), map_location="cpu", mmap=True, weights_only=False)
        self.model.mot.load_state_dict(payload["mot"], strict=True)
        self.model.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
        LOGGER.info("Loaded %s (step %s)", path, payload.get("step"))

    def _normalize_state(self, state: np.ndarray) -> torch.Tensor:
        batch = {"state": {"default": torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)}}
        batch = self.processor.action_state_transform(batch)
        return self.processor.normalizer.forward(batch)["state"]["default"]

    def _load_context(self, task: str) -> tuple[torch.Tensor, torch.Tensor]:
        prompt = DEFAULT_PROMPT.format(task=task)
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cache_path = self.text_embedding_cache_dir / f"{digest}.t5_len{self.context_len}.wan22ti2v5b.pt"
        if not cache_path.is_file():
            raise ValueError(f"No cached embedding for prompt {task!r}: {cache_path}")
        payload = torch.load(str(cache_path), map_location="cpu", weights_only=True)
        context, context_mask = payload["context"].clone(), payload["mask"].bool()
        # Match RobotVideoDataset._get: zero vectors at padding and an all-valid mask.
        context[~context_mask] = 0.0
        return context, torch.ones_like(context_mask)

    def infer(self, request: InferenceRequest) -> np.ndarray:
        state = np.asarray(request.state, dtype=np.float32)
        if not np.isfinite(state).all():
            raise ValueError("state contains non-finite values")
        image = compose_robotwin_image(
            _decode_rgb(request.primary_image, "primary_image"),
            _decode_rgb(request.wrist_image, "wrist_image"),
            _decode_rgb(request.ego_image, "ego_image"),
        )
        image = (image * 2.0 - 1.0).unsqueeze(0).to(dtype=self.model.torch_dtype, device=self.model.device)
        context, context_mask = self._load_context(request.prompt)
        with self.lock, torch.no_grad():
            action = self.model.infer_action(
                prompt=None,
                input_image=image,
                action_horizon=self.action_horizon,
                proprio=self._normalize_state(state),
                context=context,
                context_mask=context_mask,
                text_cfg_scale=1.0,
                num_inference_steps=self.num_inference_steps,
                seed=self.seed,
            )["action"]
        if action.ndim == 2:
            action = action.unsqueeze(0)
        normalizer = self.processor.normalizer.normalizers["action"]["default"]
        actions = normalizer.backward(action.detach().to(device="cpu", dtype=torch.float32)).numpy()[0]
        if actions.shape != (self.action_horizon, ACTION_DIM) or not np.isfinite(actions).all():
            raise RuntimeError(f"Invalid model output: {actions.shape}")
        # Also keeps every target inside the box the demos covered.
        return np.clip(actions, self.action_min, self.action_max)


def build_app(policy: FastWAMXArm7Policy) -> FastAPI:
    app = FastAPI(title="FastWAM xArm7 Policy")

    @app.get("/healthz")
    def healthz() -> dict[str, object]:
        return {
            "status": "ok",
            "action_horizon": policy.action_horizon,
            "num_inference_steps": policy.num_inference_steps,
            "cached_prompts": policy.num_cached_prompts,
        }

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
            "server_timing": {"infer_ms": (time.perf_counter() - started) * 1000.0},
        }

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=lambda x: Path(x).expanduser().resolve(), default=DEFAULT_BUNDLE)
    parser.add_argument("--checkpoint", default="step_030000.pt", help="Checkpoint filename inside --bundle")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8777)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    policy = FastWAMXArm7Policy(args)
    LOGGER.info(
        "Serving on %s:%d (action_horizon=%d, %d cached prompts)",
        args.host, args.port, policy.action_horizon, policy.num_cached_prompts,
    )
    uvicorn.run(build_app(policy), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
