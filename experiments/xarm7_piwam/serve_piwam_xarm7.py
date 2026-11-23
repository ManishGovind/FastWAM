#!/usr/bin/env python3
"""Serve the xArm7 3-task pi-WAM (router) checkpoint over HTTP for xarm_piwam_inference.py.

Same /healthz and /infer contract as serve_fastwam_xarm7.py (image tiling, state
normalization, action de-normalization and clipping are identical), plus:

* The model is FastWAM's joint multimodal *select* model from origin/dev
  (config: xarm7_3task_joint_3cam_384_1e-4_router_full_all_streamemb_embed_tok_cf_ahist).
  A learned router picks ONE future stream (rgb | depth | flow) per inference call
  (= per chunk); the action is read with that stream. Only RGB is needed as input.
* Router action history (K = 32): the last K executed actions in normalized space,
  front-padded at an episode start, exactly like training. The server keeps it:
  after each chunk it appends the first --replan-steps rows of the predicted
  normalized chunk (the rows the client executes), and clears it on a new
  episode_id or chunk_id == 1. --replan-steps MUST equal the client's chunk
  execution length (16).
* Every response reports the router's choice (`active_future`) and its softmax
  probabilities over the enabled streams (`router_probs`), so the client can
  record what was selected per chunk.

Code: the dev worktree (default ~/workspace/FastWAM_dev, `git worktree add
--detach ~/workspace/FastWAM_dev origin/dev`) is put first on sys.path, so the
older ~/workspace/FastWAM/src installed in the env is not used.

Bundle (--bundle, default experiments/xarm7_piwam/models_piwam): step_*.pt, config.yaml
(resolved), dataset_stats.json, xarm7_3task/ (cached T5 prompts), Wan2.2_VAE.pth.
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
from typing import Annotated, Optional

DEFAULT_CODE_ROOT = Path.home() / "workspace" / "FastWAM_dev"


def _code_root_from_argv() -> Path:
    # Must run before anything imports `fastwam`: the env's installed copy is older.
    for i, arg in enumerate(sys.argv):
        if arg == "--code-root" and i + 1 < len(sys.argv):
            return Path(sys.argv[i + 1]).expanduser().resolve()
        if arg.startswith("--code-root="):
            return Path(arg.split("=", 1)[1]).expanduser().resolve()
    return DEFAULT_CODE_ROOT


CODE_ROOT = _code_root_from_argv()
for _p in (CODE_ROOT / "src", CODE_ROOT):
    sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from hydra.utils import instantiate  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from PIL import Image  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402
import torch  # noqa: E402
import torchvision.transforms.functional as transforms_F  # noqa: E402
import uvicorn  # noqa: E402

import fastwam  # noqa: E402
from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT  # noqa: E402
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json  # noqa: E402

LOGGER = logging.getLogger("fastwam.xarm7.piwam_server")
STATE_DIM = ACTION_DIM = 7
EXPECTED_IMAGE_SIZE = 480
DEFAULT_BUNDLE = Path(__file__).resolve().parent / "models_piwam"


class InferenceRequest(BaseModel):
    """Current state plus post-crop 480x480 base64 JPEG/PNG RGB images.

    episode_id / chunk_id drive the router action history (cleared on a new
    episode or chunk 1). Without them every request is treated as chunk 1.
    """

    prompt: Annotated[str, Field(min_length=1)]
    state: Annotated[list[float], Field(min_length=STATE_DIM, max_length=STATE_DIM)]
    primary_image: Annotated[str, Field(min_length=1)]
    wrist_image: Annotated[str, Field(min_length=1)]
    ego_image: Annotated[str, Field(min_length=1)]
    episode_id: Optional[str] = None
    chunk_id: Optional[int] = None


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
        image, size=size_hw, interpolation=transforms_F.InterpolationMode.BILINEAR, antialias=True
    )


def compose_robotwin_image(primary: torch.Tensor, wrist: torch.Tensor, ego: torch.Tensor) -> torch.Tensor:
    """Same ops as training: transform to 240x320, then the robotwin tiling (as serve_fastwam_xarm7.py)."""
    primary, wrist, ego = (_resize(image, [240, 320]) for image in (primary, wrist, ego))
    bottom = torch.cat([_resize(wrist, [128, 160]), _resize(ego, [128, 160])], dim=-1)
    return torch.cat([_resize(primary, [256, 320]), bottom], dim=-2)  # [C, 384, 320]


class ActionHistory:
    """Last K executed actions (normalized), front-padded with zeros + is_pad, as in training."""

    def __init__(self, k: int, dim: int, replan: int) -> None:
        self.k, self.dim, self.replan = k, dim, replan
        self.rows: list[torch.Tensor] = []
        self.episode: Optional[str] = None

    def reset(self) -> None:
        self.rows.clear()

    def push(self, normalized_chunk: torch.Tensor) -> None:
        self.rows.extend(normalized_chunk.detach().float().reshape(-1, self.dim)[: self.replan].cpu())

    def kwargs(self) -> dict[str, torch.Tensor]:
        rows = self.rows[-self.k:]
        n_pad = self.k - len(rows)
        act = torch.cat([torch.zeros(n_pad, self.dim)] + ([torch.stack(rows)] if rows else []), dim=0)
        pad = torch.cat([torch.ones(n_pad), torch.zeros(len(rows))])
        return {"action_history": act, "action_history_is_pad": pad}


class PiWAMXArm7Policy:
    def __init__(self, args: argparse.Namespace) -> None:
        bundle = args.bundle
        self.cfg = OmegaConf.load(bundle / "config.yaml")
        self.text_embedding_cache_dir = bundle / "xarm7_3task"
        self.context_len = int(self.cfg.data.train.context_len)
        self.action_horizon = int(self.cfg.data.train.num_frames) - 1
        self.num_video_frames = (int(self.cfg.data.train.num_frames) - 1) // int(
            self.cfg.data.train.action_video_freq_ratio
        ) + 1
        self.num_inference_steps = args.num_inference_steps
        self.seed = args.seed
        self.active_future = None if args.active_future in (None, "router") else args.active_future
        self.lock = threading.Lock()
        self.num_cached_prompts = len(list(self.text_embedding_cache_dir.glob("*.pt")))
        if self.num_cached_prompts == 0:
            raise FileNotFoundError(f"No cached text embeddings in {self.text_embedding_cache_dir}")

        vae_link = bundle / str(self.cfg.model.model_id) / "Wan2.2_VAE.pth"
        vae_link.parent.mkdir(parents=True, exist_ok=True)
        if vae_link.is_symlink() and not vae_link.exists():
            vae_link.unlink()
        if not vae_link.exists():
            vae_link.symlink_to((bundle / "Wan2.2_VAE.pth").resolve())
        os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = str(bundle)
        os.environ["DIFFSYNTH_SKIP_DOWNLOAD"] = "true"

        model_cfg = OmegaConf.create(OmegaConf.to_container(self.cfg.model, resolve=True))
        model_cfg.load_text_encoder = False
        model_cfg.skip_dit_load_from_pretrain = True
        # Build on the GPU in bf16 (CPU fp32 construction OOM-killed the box for the baseline).
        torch.cuda.set_device(torch.device(args.device))
        prev_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            with torch.device(args.device):
                self.model = instantiate(model_cfg, model_dtype=torch.bfloat16, device=args.device)
        finally:
            torch.set_default_dtype(prev_dtype)
        self.step = self._load_checkpoint(bundle / args.checkpoint)
        self.model = self.model.to(args.device).eval()
        # Override only after building: counterfactual-router models must be built with the
        # full layout (as the R1 Lite server on origin/real-world-mm does).
        self.trained_router_layout = str(getattr(self.model, "router_layout", "short"))
        if args.router_layout:
            self.model.router_layout = args.router_layout
        self.router_layout = str(getattr(self.model, "router_layout", "short"))
        LOGGER.info("router_layout=%s (trained: %s)", self.router_layout, self.trained_router_layout)

        self.enabled_streams = tuple(self.model._enabled_streams())
        self.history = None
        if getattr(self.model, "router_action_history", False):
            self.history = ActionHistory(
                int(self.model.router_action_history_steps), int(self.model.action_expert.action_dim), args.replan_steps
            )
        self._capture_router()

        self.processor = instantiate(self.cfg.data.train.processor).eval()
        self.processor.set_normalizer_from_stats(load_dataset_stats_from_json(str(bundle / "dataset_stats.json")))
        action_stats = self.processor.normalizer.normalizers["action"]["default"].get_stats()
        self.action_min = action_stats["min"].detach().cpu().numpy().astype(np.float32)
        self.action_max = action_stats["max"].detach().cpu().numpy().astype(np.float32)
        if self.action_min.shape != (ACTION_DIM,) or self.action_max.shape != (ACTION_DIM,):
            raise ValueError("Expected 7-D global action min/max in dataset statistics")

    def _load_checkpoint(self, path: Path) -> Optional[int]:
        payload = torch.load(str(path), map_location="cpu", mmap=True, weights_only=False)
        self.model.mot.load_state_dict(payload["mot"], strict=True)
        self.model.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
        if getattr(self.model, "modality_router", None) is None or "modality_router" not in payload:
            raise RuntimeError("Expected a router model and `modality_router` weights in the checkpoint")
        self.model.modality_router.load_state_dict(payload["modality_router"], strict=True)
        if payload.get("modality_select", "router") != getattr(self.model, "modality_select", None):
            raise RuntimeError(
                f"Checkpoint modality_select={payload.get('modality_select')!r} != model {self.model.modality_select!r}"
            )
        LOGGER.info("Loaded %s (step %s): mot + proprio_encoder + modality_router, strict", path, payload.get("step"))
        return payload.get("step")

    def _capture_router(self) -> None:
        """Wrap _sample_modality_select to keep the router's log-probs of the latest call."""
        self.last_router_probs: Optional[list[float]] = None
        original = self.model._sample_modality_select

        def wrapped(*a, **k):
            choices, logits, log_probs = original(*a, **k)
            self.last_router_probs = None if log_probs is None else log_probs.exp()[0].float().cpu().tolist()
            return choices, logits, log_probs

        self.model._sample_modality_select = wrapped

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
        context[~context_mask] = 0.0  # as RobotVideoDataset._get: zero padding + all-valid mask
        return context, torch.ones_like(context_mask)

    def infer(self, request: InferenceRequest) -> dict[str, object]:
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
        kwargs = dict(
            prompt=None,
            input_image=image,
            action_horizon=self.action_horizon,
            num_video_frames=self.num_video_frames,
            proprio=self._normalize_state(state),
            context=context,
            context_mask=context_mask,
            text_cfg_scale=1.0,
            num_inference_steps=self.num_inference_steps,
            seed=self.seed,
            active_future=self.active_future,
        )
        with self.lock, torch.no_grad():
            history_len = 0
            if self.history is not None:
                if request.chunk_id in (None, 1) or request.episode_id != self.history.episode:
                    self.history.reset()
                    self.history.episode = request.episode_id
                history_len = min(len(self.history.rows), self.history.k)
                kwargs.update(self.history.kwargs())
            self.last_router_probs = None
            out = self.model.infer_action(**kwargs)
            action = out["action"]
            if self.history is not None:
                self.history.push(action)
            active_future = out.get("active_future")
            router_probs = self.last_router_probs
        if action.ndim == 2:
            action = action.unsqueeze(0)
        normalizer = self.processor.normalizer.normalizers["action"]["default"]
        actions = normalizer.backward(action.detach().to(device="cpu", dtype=torch.float32)).numpy()[0]
        if actions.shape != (self.action_horizon, ACTION_DIM) or not np.isfinite(actions).all():
            raise RuntimeError(f"Invalid model output: {actions.shape}")
        probs = None if router_probs is None else dict(zip(self.enabled_streams, router_probs))
        LOGGER.info(
            "episode=%s chunk=%s history=%d/%s active_future=%s probs=%s",
            request.episode_id, request.chunk_id, history_len, self.history.k if self.history else 0,
            active_future, None if probs is None else {k: round(v, 3) for k, v in probs.items()},
        )
        return {
            "actions": np.clip(actions, self.action_min, self.action_max),
            "active_future": active_future,
            "router_probs": probs,
            "history_len": history_len,
            "router_layout": self.router_layout,
        }


def build_app(policy: PiWAMXArm7Policy) -> FastAPI:
    app = FastAPI(title="pi-WAM xArm7 Policy")

    @app.get("/healthz")
    def healthz() -> dict[str, object]:
        return {
            "status": "ok",
            "checkpoint_step": policy.step,
            "action_horizon": policy.action_horizon,
            "num_inference_steps": policy.num_inference_steps,
            "cached_prompts": policy.num_cached_prompts,
            "enabled_streams": list(policy.enabled_streams),
            "active_future": policy.active_future or "router",
            "router_layout": policy.router_layout,
            "trained_router_layout": policy.trained_router_layout,
            "action_history_k": policy.history.k if policy.history else 0,
            "replan_steps": policy.history.replan if policy.history else None,
        }

    @app.post("/infer")
    def infer(request: InferenceRequest) -> dict[str, object]:
        started = time.perf_counter()
        try:
            result = policy.infer(request)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Inference failed")
            raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc
        return {
            "actions": result["actions"].tolist(),
            "active_future": result["active_future"],
            "router_probs": result["router_probs"],
            "history_len": result["history_len"],
            "router_layout": result["router_layout"],
            "server_timing": {"infer_ms": (time.perf_counter() - started) * 1000.0},
        }

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--code-root", type=lambda x: Path(x).expanduser().resolve(), default=DEFAULT_CODE_ROOT,
                        help="FastWAM checkout with the pi-WAM code (origin/dev worktree)")
    parser.add_argument("--bundle", type=lambda x: Path(x).expanduser().resolve(), default=DEFAULT_BUNDLE)
    parser.add_argument("--checkpoint", default="step_020000.pt", help="Checkpoint filename inside --bundle")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8780)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--replan-steps", type=int, default=16,
                        help="Actions the client executes per chunk (client CHUNK_EXECUTE_STEPS); "
                             "these rows feed the router action history")
    parser.add_argument("--active-future", choices=["router", "rgb", "depth", "flow", "random"], default="router",
                        help="router (default) = learned per-chunk choice; others force/randomize the stream")
    parser.add_argument("--router-layout", choices=("short", "full"), default=None,
                        help="Inference layout (default: as trained, i.e. full for this bundle). full = all three "
                             "futures in the MoT, only the chosen one denoised; short = only the chosen future "
                             "(same action up to numerics, ~3x fewer video tokens, faster)")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    LOGGER.info("fastwam code: %s", Path(fastwam.__file__).resolve())
    if not Path(fastwam.__file__).resolve().is_relative_to(args.code_root):
        raise SystemExit(f"fastwam imported from {fastwam.__file__}, not --code-root {args.code_root}")
    policy = PiWAMXArm7Policy(args)
    LOGGER.info(
        "Serving step %s on %s:%d (action_horizon=%d, streams=%s, router_layout=%s, history K=%s, replan=%s, "
        "%d cached prompts)",
        policy.step, args.host, args.port, policy.action_horizon, policy.enabled_streams, policy.router_layout,
        policy.history.k if policy.history else 0, args.replan_steps, policy.num_cached_prompts,
    )
    uvicorn.run(build_app(policy), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
