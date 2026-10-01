"""Representational analysis helpers for LIBERO joint unimodal FastWAM.

Used by ``notebooks/libero_plus_failure_analysis.ipynb``.

1. Linear CKA between Video / Action experts (and across RGB/Depth/Flow).
2. Mixed-attention probes: Action queries attending to Video keys.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

REPO = Path("/data/mgovind/FastWAM").resolve()
CACHE_DIR = REPO / "notebooks" / "outputs" / "repr_joint_unimodal"

PHASE_NAMES = ("free_space", "pregrasp_contact", "post_manip")
PHASE_LABELS = {
    "free_space": "Phase 1: Free-space alignment",
    "pregrasp_contact": "Phase 2: Pre-grasp & contact",
    "post_manip": "Phase 3: Post-manipulation",
}

# Checkpoints used by the LIBERO-Plus joint unimodal evals in the failure notebook.
JOINT_UNIMODAL = {
    "RGB": {
        "task": "libero_joint_2cam224_1e-4",
        "run_dir": REPO / "runs" / "libero_joint_2cam224_1e-4" / "2026-08-23_17-22-15",
        "ckpt": REPO
        / "runs"
        / "libero_joint_2cam224_1e-4"
        / "2026-08-23_17-22-15"
        / "checkpoints"
        / "weights"
        / "step_021700.pt",
    },
    "Depth": {
        "task": "libero_joint_2cam224_1e-4_depth",
        "run_dir": REPO / "runs" / "libero_joint_2cam224_1e-4_depth" / "2026-08-21_23-21-22",
        "ckpt": REPO
        / "runs"
        / "libero_joint_2cam224_1e-4_depth"
        / "2026-08-21_23-21-22"
        / "checkpoints"
        / "weights"
        / "step_021700.pt",
    },
    "Flow": {
        "task": "libero_joint_2cam224_1e-4_flow",
        "run_dir": REPO / "runs" / "libero_joint_2cam224_1e-4_flow" / "2026-08-22_14-11-53",
        # Plus eval for Flow used step_021000 (not 021700).
        "ckpt": REPO
        / "runs"
        / "libero_joint_2cam224_1e-4_flow"
        / "2026-08-22_14-11-53"
        / "checkpoints"
        / "weights"
        / "step_021700.pt",
    },
}


def pick_device() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    best_i, best_free = 0, -1
    for i in range(torch.cuda.device_count()):
        with torch.cuda.device(i):
            free, _total = torch.cuda.mem_get_info()
        if free > best_free:
            best_free, best_i = int(free), i
    return f"cuda:{best_i}"


def linear_cka(x: torch.Tensor, y: torch.Tensor) -> float:
    """Linear CKA between matrices with matching sample dim.

    Args:
        x: [n, p]
        y: [n, q]
    """
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError(f"CKA expects 2D matrices, got {tuple(x.shape)} and {tuple(y.shape)}")
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"CKA sample counts must match, got n={x.shape[0]} vs {y.shape[0]}")
    if x.shape[0] < 2:
        raise ValueError("CKA needs at least 2 samples after flattening")

    x = x.detach().float()
    y = y.detach().float()
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)

    n, p = x.shape
    q = y.shape[1]
    # Use the cheaper Gram form when n is the small axis.
    if n <= max(p, q):
        gram_x = x @ x.T
        gram_y = y @ y.T
        hsic_xy = (gram_x * gram_y).sum()
        hsic_xx = (gram_x * gram_x).sum()
        hsic_yy = (gram_y * gram_y).sum()
    else:
        xtx = x.T @ x
        yty = y.T @ y
        xty = x.T @ y
        hsic_xy = (xty * xty).sum()
        hsic_xx = (xtx * xtx).sum()
        hsic_yy = (yty * yty).sum()

    denom = torch.sqrt(hsic_xx.clamp(min=0) * hsic_yy.clamp(min=0)) + 1e-12
    return float((hsic_xy / denom).item())


def cka_matrix(feats_a: torch.Tensor, feats_b: torch.Tensor) -> np.ndarray:
    """Layer-to-layer CKA.

    Args:
        feats_a: [n, n_layers_a, dim_a] or list of [n, dim]
        feats_b: [n, n_layers_b, dim_b]
    Returns:
        [n_layers_a, n_layers_b] numpy array
    """
    if isinstance(feats_a, (list, tuple)):
        feats_a = torch.stack(list(feats_a), dim=1)
    if isinstance(feats_b, (list, tuple)):
        feats_b = torch.stack(list(feats_b), dim=1)
    n_a = int(feats_a.shape[1])
    n_b = int(feats_b.shape[1])
    out = np.zeros((n_a, n_b), dtype=np.float64)
    for i in range(n_a):
        xi = feats_a[:, i, :].reshape(feats_a.shape[0], -1)
        for j in range(n_b):
            yj = feats_b[:, j, :].reshape(feats_b.shape[0], -1)
            out[i, j] = linear_cka(xi, yj)
    return out


def compose_task_cfg(task: str):
    configs_root = str((REPO / "configs").resolve())
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(version_base="1.3", config_dir=configs_root):
        cfg = compose(
            config_name="sim_libero.yaml",
            overrides=[
                f"task={task}",
                "model.skip_dit_load_from_pretrain=true",
                "model.load_text_encoder=false",
                "model.mot_checkpoint_mixed_attn=false",
            ],
        )
    return cfg


def load_joint_model(cfg, ckpt: Path, device: str):
    model_dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
    model = instantiate(cfg.model, model_dtype=model_dtype, device=device)
    model.load_checkpoint(str(ckpt))
    model = model.to(device).eval()
    return model


def load_dataset(cfg, stats_path: Path):
    return instantiate(cfg.data.train, pretrained_norm_stats=str(stats_path))


def collate_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    batch: dict[str, Any] = {}
    for key in samples[0]:
        vals = [s[key] for s in samples]
        if torch.is_tensor(vals[0]):
            batch[key] = torch.stack(vals, dim=0)
        else:
            batch[key] = vals
    return batch


def _prompt_instruction(prompt: str) -> str:
    marker = "executing the following instruction: "
    if marker in prompt:
        return prompt.split(marker, 1)[1].strip()
    return prompt


@torch.no_grad()
def _action_to_video_weights(
    q_action: torch.Tensor,
    k_video: torch.Tensor,
    attn_mask_action_to_video: torch.Tensor,
    num_heads: int,
) -> torch.Tensor:
    """Softmax attention from action queries onto video keys.

    Args:
        q_action: [B, Sa, H*Dh]
        k_video: [B, Sv, H*Dh]
        attn_mask_action_to_video: [Sa, Sv] True = allowed
    Returns:
        weights [B, H, Sa, Sv]
    """
    q = rearrange(q_action.float(), "b s (n d) -> b n s d", n=num_heads)
    k = rearrange(k_video.float(), "b s (n d) -> b n s d", n=num_heads)
    scale = float(q.shape[-1]) ** -0.5
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    mask = attn_mask_action_to_video.to(device=scores.device, dtype=torch.bool)
    scores = scores.masked_fill(~mask.view(1, 1, *mask.shape), torch.finfo(scores.dtype).min)
    return torch.softmax(scores, dim=-1)


@dataclass
class ProbeBatch:
    video_tokens: list[torch.Tensor]  # L * [B, Sv, Dv] cpu fp16
    action_tokens: list[torch.Tensor]  # L * [B, Sa, Da]
    action_to_video: Optional[list[torch.Tensor]]  # L * [B, H, Sa, Sv] cpu fp16
    attn_layer_ids: list[int] = field(default_factory=list)
    grid: dict[str, int] = field(default_factory=dict)
    first_frame: Optional[torch.Tensor] = None  # [B, 3, H, W] in [0, 1]


class MoTProbe:
    """Monkey-patch MoT mixed attention to capture residual tokens + Action→Video weights."""

    def __init__(
        self,
        mot,
        capture_attn: bool = True,
        capture_tokens: bool = True,
        attn_layers: Optional[list[int]] = None,
    ):
        self.mot = mot
        self.capture_attn = bool(capture_attn)
        self.capture_tokens = bool(capture_tokens)
        self.attn_layers = None if attn_layers is None else {int(i) for i in attn_layers}
        self.video_tokens: list[torch.Tensor] = []
        self.action_tokens: list[torch.Tensor] = []
        self.action_to_video: list[torch.Tensor] = []
        self.attn_layer_ids: list[int] = []
        self._layer_idx = 0
        self._orig = None

    def __enter__(self):
        self.video_tokens = []
        self.action_tokens = []
        self.action_to_video = []
        self.attn_layer_ids = []
        self._layer_idx = 0
        self._orig = self.mot._forward_joint_layer
        self.mot._forward_joint_layer = self._forward_joint_layer
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._orig is not None:
            self.mot._forward_joint_layer = self._orig
        self._orig = None
        return False

    def _forward_joint_layer(
        self,
        video_block,
        action_block,
        video_tokens,
        action_tokens,
        video_freqs,
        action_freqs,
        video_t_mod,
        action_t_mod,
        video_context,
        video_context_mask,
        action_context,
        action_context_mask,
        attention_mask,
    ):
        mot = self.mot
        (
            q_video,
            k_video,
            v_video,
            residual_video,
            gate_msa_video,
            shift_mlp_video,
            scale_mlp_video,
            gate_mlp_video,
            _video_checkpointing,
        ) = mot._build_expert_attention_io(
            expert=mot.mixtures["video"],
            block=video_block,
            x=video_tokens,
            freqs=video_freqs,
            t_mod=video_t_mod,
        )
        (
            q_action,
            k_action,
            v_action,
            residual_action,
            gate_msa_action,
            shift_mlp_action,
            scale_mlp_action,
            gate_mlp_action,
            _action_checkpointing,
        ) = mot._build_expert_attention_io(
            expert=mot.mixtures["action"],
            block=action_block,
            x=action_tokens,
            freqs=action_freqs,
            t_mod=action_t_mod,
        )

        from fastwam.models.wan22.wan_video_dit import flash_attention

        mixed = flash_attention(
            q=torch.cat([q_video, q_action], dim=1),
            k=torch.cat([k_video, k_action], dim=1),
            v=torch.cat([v_video, v_action], dim=1),
            num_heads=mot.num_heads,
            ctx_mask=attention_mask.to(device=q_video.device),
        )
        video_seq_len = video_tokens.shape[1]
        x_video = mot._apply_expert_post_block_tensor(
            block=video_block,
            residual_x=residual_video,
            mixed_attn_out=mixed[:, :video_seq_len],
            gate_msa=gate_msa_video,
            shift_mlp=shift_mlp_video,
            scale_mlp=scale_mlp_video,
            gate_mlp=gate_mlp_video,
            context=video_context,
            context_mask=video_context_mask,
        )
        x_action = mot._apply_expert_post_block_tensor(
            block=action_block,
            residual_x=residual_action,
            mixed_attn_out=mixed[:, video_seq_len:],
            gate_msa=gate_msa_action,
            shift_mlp=shift_mlp_action,
            scale_mlp=scale_mlp_action,
            gate_mlp=gate_mlp_action,
            context=action_context,
            context_mask=action_context_mask,
        )

        if self.capture_tokens:
            self.video_tokens.append(x_video.detach().to(dtype=torch.float16, device="cpu"))
            self.action_tokens.append(x_action.detach().to(dtype=torch.float16, device="cpu"))
        if self.capture_attn and (self.attn_layers is None or self._layer_idx in self.attn_layers):
            action_mask = attention_mask[video_seq_len:, :video_seq_len]
            weights = _action_to_video_weights(
                q_action=q_action,
                k_video=k_video,
                attn_mask_action_to_video=action_mask,
                num_heads=int(mot.num_heads),
            )
            self.action_to_video.append(weights.detach().to(dtype=torch.float16, device="cpu"))
            self.attn_layer_ids.append(int(self._layer_idx))
        self._layer_idx += 1
        return x_video, x_action


@torch.no_grad()
def forward_probe(
    model,
    batch: dict[str, Any],
    *,
    sigma: float = 0.5,
    seed: int = 0,
    capture_attn: bool = True,
    capture_tokens: bool = True,
    attn_layers: Optional[list[int]] = None,
) -> ProbeBatch:
    """One joint denoise step with matching GT video/action frames."""
    model.eval()
    inputs = model.build_inputs(batch, tiled=False)
    latents = inputs["input_latents"]
    action = inputs["action"]
    device = latents.device
    dtype = latents.dtype

    g = torch.Generator(device="cpu").manual_seed(int(seed))
    noise_video = torch.randn(latents.shape, generator=g, dtype=torch.float32).to(device=device, dtype=dtype)
    noise_action = torch.randn(action.shape, generator=g, dtype=torch.float32).to(device=device, dtype=dtype)

    timestep = torch.full(
        (latents.shape[0],),
        float(sigma) * float(model.train_video_scheduler.num_train_timesteps),
        device=device,
        dtype=dtype,
    )
    noisy_video = model.train_video_scheduler.add_noise(latents, noise_video, timestep)
    if inputs["first_frame_latents"] is not None:
        noisy_video[:, :, 0:1] = inputs["first_frame_latents"]
    noisy_action = model.train_action_scheduler.add_noise(action, noise_action, timestep)

    patch_t, patch_h, patch_w = (int(size) for size in model.video_expert.patch_size)
    latent_t, latent_h, latent_w = noisy_video.shape[-3:]
    tokens_per_frame = (latent_h // patch_h) * (latent_w // patch_w)
    attention_mask = model._build_mot_attention_mask(
        video_seq_len=(latent_t // patch_t) * tokens_per_frame,
        action_seq_len=noisy_action.shape[1],
        video_tokens_per_frame=tokens_per_frame,
        device=device,
    )

    video_preview = ((batch["video"][:, :, 0].float() * 0.5) + 0.5).clamp(0, 1).cpu()

    with MoTProbe(
        model.mot,
        capture_attn=capture_attn,
        capture_tokens=capture_tokens,
        attn_layers=attn_layers,
    ) as probe:
        _ = model._joint_denoise_core(
            latents_video=noisy_video,
            latents_action=noisy_action,
            timestep_video=timestep,
            timestep_action=timestep,
            context=inputs["context"],
            context_mask=inputs["context_mask"],
            attention_mask=attention_mask,
            fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
            action_condition=action,
        )

    grid = {
        "latent_t": int(latent_t),
        "latent_h": int(latent_h),
        "latent_w": int(latent_w),
        "patch_h": int(patch_h),
        "patch_w": int(patch_w),
        "token_h": int(latent_h // patch_h),
        "token_w": int(latent_w // patch_w),
        "tokens_per_frame": int(tokens_per_frame),
        "num_layers": int(model.mot.num_layers),
        "num_heads": int(model.mot.num_heads),
    }
    return ProbeBatch(
        video_tokens=probe.video_tokens,
        action_tokens=probe.action_tokens,
        action_to_video=probe.action_to_video if capture_attn else None,
        attn_layer_ids=list(probe.attn_layer_ids),
        grid=grid,
        first_frame=video_preview,
    )


def _pool_tokens(tokens: torch.Tensor) -> torch.Tensor:
    """[B, S, D] -> [B, D] mean over sequence."""
    return tokens.float().mean(dim=1)


def _frame_pool(tokens: torch.Tensor, tokens_per_frame: int) -> torch.Tensor:
    """[B, F*tpf, D] -> [B, F, D]."""
    b, s, d = tokens.shape
    if s % tokens_per_frame != 0:
        raise ValueError(f"seq {s} not divisible by tokens_per_frame {tokens_per_frame}")
    f = s // tokens_per_frame
    return tokens.float().view(b, f, tokens_per_frame, d).mean(dim=2)


def extract_model(
    name: str,
    *,
    indices: list[int],
    device: str,
    sigma: float = 0.5,
    seed: int = 0,
    batch_size: int = 1,
    attn_sample_limit: int = 2,
    cache_dir: Path = CACHE_DIR,
    reuse_cache: bool = True,
) -> dict[str, Any]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / f"{name.lower()}_features.pt"
    if reuse_cache and out_path.exists():
        print(f"[{name}] loading cache {out_path}")
        try:
            return torch.load(out_path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(out_path, map_location="cpu")

    spec = JOINT_UNIMODAL[name]
    ckpt = Path(spec["ckpt"])
    stats = Path(spec["run_dir"]) / "dataset_stats.json"
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)
    if not stats.exists():
        raise FileNotFoundError(stats)

    print(f"[{name}] composing cfg={spec['task']}")
    cfg = compose_task_cfg(spec["task"])
    print(f"[{name}] loading dataset")
    dataset = load_dataset(cfg, stats)
    n_ds = len(dataset)
    indices = [int(i) % n_ds for i in indices]
    print(f"[{name}] dataset len={n_ds}, loading model {ckpt}")
    model = load_joint_model(cfg, ckpt, device=device)

    video_pooled = []
    action_pooled = []
    video_frame_pooled = []
    attn_maps = []
    first_frames = []
    prompts = []
    grid = None

    n = len(indices)
    for start in range(0, n, batch_size):
        chunk = indices[start : start + batch_size]
        samples = [dataset[int(i)] for i in chunk]
        batch = collate_samples(samples)
        capture_attn = start < attn_sample_limit
        print(
            f"[{name}] forward samples {start}:{start + len(chunk)} "
            f"attn={capture_attn}",
            flush=True,
        )
        probed = forward_probe(
            model,
            batch,
            sigma=sigma,
            seed=seed + start,
            capture_attn=capture_attn,
        )
        if grid is None:
            grid = probed.grid
        tpf = int(grid["tokens_per_frame"])
        v_pool, a_pool, v_frame = [], [], []
        for layer_v, layer_a in zip(probed.video_tokens, probed.action_tokens):
            v_pool.append(_pool_tokens(layer_v))
            a_pool.append(_pool_tokens(layer_a))
            v_frame.append(_frame_pool(layer_v, tpf))
        # [B, L, D]
        video_pooled.append(torch.stack(v_pool, dim=1))
        action_pooled.append(torch.stack(a_pool, dim=1))
        video_frame_pooled.append(torch.stack(v_frame, dim=1))
        if capture_attn and probed.action_to_video:
            # keep at most remaining slots, head-mean later in notebook
            take = min(len(chunk), attn_sample_limit - len(attn_maps))
            attn_layer = torch.stack(probed.action_to_video, dim=1)  # [B, L, H, Sa, Sv]
            for b_i in range(take):
                attn_maps.append(attn_layer[b_i].contiguous())
                first_frames.append(probed.first_frame[b_i])
                prompt = samples[b_i].get("prompt", "")
                prompts.append(_prompt_instruction(prompt) if isinstance(prompt, str) else str(prompt))

        del probed, batch, samples
        gc.collect()
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()

    payload = {
        "name": name,
        "ckpt": str(ckpt),
        "task": spec["task"],
        "indices": list(indices),
        "sigma": float(sigma),
        "grid": grid,
        "video_pooled": torch.cat(video_pooled, dim=0),  # [N, L, Dv]
        "action_pooled": torch.cat(action_pooled, dim=0),  # [N, L, Da]
        "video_frame_pooled": torch.cat(video_frame_pooled, dim=0),  # [N, L, F, Dv]
        "attn_maps": attn_maps,  # list of [L, H, Sa, Sv]
        "first_frames": first_frames,  # list of [3, H, W]
        "prompts": prompts,
    }
    torch.save(payload, out_path)
    print(f"[{name}] wrote {out_path}")

    del model, dataset, cfg
    gc.collect()
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return payload


def extract_all(
    *,
    n_samples: int = 16,
    sample_stride: int = 2048,
    device: Optional[str] = None,
    sigma: float = 0.5,
    seed: int = 0,
    batch_size: int = 1,
    models: Optional[list[str]] = None,
    reuse_cache: bool = True,
) -> dict[str, dict[str, Any]]:
    device = device or pick_device()
    names = models or list(JOINT_UNIMODAL.keys())
    # Shared indices so RGB/Depth/Flow see the same episode windows.
    indices = [int(i * sample_stride) for i in range(n_samples)]
    print(f"device={device} indices[:4]={indices[:4]} n={n_samples}")
    out = {}
    for name in names:
        out[name] = extract_model(
            name,
            indices=indices,
            device=device,
            sigma=sigma,
            seed=seed,
            batch_size=batch_size,
            reuse_cache=reuse_cache,
        )
    return out


def head_mean_attn(attn: torch.Tensor) -> torch.Tensor:
    """[L, H, Sa, Sv] -> [L, Sa, Sv] or [H, Sa, Sv] -> [Sa, Sv]."""
    if attn.ndim == 4:
        return attn.float().mean(dim=1)
    if attn.ndim == 3:
        return attn.float().mean(dim=0)
    raise ValueError(f"Unexpected attn shape {tuple(attn.shape)}")


def reshape_video_attn(attn_sv: torch.Tensor, grid: dict[str, int]) -> torch.Tensor:
    """[Sa, Sv] or [Sv] -> [Sa, F, Ht, Wt] (or [F, Ht, Wt] if no Sa)."""
    tpf = int(grid["tokens_per_frame"])
    ht, wt = int(grid["token_h"]), int(grid["token_w"])
    if ht * wt != tpf:
        raise ValueError(f"token grid {ht}x{wt} != tpf {tpf}")
    if attn_sv.ndim == 1:
        f = attn_sv.numel() // tpf
        return attn_sv.float().view(f, ht, wt)
    if attn_sv.ndim == 2:
        f = attn_sv.shape[1] // tpf
        return attn_sv.float().view(attn_sv.shape[0], f, ht, wt)
    raise ValueError(f"Unexpected spatial attn shape {tuple(attn_sv.shape)}")


def camera_split_mass(spatial: torch.Tensor) -> dict[str, torch.Tensor]:
    """Split token width into agentview (left) / wrist (right).

    ``spatial``: [F, Ht, Wt] or [L, F, Ht, Wt]
    """
    wt = spatial.shape[-1]
    mid = wt // 2
    agent = spatial[..., :, :mid].sum(dim=(-2, -1))
    wrist = spatial[..., :, mid:].sum(dim=(-2, -1))
    return {"agentview": agent, "wrist": wrist}


def overlay_heatmap(frame: np.ndarray, heat: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """frame [H,W,3] in [0,1], heat [h,w] >= 0. Returns uint8 RGB."""
    import matplotlib.cm as cm

    heat = heat.astype(np.float32)
    heat = heat - heat.min()
    denom = float(heat.max()) + 1e-8
    heat = heat / denom
    heat_img = torch.from_numpy(heat)[None, None]
    h, w = frame.shape[:2]
    heat_up = F.interpolate(heat_img, size=(h, w), mode="bilinear", align_corners=False)
    heat_up = heat_up[0, 0].numpy()
    color = cm.inferno(heat_up)[..., :3]
    blend = (1.0 - alpha) * frame + alpha * color
    return np.clip(blend * 255.0, 0, 255).astype(np.uint8)


def denorm_frame(frame: torch.Tensor) -> np.ndarray:
    """[3,H,W] in [0,1] -> [H,W,3] float."""
    return frame.detach().float().clamp(0, 1).permute(1, 2, 0).cpu().numpy()


def pick_episode(dataset, min_len: int = 80) -> tuple[int, int, int, int]:
    """Return (episode_idx, frame_from, frame_to, length) for a usable episode."""
    epi = dataset.lerobot_dataset.episode_data_index
    n_ep = int(epi["from"].shape[0])
    obs = int(dataset.lerobot_dataset.obs_size)
    for i in range(n_ep):
        start = int(epi["from"][i].item())
        end = int(epi["to"][i].item())
        length = end - start
        if length >= max(min_len, obs + 1):
            return i, start, end, length
    start = int(epi["from"][0].item())
    end = int(epi["to"][0].item())
    return 0, start, end, end - start


def episode_window_indices(
    dataset,
    episode_idx: int,
    n_windows: int = 8,
) -> tuple[list[int], dict[str, int]]:
    epi = dataset.lerobot_dataset.episode_data_index
    start = int(epi["from"][episode_idx].item())
    end = int(epi["to"][episode_idx].item())
    obs = int(dataset.lerobot_dataset.obs_size)
    last = end - obs
    if last < start:
        raise ValueError(
            f"Episode {episode_idx} is shorter than the model window "
            f"(len={end - start}, obs_size={obs})"
        )
    if n_windows <= 1:
        idxs = [start]
    else:
        idxs = np.linspace(start, last, num=int(n_windows), dtype=np.int64).tolist()
        idxs = [int(i) for i in idxs]
    meta = {
        "episode_idx": int(episode_idx),
        "frame_from": start,
        "frame_to": end,
        "length": end - start,
        "obs_size": obs,
        "n_windows": len(idxs),
    }
    return idxs, meta


def cond_spatial_from_attn(attn_bhsasv: torch.Tensor, grid: dict[str, int]) -> torch.Tensor:
    """[B, H, Sa, Sv] -> [Ht, Wt] condition-frame map (batch 0, head+query mean)."""
    sv = head_mean_attn(attn_bhsasv[0]).mean(dim=0)  # [Sv]
    spatial = reshape_video_attn(sv, grid)  # [F, Ht, Wt]
    return spatial[0]


def extract_episode_attention(
    name: str,
    *,
    episode_idx: Optional[int] = None,
    n_windows: int = 8,
    min_episode_len: int = 80,
    device: str,
    sigma: float = 0.5,
    seed: int = 0,
    cache_dir: Path = CACHE_DIR,
    reuse_cache: bool = True,
    attn_layers: Optional[list[int]] = None,
) -> dict[str, Any]:
    """Action→Video attention along one full episode (evenly spaced windows)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    layers_key = "init_mid_last" if attn_layers is None else "_".join(str(i) for i in attn_layers)
    out_path = cache_dir / f"{name.lower()}_episode_attn_ep{episode_idx if episode_idx is not None else 'auto'}_{layers_key}.pt"
    if reuse_cache and out_path.exists():
        print(f"[{name}] loading episode attn cache {out_path}")
        try:
            return torch.load(out_path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(out_path, map_location="cpu")

    spec = JOINT_UNIMODAL[name]
    ckpt = Path(spec["ckpt"])
    stats = Path(spec["run_dir"]) / "dataset_stats.json"
    cfg = compose_task_cfg(spec["task"])
    dataset = load_dataset(cfg, stats)
    if episode_idx is None:
        episode_idx, _, _, _ = pick_episode(dataset, min_len=min_episode_len)
    indices, ep_meta = episode_window_indices(dataset, episode_idx, n_windows=n_windows)
    model = load_joint_model(cfg, ckpt, device=device)
    n_layers = int(model.mot.num_layers)
    if attn_layers is None:
        attn_layers = [0, n_layers // 2, n_layers - 1]

    first_frames = []
    cond_maps = []  # [T, n_sel, Ht, Wt]
    layer_ids = None
    grid = None
    instruction = ""

    for t, idx in enumerate(indices):
        sample = dataset[int(idx)]
        if t == 0:
            prompt = sample.get("prompt", "")
            instruction = _prompt_instruction(prompt) if isinstance(prompt, str) else str(prompt)
        batch = collate_samples([sample])
        print(f"[{name}] episode {episode_idx} window {t + 1}/{len(indices)} idx={idx}", flush=True)
        probed = forward_probe(
            model,
            batch,
            sigma=sigma,
            seed=seed + t,
            capture_attn=True,
            capture_tokens=False,
            attn_layers=attn_layers,
        )
        if grid is None:
            grid = probed.grid
            layer_ids = list(probed.attn_layer_ids)
        maps = []
        for attn in probed.action_to_video:
            maps.append(cond_spatial_from_attn(attn, grid))
        cond_maps.append(torch.stack(maps, dim=0))
        first_frames.append(probed.first_frame[0].contiguous())
        del probed, batch, sample
        gc.collect()
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()

    payload = {
        "name": name,
        "ckpt": str(ckpt),
        "instruction": instruction,
        "episode": ep_meta,
        "indices": list(indices),
        "steps": [int(i - ep_meta["frame_from"]) for i in indices],
        "layers": list(layer_ids or attn_layers),
        "grid": grid,
        "first_frames": first_frames,
        "cond_maps": torch.stack(cond_maps, dim=0),  # [T, n_layers, Ht, Wt]
        "sigma": float(sigma),
    }
    torch.save(payload, out_path)
    print(f"[{name}] wrote {out_path}")
    del model, dataset, cfg
    gc.collect()
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return payload


def extract_episode_all(
    *,
    episode_idx: Optional[int] = None,
    n_windows: int = 8,
    device: Optional[str] = None,
    sigma: float = 0.5,
    models: Optional[list[str]] = None,
    reuse_cache: bool = True,
) -> dict[str, dict[str, Any]]:
    device = device or pick_device()
    names = models or list(JOINT_UNIMODAL.keys())
    print(f"episode attn  device={device} episode_idx={episode_idx} n_windows={n_windows}")
    out = {}
    for name in names:
        out[name] = extract_episode_attention(
            name,
            episode_idx=episode_idx,
            n_windows=n_windows,
            device=device,
            sigma=sigma,
            reuse_cache=reuse_cache,
        )
    return out


def _episode_action_array(raw: dict[str, Any]) -> np.ndarray:
    if "action" in raw:
        act = raw["action"]
    else:
        keys = [k for k in raw if "action" in str(k).lower() and "is_pad" not in str(k)]
        if not keys:
            raise KeyError(f"No action column in episode parquet; keys={list(raw)[:20]}")
        act = raw[keys[0]]
    if torch.is_tensor(act):
        act = act.detach().cpu().numpy()
    act = np.asarray(act)
    if act.ndim == 3:
        act = act[:, 0]
    if act.ndim != 2:
        raise ValueError(f"Expected action [T, A], got {act.shape}")
    return act.astype(np.float64)


def _episode_action_state(raw: dict[str, Any]) -> tuple[np.ndarray, Optional[np.ndarray]]:
    action = _episode_action_array(raw)
    state = None
    for key in ("observation.state", "observation.states.ee_state"):
        if key not in raw:
            continue
        st = raw[key]
        if torch.is_tensor(st):
            st = st.detach().cpu().numpy()
        st = np.asarray(st)
        if st.ndim == 3:
            st = st[:, 0]
        if st.ndim == 2 and st.shape[0] >= 2:
            state = st.astype(np.float64)
        break
    return action, state


def segment_episode_phases(
    action: np.ndarray,
    state: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Label frames: 0 free-space, 1 pre-grasp/contact, 2 post-manipulation.

    LIBERO action gripper is 1=open, 0=closed. Speed is proprio xyz when present.
    """
    if action.shape[0] < 8:
        raise ValueError(f"Episode too short to segment: T={action.shape[0]}")

    if state is not None and state.shape[0] >= 2:
        n = min(action.shape[0], state.shape[0])
        eef = state[:n, :3]
        speed = np.linalg.norm(np.diff(eef, axis=0, prepend=eef[:1]), axis=-1)
        action = action[:n]
    else:
        xyz = action[:, :3]
        speed = np.linalg.norm(xyz, axis=-1)
        if float(np.median(speed)) > 0.2:
            speed = np.linalg.norm(np.diff(xyz, axis=0, prepend=xyz[:1]), axis=-1)

    kernel = int(np.clip(round(len(speed) / 16), 3, 9))
    if kernel % 2 == 0:
        kernel += 1
    pad = kernel // 2
    speed_s = np.convolve(np.pad(speed, (pad, pad), mode="edge"), np.ones(kernel) / kernel, mode="valid")

    grip = action[:, -1].astype(np.float64)
    g_span = float(grip.max() - grip.min())
    if g_span < 1e-6:
        closedness = np.zeros_like(grip)
    elif float(np.mean(grip[: max(3, len(grip) // 10)])) > float(np.mean(grip)):
        closedness = (grip.max() - grip) / g_span
    else:
        closedness = (grip - grip.min()) / g_span

    dg = np.diff(closedness, prepend=closedness[0])
    close_at = np.flatnonzero(dg > 0.4)
    if close_at.size:
        grasp_t = int(close_at[0])
    elif float(closedness.max()) > 0.5:
        grasp_t = max(1, int(np.argmax(dg)))
    else:
        mid0, mid1 = len(speed_s) // 5, max(len(speed_s) // 5 + 1, 4 * len(speed_s) // 5)
        grasp_t = int(mid0 + np.argmin(speed_s[mid0:mid1]))

    pre_end = max(2, grasp_t)
    peak = int(np.argmax(speed_s[:pre_end]))
    peak_v = float(speed_s[peak])
    v_cut = 0.55 * peak_v if peak_v > 0 else float(np.quantile(speed_s, 0.5))
    t0 = peak
    while t0 < grasp_t and speed_s[t0] > v_cut:
        t0 += 1
    t0 = min(t0, max(0, grasp_t - 2))
    t1 = min(len(speed_s) - 1, grasp_t + max(4, len(speed_s) // 20))
    if t0 < 2:
        t0 = max(2, grasp_t // 2)

    labels = np.zeros(len(speed_s), dtype=np.int64)
    labels[t0 : t1 + 1] = 1
    labels[t1 + 1 :] = 2
    meta = {
        "grasp_t": int(grasp_t),
        "phase2_span": (int(t0), int(t1)),
        "speed": speed_s,
        "gripper_norm": closedness,
        "counts": {PHASE_NAMES[i]: int((labels == i).sum()) for i in range(3)},
    }
    return labels, meta


def collect_phase_windows(
    dataset,
    *,
    n_episodes: int = 10,
    per_phase: int = 2,
    min_len: int = 80,
    seed: int = 0,
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    """Evenly sample model windows whose *current* frame falls in each phase."""
    epi = dataset.lerobot_dataset.episode_data_index
    n_ep = int(epi["from"].shape[0])
    obs = int(dataset.lerobot_dataset.obs_size)
    rng = np.random.default_rng(int(seed))
    buckets: dict[int, list[dict[str, Any]]] = {0: [], 1: [], 2: []}
    phase_meta: dict[int, dict[str, Any]] = {}
    scanned = 0
    for ep in range(n_ep):
        if all(len(buckets[p]) >= n_episodes * per_phase for p in range(3)):
            break
        start = int(epi["from"][ep].item())
        end = int(epi["to"][ep].item())
        length = end - start
        if length < max(min_len, obs + 1):
            continue
        raw = dataset.lerobot_dataset.multi_dataset.get_episode_data(ep)
        action, state = _episode_action_state(raw)
        T = min(int(action.shape[0]), length)
        labels, meta = segment_episode_phases(action[:T], None if state is None else state[:T])
        last_start = end - obs
        scanned += 1
        phase_meta[ep] = {
            "frame_from": start,
            "frame_to": end,
            "length": length,
            **{k: meta[k] for k in ("grasp_t", "phase2_span", "counts")},
        }
        if "_example" not in phase_meta:
            phase_meta["_example"] = {
                "episode": int(ep),
                "speed": meta["speed"],
                "gripper_norm": meta["gripper_norm"],
                "labels": labels,
                "grasp_t": meta["grasp_t"],
                "phase2_span": meta["phase2_span"],
            }
        for phase in range(3):
            if len(buckets[phase]) >= n_episodes * per_phase:
                continue
            offsets = np.flatnonzero(labels == phase)
            offsets = offsets[(start + offsets) <= last_start]
            if offsets.size == 0:
                continue
            n_take = min(int(per_phase), int(offsets.size))
            if n_take == 1:
                chosen = [int(offsets[len(offsets) // 2])]
            else:
                pick = np.linspace(0, len(offsets) - 1, n_take)
                chosen = [int(offsets[int(round(i))]) for i in pick]
            for off in chosen:
                buckets[phase].append(
                    {
                        "idx": int(start + off),
                        "episode": int(ep),
                        "step": int(off),
                        "phase": int(phase),
                        "phase_name": PHASE_NAMES[phase],
                    }
                )
        if scanned >= n_episodes * 4 and all(len(buckets[p]) >= per_phase for p in range(3)):
            # enough diversity even if some phases are short
            if all(len(buckets[p]) >= n_episodes for p in range(3)):
                break

    windows = buckets[0] + buckets[1] + buckets[2]
    print(
        "phase windows:",
        {PHASE_NAMES[p]: len(buckets[p]) for p in range(3)},
        "from",
        scanned,
        "episodes",
    )
    del rng
    return windows, phase_meta


def extract_phase_cka(
    name: str,
    windows: list[dict[str, Any]],
    *,
    device: str,
    sigma: float = 0.5,
    seed: int = 0,
    cache_dir: Path = CACHE_DIR,
    reuse_cache: bool = True,
) -> dict[str, Any]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / f"{name.lower()}_phase_cka_v2_{len(windows)}win.pt"
    if reuse_cache and out_path.exists():
        print(f"[{name}] loading phase CKA cache {out_path}")
        try:
            return torch.load(out_path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(out_path, map_location="cpu")

    spec = JOINT_UNIMODAL[name]
    ckpt = Path(spec["ckpt"])
    stats = Path(spec["run_dir"]) / "dataset_stats.json"
    cfg = compose_task_cfg(spec["task"])
    dataset = load_dataset(cfg, stats)
    model = load_joint_model(cfg, ckpt, device=device)

    video_pooled, video_cond, action_pooled = [], [], []
    phases, steps, episodes = [], [], []
    grid = None

    for t, win in enumerate(windows):
        sample = dataset[int(win["idx"])]
        batch = collate_samples([sample])
        print(
            f"[{name}] phase={win['phase_name']} ep={win['episode']} "
            f"step={win['step']}  {t + 1}/{len(windows)}",
            flush=True,
        )
        probed = forward_probe(
            model,
            batch,
            sigma=sigma,
            seed=seed + t,
            capture_attn=False,
            capture_tokens=True,
        )
        if grid is None:
            grid = probed.grid
        tpf = int(grid["tokens_per_frame"])
        v_all, v_cond, a_all = [], [], []
        for layer_v, layer_a in zip(probed.video_tokens, probed.action_tokens):
            v_all.append(_pool_tokens(layer_v))
            v_cond.append(_frame_pool(layer_v, tpf)[:, 0, :])
            a_all.append(_pool_tokens(layer_a))
        video_pooled.append(torch.stack(v_all, dim=1)[0])
        video_cond.append(torch.stack(v_cond, dim=1)[0])
        action_pooled.append(torch.stack(a_all, dim=1)[0])
        phases.append(int(win["phase"]))
        steps.append(int(win["step"]))
        episodes.append(int(win["episode"]))
        del probed, batch, sample
        gc.collect()
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()

    payload = {
        "name": name,
        "ckpt": str(ckpt),
        "windows": windows,
        "phases": torch.tensor(phases, dtype=torch.long),
        "steps": torch.tensor(steps, dtype=torch.long),
        "episodes": torch.tensor(episodes, dtype=torch.long),
        "video_pooled": torch.stack(video_pooled, dim=0),
        "video_cond_pooled": torch.stack(video_cond, dim=0),
        "action_pooled": torch.stack(action_pooled, dim=0),
        "grid": grid,
        "sigma": float(sigma),
        "phase_names": list(PHASE_NAMES),
    }
    torch.save(payload, out_path)
    print(f"[{name}] wrote {out_path}")
    del model, dataset, cfg
    gc.collect()
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return payload


def extract_phase_cka_all(
    *,
    n_episodes: int = 10,
    per_phase: int = 2,
    device: Optional[str] = None,
    sigma: float = 0.5,
    models: Optional[list[str]] = None,
    reuse_cache: bool = True,
    seed: int = 0,
) -> dict[str, dict[str, Any]]:
    device = device or pick_device()
    names = models or list(JOINT_UNIMODAL.keys())
    n_win = n_episodes * per_phase * 3
    meta_path = CACHE_DIR / f"phase_windows_v2_{n_win}win.pt"
    if reuse_cache and meta_path.exists():
        print(f"loading phase windows {meta_path}")
        try:
            packed = torch.load(meta_path, map_location="cpu", weights_only=False)
        except TypeError:
            packed = torch.load(meta_path, map_location="cpu")
        windows, phase_meta = packed["windows"], packed["phase_meta"]
    else:
        spec = JOINT_UNIMODAL[names[0]]
        cfg = compose_task_cfg(spec["task"])
        stats = Path(spec["run_dir"]) / "dataset_stats.json"
        print(f"phase CKA  device={device} collecting windows on {names[0]}")
        dataset = load_dataset(cfg, stats)
        windows, phase_meta = collect_phase_windows(
            dataset,
            n_episodes=n_episodes,
            per_phase=per_phase,
            seed=seed,
        )
        del dataset, cfg
        gc.collect()
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        torch.save({"windows": windows, "phase_meta": phase_meta}, meta_path)
        print(f"wrote {meta_path}")
    print(f"phase CKA  device={device} n_windows={len(windows)}")
    out = {"_windows": windows, "_phase_meta": phase_meta}
    for name in names:
        out[name] = extract_phase_cka(
            name,
            windows,
            device=device,
            sigma=sigma,
            seed=seed,
            reuse_cache=reuse_cache,
        )
    return out


def phase_cka_matrices(feat: dict[str, Any], *, use_cond_video: bool = True) -> dict[str, np.ndarray]:
    phases = feat["phases"].cpu().numpy()
    video = feat["video_cond_pooled" if use_cond_video else "video_pooled"].float()
    action = feat["action_pooled"].float()
    out = {}
    for p, pname in enumerate(feat.get("phase_names", PHASE_NAMES)):
        mask = phases == p
        n = int(mask.sum())
        if n < 2:
            print(f"  skip {feat['name']} {pname}: n={n}")
            continue
        out[str(pname)] = cka_matrix(video[mask], action[mask])
        out[f"{pname}_n"] = n
    return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=16)
    parser.add_argument("--sample-stride", type=int, default=2048)
    parser.add_argument("--sigma", type=float, default=0.5)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--no-reuse-cache", action="store_true")
    parser.add_argument("--episode", type=int, default=None)
    parser.add_argument("--n-windows", type=int, default=8)
    parser.add_argument("--episode-attn-only", action="store_true")
    parser.add_argument("--phase-cka-only", action="store_true")
    parser.add_argument("--n-episodes", type=int, default=10)
    parser.add_argument("--per-phase", type=int, default=2)
    args = parser.parse_args()
    if args.phase_cka_only:
        extract_phase_cka_all(
            n_episodes=args.n_episodes,
            per_phase=args.per_phase,
            device=args.device,
            sigma=args.sigma,
            models=args.models,
            reuse_cache=not args.no_reuse_cache,
        )
    elif args.episode_attn_only:
        extract_episode_all(
            episode_idx=args.episode,
            n_windows=args.n_windows,
            device=args.device,
            sigma=args.sigma,
            models=args.models,
            reuse_cache=not args.no_reuse_cache,
        )
    else:
        extract_all(
            n_samples=args.n_samples,
            sample_stride=args.sample_stride,
            device=args.device,
            sigma=args.sigma,
            batch_size=args.batch_size,
            models=args.models,
            reuse_cache=not args.no_reuse_cache,
        )
