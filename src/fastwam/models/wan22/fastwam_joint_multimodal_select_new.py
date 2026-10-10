"""Multimodal FastWAM with per-sample modality selection (random / gt / router).

Extends ``FastWAMJointMultimodal``. Training picks one future stream per sample and
runs MoT as R0 + that future only. The base multistream (all futures) model lives
in ``fastwam_joint_multimodal.py``.

``_new`` variant of ``fastwam_joint_multimodal_select.py``. Only the inline
(``modality_select=router``) path changes:

* router features are detached, so the REINFORCE / entropy / collapse terms train
  the router only and do not leak into ``proprio_encoder`` via the context token;
* entropy and collapse-KL use the same tempered logits as sampling;
* the REINFORCE reward is configurable (``router_reward``): ``action`` (default,
  matches the action-error oracle) or ``task`` (video + action, with the video term
  weighted by ``video_weight`` exactly as in ``loss_total``);
* the baseline is configurable (``router_baseline``): ``batch`` mean, or a running
  per-timestep-bucket EMA (``timestep_ema``) that removes the noise-level variance
  from the advantage; the advantage is optionally std-normalized;
* the router loss is only added to ``loss_total`` in train mode (eval-loss numbers
  stay comparable with the other select modes);
* ``router_warmup_steps`` (optimizer steps; default 0 = off): for the first N steps the
  stream is drawn uniformly at random (like ``modality_select=random``) and the router
  loss is multiplied by 0, so every branch trains equally before the router starts
  choosing. Prevents the early rich-get-richer collapse onto whichever branch happens
  to be ahead. Counted in ``training_loss`` calls / ``router_grad_accum``; restored
  from the checkpoint ``step`` when resuming from a ``.pt``;
* ``modality_prompt`` (default off): after the stream is chosen, the MoT context (video
  and action cross-attention) is told which modality to predict. The router itself still
  sees ``[instruction | proprio]``.
  ``modality_prompt_mode=joint`` (default, as in the reference WAM paper): the condition
  is appended to the instruction and encoded together, ``T5("<instruction> Predict the
  depth video.")`` -> ``[combined | proprio]``. ``separate``: the condition is encoded alone
  and its tokens are inserted, ``[instruction | condition | proprio]``. Training uses the
  text cache (``scripts/precompute_modality_text_embeds.py``); inference encodes with
  the loaded text encoder when available. ``modality_prompt_target``: ``both`` (default,
  as in the paper) feeds the conditioned context to both experts; ``video`` feeds it to
  the video expert only, and the action expert keeps ``[instruction | proprio]``;
* ``router_layout`` (default ``short``): ``short`` builds only ``[R0 | chosen future | A]``;
  ``full`` builds the MM layout ``[R0 | R_fut | D_fut | F_fut | A]`` and a per-sample
  one-hot mask lets the action attend only R0 + the chosen future.
  ``router_video_loss`` (``chosen`` / ``all``) picks which futures get a video loss in
  ``full``. Inference uses the same layout and one-hot code as training: all three
  futures are in the MoT, only the chosen one is denoised, and the action attends only to
  it (same action as the short layout up to numerics; ~3x the video tokens per step);
* sampling vs argmax, the loss gate and the EMA update all key off
  ``modality_router.training`` instead of ``self.training``: the trainer puts the
  top-level model in eval mode and only re-enables train mode on submodules, so
  ``self.training`` is always False during training.
"""

from __future__ import annotations

import contextlib
import hashlib
from pathlib import Path
from typing import Any, Optional, Sequence

import torch
import torch.nn.functional as F

from fastwam.utils.logging_config import get_logger

from .fastwam_joint_multimodal import (
    ACTION_ATTEND_MODES,
    FastWAMJointMultimodal,
    _STREAM_KEYS,
)
from .modality_router_new import ModalityRouter, collapse_kl

logger = get_logger(__name__)

MODALITY_SELECT_MODES = ("all", "random", "router", "gt")
_ROUTER_FEATURE_GROUPS = ("r0", "context")
ROUTER_REWARDS = ("action", "task")
ROUTER_BASELINES = ("batch", "timestep_ema")
ROUTER_LAYOUTS = ("short", "full")
ROUTER_VIDEO_LOSSES = ("chosen", "all")
# Additive attention bias that hides a future from the action (finite: bf16-safe, and
# its straight-through gradient stays finite for the Gumbel subclass).
_GATE_OFF = -1.0e4
DEFAULT_MODALITY_SENTENCES = {
    "rgb": "Predict the RGB video.",
    "depth": "Predict the depth video.",
    "flow": "Predict the flow video.",
}
MODALITY_PROMPT_MODES = ("joint", "separate")
MODALITY_PROMPT_TARGETS = ("both", "video")

class FastWAMJointMultimodalSelectNew(FastWAMJointMultimodal):
    """Like FastWAMJointMultimodal, with modality_select=random|gt|router|all."""

    @classmethod
    def from_wan22_pretrained(cls, **kwargs):
        # Selection / router kwargs (not on the base multimodal factory).
        modality_select = str(kwargs.pop("modality_select", "all")).lower()
        stream_router_temperature = float(kwargs.pop("stream_router_temperature", 1.0))
        router_entropy_coef = float(kwargs.pop("router_entropy_coef", 0.01))
        router_collapse_kl = float(kwargs.pop("router_collapse_kl", 0.1))
        router_hidden = int(kwargs.pop("router_hidden", 256))
        router_dropout = float(kwargs.pop("router_dropout", 0.1))
        router_video_size = kwargs.pop("router_video_size", None)
        router_reward = str(kwargs.pop("router_reward", "action")).lower()
        router_baseline = str(kwargs.pop("router_baseline", "batch")).lower()
        router_baseline_bins = int(kwargs.pop("router_baseline_bins", 10))
        router_baseline_momentum = float(kwargs.pop("router_baseline_momentum", 0.99))
        router_adv_normalize = bool(kwargs.pop("router_adv_normalize", True))
        router_warmup_steps = int(kwargs.pop("router_warmup_steps", 0))
        router_grad_accum = int(kwargs.pop("router_grad_accum", 1))
        router_layout = str(kwargs.pop("router_layout", "short")).lower()
        router_video_loss = str(kwargs.pop("router_video_loss", "chosen")).lower()
        modality_prompt = bool(kwargs.pop("modality_prompt", False))
        modality_prompt_sentences = kwargs.pop("modality_prompt_sentences", None)
        modality_prompt_cache_dir = str(kwargs.pop("modality_prompt_cache_dir", "./data/text_embeds_cache/libero_modality"))
        modality_prompt_context_len = int(kwargs.pop("modality_prompt_context_len", 128))
        modality_prompt_enc_id = str(kwargs.pop("modality_prompt_enc_id", "wan22ti2v5b"))
        modality_prompt_mode = str(kwargs.pop("modality_prompt_mode", "joint")).lower()
        modality_prompt_target = str(kwargs.pop("modality_prompt_target", "both")).lower()
        if modality_prompt_target not in MODALITY_PROMPT_TARGETS:
            raise ValueError(
                f"`modality_prompt_target` must be one of {MODALITY_PROMPT_TARGETS}, got {modality_prompt_target!r}"
            )
        if modality_prompt_mode not in MODALITY_PROMPT_MODES:
            raise ValueError(
                f"`modality_prompt_mode` must be one of {MODALITY_PROMPT_MODES}, got {modality_prompt_mode!r}"
            )
        if router_layout not in ROUTER_LAYOUTS:
            raise ValueError(f"`router_layout` must be one of {ROUTER_LAYOUTS}, got {router_layout!r}")
        if router_video_loss not in ROUTER_VIDEO_LOSSES:
            raise ValueError(
                f"`router_video_loss` must be one of {ROUTER_VIDEO_LOSSES}, got {router_video_loss!r}"
            )
        if router_warmup_steps < 0:
            raise ValueError(f"`router_warmup_steps` must be >= 0, got {router_warmup_steps}")
        if router_grad_accum <= 0:
            raise ValueError(f"`router_grad_accum` must be positive, got {router_grad_accum}")
        if router_reward not in ROUTER_REWARDS:
            raise ValueError(f"`router_reward` must be one of {ROUTER_REWARDS}, got {router_reward!r}")
        if router_baseline not in ROUTER_BASELINES:
            raise ValueError(
                f"`router_baseline` must be one of {ROUTER_BASELINES}, got {router_baseline!r}"
            )
        if router_baseline_bins <= 0:
            raise ValueError(f"`router_baseline_bins` must be positive, got {router_baseline_bins}")
        if not 0.0 <= router_baseline_momentum < 1.0:
            raise ValueError(
                f"`router_baseline_momentum` must be in [0, 1), got {router_baseline_momentum}"
            )
        # Peek before parent consumes it (tokenizer is None when load_text_encoder=false).
        text_context_len = int(kwargs.get("tokenizer_max_len", 128))
        # Parent builds streams / loss lambdas (and ignores legacy stream_router_*).
        model = super().from_wan22_pretrained(**kwargs)
        model.modality_select = cls._validate_modality_select(modality_select)
        model.stream_router_temperature = float(stream_router_temperature)
        model.router_entropy_coef = float(router_entropy_coef)
        model.router_collapse_kl = float(router_collapse_kl)
        model.router_reward = router_reward
        model.router_baseline = router_baseline
        model.router_baseline_bins = router_baseline_bins
        model.router_baseline_momentum = router_baseline_momentum
        model.router_adv_normalize = router_adv_normalize
        model.router_warmup_steps = router_warmup_steps
        model.router_grad_accum = router_grad_accum
        # training_loss calls with the router in train mode (micro-batches).
        model._router_calls = 0
        model.router_layout = router_layout
        model.router_video_loss = router_video_loss
        # Per-sample [B, K] additive gate on (action rows, future cols); see
        # _build_multistream_mot_attention_mask / _action_stream_gate.
        model._action_stream_log_gate = None
        model.modality_prompt = modality_prompt
        model.modality_prompt_mode = modality_prompt_mode
        model.modality_prompt_target = modality_prompt_target
        model.modality_prompt_sentences = dict(DEFAULT_MODALITY_SENTENCES)
        if modality_prompt_sentences:
            model.modality_prompt_sentences.update(
                {str(k): str(v) for k, v in dict(modality_prompt_sentences).items()}
            )
        model._modality_prompt_cache_dir = modality_prompt_cache_dir
        model._modality_prompt_context_len = modality_prompt_context_len
        model._modality_prompt_enc_id = modality_prompt_enc_id
        model._joint_prompt_cache = {}  # combined text -> (context [L, D], mask [L]) on CPU
        model._modality_prompt_tokens = None  # separate mode: [S, K, D] canonical order
        model._modality_prompt_mask = None  # [S, K]
        if modality_prompt and modality_prompt_mode == "separate":
            model._modality_prompt_tokens, model._modality_prompt_mask, model.modality_prompt_sentences = (
                cls._load_modality_prompts(
                    sentences=modality_prompt_sentences,
                    cache_dir=modality_prompt_cache_dir,
                    context_len=modality_prompt_context_len,
                    enc_id=modality_prompt_enc_id,
                )
            )
        # Running mean reward per action-timestep bucket (timestep_ema baseline).
        # Not checkpointed: it re-warms in a few hundred steps after resume.
        model._router_baseline_ema = None
        model._router_baseline_seen = None

        model.modality_router = None
        if model.modality_select == "router":
            model.modality_router = cls._build_modality_router(
                model,
                hidden=router_hidden,
                dropout=router_dropout,
                video_size=router_video_size,
                text_context_len=text_context_len,
            )

        logger.info(
            "FastWAMJointMultimodalSelectNew streams=%s modality_select=%s "
            "router_T=%.3f entropy=%.4f collapse_kl=%.4f reward=%s baseline=%s "
            "(bins=%d momentum=%.3f) adv_normalize=%s warmup_steps=%d (grad_accum=%d) "
            "layout=%s video_loss=%s modality_prompt=%s (%s, target=%s) "
            "loss lambdas: rgb=%.4f depth=%.4f flow=%.4f action=%.4f",
            list(model.enabled_video_streams),
            model.modality_select,
            model.stream_router_temperature,
            model.router_entropy_coef,
            model.router_collapse_kl,
            model.router_reward,
            model.router_baseline,
            model.router_baseline_bins,
            model.router_baseline_momentum,
            model.router_adv_normalize,
            model.router_warmup_steps,
            model.router_grad_accum,
            model.router_layout,
            model.router_video_loss,
            model.modality_prompt,
            model.modality_prompt_mode,
            model.modality_prompt_target,
            model.loss_lambda_rgb,
            model.loss_lambda_depth,
            model.loss_lambda_flow,
            model.loss_lambda_action,
        )
        return model

    @staticmethod
    def _build_modality_router(
        model: "FastWAMJointMultimodalSelectNew",
        *,
        hidden: int,
        dropout: float,
        video_size: Optional[Sequence[int]] = None,
        text_context_len: int = 128,
    ) -> ModalityRouter:
        """Build router over full training R0 map + full MoT context sequence."""
        enabled = model._enabled_streams()
        z_dim = 48
        vae = getattr(model, "vae", None)
        if vae is not None and hasattr(vae, "model") and hasattr(vae.model, "z_dim"):
            z_dim = int(vae.model.z_dim)
        up = int(getattr(vae, "upsampling_factor", 16)) if vae is not None else 16

        if video_size is None:
            raise ValueError(
                "modality_select=router requires `router_video_size` [H, W] "
                "(same as data.train.processor.video_size) so R0 flatten dim matches training."
            )
        if len(video_size) != 2:
            raise ValueError(f"`router_video_size` must be [H, W], got {video_size!r}")
        height, width = int(video_size[0]), int(video_size[1])
        if height % up != 0 or width % up != 0:
            raise ValueError(
                f"`router_video_size` HxW=({height},{width}) must be divisible by "
                f"VAE upsampling_factor={up}"
            )
        r0_dim = z_dim * (height // up) * (width // up)

        # Same context length MoT sees: tokenizer_max_len (+1 proprio token if enabled).
        text_len = int(text_context_len)
        if text_len <= 0:
            raise ValueError(f"`text_context_len` must be positive, got {text_len}")
        context_len = text_len + (1 if getattr(model, "proprio_encoder", None) is not None else 0)
        context_dim = int(getattr(model, "text_dim", 4096))

        router = ModalityRouter(
            r0_dim=r0_dim,
            context_dim=context_dim,
            context_len=context_len,
            num_classes=len(enabled),
            hidden=hidden,
            dropout=dropout,
        ).to(device=model.device, dtype=torch.float32)
        logger.info(
            "Built ModalityRouter classes=%s r0_dim=%d context=[L=%d,D=%d] "
            "video_size=%sx%s hidden=%d (full R0 + context, no pool)",
            list(enabled),
            r0_dim,
            context_len,
            context_dim,
            height,
            width,
            hidden,
        )
        return router

    @staticmethod
    def _normalize_enabled_streams(streams: Optional[Sequence[str]]) -> tuple[str, ...]:
        if streams is None:
            return _STREAM_KEYS
        seen: list[str] = []
        for raw in streams:
            name = str(raw).lower()
            if name not in _STREAM_KEYS:
                raise ValueError(
                    f"`enabled_video_streams` must be a subset of {_STREAM_KEYS}, got {streams!r}"
                )
            if name not in seen:
                seen.append(name)
        if "rgb" not in seen:
            raise ValueError("`enabled_video_streams` must include 'rgb' (R0 condition).")
        return tuple(name for name in _STREAM_KEYS if name in seen)

    def _enabled_streams(self) -> tuple[str, ...]:
        return getattr(self, "enabled_video_streams", _STREAM_KEYS)

    @staticmethod
    def _validate_modality_select(modality_select: str) -> str:
        mode = str(modality_select).lower()
        if mode not in MODALITY_SELECT_MODES:
            raise ValueError(
                f"`modality_select` must be one of {MODALITY_SELECT_MODES}, got {modality_select!r}"
            )
        return mode

    @staticmethod
    def _validate_action_attend_mode(action_attend_mode: str) -> str:
        mode = str(action_attend_mode).lower()
        if mode not in ACTION_ATTEND_MODES:
            raise ValueError(
                f"`action_attend_mode` must be one of {ACTION_ATTEND_MODES}, got {action_attend_mode!r}"
            )
        return mode

    def _sample_modality_select(
        self,
        enabled: Sequence[str],
        batch_size: int,
        device: torch.device,
        *,
        r0_latent: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        selected_modality: Optional[torch.Tensor] = None,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Return (choices [B], logits [B,K] or None, log_probs [B,K] or None).

        For ``router`` the returned logits are already divided by the temperature, so
        ``log_probs == log_softmax(logits)`` and entropy / collapse-KL see the same
        distribution that sampling uses.

        ``all``: (None, None, None)
        ``random``: uniform hard indices
        ``gt``: hard indices from ``selected_modality`` (offline action-error oracle)
        ``router``: hard indices from learned router over R0 + MoT context
        """
        mode = getattr(self, "modality_select", "all")
        enabled_t = tuple(enabled)
        if mode == "all":
            return None, None, None
        if mode == "random":
            if not enabled_t:
                raise ValueError("Cannot sample modality_select=random with no enabled streams.")
            if batch_size <= 0:
                raise ValueError(f"`batch_size` must be positive, got {batch_size}")
            choices = torch.randint(len(enabled_t), (batch_size,), device=device)
            return choices, None, None
        if mode == "gt":
            if selected_modality is None:
                raise ValueError(
                    "modality_select=gt requires sample['selected_modality'] "
                    "(set data.train.modality_label_path)."
                )
            choices = selected_modality
            if choices.ndim == 0:
                choices = choices.view(1)
            choices = choices.to(device=device, dtype=torch.long).view(-1)
            if int(choices.shape[0]) != batch_size:
                raise ValueError(
                    f"`selected_modality` batch {tuple(choices.shape)} != batch_size={batch_size}"
                )
            # Labels are canonical rgb=0, depth=1, flow=2.
            if int(choices.min()) < 0 or int(choices.max()) >= len(_STREAM_KEYS):
                raise ValueError(
                    f"`selected_modality` must be in [0, {len(_STREAM_KEYS)}), "
                    f"got min={int(choices.min())} max={int(choices.max())}"
                )
            if enabled_t != _STREAM_KEYS:
                canon_to_enabled = {name: i for i, name in enumerate(enabled_t)}
                mapped = []
                for c in choices.tolist():
                    name = _STREAM_KEYS[int(c)]
                    if name not in canon_to_enabled:
                        raise ValueError(
                            f"GT label stream {name!r} is not in enabled streams {enabled_t}"
                        )
                    mapped.append(canon_to_enabled[name])
                choices = torch.tensor(mapped, device=device, dtype=torch.long)
            return choices, None, None
        if mode == "router":
            if self.modality_router is None:
                raise RuntimeError("modality_select=router but modality_router is not built.")
            if r0_latent is None or context is None or context_mask is None:
                raise ValueError(
                    "modality_select=router requires r0_latent, context, context_mask "
                    "(same tensors as MoT)."
                )
            feats = self._build_router_features(
                r0_latent=r0_latent,
                context=context,
                context_mask=context_mask,
            )
            temp = max(float(getattr(self, "stream_router_temperature", 1.0)), 1e-4)
            logits = self.modality_router(feats) / temp  
            log_probs = F.log_softmax(logits, dim=-1)
            # Router's own mode, not self.training: the trainer calls model.eval()
            # and then re-enables train mode only on dit / proprio_encoder /
            # modality_router, so self.training is False during training.
            if self._router_in_warmup():
                # Warm-up: uniform routing so every branch trains equally.
                choices = torch.randint(logits.shape[-1], (logits.shape[0],), device=logits.device)
            elif self._router_is_training():
                choices = torch.multinomial(log_probs.exp(), num_samples=1).squeeze(-1)
            else:
                choices = logits.argmax(dim=-1)
            return choices, logits, log_probs
        raise ValueError(f"Unhandled modality_select={mode!r}")

    def _build_router_features(
        self,
        *,
        r0_latent: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Same R0 + MoT context FastWAM training uses — no adaptive pool, no token mean.

        Detached: the context carries the trainable proprio token, and the router
        losses must not update ``proprio_encoder`` (only MoT losses should).
        """
        if r0_latent.ndim != 4:
            raise ValueError(f"`r0_latent` must be [B,C,H,W], got {tuple(r0_latent.shape)}")
        if context.ndim != 3:
            raise ValueError(f"`context` must be [B,L,D], got {tuple(context.shape)}")
        r0 = r0_latent.detach().float().flatten(1)
        ctx = context.detach().float()
        mask = context_mask.bool()
        router = self.modality_router
        if r0.shape[-1] != router.r0_dim:
            raise ValueError(
                f"Router `r0` dim mismatch: got {r0.shape[-1]}, expected {router.r0_dim} "
                f"(check router_video_size vs actual video HxW)."
            )
        if ctx.shape[1] != router.context_len or ctx.shape[2] != router.context_dim:
            raise ValueError(
                f"Router `context` shape mismatch: got {tuple(ctx.shape[1:])}, "
                f"expected ({router.context_len}, {router.context_dim})."
            )
        if mask.shape != ctx.shape[:2]:
            raise ValueError(
                f"`context_mask` must be [B,L]={tuple(ctx.shape[:2])}, got {tuple(mask.shape)}"
            )
        return {"r0": r0, "context": ctx, "context_mask": mask}

    def _router_is_training(self) -> bool:
        """Train/eval mode of the router (see ``_apply_dit_only_train_mode``)."""
        router = getattr(self, "modality_router", None)
        return bool(router is not None and router.training)

    def _router_in_warmup(self) -> bool:
        """True during the first ``router_warmup_steps`` optimizer steps (train mode only)."""
        warmup = int(getattr(self, "router_warmup_steps", 0))
        if warmup <= 0 or not self._router_is_training():
            return False
        steps_done = int(getattr(self, "_router_calls", 0)) // int(getattr(self, "router_grad_accum", 1))
        return steps_done < warmup

    def _router_tick(self) -> None:
        """Count one training_loss call (micro-batch) with the router in train mode."""
        if self._router_is_training():
            self._router_calls = int(getattr(self, "_router_calls", 0)) + 1

    @torch.no_grad()
    def _router_reward_baseline(
        self,
        reward: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        """Per-sample REINFORCE baseline ``[B]`` for a detached reward ``[B]``.

        ``batch``: batch mean (the original behaviour).
        ``timestep_ema``: running mean reward of the sample's action-noise bucket
        (``sigma = t / num_train_timesteps`` split into ``router_baseline_bins``
        equal bins). Per-sample loss is dominated by the sampled noise level, so
        this removes most of the variance that is unrelated to the modality pick.
        Buckets with no history yet fall back to the batch mean. The EMA is only
        updated in train mode (router's own mode), after the baseline is read
        (no self-leakage).
        """
        batch_mean = reward.mean().expand_as(reward)
        if self.router_baseline == "batch":
            return batch_mean

        bins = int(self.router_baseline_bins)
        if self._router_baseline_ema is None or self._router_baseline_ema.device != reward.device:
            self._router_baseline_ema = torch.zeros(bins, device=reward.device, dtype=torch.float32)
            self._router_baseline_seen = torch.zeros(bins, device=reward.device, dtype=torch.bool)
        bucket = (sigma * bins).long().clamp_(0, bins - 1)
        seen = self._router_baseline_seen[bucket]
        baseline = torch.where(seen, self._router_baseline_ema[bucket], batch_mean)

        if self._router_is_training():
            m = float(self.router_baseline_momentum)
            for b in bucket.unique().tolist():
                r_b = reward[bucket == b].float().mean()
                if bool(self._router_baseline_seen[b]):
                    self._router_baseline_ema[b] = m * self._router_baseline_ema[b] + (1.0 - m) * r_b
                else:
                    self._router_baseline_ema[b] = r_b
                    self._router_baseline_seen[b] = True
        return baseline

    @staticmethod
    def _load_modality_prompts(
        *,
        sentences: Optional[dict],
        cache_dir: str,
        context_len: int,
        enc_id: str,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """Load the cached T5 tokens of one sentence per stream (canonical order).

        Same cache / naming / prompt template as the task instructions
        (``RobotVideoDataset``). Only the real (non-zero) tokens are kept, padded to the
        longest sentence.
        """
        # Same wrapping as the instructions (dataset + precompute script), so the cache key
        # and the encoding match exactly.
        from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT

        sent = dict(DEFAULT_MODALITY_SENTENCES)
        if sentences:
            sent.update({str(k): str(v) for k, v in dict(sentences).items()})
        rows = []
        for name in _STREAM_KEYS:
            text = sent[name]
            hashed = hashlib.sha256(DEFAULT_PROMPT.format(task=text).encode("utf-8")).hexdigest()
            path = Path(cache_dir) / f"{hashed}.t5_len{context_len}.{enc_id}.pt"
            if not path.is_file():
                raise FileNotFoundError(
                    f"Missing modality prompt cache {path} for {name!r}: {text!r}. Run "
                    f"`python scripts/precompute_text_embeds.py task=<task> override_instruction=\"{text}\"`."
                )
            payload = torch.load(path, map_location="cpu")
            ctx = payload["context"].float()
            real = (ctx.abs().amax(-1) > 0) & payload["mask"].bool()
            rows.append(ctx[real])
        k = max(int(r.shape[0]) for r in rows)
        tokens = torch.zeros(len(rows), k, rows[0].shape[-1])
        mask = torch.zeros(len(rows), k, dtype=torch.bool)
        for i, r in enumerate(rows):
            tokens[i, : r.shape[0]] = r
            mask[i, : r.shape[0]] = True
        logger.info(
            "Loaded modality prompts (%d tokens max): %s",
            k, {n: int(m.sum()) for n, m in zip(_STREAM_KEYS, mask)},
        )
        return tokens, mask, sent

    def _joint_prompt_text(self, prompt: str, stream: str) -> str:
        """Instruction + condition, e.g. ``"... instruction: <task> Predict the depth video."``."""
        return f"{prompt} {self.modality_prompt_sentences[stream]}"

    def _load_joint_prompt(self, text: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Cached T5 context of ``text`` (same cache / naming as the instructions)."""
        hit = self._joint_prompt_cache.get(text)
        if hit is not None:
            return hit
        hashed = hashlib.sha256(text.encode("utf-8")).hexdigest()
        path = (
            Path(self._modality_prompt_cache_dir)
            / f"{hashed}.t5_len{self._modality_prompt_context_len}.{self._modality_prompt_enc_id}.pt"
        )
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing joint modality prompt cache {path} for {text!r}. Run "
                "`python scripts/precompute_modality_text_embeds.py task=<task>`."
            )
        payload = torch.load(path, map_location="cpu")
        hit = (payload["context"], payload["mask"].bool())
        self._joint_prompt_cache[text] = hit
        return hit

    def _modality_prompt_context(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        stream_names: Sequence[str],
        prompts: Optional[Sequence[str]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """MoT context told which modality to predict (per sample).

        ``stream_names``: one canonical stream name per sample (or one for the batch).
        ``joint``: ``[T5(instruction + condition) | proprio]`` (needs ``prompts``);
        ``separate``: ``[instruction | condition | proprio]``. No-op when off.
        """
        if not getattr(self, "modality_prompt", False):
            return context, context_mask
        if len(stream_names) == 1 and context.shape[0] > 1:
            stream_names = list(stream_names) * int(context.shape[0])
        if len(stream_names) != context.shape[0]:
            raise ValueError(f"{len(stream_names)} stream names for batch {context.shape[0]}")
        n_tail = 1 if getattr(self, "proprio_encoder", None) is not None else 0  # proprio token stays last
        if self.modality_prompt_mode == "joint":
            if prompts is None:
                raise ValueError("modality_prompt_mode=joint needs the instruction `prompt` strings.")
            prompts = list(prompts)
            if len(prompts) == 1 and context.shape[0] > 1:
                prompts = prompts * int(context.shape[0])
            pairs = [self._load_joint_prompt(self._joint_prompt_text(p, n)) for p, n in zip(prompts, stream_names)]
            jctx = torch.stack([c for c, _ in pairs]).to(device=context.device, dtype=context.dtype)
            jmsk = torch.stack([m for _, m in pairs]).to(device=context_mask.device, dtype=context_mask.dtype)
            tail = context[:, context.shape[1] - n_tail :]
            mtail = context_mask[:, context.shape[1] - n_tail :]
            return torch.cat([jctx, tail], dim=1), torch.cat([jmsk, mtail], dim=1)
        idx = torch.tensor([_STREAM_KEYS.index(str(n)) for n in stream_names], device="cpu")
        tok = self._modality_prompt_tokens[idx].to(device=context.device, dtype=context.dtype)
        msk = self._modality_prompt_mask[idx].to(device=context_mask.device, dtype=context_mask.dtype)
        head, tail = context[:, : context.shape[1] - n_tail], context[:, context.shape[1] - n_tail :]
        mhead, mtail = context_mask[:, : context.shape[1] - n_tail], context_mask[:, context.shape[1] - n_tail :]
        return torch.cat([head, tok, tail], dim=1), torch.cat([mhead, msk, mtail], dim=1)

    def _action_side_context(
        self,
        plain_context: torch.Tensor,
        plain_context_mask: torch.Tensor,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Context for the action expert: ``None`` = same as the video expert; with
        ``modality_prompt_target=video`` the plain ``[instruction | proprio]``."""
        if getattr(self, "modality_prompt", False) and getattr(self, "modality_prompt_target", "both") == "video":
            return plain_context, plain_context_mask
        return None, None

    def _infer_modality_context(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        stream: str,
        *,
        prompt: Optional[str],
        proprio: Optional[torch.Tensor],
        prompt_text: Optional[str] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Inference MoT context for the chosen ``stream`` (router or fixed active_future).

        ``prompt_text``: the instruction string when only the cached ``context`` was passed
        (e.g. the trainer's periodic eval), used to look up the cached joint condition.
        """
        if not getattr(self, "modality_prompt", False):
            return context, context_mask
        if self.modality_prompt_mode == "joint" and prompt is not None and getattr(self, "text_encoder", None) is not None:
            # Eval path: encode instruction + condition together, then append proprio as usual.
            return self._resolve_infer_context(
                prompt=self._joint_prompt_text(prompt, stream), proprio=proprio, context=None, context_mask=None
            )
        text = prompt if prompt is not None else prompt_text
        return self._modality_prompt_context(
            context, context_mask, [stream], prompts=None if text is None else [text]
        )

    def _infer_layout(self, active_future: Optional[str], action_attend_mode: str):
        """(gate context, loop active_future, streams to denoise) for inference.

        ``full`` + a chosen stream (same one-hot code as training): all three futures are
        in the MoT, only the chosen one is denoised, and the action attends only to it.
        Otherwise: the existing behaviour (short layout for a chosen stream).
        """
        if getattr(self, "router_layout", "short") != "full" or active_future is None:
            return contextlib.nullcontext(), active_future, None
        if action_attend_mode != "all":
            logger.warning("router_layout=full needs action_attend_mode='all'; using the short layout.")
            return contextlib.nullcontext(), active_future, None
        enabled = self._enabled_streams()
        choice = torch.tensor([enabled.index(active_future)], device=self.device)
        gate = self._action_stream_gate(self._one_hot_gate(choice, len(enabled)).to(self.device))
        return gate, None, (active_future,)

    @contextlib.contextmanager
    def _action_stream_gate(self, log_gate: torch.Tensor):
        """Within this block, the MoT mask hides futures from the action per sample."""
        self._action_stream_log_gate = log_gate
        try:
            yield
        finally:
            self._action_stream_log_gate = None

    @staticmethod
    def _one_hot_gate(choices: torch.Tensor, num_streams: int) -> torch.Tensor:
        """(1 0 0) -> [0, -1e4, -1e4]: chosen future visible to the action, others hidden."""
        hard = F.one_hot(choices.long(), num_streams).bool()
        return torch.where(hard, 0.0, _GATE_OFF).float()

    @staticmethod
    def _slice_batch(x: Optional[torch.Tensor], idx: torch.Tensor) -> Optional[torch.Tensor]:
        if x is None:
            return None
        return x.index_select(0, idx)

    # Which futures the action may read, per ``action_attend_mode`` (ablations).
    _ACTION_SEES = {
        "all": ("rgb", "depth", "flow"),
        "rgb": ("rgb",),
        "depth": ("depth",),
        "flow": ("flow",),
        "rgb_depth": ("rgb", "depth"),
        "rgb_flow": ("rgb", "flow"),
        "depth_flow": ("depth", "flow"),
        "cond_only": (),
    }

    def _mot_blocks(
        self,
        stream_seq_lens: list[int],
        action_seq_len: int,
        video_tokens_per_frame: int,
        enabled_streams: Optional[Sequence[str]],
    ) -> tuple[tuple[str, ...], list[str], list[int]]:
        """Blocks of the MoT sequence, in order, with their token lengths.

        ``R0 | rgb | depth | flow | A``: R0 is frame 0 of the rgb stream, each stream
        block is that stream's future tokens (may be empty for rgb in the short layout).
        """
        enabled = tuple(enabled_streams) if enabled_streams is not None else self._enabled_streams()
        if list(enabled) != [n for n in _STREAM_KEYS if n in enabled]:
            raise ValueError(f"`enabled_streams` must be in canonical order, got {enabled}")
        if "rgb" not in enabled:
            raise ValueError("`enabled_streams` must include 'rgb'.")
        if len(stream_seq_lens) != len(enabled):
            raise ValueError(f"stream_seq_lens length {len(stream_seq_lens)} != enabled streams {enabled}")
        tpf = int(video_tokens_per_frame)
        if tpf <= 0 or any(int(n) <= 0 for n in stream_seq_lens):
            raise ValueError(f"Bad lengths: tokens_per_frame={tpf}, stream_seq_lens={stream_seq_lens}")
        if any(int(n) % tpf for n in stream_seq_lens):
            raise ValueError(f"stream_seq_lens {stream_seq_lens} must be multiples of tokens_per_frame {tpf}")
        lens = dict(zip(enabled, (int(n) for n in stream_seq_lens)))
        names = ["R0", *enabled, "A"]
        sizes = [tpf, *(lens[n] - tpf if n == "rgb" else lens[n] for n in enabled), int(action_seq_len)]
        return enabled, names, sizes

    @torch.no_grad()
    def _build_multistream_mot_attention_mask_base(
        self,
        stream_seq_lens: list[int],
        action_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
        action_attend_mode: str = "all",
        enabled_streams: Optional[Sequence[str]] = None,
    ) -> torch.Tensor:
        """Bool ``[S, S]`` mask (row may attend column), built from a block table:

                    R0   rgb   depth  flow   A
            R0      ✓
            rgb     ✓    ✓
            depth   ✓          ✓
            flow    ✓                 ✓
            A       ✓    (futures allowed by action_attend_mode)  ✓

        Futures never see each other; video never sees the action.
        """
        mode = self._validate_action_attend_mode(action_attend_mode)
        enabled, names, sizes = self._mot_blocks(stream_seq_lens, action_seq_len, video_tokens_per_frame, enabled_streams)
        allow = {"R0": {"R0"}, "A": {"R0", "A", *self._ACTION_SEES[mode]}}
        allow.update({n: {"R0", n} for n in enabled})
        block = torch.tensor([[col in allow[row] for col in names] for row in names], dtype=torch.bool)
        reps = torch.tensor(sizes)
        return block.repeat_interleave(reps, 0).repeat_interleave(reps, 1).to(device)

    def _build_multistream_mot_attention_mask(
        self,
        stream_seq_lens: list[int],
        action_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
        action_attend_mode: str = "all",
        enabled_streams: Optional[Sequence[str]] = None,
    ) -> torch.Tensor:
        """Bool mask (``_base``), or inside ``_action_stream_gate`` a per-sample float mask
        ``[B, S, S]``: the gate ``[B, K]`` (0 = visible, -1e4 = hidden) is added on the
        action row, over each stream's future block.

        Not under ``no_grad``: the Gumbel subclass needs gradient through the gate.
        """
        base = self._build_multistream_mot_attention_mask_base(
            stream_seq_lens=stream_seq_lens,
            action_seq_len=action_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            device=device,
            action_attend_mode=action_attend_mode,
            enabled_streams=enabled_streams,
        )
        log_gate = getattr(self, "_action_stream_log_gate", None)
        if log_gate is None:
            return base
        if action_attend_mode != "all":
            raise ValueError("Per-sample action gate requires action_attend_mode='all'.")
        enabled, _names, sizes = self._mot_blocks(stream_seq_lens, action_seq_len, video_tokens_per_frame, enabled_streams)
        if log_gate.shape[-1] != len(enabled):
            raise ValueError(f"log_gate K={log_gate.shape[-1]} != streams {enabled}")
        g = log_gate.float()
        zero = g.new_zeros(g.shape[0], 1)
        block_bias = torch.cat([zero, g, zero], dim=1)  # [B, R0 | streams | A]
        col_bias = block_bias.repeat_interleave(torch.tensor(sizes, device=g.device), dim=1)  # [B, S]
        is_action_row = torch.zeros(base.shape[-1], device=device)
        is_action_row[base.shape[-1] - int(action_seq_len):] = 1.0
        base_f = torch.zeros(base.shape, device=device).masked_fill(~base, float("-inf"))
        return (base_f[None] + is_action_row[None, :, None] * col_bias[:, None, :]).to(self.torch_dtype)

    def _encode_stream_latents(
        self,
        video: torch.Tensor,
        tiled: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], bool]:
        if video.ndim != 5:
            raise ValueError(f"Stream video must be [B,3,T,H,W], got {tuple(video.shape)}")
        if video.shape[1] != 3:
            raise ValueError(f"Stream video channel dim must be 3, got {tuple(video.shape)}")

        _, _, num_frames, height, width = video.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(f"Video spatial dims must be multiples of 16, got H={height}, W={width}")
        if num_frames % 4 != 1:
            raise ValueError(f"Video T must satisfy T % 4 == 1, got T={num_frames}")
        if num_frames <= 1:
            raise ValueError(f"Video T must be > 1, got T={num_frames}")

        input_video = video.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        input_latents = self._encode_video_latents(input_video, tiled=tiled)
        first_frame_latents = None
        fuse_flag = False
        if getattr(self.video_expert, "fuse_vae_embedding_in_latents", False):
            first_frame_latents = input_latents[:, :, 0:1]
            fuse_flag = True
        return input_latents, first_frame_latents, fuse_flag

    def _build_shared_context_action(
        self,
        sample: dict[str, Any],
        batch_size: int,
        num_frames: int,
    ) -> dict[str, Any]:
        if "action" not in sample:
            raise ValueError("`sample['action']` is required for multimodal training.")
        action = sample["action"]
        if action.ndim != 3:
            raise ValueError(f"`sample['action']` must be [B,T,a_dim], got {tuple(action.shape)}")
        action_horizon = int(action.shape[1])
        if action_horizon % (num_frames - 1) != 0:
            raise ValueError(
                f"`sample['action']` temporal dim must be divisible by video transitions "
                f"({num_frames - 1}), got {action_horizon}"
            )

        action_is_pad = sample.get("action_is_pad", None)
        if action_is_pad is not None:
            if action_is_pad.ndim != 2 or action_is_pad.shape != (batch_size, action_horizon):
                raise ValueError(
                    f"`action_is_pad` shape mismatch: got {tuple(action_is_pad.shape)} "
                    f"vs expected ({batch_size}, {action_horizon})"
                )

        image_is_pad = sample.get("image_is_pad", None)
        if image_is_pad is not None:
            if image_is_pad.ndim != 2 or image_is_pad.shape != (batch_size, num_frames):
                raise ValueError(
                    f"`image_is_pad` shape mismatch: got {tuple(image_is_pad.shape)} "
                    f"vs expected ({batch_size}, {num_frames})"
                )

        context = sample.get("context")
        context_mask = sample.get("context_mask")
        if context is None and context_mask is None:
            prompt = sample.get("prompt")
            if prompt is None:
                raise ValueError("Multimodal training requires `context/context_mask` or `prompt`.")
            context, context_mask = self.encode_prompt(prompt)
        elif context is None or context_mask is None:
            raise ValueError("`context` and `context_mask` must both exist when either is provided.")

        if context.ndim != 3 or context_mask.ndim != 2:
            raise ValueError(
                f"`context/context_mask` must be [B,L,D]/[B,L], got "
                f"{tuple(context.shape)} and {tuple(context_mask.shape)}"
            )
        context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)

        proprio = sample.get("proprio", None)
        if self.proprio_encoder is not None:
            if proprio is None:
                raise ValueError("`sample['proprio']` is required when `proprio_dim` is enabled.")
            if proprio.ndim != 3:
                raise ValueError(f"`sample['proprio']` must be [B,T,d], got {tuple(proprio.shape)}")
            if proprio.shape[2] != self.proprio_dim:
                raise ValueError(
                    f"`sample['proprio']` last dim must be {self.proprio_dim}, got {proprio.shape[2]}"
                )
            proprio = proprio[:, 0, :]
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio.to(device=self.device, dtype=self.torch_dtype),
            )

        action = action.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        if action_is_pad is not None:
            action_is_pad = action_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if image_is_pad is not None:
            image_is_pad = image_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)

        return {
            "context": context,
            "context_mask": context_mask,
            "action": action,
            "action_is_pad": action_is_pad,
            "image_is_pad": image_is_pad,
        }

    def _joint_denoise_multistream(
        self,
        noisy_latents: dict[str, torch.Tensor],
        latents_action: torch.Tensor,
        timestep_video: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
        action_condition: Optional[torch.Tensor] = None,
        action_attend_mode: str = "all",
        active_future: Optional[str] = None,
        action_context: Optional[torch.Tensor] = None,
        action_context_mask: Optional[torch.Tensor] = None,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Joint denoise over multistream MoT.

        ``action_context``: text context for the action expert's cross-attention
        (default: the same ``context`` as the video expert).

        ``active_future=None``: all ``enabled_video_streams`` futures in MoT
        (training default when ``modality_select=all``, and inference).

        ``active_future`` in {rgb, depth, flow}: only that future is active —
          - rgb:   [R0 | R_fut | A]
          - depth: [R0 | D_fut | A]
          - flow:  [R0 | F_fut | A]
        """
        action_attend_mode = self._validate_action_attend_mode(action_attend_mode)
        enabled_all = self._enabled_streams()
        if active_future is None:
            mot_streams = enabled_all
            rgb_tokens_mode = "full"
            attend_mode = action_attend_mode
        else:
            active = str(active_future).lower()
            if active not in enabled_all:
                raise ValueError(
                    f"`active_future`={active!r} must be one of enabled streams {enabled_all}"
                )
            if active == "rgb":
                mot_streams = ("rgb",)
                rgb_tokens_mode = "full"
                attend_mode = "all"
            else:
                # R0 condition + selected non-RGB future only.
                mot_streams = ("rgb", active)
                rgb_tokens_mode = "cond_only"
                attend_mode = active

        missing_latents = [name for name in mot_streams if name not in noisy_latents]
        if missing_latents:
            raise ValueError(f"Missing noisy latents for MoT streams: {missing_latents}")
        if "rgb" not in noisy_latents:
            raise ValueError("RGB latents are required for R0 conditioning.")

        # Always prepare RGB (R0 and optionally R_fut). Prepare other MoT streams as needed.
        prepare_names = tuple(dict.fromkeys(("rgb",) + tuple(mot_streams)))
        prepared = {}
        for name in prepare_names:
            (
                video_tokens,
                t_video,
                t_mod_video,
                context_video,
                context_mask_video,
                freqs_video,
                f_video,
                h_video,
                w_video,
                tokens_per_frame,
            ) = self.video_expert.prepare(
                x=noisy_latents[name],
                timestep=timestep_video,
                context=context,
                context_mask=context_mask,
                action=action_condition,
                fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
            )
            prepared[name] = {
                "tokens": video_tokens,
                "t": t_video,
                "t_mod": t_mod_video,
                "context": context_video,
                "context_mask": context_mask_video,
                "freqs": freqs_video,
                "f": f_video,
                "h": h_video,
                "w": w_video,
                "tokens_per_frame": tokens_per_frame,
            }

        ref = prepared["rgb"]
        tpf = int(ref["tokens_per_frame"])
        for name in prepare_names:
            if name == "rgb":
                continue
            cur = prepared[name]
            if (cur["f"], cur["h"], cur["w"], cur["tokens_per_frame"]) != (
                ref["f"],
                ref["h"],
                ref["w"],
                ref["tokens_per_frame"],
            ):
                raise ValueError(
                    f"Stream '{name}' grid mismatch vs rgb: "
                    f"got fhw/tpf={(cur['f'], cur['h'], cur['w'], cur['tokens_per_frame'])} "
                    f"vs {(ref['f'], ref['h'], ref['w'], ref['tokens_per_frame'])}"
                )
            if cur["tokens"].shape[1] <= tpf:
                raise ValueError(
                    f"Stream '{name}' must have future tokens beyond frame0, "
                    f"got seq_len={cur['tokens'].shape[1]}, tpf={tpf}"
                )

        def _futures_only(p: dict[str, Any]) -> dict[str, Any]:
            return {
                "tokens": p["tokens"][:, tpf:],
                "t": p["t"][:, tpf:],
                "t_mod": p["t_mod"][:, tpf:],
                "context": p["context"],
                "context_mask": p["context_mask"][:, tpf:],
                "freqs": p["freqs"][tpf:],
                "f": int(p["f"]) - 1,
                "h": p["h"],
                "w": p["w"],
                "tokens_per_frame": tpf,
                "drop_condition_tokens": True,
            }

        def _condition_only(p: dict[str, Any]) -> dict[str, Any]:
            return {
                "tokens": p["tokens"][:, :tpf],
                "t": p["t"][:, :tpf],
                "t_mod": p["t_mod"][:, :tpf],
                "context": p["context"],
                "context_mask": p["context_mask"][:, :tpf],
                "freqs": p["freqs"][:tpf],
                "f": 1,
                "h": p["h"],
                "w": p["w"],
                "tokens_per_frame": tpf,
                "drop_condition_tokens": False,
            }

        mot_parts: dict[str, dict[str, Any]] = {}
        if rgb_tokens_mode == "full":
            mot_parts["rgb"] = {**prepared["rgb"], "drop_condition_tokens": False}
        else:
            mot_parts["rgb"] = _condition_only(prepared["rgb"])
        for name in mot_streams:
            if name == "rgb":
                continue
            mot_parts[name] = _futures_only(prepared[name])

        video_tokens = torch.cat([mot_parts[n]["tokens"] for n in mot_streams], dim=1)
        t_mod_video = torch.cat([mot_parts[n]["t_mod"] for n in mot_streams], dim=1)
        freqs_video = torch.cat([mot_parts[n]["freqs"] for n in mot_streams], dim=0)
        context_mask_video = torch.cat(
            [mot_parts[n]["context_mask"] for n in mot_streams], dim=1
        )
        context_video = mot_parts["rgb"]["context"]
        
        
        (
            action_tokens,
            _t_action,
            t_mod_action,
            context_action,
            context_mask_action,
            freqs_action,
        ) = self.action_expert.prepare(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context if action_context is None else action_context,
            context_mask=context_mask if action_context is None else action_context_mask,
        )

        stream_seq_lens = [int(mot_parts[n]["tokens"].shape[1]) for n in mot_streams]
        attention_mask = self._build_multistream_mot_attention_mask(
            stream_seq_lens=stream_seq_lens,
            action_seq_len=action_tokens.shape[1],
            video_tokens_per_frame=tpf,
            device=video_tokens.device,
            action_attend_mode=attend_mode,
            enabled_streams=mot_streams,
        )

        video_tokens_out, action_tokens_out = self.mot.forward_joint_core(
            video_tokens=video_tokens,
            action_tokens=action_tokens,
            video_freqs=freqs_video,
            action_freqs=freqs_action,
            video_t_mod=t_mod_video,
            action_t_mod=t_mod_action,
            video_context=context_video,
            video_context_mask=context_mask_video,
            action_context=context_action,
            action_context_mask=context_mask_action,
            attention_mask=attention_mask,
        )

        pred_video = {}
        offset = 0
        for name, seq_len in zip(mot_streams, stream_seq_lens):
            chunk = video_tokens_out[:, offset : offset + seq_len]
            part = mot_parts[name]
            if name == "rgb" and rgb_tokens_mode == "cond_only":
                # R0-only tokens are conditioning; no future video pred from this branch.
                offset += seq_len
                continue
            pred = self.video_expert.post(
                chunk,
                part["t"],
                part["f"],
                part["h"],
                part["w"],
            )
            if part["drop_condition_tokens"]:
                cond = noisy_latents[name][:, :, 0:1]
                pred = torch.cat([cond, pred], dim=2)
            pred_video[name] = pred
            offset += seq_len

        pred_action = self.action_expert.post(action_tokens_out)
        return pred_video, pred_action

    def training_loss(self, sample, tiled: bool = False):
        enabled = self._enabled_streams()
        missing = [f"video_{k}" for k in enabled if f"video_{k}" not in sample]
        if missing:
            raise ValueError(
                f"FastWAMJointMultimodal.training_loss requires enabled video keys {list(enabled)}; "
                f"missing {missing}."
            )

        stream_latents: dict[str, torch.Tensor] = {}
        first_frames: dict[str, Optional[torch.Tensor]] = {}
        fuse_flag = False
        for name in enabled:
            latents, first_frame, fuse = self._encode_stream_latents(
                sample[f"video_{name}"], tiled=tiled
            )
            stream_latents[name] = latents
            first_frames[name] = first_frame
            fuse_flag = fuse_flag or fuse

        batch_size = stream_latents["rgb"].shape[0]
        num_frames = int(sample["video_rgb"].shape[2])
        for name in enabled:
            if name == "rgb":
                continue
            if stream_latents[name].shape != stream_latents["rgb"].shape:
                raise ValueError(
                    f"Latent shape mismatch for '{name}': "
                    f"{tuple(stream_latents[name].shape)} vs rgb {tuple(stream_latents['rgb'].shape)}"
                )

        shared = self._build_shared_context_action(
            sample=sample,
            batch_size=batch_size,
            num_frames=num_frames,
        )
        context = shared["context"]
        context_mask = shared["context_mask"]
        action = shared["action"]
        action_is_pad = shared["action_is_pad"]
        image_is_pad = shared["image_is_pad"]

        timestep_video = self.train_video_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=stream_latents["rgb"].dtype,
        )
        noisy_latents: dict[str, torch.Tensor] = {}
        target_video: dict[str, torch.Tensor] = {}
        for name in enabled:
            noise = torch.randn_like(stream_latents[name])
            noisy = self.train_video_scheduler.add_noise(
                stream_latents[name], noise, timestep_video
            )
            target_video[name] = self.train_video_scheduler.training_target(
                stream_latents[name], noise, timestep_video
            )
            if first_frames[name] is not None:
                noisy[:, :, 0:1] = first_frames[name]
            noisy_latents[name] = noisy

        noise_action = torch.randn_like(action)
        timestep_action = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=action.dtype,
        )
        noisy_action = self.train_action_scheduler.add_noise(action, noise_action, timestep_action)
        target_action = self.train_action_scheduler.training_target(
            action, noise_action, timestep_action
        )

        selected_modality = sample.get("selected_modality", None)
        if selected_modality is not None and not torch.is_tensor(selected_modality):
            selected_modality = torch.as_tensor(selected_modality)
        modality_choices, router_logits, router_log_probs = self._sample_modality_select(
            enabled=enabled,
            batch_size=batch_size,
            device=stream_latents["rgb"].device,
            r0_latent=stream_latents["rgb"][:, :, 0],
            context=context,
            context_mask=context_mask,
            selected_modality=selected_modality,
        )
        # MoT context: instruction (+ chosen modality sentence) + proprio. The router above
        # saw the context without the sentence.
        ctx_mot, ctx_mask_mot = context, context_mask
        act_ctx, act_ctx_mask = self._action_side_context(context, context_mask)
        if modality_choices is not None:
            prompts = sample.get("prompt")
            if isinstance(prompts, str):
                prompts = [prompts]
            ctx_mot, ctx_mask_mot = self._modality_prompt_context(
                context, context_mask, [enabled[int(i)] for i in modality_choices.tolist()], prompts=prompts
            )

        include_initial_video_step = first_frames["rgb"] is None
        lambdas = {
            "rgb": float(self.loss_lambda_rgb),
            "depth": float(self.loss_lambda_depth),
            "flow": float(self.loss_lambda_flow),
        }
        loss_dict: dict[str, float] = {f"loss_video_{name}": 0.0 for name in _STREAM_KEYS}
        # float32 like FastWAM.training_loss (no bf16 rounding of the timestep weight).
        video_weight = self.train_video_scheduler.training_weight(timestep_video).to(
            device=stream_latents["rgb"].device,
            dtype=torch.float32,
        )
        loss_video_per_sample = torch.zeros(
            batch_size, device=stream_latents["rgb"].device, dtype=torch.float32
        )
        action_loss_per_sample = torch.zeros(
            batch_size, device=stream_latents["rgb"].device, dtype=torch.float32
        )
        # Detached accumulators for per-stream logging (sum of weighted sample losses).
        stream_loss_acc = {name: 0.0 for name in enabled}
        stream_loss_count = {name: 0 for name in enabled}

        def _run_group(
            idx: torch.Tensor,
            *,
            active_future: Optional[str],
            full_choices: Optional[torch.Tensor] = None,
        ) -> None:
            """``full_choices``: MM layout with a per-sample one-hot action mask."""
            nonlocal loss_video_per_sample, action_loss_per_sample
            if idx.numel() == 0:
                return

            # Only ship latents needed for this MoT topology.
            if full_choices is not None:
                needed = enabled
            elif active_future is None:
                needed = enabled
            elif active_future == "rgb":
                needed = ("rgb",)
            else:
                needed = ("rgb", active_future)
            noisy_sub = {k: noisy_latents[k].index_select(0, idx) for k in needed}
            action_cond = action.index_select(0, idx) if action is not None else None
            sub_choices = None if full_choices is None else full_choices.index_select(0, idx)
            gate = (
                contextlib.nullcontext()
                if sub_choices is None
                else self._action_stream_gate(self._one_hot_gate(sub_choices, len(enabled)).to(idx.device))
            )
            with gate:
                pred_video, pred_action = self._joint_denoise_multistream(
                    noisy_latents=noisy_sub,
                    latents_action=noisy_action.index_select(0, idx),
                    timestep_video=timestep_video.index_select(0, idx),
                    timestep_action=timestep_action.index_select(0, idx),
                    context=ctx_mot.index_select(0, idx),
                    context_mask=self._slice_batch(ctx_mask_mot, idx),
                    fuse_vae_embedding_in_latents=fuse_flag,
                    action_condition=action_cond,
                    active_future=None if sub_choices is not None else active_future,
                    action_context=None if act_ctx is None else act_ctx.index_select(0, idx),
                    action_context_mask=self._slice_batch(act_ctx_mask, idx),
                )

            video_names = (
                tuple(pred_video.keys())
                if active_future is None
                else (active_future,)
            )
            image_pad_sub = self._slice_batch(image_is_pad, idx)
            w_sub = video_weight.float().index_select(0, idx)
            for name in video_names:
                pred = pred_video[name]
                tgt = target_video[name].index_select(0, idx)
                if first_frames[name] is not None:
                    pred = pred[:, :, 1:]
                    tgt = tgt[:, :, 1:]
                loss_ps = self._compute_video_loss_per_sample(
                    pred_video=pred,
                    target_video=tgt,
                    image_is_pad=image_pad_sub,
                    include_initial_video_step=include_initial_video_step,
                )
                weighted_ps = lambdas[name] * loss_ps.float()
                n_used = int(idx.numel())
                if sub_choices is not None and self.router_video_loss == "chosen":
                    # Full layout: only the chosen future of each sample gets a video loss.
                    sel = (sub_choices == enabled.index(name)).float()
                    weighted_ps = weighted_ps * sel
                    n_used = int(sel.sum().item())
                loss_video_per_sample[idx] = loss_video_per_sample[idx] + weighted_ps
                stream_loss_acc[name] += float((weighted_ps * w_sub).sum().detach().item())
                stream_loss_count[name] += n_used

            tgt_a = target_action.index_select(0, idx)
            action_loss_token = F.mse_loss(
                pred_action.float(), tgt_a.float(), reduction="none"
            ).mean(dim=2)
            action_pad_sub = self._slice_batch(action_is_pad, idx)
            if action_pad_sub is not None:
                valid = (~action_pad_sub).to(
                    device=action_loss_token.device, dtype=action_loss_token.dtype
                )
                valid_sum = valid.sum(dim=1).clamp(min=1.0)
                a_ps = (action_loss_token * valid).sum(dim=1) / valid_sum
            else:
                a_ps = action_loss_token.mean(dim=1)
            action_loss_per_sample[idx] = a_ps.float()

        if modality_choices is None:
            all_idx = torch.arange(batch_size, device=stream_latents["rgb"].device)
            _run_group(all_idx, active_future=None)
        elif self.router_layout == "full":
            all_idx = torch.arange(batch_size, device=stream_latents["rgb"].device)
            _run_group(all_idx, active_future=None, full_choices=modality_choices)
        else:
            for stream_idx, name in enumerate(enabled):
                idx = (modality_choices == stream_idx).nonzero(as_tuple=True)[0]
                _run_group(idx, active_future=name)

        loss_video_total = (loss_video_per_sample * video_weight.float()).mean()
        for name in enabled:
            if stream_loss_count[name] > 0:
                loss_dict[f"loss_video_{name}"] = (
                    stream_loss_acc[name] / float(stream_loss_count[name])
                )

        action_weight = self.train_action_scheduler.training_weight(timestep_action).to(
            action_loss_per_sample.device, dtype=action_loss_per_sample.dtype
        )
        loss_action = (action_loss_per_sample * action_weight.float()).mean()
        loss_action_weighted = self.loss_lambda_action * loss_action
        loss_dict["loss_action"] = float(loss_action_weighted.detach().item())
        loss_dict["loss_video"] = float(loss_video_total.detach().item())
        if modality_choices is not None:
            canon = torch.tensor(
                [_STREAM_KEYS.index(enabled[int(i)]) for i in modality_choices.tolist()],
                device=modality_choices.device,
                dtype=torch.float32,
            )
            loss_dict["modality_select_idx"] = float(canon.mean().item())

        loss_total = loss_video_total + loss_action_weighted

        # Learned router: REINFORCE on a per-sample reward + entropy / collapse KL.
        if (
            getattr(self, "modality_select", "all") == "router"
            and router_logits is not None
            and router_log_probs is not None
            and modality_choices is not None
        ):
            # Same per-sample terms (and weights) that enter loss_total.
            action_ps = self.loss_lambda_action * (
                action_loss_per_sample * action_weight.float()
            )
            if self.router_reward == "action":
                # Matches the offline oracle (action error). Video losses of different
                # streams are on different scales and would bias the pick.
                reward_ps = action_ps
            else:
                reward_ps = loss_video_per_sample * video_weight.float() + action_ps
            reward_ps = reward_ps.detach()

            sigma = (
                timestep_action.detach().float()
                / float(self.train_action_scheduler.num_train_timesteps)
            ).clamp(0.0, 1.0)
            baseline = self._router_reward_baseline(reward_ps, sigma)
            advantage = reward_ps - baseline
            if self.router_adv_normalize and advantage.numel() > 1:
                advantage = advantage / advantage.std().clamp_min(1e-6)

            chosen_log_prob = router_log_probs.gather(
                1, modality_choices.unsqueeze(1)
            ).squeeze(1)
            # Minimize E[reward] (a loss); high advantage → lower chosen log-prob.
            pg_loss = (chosen_log_prob * advantage).mean()
            entropy = -(router_log_probs.exp() * router_log_probs).sum(dim=-1).mean()
            collapse = collapse_kl(router_logits)
            router_loss = (
                pg_loss
                - float(self.router_entropy_coef) * entropy
                + float(self.router_collapse_kl) * collapse
            )
            in_warmup = self._router_in_warmup()
            if self._router_is_training():
                # x0 in warm-up: router gets zero grads (keeps DeepSpeed happy) and stays put.
                loss_total = loss_total + (0.0 if in_warmup else 1.0) * router_loss
            loss_dict["router_warmup"] = float(in_warmup)
            loss_dict["loss_router"] = float(router_loss.detach().item())
            loss_dict["router_pg"] = float(pg_loss.detach().item())
            loss_dict["router_reward"] = float(reward_ps.mean().item())
            loss_dict["router_adv_abs"] = float(advantage.abs().mean().item())
            loss_dict["router_entropy"] = float(entropy.detach().item())
            loss_dict["router_collapse_kl"] = float(collapse.detach().item())
            probs = router_log_probs.exp().detach().mean(0)
            for i, name in enumerate(enabled):
                loss_dict[f"router_p_{name}"] = float(probs[i].item())
                # Fraction of the batch actually routed to each stream.
                loss_dict[f"router_pick_{name}"] = float(
                    (modality_choices == i).float().mean().item()
                )
            self._router_tick()

        return loss_total, loss_dict

    @staticmethod
    def _validate_infer_input_image(
        input_image: torch.Tensor,
        num_video_frames: int,
        *,
        check_fn,
    ) -> tuple[torch.Tensor, int, int]:
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        _, _, height, width = input_image.shape
        checked_h, checked_w, checked_t = check_fn(height, width, num_video_frames)
        if (checked_h, checked_w) != (height, width):
            raise ValueError(
                f"`input_image` must be resized before infer, expected multiples of 16 "
                f"but got HxW=({height},{width})"
            )
        if checked_t != num_video_frames:
            raise ValueError(
                f"`num_video_frames` must satisfy T % 4 == 1, got {num_video_frames}"
            )
        return input_image, height, width

    def _resolve_infer_context(
        self,
        *,
        prompt: Optional[str],
        proprio: Optional[torch.Tensor],
        context: Optional[torch.Tensor],
        context_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")

        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must be both provided together.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got "
                    f"{tuple(context.shape)} and {tuple(context_mask.shape)}"
                )
            context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)

        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError("`proprio` was provided but `proprio_dim=None` so `proprio_encoder` is disabled.")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            elif proprio.ndim == 2 and proprio.shape[0] == 1:
                pass
            else:
                raise ValueError(f"`proprio` must be [D] or [1,D], got shape {tuple(proprio.shape)}")
            if proprio.shape[1] != self.proprio_dim:
                raise ValueError(f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}")
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio,
            )
        return context, context_mask

    def _init_multistream_infer_latents(
        self,
        *,
        num_video_frames: int,
        height: int,
        width: int,
        action_horizon: int,
        input_image: torch.Tensor,
        seed: Optional[int],
        rand_device: str,
        tiled: bool,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict[str, torch.Tensor], bool]:
        latent_t = (num_video_frames - 1) // self.vae.temporal_downsample_factor + 1
        latent_h = height // self.vae.upsampling_factor
        latent_w = width // self.vae.upsampling_factor
        latent_shape = (1, self.vae.model.z_dim, latent_t, latent_h, latent_w)

        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_input_image_latents_tensor(
            input_image=input_image, tiled=tiled
        )
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))

        latents_video: dict[str, torch.Tensor] = {}
        first_frames: dict[str, torch.Tensor] = {}
        enabled = self._enabled_streams()
        for stream_idx, name in enumerate(enabled):
            generator = None
            if seed is not None:
                generator = torch.Generator(device=rand_device).manual_seed(int(seed) + stream_idx)
            noisy = torch.randn(
                latent_shape,
                generator=generator,
                device=rand_device,
                dtype=torch.float32,
            ).to(device=self.device, dtype=self.torch_dtype)
            noisy[:, :, 0:1] = first_frame_latents.clone()
            latents_video[name] = noisy
            first_frames[name] = first_frame_latents.clone()

        action_generator = None
        if seed is not None:
            action_generator = torch.Generator(device=rand_device).manual_seed(int(seed) + len(enabled))
        latents_action = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=action_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        return latents_video, latents_action, first_frames, fuse_flag

    def _decode_stream_videos(
        self,
        latents_video: dict[str, torch.Tensor],
        *,
        tiled: bool,
        streams: Optional[Sequence[str]] = None,
    ) -> dict[str, Any]:
        """Decode ``streams`` (default: all enabled). ``video`` = rgb if decoded, else the
        first decoded stream."""
        names = tuple(streams) if streams is not None else self._enabled_streams()
        decoded = {name: self._decode_latents(latents_video[name], tiled=tiled) for name in names}
        decoded["video"] = decoded["rgb"] if "rgb" in decoded else decoded[names[0]]
        return decoded

    def _run_multistream_infer_loop(
        self,
        *,
        latents_video: dict[str, torch.Tensor],
        latents_action: torch.Tensor,
        first_frames: dict[str, torch.Tensor],
        fuse_flag: bool,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        num_inference_steps: int,
        sigma_shift: Optional[float],
        action_condition: Optional[torch.Tensor],
        action_attend_mode: str = "all",
        active_future: Optional[str] = None,
        update_streams: Optional[Sequence[str]] = None,
        action_context: Optional[torch.Tensor] = None,
        action_context_mask: Optional[torch.Tensor] = None,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """``update_streams``: only these futures are denoised (stepped); the others stay
        in the MoT as their initial noise. ``None`` = step every future in the layout."""
        action_attend_mode = self._validate_action_attend_mode(action_attend_mode)
        if active_future is None:
            video_keys = self._enabled_streams()
        elif active_future == "rgb":
            video_keys = ("rgb",)
        else:
            video_keys = ("rgb", str(active_future).lower())
        latents_in = {k: latents_video[k] for k in video_keys}

        infer_timesteps_video, infer_deltas_video = self.infer_video_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_video["rgb"].dtype,
            shift_override=sigma_shift,
        )
        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        for step_t_video, step_delta_video, step_t_action, step_delta_action in zip(
            infer_timesteps_video,
            infer_deltas_video,
            infer_timesteps_action,
            infer_deltas_action,
        ):
            timestep_video = step_t_video.unsqueeze(0).to(
                dtype=latents_video["rgb"].dtype, device=self.device
            )
            timestep_action = step_t_action.unsqueeze(0).to(
                dtype=latents_action.dtype, device=self.device
            )
            pred_video, pred_action = self._joint_denoise_multistream(
                noisy_latents=latents_in,
                latents_action=latents_action,
                timestep_video=timestep_video,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                fuse_vae_embedding_in_latents=fuse_flag,
                action_condition=action_condition,
                action_attend_mode=action_attend_mode,
                active_future=active_future,
                action_context=action_context,
                action_context_mask=action_context_mask,
            )
            for name, pred in pred_video.items():
                if update_streams is not None and name not in update_streams:
                    continue  # inactive future: present in the MoT, never denoised
                latents_video[name] = self.infer_video_scheduler.step(
                    pred, step_delta_video, latents_video[name]
                )
                latents_video[name][:, :, 0:1] = first_frames[name].clone()
                latents_in[name] = latents_video[name]
            latents_action = self.infer_action_scheduler.step(
                pred_action, step_delta_action, latents_action
            )
        return latents_video, latents_action

    def _resolve_router_active_future(
        self,
        *,
        first_frame_latents: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> str:
        """Argmax modality from the learned router using the same R0 + MoT context."""
        if self.modality_router is None:
            raise RuntimeError("modality_select=router but modality_router is missing.")
        enabled = self._enabled_streams()
        r0 = first_frame_latents
        if r0.ndim == 5:
            r0 = r0[:, :, 0]
        choices, _, _ = self._sample_modality_select(
            enabled=enabled,
            batch_size=int(r0.shape[0]),
            device=r0.device,
            r0_latent=r0,
            context=context,
            context_mask=context_mask,
        )
        assert choices is not None
        return enabled[int(choices[0].item())]

    def _resolve_infer_active_future(
        self,
        *,
        active_future: Optional[str],
        action_attend_mode: str,
        first_frame_latents: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> Optional[str]:
        """Pick MoT future stream for infer — same contract as training ``active_future``.

        Priority:
          1. Explicit ``active_future`` in {rgb, depth, flow}, or ``random`` (uniform per call)
             (``all`` / ``none`` / empty → all enabled futures, same as ``None``)
          2. Learned router when ``modality_select=router`` and attend mode is ``all``
          3. ``None`` → all enabled futures in MoT
        """
        if active_future is not None:
            active = str(active_future).lower().strip()
            if active in ("", "all", "none", "null"):
                return None
            enabled = self._enabled_streams()
            if active == "random":
                # Uniform random stream per inference call (= per replan), as in training.
                chosen = enabled[int(torch.randint(len(enabled), (1,)).item())]
                logger.info("Random active_future=%s", chosen)
                return chosen
            if active not in enabled:
                raise ValueError(
                    f"`active_future`={active!r} must be one of {enabled} or 'all'"
                )
            return active
        if getattr(self, "modality_select", "all") == "router" and action_attend_mode == "all":
            chosen = self._resolve_router_active_future(
                first_frame_latents=first_frame_latents,
                context=context,
                context_mask=context_mask,
            )
            logger.info("Router selected active_future=%s", chosen)
            return chosen
        if getattr(self, "modality_prompt", False) and getattr(self, "modality_select", "all") in ("random", "gt"):
            # Trained on one conditioned stream per sample; "all futures, no condition" was
            # never seen. Default to rgb when no stream is given (e.g. the trainer's eval).
            logger.info("No active_future given with modality_prompt on; defaulting to rgb.")
            return "rgb"
        return None

    @torch.no_grad()
    def infer_action(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        num_video_frames: int,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        compile_action_infer: bool = False,
        action_attend_mode: str = "all",
        active_future: Optional[str] = None,
        prompt_text: Optional[str] = None,
    ) -> dict[str, Any]:
        del compile_action_infer, negative_prompt, text_cfg_scale
        self.eval()
        action_attend_mode = self._validate_action_attend_mode(action_attend_mode)
        if action_attend_mode != "all":
            logger.info(
                "FastWAMJointMultimodal.infer_action action_attend_mode=%s (infer ablation)",
                action_attend_mode,
            )

        input_image, height, width = self._validate_infer_input_image(
            input_image,
            num_video_frames,
            check_fn=self._check_resize_height_width,
        )
        context, context_mask = self._resolve_infer_context(
            prompt=prompt,
            proprio=proprio,
            context=context,
            context_mask=context_mask,
        )
        latents_video, latents_action, first_frames, fuse_flag = self._init_multistream_infer_latents(
            num_video_frames=num_video_frames,
            height=height,
            width=width,
            action_horizon=action_horizon,
            input_image=input_image,
            seed=seed,
            rand_device=rand_device,
            tiled=tiled,
        )
        active_future = self._resolve_infer_active_future(
            active_future=active_future,
            action_attend_mode=action_attend_mode,
            first_frame_latents=first_frames["rgb"],
            context=context,
            context_mask=context_mask,
        )
        plain_context, plain_context_mask = context, context_mask
        if active_future is not None:
            logger.info("infer_action active_future=%s", active_future)
            context, context_mask = self._infer_modality_context(
                context, context_mask, active_future, prompt=prompt, proprio=proprio, prompt_text=prompt_text
            )
        elif getattr(self, "modality_prompt", False):
            logger.warning("infer_action: modality_prompt is on but no stream was chosen; running without it.")
        gate, loop_future, update_streams = self._infer_layout(active_future, action_attend_mode)
        with gate:
            _, latents_action = self._run_multistream_infer_loop(
                latents_video=latents_video,
                latents_action=latents_action,
                first_frames=first_frames,
                fuse_flag=fuse_flag,
                context=context,
                context_mask=context_mask,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                action_condition=None,
                action_attend_mode=action_attend_mode,
                active_future=loop_future,
                update_streams=update_streams,
                action_context=self._action_side_context(plain_context, plain_context_mask)[0],
                action_context_mask=self._action_side_context(plain_context, plain_context_mask)[1],
            )
        return {
            "action": latents_action[0].detach().to(device="cpu", dtype=torch.float32),
            "action_attend_mode": action_attend_mode,
            "active_future": active_future,
        }

    @torch.no_grad()
    def infer_joint(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_video_frames: int,
        action_horizon: int,
        action: Optional[torch.Tensor] = None,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        test_action_with_infer_action: bool = False,
        compile_action_infer: bool = False,
        action_attend_mode: str = "all",
        active_future: Optional[str] = None,
        prompt_text: Optional[str] = None,
    ) -> dict[str, Any]:
        del compile_action_infer, negative_prompt, text_cfg_scale, test_action_with_infer_action
        self.eval()
        action_attend_mode = self._validate_action_attend_mode(action_attend_mode)
        if action_attend_mode != "all":
            logger.info(
                "FastWAMJointMultimodal.infer_joint action_attend_mode=%s (infer ablation)",
                action_attend_mode,
            )

        input_image, height, width = self._validate_infer_input_image(
            input_image,
            num_video_frames,
            check_fn=self._check_resize_height_width,
        )
        if action is not None:
            if action.ndim == 2:
                action = action.unsqueeze(0)
            if action.ndim != 3 or action.shape[0] != 1 or action.shape[1] != action_horizon:
                raise ValueError(
                    f"`action` must have shape [1, T, a_dim] or [T, a_dim], got {tuple(action.shape)} "
                    f"with action_horizon={action_horizon}"
                )
            action = action.to(device=self.device, dtype=self.torch_dtype)

        context, context_mask = self._resolve_infer_context(
            prompt=prompt,
            proprio=proprio,
            context=context,
            context_mask=context_mask,
        )
        latents_video, latents_action, first_frames, fuse_flag = self._init_multistream_infer_latents(
            num_video_frames=num_video_frames,
            height=height,
            width=width,
            action_horizon=action_horizon,
            input_image=input_image,
            seed=seed,
            rand_device=rand_device,
            tiled=tiled,
        )
        active_future = self._resolve_infer_active_future(
            active_future=active_future,
            action_attend_mode=action_attend_mode,
            first_frame_latents=first_frames["rgb"],
            context=context,
            context_mask=context_mask,
        )
        plain_context, plain_context_mask = context, context_mask
        if active_future is not None:
            logger.info("infer_joint active_future=%s", active_future)
            context, context_mask = self._infer_modality_context(
                context, context_mask, active_future, prompt=prompt, proprio=proprio, prompt_text=prompt_text
            )
        elif getattr(self, "modality_prompt", False):
            logger.warning("infer_joint: modality_prompt is on but no stream was chosen; running without it.")
        gate, loop_future, update_streams = self._infer_layout(active_future, action_attend_mode)
        with gate:
            latents_video, latents_action = self._run_multistream_infer_loop(
                latents_video=latents_video,
                latents_action=latents_action,
                first_frames=first_frames,
                fuse_flag=fuse_flag,
                context=context,
                context_mask=context_mask,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                action_condition=action,
                action_attend_mode=action_attend_mode,
                active_future=loop_future,
                update_streams=update_streams,
                action_context=self._action_side_context(plain_context, plain_context_mask)[0],
                action_context_mask=self._action_side_context(plain_context, plain_context_mask)[1],
            )
        # Decode only the futures that were actually denoised: with a chosen stream that is
        # just that stream (short: the others are not in the MoT; full: never stepped).
        if update_streams is not None:
            denoised = tuple(update_streams)
        elif active_future is not None:
            denoised = (active_future,)
        else:
            denoised = self._enabled_streams()
        decoded = self._decode_stream_videos(latents_video, tiled=tiled, streams=denoised)
        return {
            "video": decoded[active_future] if active_future is not None else decoded["video"],
            "video_rgb": decoded.get("rgb"),
            "video_depth": decoded.get("depth"),
            "video_flow": decoded.get("flow"),
            "action": latents_action[0].detach().to(device="cpu", dtype=torch.float32),
            "action_attend_mode": action_attend_mode,
            "active_future": active_future,
        }

    @torch.no_grad()
    def infer(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_frames: int,
        action: Optional[torch.Tensor] = None,
        action_horizon: Optional[int] = None,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 5.0,
        action_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        action_attend_mode: str = "all",
        prompt_text: Optional[str] = None,
    ) -> dict[str, Any]:
        del action_cfg_scale
        if action_horizon is None:
            raise ValueError("`action_horizon` is required for FastWAMJointMultimodal.infer.")
        return self.infer_joint(
            prompt=prompt,
            input_image=input_image,
            num_video_frames=num_frames,
            action_horizon=action_horizon,
            action=action,
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            negative_prompt=negative_prompt,
            text_cfg_scale=text_cfg_scale,
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
            seed=seed,
            rand_device=rand_device,
            tiled=tiled,
            action_attend_mode=action_attend_mode,
            prompt_text=prompt_text,
        )

    def save_checkpoint(self, path, optimizer=None, step=None):
        payload = {
            "mot": self.mot.state_dict(),
            "step": step,
            "torch_dtype": str(self.torch_dtype),
        }
        if self.proprio_encoder is not None:
            payload["proprio_encoder"] = self.proprio_encoder.state_dict()
        if getattr(self, "modality_router", None) is not None:
            payload["modality_router"] = self.modality_router.state_dict()
            payload["modality_select"] = getattr(self, "modality_select", "all")
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, path)

    def load_checkpoint(self, path, optimizer=None):
        payload = super().load_checkpoint(path, optimizer=optimizer)
        router = getattr(self, "modality_router", None)
        if router is not None:
            if "modality_router" in payload:
                router.load_state_dict(payload["modality_router"], strict=True)
            else:
                logger.warning(
                    "Checkpoint has no `modality_router` weights; keeping current router params."
                )
        elif "modality_router" in payload:
            logger.warning(
                "Checkpoint contains `modality_router` but current model has none; ignoring."
            )
        step = payload.get("step") if isinstance(payload, dict) else None
        if step is not None:
            # Resume the warm-up clock where the checkpoint left off.
            self._router_calls = int(step) * int(getattr(self, "router_grad_accum", 1))
        return payload
