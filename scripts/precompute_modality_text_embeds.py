"""Precompute T5 embeddings for the modality conditions of the ``_new`` select models.

Writes into ``model.modality_prompt_cache_dir`` (its own folder, default
``./data/text_embeds_cache/libero_modality``), with the same file naming and prompt
template as ``scripts/precompute_text_embeds.py``:

* joint mode: every dataset instruction + each condition, i.e.
  ``DEFAULT_PROMPT.format(task=task) + " " + condition``
  (e.g. "... instruction: put the bowl on the plate Predict the depth video.");
* separate mode: each condition alone, ``DEFAULT_PROMPT.format(task=condition)``.

Conditions come from ``model.modality_prompt_sentences`` (rgb / depth / flow).
Existing cache files are never overwritten.

Usage::

    python scripts/precompute_modality_text_embeds.py task=libero_joint_2cam224_1e-4_router_new
"""

import hashlib
import logging
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from fastwam.models.wan22.helpers.loader import _load_registered_model, _resolve_configs
from fastwam.models.wan22.wan_video_text_encoder import HuggingfaceTokenizer
from fastwam.utils.config_resolvers import register_default_resolvers
from fastwam.utils.logging_config import get_logger, setup_logging

from precompute_text_embeds import (  # noqa: E402  (same folder)
    DEFAULT_BATCH_SIZE,
    DEFAULT_MODEL_ID,
    DEFAULT_TOKENIZER_MODEL_ID,
    _atomic_torch_save,
    _collect_dataset_settings,
    _model_id_to_enc_id,
    _read_unique_prompts,
    _resolve_context_len,
)

register_default_resolvers()
logger = get_logger(__name__)
STREAMS = ("rgb", "depth", "flow")


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    setup_logging(log_level=logging.INFO)
    model_cfg = cfg.model
    sentences = OmegaConf.to_container(model_cfg.get("modality_prompt_sentences"), resolve=True)
    if not sentences or any(s not in sentences for s in STREAMS):
        raise ValueError(f"`model.modality_prompt_sentences` must define {STREAMS}, got {sentences}")

    dataset_dirs, _instruction_cache_dirs, context_lens = _collect_dataset_settings(cfg.data)
    context_len = _resolve_context_len(context_lens)
    out_dir = Path(str(model_cfg.get("modality_prompt_cache_dir", "./data/text_embeds_cache/libero_modality"))).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dirs = [out_dir]
    logger.info("writing modality-condition caches to %s", out_dir)
    prompts = _read_unique_prompts(dataset_dirs)
    texts = [f"{p} {sentences[s]}" for p in prompts for s in STREAMS]  # joint
    texts += [DEFAULT_PROMPT.format(task=sentences[s]) for s in STREAMS]  # separate
    logger.info("%d instructions x %d conditions + %d standalone = %d texts", len(prompts), len(STREAMS), len(STREAMS), len(texts))

    model_id = str(model_cfg.get("model_id", DEFAULT_MODEL_ID))
    enc_id = _model_id_to_enc_id(model_id)
    todo = []
    for t in texts:
        name = f"{hashlib.sha256(t.encode('utf-8')).hexdigest()}.t5_len{context_len}.{enc_id}.pt"
        if not all((d / name).exists() for d in cache_dirs):
            todo.append(t)
    logger.info("%d already cached, %d to encode", len(texts) - len(todo), len(todo))
    if not todo:
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, text_config, _, tokenizer_config = _resolve_configs(
        model_id=model_id,
        tokenizer_model_id=str(model_cfg.get("tokenizer_model_id", DEFAULT_TOKENIZER_MODEL_ID)),
        redirect_common_files=bool(model_cfg.get("redirect_common_files", True)),
    )
    text_config.download_if_necessary()
    tokenizer_config.download_if_necessary()
    text_encoder = _load_registered_model(
        text_config.path, "wan_video_text_encoder", torch_dtype=torch.bfloat16, device=device
    ).eval()
    tokenizer = HuggingfaceTokenizer(name=tokenizer_config.path, seq_len=context_len, clean="whitespace")

    new = over_length = 0
    with torch.no_grad():
        for start in range(0, len(todo), DEFAULT_BATCH_SIZE):
            batch = todo[start : start + DEFAULT_BATCH_SIZE]
            ids, mask = tokenizer(batch, return_mask=True, add_special_tokens=True)
            ids, mask = ids.to(device), mask.to(device=device, dtype=torch.bool)
            over_length += int(mask.all(dim=1).sum().item())
            context = text_encoder(ids, mask)
            for i, text in enumerate(batch):
                payload = {
                    "context": context[i].detach().to(device="cpu", dtype=torch.bfloat16).contiguous(),
                    "mask": mask[i].detach().to(device="cpu", dtype=torch.bool).contiguous(),
                }
                name = f"{hashlib.sha256(text.encode('utf-8')).hexdigest()}.t5_len{context_len}.{enc_id}.pt"
                for d in cache_dirs:
                    if not (d / name).exists():
                        _atomic_torch_save(payload, d / name)
                        new += 1
    logger.info("wrote %d new cache files (%d texts hit the %d-token limit)", new, over_length, context_len)


if __name__ == "__main__":
    main()
