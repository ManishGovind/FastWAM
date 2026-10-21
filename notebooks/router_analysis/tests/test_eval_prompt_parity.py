"""Eval vs training context of the joint modality prompt (CPU, no model weights).

Training builds the video-expert context with ``_modality_prompt_context`` from the cached
T5("<instruction> Predict the X video."): raw T5 tokens, padding MASKED OUT, then proprio.
Eval (``_infer_modality_context``) must give the identical tensors. It used to re-encode with
``encode_prompt``, which zeroes the padding and sets an all-ones mask (padding visible).

    python notebooks/router_analysis/tests/test_eval_prompt_parity.py
"""

import os

import torch
import torch.nn as nn

from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from fastwam.models.wan22.fastwam_joint_multimodal_select_new import FastWAMJointMultimodalSelectNew

os.chdir("/data/mgovind/FastWAM")
SENT = {"rgb": "Predict the RGB video.", "depth": "Predict the depth video.", "flow": "Predict the flow video."}


def stub():
    m = object.__new__(FastWAMJointMultimodalSelectNew)
    nn.Module.__init__(m)
    m.modality_prompt = True
    m.modality_prompt_mode = "joint"
    m.modality_prompt_target = "video"
    m.modality_prompt_sentences = dict(SENT)
    m._joint_prompt_cache = {}
    m._modality_prompt_cache_dir = "./data/text_embeds_cache/libero_modality"
    m._modality_prompt_context_len = 128
    m._modality_prompt_enc_id = "wan22ti2v5b"
    m.proprio_encoder = nn.Identity()  # only checked for "is not None" (proprio token is last)
    m.text_encoder = None
    m.tokenizer = None
    return m


def main():
    m = stub()
    prompt = DEFAULT_PROMPT.format(task="pick up the black bowl between the plate and the ramekin and place it on the plate")
    # Plain eval context as `encode_prompt` + proprio makes it: zeroed padding, all-ones mask.
    plain = torch.zeros(1, 129, 4096, dtype=torch.bfloat16)
    plain[0, :20] = torch.randn(20, 4096).to(torch.bfloat16)
    plain[0, 128] = torch.randn(4096).to(torch.bfloat16)  # proprio token
    plain_mask = torch.ones(1, 129, dtype=torch.bool)

    for stream in SENT:
        ev_ctx, ev_mask = m._infer_modality_context(plain, plain_mask, stream, prompt=prompt, proprio=None)
        tr_ctx, tr_mask = m._modality_prompt_context(plain, plain_mask, [stream], prompts=[prompt])
        cached, cmask = m._load_joint_prompt(f"{prompt} {SENT[stream]}")
        assert torch.equal(ev_ctx, tr_ctx) and torch.equal(ev_mask, tr_mask), stream
        assert ev_ctx.shape == (1, 129, 4096), ev_ctx.shape
        assert torch.equal(ev_ctx[0, :128], cached.to(ev_ctx.dtype)), stream  # exact cached T5 tokens
        assert torch.equal(ev_ctx[0, 128], plain[0, 128]), stream  # proprio kept last
        n_real = int(cmask.sum())
        assert ev_mask[0, :128].sum() == n_real and not ev_mask[0, n_real:128].any(), "padding must be masked"
        assert bool(ev_mask[0, 128]), "proprio visible"
        print(f"{stream:5s}: eval == training context; {n_real} real tokens, {128 - n_real} padding masked, proprio last")
    print("OK")


if __name__ == "__main__":
    main()
