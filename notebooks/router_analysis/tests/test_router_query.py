"""Query router (``router_arch: transformer_query`` on ``router_input: embed`` in the _new model), CPU.

  1. unit: logits [B, 3], uniform at init (zero-init head); all 128 text tokens are attended (no
     padding mask, as the video expert); permuting the modality queries permutes the logits the same way (one shared head);
     every parameter gets a gradient once the head is non-zero;
  2. real model: the task config builds QueryModalityRouter with 98 R0 tokens; one real
     training_loss step runs; features are detached (no router grad on patch_embedding,
     text_embedding, proprio_encoder); inference pick is the same for router_layout full / short.

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=. python notebooks/router_analysis/tests/test_router_query.py
"""

import os

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from torch.utils.data import default_collate

from fastwam.models.wan22.modality_router_new import QueryModalityRouter
from fastwam.trainer import Wan22Trainer

REPO = "/data/mgovind/FastWAM"
TASK = "libero_joint_2cam224_1e-4_router_full_all_streamemb_embed_transformer_query"
RUN = "runs/libero_joint_2cam224_1e-4_random_full_all/2026-10-12_16-35-37"   # no router weights


def unit():
    torch.manual_seed(0)
    r = QueryModalityRouter(dim=64, num_classes=3, num_r0_tokens=98, d=32)
    feats = {"r0": torch.randn(2, 98, 64), "text": torch.randn(2, 128, 64),
             "text_mask": torch.ones(2, 128, dtype=torch.bool), "proprio": torch.randn(2, 64)}
    out = r(feats)
    assert out.shape == (2, 3) and torch.equal(out, torch.zeros(2, 3)), out
    torch.nn.init.normal_(r.head[-1].weight)
    r.eval()
    a = r(feats)
    feats2 = dict(feats, text=feats["text"].clone())
    feats2["text"][:, 20:] = torch.randn(2, 108, 64) * 100
    d_pad = float((r(feats2) - a).abs().max())
    assert d_pad > 1e-3, d_pad   # every text token is attended (no padding mask, as the video expert)
    perm = torch.tensor([2, 0, 1])
    with torch.no_grad():
        r.queries.copy_(r.queries[:, perm])
    d_perm = float((r(feats) - a[:, perm]).abs().max())
    assert d_perm < 1e-5, d_perm
    r.train()
    r(feats).logsumexp(-1).sum().backward()
    missing = [n for n, p in r.named_parameters() if p.grad is None or float(p.grad.abs().sum()) == 0]
    assert not missing, missing
    print(f"1. unit: logits (2, 3), uniform at init; all 128 text tokens attended (trailing-token change -> diff {d_pad:.1e}); permuted queries "
          f"-> permuted logits (max diff {d_perm:.1e}); all parameters get grad  OK")


def main():
    os.chdir(REPO)
    unit()
    with initialize_config_dir(version_base="1.3", config_dir=f"{REPO}/configs"):
        cfg = compose(config_name="train", overrides=[f"task={TASK}"])
    model = instantiate(cfg.model, model_dtype=torch.float32, device="cpu")
    model.load_checkpoint(f"{RUN}/checkpoints/weights/step_021700.pt")
    model._vae_encode_compiled = lambda v, s: model.vae.model.encode(v, s)
    Wan22Trainer._apply_dit_only_train_mode(model)
    router = model.modality_router
    assert model.router_arch == "transformer_query" and isinstance(router, QueryModalityRouter)
    assert router.r0_pos.shape[1] == 98 and router.queries.shape[1] == 3
    model.router_warmup_steps = 0
    ds = instantiate(cfg.data.train, pretrained_norm_stats=f"{RUN}/dataset_stats.json")
    batch = default_collate([ds[0], ds[150000]])

    cap = {}
    build_feats = model._build_router_features

    def build_wrap(**kw):
        cap["in"] = kw
        return build_feats(**kw)

    model._build_router_features = build_wrap
    torch.manual_seed(0)
    loss, _ = model.training_loss(batch)
    model._build_router_features = build_feats
    print(f"2a. training_loss step OK (loss {float(loss):.4f})")

    model.zero_grad(set_to_none=True)
    torch.nn.init.normal_(router.head[-1].weight)   # zero-init head blocks grads to earlier layers
    feats = model._build_router_features(**cap["in"])
    assert not any(v.requires_grad for v in feats.values() if torch.is_tensor(v))
    router(feats).logsumexp(-1).sum().backward()
    ve = model.video_expert
    for name, p in (("patch_embedding", ve.patch_embedding.weight), ("text_embedding", ve.text_embedding[0].weight),
                    ("proprio_encoder", model.proprio_encoder.weight)):
        assert p.grad is None, name
    assert router.queries.grad is not None and router.r0_proj[1].weight.grad is not None
    print("2b. features detached: no grad on patch/text embedding or proprio_encoder; router gets grad  OK")

    model.eval()
    kw = dict(first_frame_latents=cap["in"]["r0_latent"][:1], context=cap["in"]["context"][:1].detach(),
              context_mask=cap["in"]["context_mask"][:1])
    picks = {}
    for lay in ("full", "short"):
        model.router_layout = lay
        with torch.no_grad():
            picks[lay] = model._resolve_router_active_future(**kw)
    assert picks["full"] == picks["short"], picks
    n = sum(p.numel() for p in router.parameters())
    print(f"2c. inference pick full == short ({picks['full']}); router params {n / 1e6:.2f}M  OK")
    print("ALL ROUTER-QUERY CHECKS PASSED")


if __name__ == "__main__":
    main()
