"""Router on the video expert's embeddings (``router_input: embed`` in the _new model), CPU, real weights.

  1. in a real training_loss call, the router's R0 tokens == the rgb stream's video_tokens[:, :tpf]
     and its [text | proprio] tokens == context_video (exactly what the video expert sees);
  2. features are detached: the router loss reaches the router only (no grad on patch_embedding,
     text_embedding, proprio_encoder);
  3. R0 attention-pool weights are [B, 98] and sum to 1; inference pick is the same for
     router_layout full / short;
  4. the router is saved in the checkpoint payload; router_input=raw still loads the old
     router_full_all checkpoint (strict router load).

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=. python notebooks/router_analysis/tests/test_router_embed.py
"""

import gc
import os

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from torch.utils.data import default_collate

from fastwam.models.wan22.modality_router_new import EmbedModalityRouter, ModalityRouter
from fastwam.trainer import Wan22Trainer

REPO = "/data/mgovind/FastWAM"
TASK = "libero_joint_2cam224_1e-4_router_full_all_streamemb_embed"
RUN = "runs/libero_joint_2cam224_1e-4_random_full_all/2026-10-12_16-35-37"   # no router weights
OLD_ROUTER_TASK = "libero_joint_2cam224_1e-4_router_full_all"
OLD_ROUTER_RUN = "runs/libero_joint_2cam224_1e-4_router_full_all/2026-10-12_22-29-10"


def build(task, run):
    with initialize_config_dir(version_base="1.3", config_dir=f"{REPO}/configs"):
        cfg = compose(config_name="train", overrides=[f"task={task}"])
    model = instantiate(cfg.model, model_dtype=torch.float32, device="cpu")
    model.load_checkpoint(f"{run}/checkpoints/weights/step_021700.pt")
    model._vae_encode_compiled = lambda v, s: model.vae.model.encode(v, s)
    Wan22Trainer._apply_dit_only_train_mode(model)
    return cfg, model


def main():
    os.chdir(REPO)
    cfg, model = build(TASK, RUN)
    assert model.router_input == "embed" and isinstance(model.modality_router, EmbedModalityRouter)
    model.router_warmup_steps = 0
    ds = instantiate(cfg.data.train, pretrained_norm_stats=f"{RUN}/dataset_stats.json")
    batch = default_collate([ds[0], ds[150000]])

    # 1. capture router features and the rgb stream's prepare() outputs in one real training step
    cap = {}
    build_feats, prepare = model._build_router_features, model.video_expert.prepare

    def build_wrap(**kw):
        cap["in"] = kw
        cap["feats"] = build_feats(**kw)
        return cap["feats"]

    def prepare_wrap(**kw):
        out = prepare(**kw)
        cap.setdefault("prepare", out)   # first call = rgb stream
        return out

    model._build_router_features, model.video_expert.prepare = build_wrap, prepare_wrap
    torch.manual_seed(0)
    model.training_loss(batch)
    model._build_router_features, model.video_expert.prepare = build_feats, prepare
    f = cap["feats"]
    tokens, _, _, ctx_video, _, _, _, _, _, tpf = cap["prepare"]
    assert f["r0"].shape == (2, tpf, model.video_expert.hidden_dim), f["r0"].shape
    d_r0 = float((f["r0"] - tokens[:, :tpf]).abs().max())
    d_ctx = float((torch.cat([f["text"], f["proprio"][:, None]], 1) - ctx_video).abs().max())
    assert d_r0 == 0.0 and d_ctx == 0.0, (d_r0, d_ctx)
    print(f"1. router r0 {tuple(f['r0'].shape)} == rgb video_tokens[:, :{tpf}], "
          f"text+proprio {tuple(ctx_video.shape)} == context_video (max diff {d_r0}, {d_ctx})  OK")

    # 2. detached: router-only gradient
    model.zero_grad(set_to_none=True)
    feats = model._build_router_features(**cap["in"])
    assert cap["in"]["context"].requires_grad, "context should carry the proprio_encoder graph"
    assert not any(v.requires_grad for v in feats.values())
    model.modality_router(feats).logsumexp(-1).sum().backward()
    ve = model.video_expert
    for name, p in (("patch_embedding", ve.patch_embedding.weight), ("text_embedding", ve.text_embedding[0].weight),
                    ("proprio_encoder", model.proprio_encoder.weight)):
        assert p.grad is None, name
    g = model.modality_router.q.grad
    assert g is not None and float(g.abs().sum()) > 0
    print("2. features detached: no grad on patch/text embedding or proprio_encoder; router gets grad  OK")

    # 3. pooling weights, inference pick full == short
    w = model.modality_router.last_attn
    assert w.shape == (2, tpf) and torch.allclose(w.sum(1), torch.ones(2))
    model.eval()
    kw = dict(first_frame_latents=cap["in"]["r0_latent"][:1], context=cap["in"]["context"][:1].detach(),
              context_mask=cap["in"]["context_mask"][:1])
    picks = {}
    for lay in ("full", "short"):
        model.router_layout = lay
        with torch.no_grad():
            picks[lay] = model._resolve_router_active_future(**kw)
    assert picks["full"] == picks["short"], picks
    print(f"3. attention weights {tuple(w.shape)} sum to 1; inference pick full == short ({picks['full']})  OK")

    # 4. checkpoint: router state saved; raw router still loads old checkpoint
    keys = set(model.modality_router.state_dict())
    assert {"q", "r0_proj.1.weight", "text_proj.1.weight", "proprio_proj.1.weight"} <= keys, keys
    del model, ds, cap, f, feats
    gc.collect()
    _, old = build(OLD_ROUTER_TASK, OLD_ROUTER_RUN)
    assert old.router_input == "raw" and type(old.modality_router) is ModalityRouter
    print("4. embed router state_dict has q / r0 / text / proprio projections; "
          "router_input=raw loads the old router_full_all checkpoint (strict)  OK")
    print("ALL ROUTER-EMBED CHECKS PASSED")


if __name__ == "__main__":
    main()
