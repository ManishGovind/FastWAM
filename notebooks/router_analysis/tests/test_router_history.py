"""Router with RGB observation history (router_history: true, embed router), CPU, real weights + data.

  1. training: the router's history tokens == video_expert.patch_embedding of the VAE latents of
     video_rgb_history except the first (H/4 latent frames); shape [B, 98*H/4, 3072]; pooling weights sum to 1;
  2. detached: the router loss reaches the router (incl. q_hist) only, not the expert / proprio;
  3. inference: infer_action(history_images=<the same clip>) gives the router exactly the training
     history latent and features (same pick as the router on the training features);
  4. eval buffer (RouterHistoryBuffer): frames t-4(H+1) (lead-in), .., t-4 at stride 4 (no R0), first frame repeated
     before the episode has 4H steps (same rule as the dataset episode-start clamp).

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=. python notebooks/router_analysis/tests/test_router_history.py
"""
import os
import types

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from torch.utils.data import default_collate

from experiments.libero.eval_infer import RouterHistoryBuffer
from fastwam.trainer import Wan22Trainer

REPO = "/data/mgovind/FastWAM"
TASK = "libero_joint_2cam224_1e-4_router_full_all_streamemb_embed_hist"
RUN = "runs/libero_joint_2cam224_1e-4_random_full_all/2026-10-12_16-35-37"   # no router weights


def main():
    os.chdir(REPO)
    with initialize_config_dir(version_base="1.3", config_dir=f"{REPO}/configs"):
        cfg = compose(config_name="train", overrides=[f"task={TASK}"])

    # 4. buffer indexing (cheap, first)
    H = int(cfg.data.train.image_history_frames)
    buf = RouterHistoryBuffer(types.SimpleNamespace(router_history=True), cfg, lambda t: torch.full((1, 1, 1), float(t)))
    assert (buf.n, buf.s) == (H, 4)
    for t in range(41):
        buf.push(t)
    assert buf.clip().flatten().tolist() == [40 - 4 * k for k in range(H + 1, 0, -1)]   # lead-in .. t-4, no t
    buf.frames = []
    for t in range(6):
        buf.push(t)
    assert buf.clip().flatten().tolist() == [max(0, 5 - 4 * k) for k in range(H + 1, 0, -1)]
    print(f"4. eval buffer: stride 4, {H + 1} frames t-{4 * (H + 1)} .. t-4 (no R0), first frame repeated at the episode start  OK")

    model = instantiate(cfg.model, model_dtype=torch.float32, device="cpu")
    model.load_checkpoint(f"{RUN}/checkpoints/weights/step_021700.pt")
    model._vae_encode_compiled = lambda v, s: model.vae.model.encode(v, s)
    Wan22Trainer._apply_dit_only_train_mode(model)
    model.router_warmup_steps = 0
    assert model.router_history and model.modality_router.history
    ds = instantiate(cfg.data.train, pretrained_norm_stats=f"{RUN}/dataset_stats.json")
    batch = default_collate([ds._get(70000), ds._get(200000)])
    assert batch["video_rgb_history"].shape == (2, 3, H + 1, 224, 448)
    assert model.router_history_frames == H

    # 1. training features
    cap = {}
    build = model._build_router_features

    def wrap(**kw):
        cap["in"], cap["feats"] = kw, build(**kw)
        return cap["feats"]

    model._build_router_features = wrap
    torch.manual_seed(0)
    model.training_loss(batch)
    model._build_router_features = build
    hist = cap["feats"]["hist"]
    ve = model.video_expert
    with torch.no_grad():
        lat = model._encode_video_latents(batch["video_rgb_history"].float(), tiled=False)[:, :, 1:]
        ref = ve.patch_embedding(lat).flatten(2).transpose(1, 2)
    assert lat.shape[2] == H // 4
    assert hist.shape == (2, 98 * (H // 4), 3072), hist.shape
    assert float((hist - ref).abs().max()) < 1e-5
    w = model.modality_router.last_attn_hist
    assert w.shape == (2, 98 * (H // 4)) and torch.allclose(w.sum(1), torch.ones(2))
    print(f"1. H={H}: history tokens {tuple(hist.shape)} == patch_embedding(VAE(history)[:, :, 1:]); pooling sums to 1  OK")

    # 2. detached
    model.zero_grad(set_to_none=True)
    model._router_history_latent = lat
    feats = model._build_router_features(**cap["in"])
    assert not any(v.requires_grad for v in feats.values())
    model.modality_router(feats).logsumexp(-1).sum().backward()
    for name, p in (("patch_embedding", ve.patch_embedding.weight), ("text_embedding", ve.text_embedding[0].weight),
                    ("proprio_encoder", model.proprio_encoder.weight)):
        assert p.grad is None, name
    assert model.modality_router.q_hist.grad is not None and float(model.modality_router.q_hist.grad.abs().sum()) > 0
    print("2. router loss reaches the router (q_hist too), not the expert or proprio_encoder  OK")

    # 3. inference path gets the same history latent / features
    model.eval()
    s = ds._get(70000)
    seen = {}
    build = model._build_router_features

    def wrap2(**kw):
        seen["feats"] = build(**kw)
        return seen["feats"]

    model._build_router_features = wrap2
    with torch.no_grad():
        out = model.infer_action(prompt=None, input_image=s["video_rgb"][:, 0], action_horizon=32,
                                 num_video_frames=int(s["video_rgb"].shape[1]), proprio=s["proprio"][0],
                                 context=s["context"], context_mask=s["context_mask"], num_inference_steps=1,
                                 seed=0, history_images=s["video_rgb_history"])
    model._build_router_features = build
    d = float((seen["feats"]["hist"][0] - hist[0]).abs().max())
    assert d < 1e-4, d
    with torch.no_grad():
        train_pick = model.modality_router({k: v[:1] for k, v in cap["feats"].items()}).argmax(-1).item()
    assert out["active_future"] == model._enabled_streams()[train_pick], (out["active_future"], train_pick)
    print(f"3. infer_action(history_images=clip): same history features (max diff {d:.1e}) and pick ({out['active_future']})  OK")
    print("ALL ROUTER-HISTORY CHECKS PASSED")


if __name__ == "__main__":
    main()
