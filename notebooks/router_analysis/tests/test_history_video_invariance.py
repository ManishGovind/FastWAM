"""Router history must reach the router only: the video expert's inputs and predictions are unchanged.

Same model (router_history: true, H from the _hist task), same batch, same seed, two runs:
  (a) the real history clip, (b) the history clip replaced by random noise.
The router features differ between (a) and (b), but everything that enters the video expert
(video_expert.prepare inputs: noisy latents, timesteps, context; MoT video tokens, t_mod, video
context, the video rows of the attention mask) and the video predictions / video losses must be
bit-identical. Also prints the router inputs.

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=. python notebooks/router_analysis/tests/test_history_video_invariance.py
"""
import os

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from torch.utils.data import default_collate

from fastwam.trainer import Wan22Trainer

REPO = "/data/mgovind/FastWAM"
TASK = "libero_joint_2cam224_1e-4_router_full_all_streamemb_embed_hist"
RUN = "runs/libero_joint_2cam224_1e-4_random_full_all/2026-10-12_16-35-37"


def run(model, batch):
    cap = {"prepare": [], "core": None, "feats": None}
    prepare, core, build = model.video_expert.prepare, model.mot.forward_joint_core, model._build_router_features

    def prepare_w(**kw):
        cap["prepare"].append({k: (v.detach().clone() if torch.is_tensor(v) else v) for k, v in kw.items()})
        return prepare(**kw)

    def core_w(**kw):
        out = core(**kw)
        nv = kw["video_tokens"].shape[1]
        cap["core"] = {"video_tokens": kw["video_tokens"].clone(), "video_t_mod": kw["video_t_mod"].clone(),
                       "video_context": kw["video_context"].clone(), "video_context_mask": kw["video_context_mask"].clone(),
                       "mask_video_rows": kw["attention_mask"][..., :nv, :].clone(), "video_out": out[0].clone()}
        return out

    def build_w(**kw):
        cap["feats"] = build(**kw)
        return cap["feats"]

    model.video_expert.prepare, model.mot.forward_joint_core, model._build_router_features = prepare_w, core_w, build_w
    try:
        torch.manual_seed(0)
        with torch.no_grad():
            _, logs = model.training_loss(batch)
    finally:
        model.video_expert.prepare, model.mot.forward_joint_core, model._build_router_features = prepare, core, build
    return cap, logs


def main():
    os.chdir(REPO)
    with initialize_config_dir(version_base="1.3", config_dir=f"{REPO}/configs"):
        cfg = compose(config_name="train", overrides=[f"task={TASK}"])
    model = instantiate(cfg.model, model_dtype=torch.float32, device="cpu")
    model.load_checkpoint(f"{RUN}/checkpoints/weights/step_021700.pt")
    model._vae_encode_compiled = lambda v, s: model.vae.model.encode(v, s)
    Wan22Trainer._apply_dit_only_train_mode(model)
    model.dit.eval()
    model.router_warmup_steps = 0
    ds = instantiate(cfg.data.train, pretrained_norm_stats=f"{RUN}/dataset_stats.json")
    batch = default_collate([ds._get(70000), ds._get(200000)])
    fake = dict(batch)
    g = torch.Generator().manual_seed(1)
    fake["video_rgb_history"] = torch.rand(batch["video_rgb_history"].shape, generator=g) * 2 - 1

    a, la = run(model, batch)
    b, lb = run(model, fake)

    f = a["feats"]
    print("router inputs (detached, from the video expert's own embeddings):")
    for k, v in f.items():
        print(f"  {k:10s} {tuple(v.shape)}")
    d_hist = float((a["feats"]["hist"] - b["feats"]["hist"]).abs().max())
    assert d_hist > 0.1, "history features must differ between the two runs"
    print(f"router 'hist' features differ between real and noise history (max |diff| {d_hist:.2f})")

    assert len(a["prepare"]) == len(b["prepare"]) == 3          # rgb, depth, flow
    for i, (pa, pb) in enumerate(zip(a["prepare"], b["prepare"])):
        for k in pa:
            if torch.is_tensor(pa[k]):
                assert torch.equal(pa[k], pb[k]), (i, k)
    for k in a["core"]:
        assert torch.equal(a["core"][k], b["core"][k]), k
    print("video expert inputs identical: prepare(x, timestep, context, mask) x3 streams; MoT video tokens, "
          "t_mod, video context, video rows of the attention mask")
    print("video expert outputs identical: MoT video tokens out")
    for k in ("loss_video_rgb", "loss_video_depth", "loss_video_flow"):
        assert la[k] == lb[k], (k, la[k], lb[k])
    print("video losses identical:", {k: round(la[k], 6) for k in ("loss_video_rgb", "loss_video_depth", "loss_video_flow")})
    print(f"(action loss may differ through the router's pick: {la['loss_action']:.5f} vs {lb['loss_action']:.5f})")
    print("HISTORY -> ROUTER ONLY: VIDEO EXPERT UNCHANGED  OK")


if __name__ == "__main__":
    main()
