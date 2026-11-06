"""Stream embedding on the future TOKENS in the ROUTER model (select_new, raw router, stream_embedding_mode: token), CPU.

  1. the task config resolves to (MM mask, stream embedding, mode token); the base model default is timestep;
  2. zeroed embedding: video / action losses == flag off, exactly (same seed) -> pure addition;
  3. token mode leaves the adaLN input unchanged (video t_mod == flag off) and changes only the
     FUTURE tokens of each stream, by exactly that stream's vector (R0 / frame 0 and the action untouched);
  4. the embedding gets a gradient; it lives in model.mot.state_dict();
  5. prints the norm of the patch-embedded future tokens vs the embedding (init scale check).

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=. python notebooks/router_analysis/tests/test_mm_stream_embedding_token.py
"""
import glob
import os

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from torch.utils.data import default_collate

from fastwam.models.wan22.fastwam_joint_multimodal import _STREAM_KEYS
from fastwam.trainer import Wan22Trainer

REPO = "/data/mgovind/FastWAM"
TASK = "libero_joint_2cam224_1e-4_router_full_all_streamemb_tok"
RUN = "runs/libero_joint_2cam224_1e-4_router_full_all/2026-10-12_22-29-10"   # raw router, no stream emb
KEYS = ("loss_video_rgb", "loss_video_depth", "loss_video_flow", "loss_action")


def main():
    os.chdir(REPO)
    with initialize_config_dir(version_base="1.3", config_dir=f"{REPO}/configs"):
        cfg = compose(config_name="train", overrides=[f"task={TASK}"])
        base = compose(config_name="train", overrides=["task=libero_joint_2cam224_1e-4_router_full_all_streamemb"]).model
    m = cfg.model
    assert (m.modality_select, m.router_input, m.stream_embedding, m.stream_embedding_mode) == ("router", "raw", True, "token"), m
    assert base.stream_embedding_mode == "timestep", base.stream_embedding_mode
    print("1. config: (router, raw, stream embedding, token); ..._router_full_all_streamemb keeps timestep  OK")

    model = instantiate(cfg.model, model_dtype=torch.float32, device="cpu")
    assert model.use_stream_embedding and model.stream_embedding_mode == "token"
    ckpt = f"{RUN}/checkpoints/weights/step_021700.pt"
    model.load_checkpoint(ckpt if os.path.isfile(ckpt) else sorted(glob.glob(f"{RUN}/checkpoints/weights/*.pt"))[-1])
    model._vae_encode_compiled = lambda v, s: model.vae.model.encode(v, s)
    Wan22Trainer._apply_dit_only_train_mode(model)
    model.dit.eval()
    model.router_warmup_steps = 10**9   # uniform picks: router choice cannot differ between the compared runs
    ds = instantiate(cfg.data.train, pretrained_norm_stats=f"{RUN}/dataset_stats.json")
    batch = default_collate([ds._get(70000), ds._get(200000)])
    emb = model.video_expert.stream_embedding

    cap = {}
    core = model.mot.forward_joint_core

    def core_wrap(**kw):
        cap["tokens"], cap["t_mod"], cap["action"] = kw["video_tokens"], kw["video_t_mod"], kw["action_tokens"]
        return core(**kw)

    def run(flag):
        model.use_stream_embedding = flag
        torch.manual_seed(0)
        model.mot.forward_joint_core = core_wrap
        try:
            with torch.no_grad():
                _, logs = model.training_loss(batch)
        finally:
            model.mot.forward_joint_core = core
        return {k: logs[k] for k in KEYS}, {k: v.clone() for k, v in cap.items()}

    with torch.no_grad():
        saved = emb.detach().clone()
        emb.zero_()
    (on0, _), (off, ref) = run(True), run(False)
    assert on0 == off, (on0, off)
    print("2. zeroed embedding: losses == flag off", {k: round(v, 6) for k, v in off.items()}, " OK")

    with torch.no_grad():
        emb.copy_(saved * 50)
    on, got = run(True)
    assert any(on[k] != off[k] for k in KEYS[:3]), "embedding has no effect"
    assert torch.equal(got["t_mod"], ref["t_mod"]), "token mode must not touch the adaLN input"
    assert torch.equal(got["action"], ref["action"])
    tpf = 98
    d = got["tokens"] - ref["tokens"]                                        # [B, S_video, D]
    streams = list(model._enabled_streams())                                 # [R0+R_f | D_f | F_f]
    assert float(d[:, :tpf].abs().max()) == 0.0, "R0 changed"
    off_ = tpf
    for name in streams:
        n = 2 * tpf                                                          # T=2 future latent frames
        exp = emb[_STREAM_KEYS.index(name)].to(d.dtype)
        err = float((d[:, off_:off_ + n] - exp).abs().max())
        assert err < 1e-4, (name, err)
        off_ += n
    print("3. token mode: adaLN t_mod and action tokens unchanged; R0 unchanged; each stream's future tokens "
          "shifted by exactly its vector  OK")

    model.use_stream_embedding = True
    model.zero_grad(set_to_none=True)
    torch.manual_seed(0)
    loss, _ = model.training_loss(batch)
    loss.backward()
    assert emb.grad is not None and float(emb.grad.abs().sum()) > 0
    keys = [k for k in model.mot.state_dict() if k.endswith("stream_embedding")]
    assert keys == ["mixtures.video.stream_embedding"], keys
    print(f"4. grad norm per stream {[round(float(x), 5) for x in emb.grad.norm(dim=1)]}; in mot.state_dict()  OK")

    fut = ref["tokens"][:, tpf:]
    print(f"5. future token norm: mean {float(fut.norm(dim=-1).mean()):.2f} (median {float(fut.norm(dim=-1).median()):.2f}); "
          f"init embedding norm {float(saved.norm(dim=1).mean()):.2f}")
    print("ALL ROUTER TOKEN STREAM-EMBEDDING CHECKS PASSED")


if __name__ == "__main__":
    main()
