"""Stream embedding (``stream_embedding: true`` in the _new model), CPU, real weights.

  1. zeroed embedding: training loss with the flag on == flag off, exactly (same seed);
  2. the embedding is inside model.mot (trained by the trainer's freeze logic, saved in
     checkpoints) and gets a non-zero gradient;
  3. with a NON-zero embedding, full and short inference layouts give the same actions (fp32);
  4. the embedding is in model.mot.state_dict() (what save_checkpoint writes).

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=. python notebooks/router_analysis/tests/test_stream_embedding.py
"""

import os

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from torch.utils.data import default_collate

from fastwam.trainer import Wan22Trainer

REPO = "/data/mgovind/FastWAM"
TASK = "libero_joint_2cam224_1e-4_router_full_all_streamemb"
RUN = "runs/libero_joint_2cam224_1e-4_router_full_all/2026-10-12_22-29-10"


def main():
    os.chdir(REPO)
    with initialize_config_dir(version_base="1.3", config_dir=f"{REPO}/configs"):
        cfg = compose(config_name="train", overrides=[f"task={TASK}"])
    model = instantiate(cfg.model, model_dtype=torch.float32, device="cpu")
    model.load_checkpoint(f"{RUN}/checkpoints/weights/step_021700.pt")   # old ckpt: no embedding in it
    model._vae_encode_compiled = lambda v, s: model.vae.model.encode(v, s)
    Wan22Trainer._apply_dit_only_train_mode(model)
    emb = model.video_expert.stream_embedding
    assert model.use_stream_embedding
    with torch.no_grad():   # init is randn*0.02; zero it to check the no-op property
        emb.zero_()
    ds = instantiate(cfg.data.train, pretrained_norm_stats=f"{RUN}/dataset_stats.json")
    batch = default_collate([ds[0], ds[150000]])

    # 1. zero-init == flag off
    # (compare the video / action losses: the router's REINFORCE term has state across calls,
    #  e.g. the timestep_ema baseline, so the total loss differs between two calls anyway)
    keys = ("loss_video", "loss_video_rgb", "loss_video_depth", "loss_video_flow", "loss_action")
    torch.manual_seed(0); _, on = model.training_loss(batch)
    model.use_stream_embedding = False
    torch.manual_seed(0); _, off = model.training_loss(batch)
    model.use_stream_embedding = True
    assert all(on[k] == off[k] for k in keys), {k: (on[k], off[k]) for k in keys}
    print("1. zero-init: video/action losses flag-on == flag-off", {k: round(on[k], 6) for k in keys}, " OK")

    # 2. inside model.mot, trainable, gets gradient
    assert any(p is emb for p in model.mot.parameters()), "stream_embedding must live inside model.mot"
    assert emb.requires_grad
    model.zero_grad(set_to_none=True)
    torch.manual_seed(0); loss, _ = model.training_loss(batch); loss.backward()
    g = emb.grad
    assert g is not None and float(g.abs().sum()) > 0
    print(f"2. in model.mot, requires_grad, grad norm per stream {[round(float(x), 6) for x in g.norm(dim=1)]}  OK")

    # 3. non-zero embedding: full == short layout (fp32)
    with torch.no_grad():
        emb.copy_(torch.randn_like(emb) * 0.5)
    s = ds[150000]
    kw = dict(prompt=None, input_image=s["video_rgb"][:, 0], action_horizon=int(s["action"].shape[0]),
              num_video_frames=int(s["video_rgb"].shape[1]), proprio=s["proprio"][0], context=s["context"],
              context_mask=s["context_mask"], num_inference_steps=2, seed=42)
    model.eval()
    for st in ("rgb", "depth", "flow"):
        out = {}
        for lay in ("full", "short"):
            model.router_layout = lay
            with torch.no_grad():
                out[lay] = model.infer_action(active_future=st, **kw)["action"]
        d = float((out["full"] - out["short"]).abs().max())
        assert d < 1e-4, (st, d)
        print(f"3. non-zero embedding, {st:5s}: max |a_full - a_short| = {d:.2e}  OK")

    # 4. saved with the checkpoint: save_checkpoint stores model.mot.state_dict()
    keys = [k for k in model.mot.state_dict() if k.endswith("stream_embedding")]
    assert len(keys) == 1, keys
    assert torch.equal(model.mot.state_dict()[keys[0]], emb.detach())
    print(f"4. in model.mot.state_dict() as {keys[0]!r} (saved / loaded with the checkpoint)  OK")
    print("ALL STREAM-EMBEDDING CHECKS PASSED")


if __name__ == "__main__":
    main()
