"""Stream embedding in MM (FastWAMJointMultimodal, stream_embedding: true), CPU, real weights + data.

  1. the old MM checkpoint loads into the stream-embedding model (embedding = new parameter);
  2. zeroed embedding: video / action losses == flag off, exactly (same seed) -> pure addition;
  3. non-zero embedding changes the video losses (it is used) and gets a gradient;
  4. it lives in model.mot.state_dict() (saved with the checkpoint);
  5. the task configs resolve as intended (MM / cascade / streamemb / cascade+streamemb).

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=. python notebooks/router_analysis/tests/test_mm_stream_embedding.py
"""
import os

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from torch.utils.data import default_collate

from fastwam.trainer import Wan22Trainer

REPO = "/data/mgovind/FastWAM"
TASK = "libero_joint_2cam224_1e-4_rgb_depth_flow_crosscascade_streamemb"
RUN = "runs/libero_joint_2cam224_1e-4_rgb_depth_flow/2026-08-28_09-13-18"
KEYS = ("loss_video_rgb", "loss_video_depth", "loss_video_flow", "loss_action")


def main():
    os.chdir(REPO)
    with initialize_config_dir(version_base="1.3", config_dir=f"{REPO}/configs"):
        cfg = compose(config_name="train", overrides=[f"task={TASK}"])
        for t, want in [("rgb_depth_flow_crosscascade", ("cascade", False)), ("rgb_depth_flow_streamemb", ("none", True)),
                        ("rgb_depth_flow_crosscascade_streamemb", ("cascade", True))]:
            c = compose(config_name="train", overrides=[f"task=libero_joint_2cam224_1e-4_{t}"]).model
            assert (c.cross_stream_attention, c.stream_embedding) == want, (t, c.cross_stream_attention, c.stream_embedding)
    print("5. configs: crosscascade=(cascade, no emb), streamemb=(none, emb), crosscascade_streamemb=(cascade, emb)  OK")

    model = instantiate(cfg.model, model_dtype=torch.float32, device="cpu")
    assert model.use_stream_embedding and model.cross_stream_attention == "cascade"
    model.load_checkpoint(f"{RUN}/checkpoints/weights/step_021700.pt") if os.path.isfile(
        f"{RUN}/checkpoints/weights/step_021700.pt") else model.load_checkpoint(
        sorted(__import__("glob").glob(f"{RUN}/checkpoints/weights/*.pt"))[-1])
    print("1. old MM checkpoint loads into the stream-embedding model  OK")
    model._vae_encode_compiled = lambda v, s: model.vae.model.encode(v, s)
    Wan22Trainer._apply_dit_only_train_mode(model)
    model.dit.eval()
    ds = instantiate(cfg.data.train, pretrained_norm_stats=f"{RUN}/dataset_stats.json")
    batch = default_collate([ds._get(70000), ds._get(200000)])
    emb = model.video_expert.stream_embedding

    def losses(flag):
        model.use_stream_embedding = flag
        torch.manual_seed(0)
        with torch.no_grad():
            _, logs = model.training_loss(batch)
        return {k: logs[k] for k in KEYS}

    with torch.no_grad():
        saved = emb.detach().clone(); emb.zero_()
    on0, off = losses(True), losses(False)
    assert on0 == off, (on0, off)
    print("2. zeroed embedding: losses == flag off", {k: round(v, 6) for k, v in off.items()}, " OK")

    with torch.no_grad():
        emb.copy_(saved * 50)                       # clearly non-zero
    on = losses(True)
    assert any(on[k] != off[k] for k in KEYS[:3]), "embedding has no effect"
    model.use_stream_embedding = True
    model.zero_grad(set_to_none=True)
    torch.manual_seed(0)
    loss, _ = model.training_loss(batch)
    loss.backward()
    assert emb.grad is not None and float(emb.grad.abs().sum()) > 0
    print(f"3. non-zero embedding changes the video losses; grad norm per stream {[round(float(x), 5) for x in emb.grad.norm(dim=1)]}  OK")

    keys = [k for k in model.mot.state_dict() if k.endswith("stream_embedding")]
    assert keys == ["mixtures.video.stream_embedding"], keys
    print(f"4. in model.mot.state_dict() as {keys[0]!r}  OK")
    print("ALL MM STREAM-EMBEDDING CHECKS PASSED")


if __name__ == "__main__":
    main()
