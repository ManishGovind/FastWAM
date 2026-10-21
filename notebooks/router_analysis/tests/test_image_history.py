"""RGB observation history in RobotMultimodalVideoDataset, CPU, real LIBERO data.

Run for H = the _hist task config value (4) and H = 8 (two history latents):

  1. off by default: no history keys; with history on, video_rgb / depth / flow / action / proprio
     are identical to the history-off sample (history does not touch the training window);
  2. video_rgb_history is [3, H+1, 224, 448] (lead-in + H past frames, no R0);
  3. frame spacing: history frame j (j = 0..H) == R0 of the window starting (H+1-j)*4 frames earlier
     (same episode); so the last history frame is t-4;
  4. episode start: padded history frames repeat the episode's first frame, history_is_pad marks them.

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=. python notebooks/router_analysis/tests/test_image_history.py
"""
import os

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

REPO = "/data/mgovind/FastWAM"
RUN = "runs/libero_joint_2cam224_1e-4_router_full_all_streamemb_embed/2026-10-16_06-33-56"
HIST_TASK = "libero_joint_2cam224_1e-4_router_full_all_streamemb_embed_hist"
S = 4


def main():
    os.chdir(REPO)
    with initialize_config_dir(version_base="1.3", config_dir=f"{REPO}/configs"):
        cfg = compose(config_name="train", overrides=["task=libero_joint_2cam224_1e-4_router_full_all_streamemb_embed"])
        h_cfg = int(compose(config_name="train", overrides=[f"task={HIST_TASK}"]).data.train.image_history_frames)
    stats = f"{RUN}/dataset_stats.json"
    off = instantiate(cfg.data.train, pretrained_norm_stats=stats)
    for H in sorted({h_cfg, 8}):
        print(f"--- H = {H}" + (" (task config)" if H == h_cfg else ""))
        check(off, instantiate(cfg.data.train, pretrained_norm_stats=stats, image_history_frames=H), H)
    print("ALL IMAGE-HISTORY DATASET CHECKS PASSED")


def check(off, on, H):
    ep = on.lerobot_dataset.episode_data_index
    start, end = int(ep["from"][3]), int(ep["to"][3])        # one full episode
    mid = start + 60
    assert end - start > 100

    a, b = off._get(mid), on._get(mid)
    assert "video_rgb_history" not in a
    for k in ("video_rgb", "video_depth", "video_flow", "action", "proprio"):
        assert torch.equal(a[k], b[k]), k
    print("1. history off by default; history on leaves the training window unchanged  OK")

    hist = b["video_rgb_history"]
    assert hist.shape == (3, H + 1, 224, 448), hist.shape
    assert not torch.equal(hist[:, -1], b["video_rgb"][:, 0])     # R0 is not in the history
    assert not b["history_is_pad"].any()
    print(f"2. video_rgb_history {tuple(hist.shape)} (lead-in + {H} past frames, no R0)  OK")

    for j in range(H + 1):
        prev = on._get(mid - (H + 1 - j) * S)["video_rgb"][:, 0]
        d = float((hist[:, j] - prev).abs().max())
        assert d < 1e-6, (j, d)
    print(f"3. history frame j == R0 of the window (H+1-j)*{S} frames earlier: t-{(H + 1) * S} .. t-{S}  OK")

    c = on._get(start + 5)                                   # 5 frames into the episode
    first = on._get(start)["video_rgb"][:, 0]
    n_pad = int(c["history_is_pad"].sum())
    assert n_pad == H, n_pad                                  # only t-4 (frame 1) is real
    for j in range(n_pad):
        assert float((c["video_rgb_history"][:, j] - first).abs().max()) < 1e-6
    print(f"4. episode start: {n_pad} padded frames == first frame, history_is_pad marks them  OK")


if __name__ == "__main__":
    main()
