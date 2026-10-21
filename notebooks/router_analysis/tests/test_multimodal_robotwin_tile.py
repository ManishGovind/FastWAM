"""RobotMultimodalVideoDataset with concat_multi_camera='robotwin' (3 cameras per stream), CPU.

On the local R1-Lite copy (RGB + precomputed depth / flow):
  1. RGB-only groups: multimodal `video` == base RobotVideoDataset `video` (same window), exactly;
  2. 9 cameras in 3 groups: video_rgb / video_depth / video_flow are [3, 9, 384, 320],
     video_rgb == the RGB-only tile, and depth / flow frame 0 == RGB frame 0 (unimodal parity).

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=. python notebooks/router_analysis/tests/test_multimodal_robotwin_tile.py
"""

import copy
import os

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from fastwam.datasets.lerobot.robot_video_dataset import RobotVideoDataset

REPO = "/data/mgovind/FastWAM"
DATA = "./data/real-world-r1-lite"
STATS = "runs/r1_lite_tasks_joint_3cam_384_1e-4/2026-09-17_14-25-47/dataset_stats.json"
MM = "fastwam.datasets.lerobot.robot_multimodal_video_dataset.RobotMultimodalVideoDataset"


def cfg_from(base, image_keys=None, groups=None):
    c = copy.deepcopy(base)
    c.dataset_dirs = [DATA]
    c.pretrained_norm_stats = STATS
    c.text_embedding_cache_dir = "./data/text_embeds_cache/r1_lite_tasks"
    if groups is not None:
        c._target_ = MM
        c.modality_camera_groups = groups
        c.shape_meta.images = [
            {"key": k, "raw_shape": raw, "shape": [3, 240, 320]} for k, raw in image_keys
        ]
        c.processor.num_output_cameras = len(image_keys)
    c.processor.shape_meta = c.shape_meta          # yaml has ${data.train.*} interpolations
    c.processor.num_obs_steps = c.num_frames
    return c


def main():
    os.chdir(REPO)
    # Text is irrelevant here (and the local copy has no cached embedding for its instruction).
    RobotVideoDataset._get_cached_text_context = lambda self, text: (
        torch.zeros(128, 4096), torch.ones(128, dtype=torch.bool))
    base = OmegaConf.load("configs/data/r1_lite_tasks.yaml").train
    rgb = [("image", [3, 640, 640]), ("wrist_left_image", [3, 360, 640]), ("wrist_right_image", [3, 360, 640])]
    dep = [("depth", [3, 640, 640]), ("wrist_left_depth", [3, 360, 640]), ("wrist_right_depth", [3, 360, 640])]
    flo = [("flow", [3, 640, 640]), ("wrist_left_flow", [3, 360, 640]), ("wrist_right_flow", [3, 360, 640])]

    ref_ds = instantiate(cfg_from(base))
    idx = len(ref_ds) // 2
    ref = ref_ds._get(idx)["video"]          # _get: __getitem__ silently swaps in a random sample on error

    # 1. RGB only
    mm_rgb = instantiate(cfg_from(base, rgb, {"rgb": [0, 1, 2]}))._get(idx)
    assert mm_rgb["video"].shape == ref.shape == (3, 9, 384, 320), (mm_rgb["video"].shape, ref.shape)
    assert torch.equal(mm_rgb["video"], ref)
    print(f"1. RGB-only multimodal tile == base RobotVideoDataset tile {tuple(ref.shape)}  OK")

    # 2. RGB + depth + flow
    s = instantiate(cfg_from(base, rgb + dep + flo, {"rgb": [0, 1, 2], "depth": [3, 4, 5], "flow": [6, 7, 8]}))._get(idx)
    for m in ("rgb", "depth", "flow"):
        assert s[f"video_{m}"].shape == (3, 9, 384, 320), (m, s[f"video_{m}"].shape)
    assert torch.equal(s["video_rgb"], ref)
    for m in ("depth", "flow"):
        assert torch.equal(s[f"video_{m}"][:, 0], ref[:, 0])
        assert not torch.equal(s[f"video_{m}"][:, 1:], ref[:, 1:])
    print("2. rgb/depth/flow each [3, 9, 384, 320]; video_rgb == base tile; depth/flow frame0 == RGB frame0  OK")
    print("ALL ROBOTWIN-TILE CHECKS PASSED")


if __name__ == "__main__":
    main()
