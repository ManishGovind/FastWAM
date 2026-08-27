"""Multi-stream robot video dataset: RGB + depth + flow as separate videos.

Extends RobotVideoDataset without modifying it. Emits video_rgb / video_depth /
video_flow (each [C, T, H, W]) plus a `video` alias equal to video_rgb for
trainer compatibility.

By default, depth/flow frame-0 is replaced with RGB frame-0 so each stream
matches unimodal future_key packing: [RGB_t0, modality_t≥1]. Inference stays
RGB-only via the inherited Joint path.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from fastwam.utils.logging_config import get_logger

from .robot_video_dataset import DEFAULT_PROMPT, RobotVideoDataset

logger = get_logger(__name__)

_REQUIRED_MODALITIES = ("rgb", "depth", "flow")


class RobotMultimodalVideoDataset(RobotVideoDataset):
    """Pack multiple modality camera groups into separate video tensors."""

    def __init__(
        self,
        *args,
        modality_camera_groups: dict[str, Sequence[int]] | DictConfig | None = None,
        # Paste RGB frame-0 into these streams (unimodal future_key parity).
        rgb_condition_modalities: Sequence[str] | None = ("depth", "flow"),
        # Optional: zeros frame-0 instead of RGB paste (mutually exclusive prefer rgb).
        zero_condition_modalities: Sequence[str] | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if modality_camera_groups is None:
            raise ValueError(
                "`modality_camera_groups` is required, e.g. "
                "{rgb: [0, 1], depth: [2, 3], flow: [4, 5]} indexing into pixel_values."
            )
        groups = OmegaConf.to_container(modality_camera_groups, resolve=True)
        if not isinstance(groups, dict):
            raise ValueError(
                f"`modality_camera_groups` must be a dict, got {type(groups)}"
            )
        missing = [m for m in _REQUIRED_MODALITIES if m not in groups]
        if missing:
            raise ValueError(
                f"`modality_camera_groups` missing modalities {missing}; "
                f"required={list(_REQUIRED_MODALITIES)}"
            )
        self.modality_camera_groups: dict[str, list[int]] = {
            str(name): [int(i) for i in indices] for name, indices in groups.items()
        }

        if zero_condition_modalities is not None and rgb_condition_modalities is not None:
            logger.warning(
                "Both rgb_condition_modalities and zero_condition_modalities set; "
                "using rgb_condition_modalities (unimodal parity)."
            )
        self.rgb_condition_modalities = {
            str(m) for m in (rgb_condition_modalities or ())
        }
        self.zero_condition_modalities = {
            str(m) for m in (zero_condition_modalities or ())
        } - self.rgb_condition_modalities

        unknown = (self.rgb_condition_modalities | self.zero_condition_modalities) - set(
            self.modality_camera_groups
        )
        if unknown:
            raise ValueError(f"Unknown condition modalities: {sorted(unknown)}")
        if "rgb" in self.rgb_condition_modalities or "rgb" in self.zero_condition_modalities:
            raise ValueError("Cannot apply condition replacement to the rgb stream itself.")

        expected_cams = max(
            max(indices) for indices in self.modality_camera_groups.values()
        ) + 1
        processor = getattr(self.lerobot_dataset, "processor", None)
        if processor is not None:
            num_out = int(getattr(processor, "num_output_cameras", expected_cams))
            if num_out < expected_cams:
                raise ValueError(
                    f"processor.num_output_cameras={num_out} < required camera index "
                    f"{expected_cams - 1} from modality_camera_groups."
                )

        logger.info(
            "RobotMultimodalVideoDataset modalities=%s rgb_condition=%s zero_condition=%s",
            {k: v for k, v in self.modality_camera_groups.items()},
            sorted(self.rgb_condition_modalities),
            sorted(self.zero_condition_modalities),
        )

    def _subsample_pixel_values(
        self, pixel_values: torch.Tensor, image_is_pad: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        """Return cameras as [N, T_video, C, H, W] and aligned image_is_pad."""
        if pixel_values.ndim == 5:
            video = pixel_values
            if not self.lerobot_dataset.presample_images:
                video = video[:, self.video_sample_indices, :, :, :]
            num_cameras, t_video, _, _, _ = video.shape
        elif pixel_values.ndim == 4:
            video = pixel_values.unsqueeze(0)
            if not self.lerobot_dataset.presample_images:
                video = video[:, self.video_sample_indices, :, :, :]
            num_cameras, t_video, _, _, _ = video.shape
        else:
            raise ValueError(
                f"Expected pixel_values [N,T,C,H,W] or [T,C,H,W], got {tuple(pixel_values.shape)}"
            )

        if not self.lerobot_dataset.presample_images:
            image_is_pad = image_is_pad[self.video_sample_indices]
        return video, image_is_pad, num_cameras, t_video

    def _pack_camera_group(self, cameras: torch.Tensor) -> torch.Tensor:
        """cameras: [N_cam, T, C, H, W] -> [C, T, H', W'] in [-1, 1]."""
        num_cameras, t_video, _, _, _ = cameras.shape
        if self.concat_multi_camera == "robotwin":
            raise ValueError(
                "RobotMultimodalVideoDataset does not support concat_multi_camera='robotwin'."
            )
        if num_cameras > 1:
            if self.concat_multi_camera == "horizontal":
                video = torch.cat([cameras[i] for i in range(num_cameras)], dim=-1)
            elif self.concat_multi_camera == "vertical":
                video = torch.cat([cameras[i] for i in range(num_cameras)], dim=-2)
            else:
                raise ValueError(
                    f"Invalid concat_multi_camera: {self.concat_multi_camera}. "
                    "Expected one of: horizontal, vertical."
                )
        else:
            video = cameras.squeeze(0)

        video = self.resize_transform(video)
        video = self.crop_transform(video)
        video = self.normalize_transform(video)
        return video.permute(1, 0, 2, 3)  # [C, T, H, W]

    def _get(self, idx):
        sample_idx = idx
        sample = None
        for attempt in range(self.max_padding_retry + 1):
            sample = self.lerobot_dataset[sample_idx]

            if not self.skip_padding_as_possible:
                break

            has_pad = (
                bool(sample["action_is_pad"].any().item())
                or bool(sample["image_is_pad"].any().item())
                or bool(sample["proprio_is_pad"].any().item())
            )
            if not has_pad or attempt >= self.max_padding_retry:
                break
            sample_idx = np.random.randint(len(self.lerobot_dataset))

        cameras, image_is_pad, num_cameras, _t_video = self._subsample_pixel_values(
            sample["pixel_values"], sample["image_is_pad"]
        )

        videos: dict[str, torch.Tensor] = {}
        for modality, cam_indices in self.modality_camera_groups.items():
            if any(i < 0 or i >= num_cameras for i in cam_indices):
                raise ValueError(
                    f"Modality '{modality}' camera indices {cam_indices} out of range "
                    f"for num_cameras={num_cameras}."
                )
            videos[modality] = self._pack_camera_group(cameras[cam_indices])

        video_rgb = videos["rgb"]
        # Unimodal parity: [RGB_t0, depth/flow_t≥1]
        for modality in self.rgb_condition_modalities:
            videos[modality] = videos[modality].clone()
            videos[modality][:, 0] = video_rgb[:, 0]
        for modality in self.zero_condition_modalities:
            videos[modality] = videos[modality].clone()
            videos[modality][:, 0] = 0

        video_depth = videos["depth"]
        video_flow = videos["flow"]

        action = sample["action"]
        proprio = sample["proprio"][:-1, :]
        for name, video in (("rgb", video_rgb), ("depth", video_depth), ("flow", video_flow)):
            if video.shape[1] <= 1:
                raise ValueError(
                    f"`video_{name}` must have at least 2 frames, got shape {tuple(video.shape)}"
                )
            if action.shape[0] % (video.shape[1] - 1) != 0:
                raise ValueError(
                    f"`action` horizon must be divisible by `video_{name}` transitions, "
                    f"got {action.shape[0]} and {video.shape[1] - 1}"
                )

        task = sample["instruction"]
        if self.override_instruction is not None:
            task = self.override_instruction
        instruction = DEFAULT_PROMPT.format(task=task)

        data: dict[str, Any] = {
            "video": video_rgb,
            "video_rgb": video_rgb,
            "video_depth": video_depth,
            "video_flow": video_flow,
            "action": action,
            "proprio": proprio,
            "prompt": instruction,
            "image_is_pad": image_is_pad,
            "action_is_pad": sample["action_is_pad"],
            "proprio_is_pad": sample["proprio_is_pad"],
        }
        if self.use_text_embed_cache:
            context, context_mask = self._get_cached_text_context(instruction)
            context[~context_mask] = 0.0
            data["context"] = context
            data["context_mask"] = torch.ones_like(context_mask)
        return data
