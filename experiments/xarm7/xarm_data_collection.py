#!/usr/bin/env python3
"""PS5-only xArm data collection for Cosmos Policy fine-tuning.

This collector is aligned with the current Cosmos xArm inference path while
writing demonstrations in an xArm dataset layout:
- images are saved under data/demo_i/obs/*_rgb_jpeg
- joint_states is the same 7D joint-state vector used at inference time
- actions use Cosmos policy action units
- absolute_actions stores observed absolute x/y/z/rpy/gripper state
- ee_states / ee_pos / ee_ori follow the older meters+radians convention

Keyboard controls:
    r -> start/stop and save current trajectory
    d -> discard current trajectory
    h -> return arm to home pose
    q -> quit cleanly

Controller controls:
    Left stick X/Y  -> dx / dy
    L2 / R2         -> dz up / down
    Right stick X/Y -> dyaw / dpitch
    R1 / L1         -> droll
    Cross           -> close gripper
    Circle          -> open gripper
    Options         -> quit
"""

from __future__ import annotations

import argparse
import math
import os
import select
import sys
import termios
import time
import tty
from dataclasses import dataclass

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import cv2
import h5py
import numpy as np
import pygame
from xarm.core.config import x_code
from xarm.wrapper import XArmAPI


# Cameras (both capture 640x480; crop format: [width, height, x_offset, y_offset]):
#   primary = overhead exo view, /dev/video18
#   wrist   = gripper-mounted view, /dev/video4. Mounting orientation depends on
#             the gripper installed (xhand vs. traditional), so re-check the
#             crop and keep the same gripper/orientation for collection and eval.
WRISTCAM_CROP = [480, 480, 80, 0]
EXO1_CROP = [258, 258, 194, 104]
CROP_TARGET_SIZE = (480, 480)
CAPTURE_FRAME_SIZE = (640, 480)

TCP_OFFSET = [0, 0, 0, 0, 0, 0]
HOME_POSE_MM_DEG = (300.0, 0.0, 300.0, 180.0, 0.0, 0.0)

GRIPPER_OPEN_POS = 850
GRIPPER_CLOSED_POS = 0

AXIS_LEFT_X = 0
AXIS_LEFT_Y = 1
AXIS_L2 = 2
AXIS_RIGHT_X = 3
AXIS_RIGHT_Y = 4
AXIS_R2 = 5

BUTTON_CROSS = 0
BUTTON_CIRCLE = 1
BUTTON_L1 = 4
BUTTON_R1 = 5
BUTTON_OPTIONS = 9


@dataclass
class CollectionConfig:
    task_name: str = "stack the blue cube on the green cube"
    save_root: str = "/home/skapse/workspace/cosmos-xarm/xarm_collected_demos"
    resume_path: str | None = None
    robot_ip: str = "192.168.1.198"
    primary_cam_id: int = 18
    wrist_cam_id: int = 4
    warmup_frames: int = 100
    control_hz: float = 15.0
    max_linear_speed_mm_s: float = 50.0
    max_rotation_speed_deg_s: float = 5.0
    joystick_dead_zone: float = 0.05
    home_speed_mm_s: float = 200.0
    home_accel_mm_s2: float = 1000.0
    gripper_speed: int = 5000
    reduced_mode: bool = False
    reduced_tcp_boundary: list[float] | None = None
    jpeg_quality: int = 95


def safe_task_name(task_name: str) -> str:
    return (
        task_name.strip()
        .lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(".", "_")
    )


def decode_hdf5_string(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def demo_index_from_key(demo_key: str) -> int | None:
    if not demo_key.startswith("demo_"):
        return None
    suffix = demo_key[5:]
    if not suffix.isdigit():
        return None
    return int(suffix)


def get_code(result) -> int:
    return int(result[0] if isinstance(result, tuple) else result)


def describe_controller_error(err_code: int | None, controller_error: int | None, controller_warn: int | None) -> str:
    parts: list[str] = []
    if controller_error is not None and controller_error in x_code.ControllerErrorCodeMap:
        detail = x_code.ControllerErrorCodeMap[controller_error]
        title = detail.get("en", {}).get("title", f"controller_error_{controller_error}")
        desc = detail.get("en", {}).get("desc", "")
        parts.append(f"{title} ({controller_error})")
        if desc:
            parts.append(desc)
    elif controller_error is not None:
        parts.append(f"controller_error={controller_error}")

    if controller_warn is not None:
        parts.append(f"warn_code={controller_warn}")
    if err_code is not None:
        parts.append(f"api_code={err_code}")
    return "; ".join(parts) if parts else "unknown xArm error"


def handle_xarm_result(arm: XArmAPI, api_name: str, result) -> bool:
    code = get_code(result)
    if code == 0:
        return True

    print(f"[xarm] warning: {api_name} returned code={code}; attempting clean_error()")
    try:
        err_code, err_warn = arm.get_err_warn_code()
        if err_code == 0 and err_warn is not None:
            controller_error = err_warn[0] if len(err_warn) > 0 else 0
            controller_warn = err_warn[1] if len(err_warn) > 1 else 0
            print(
                "[xarm] controller status: "
                f"{describe_controller_error(code, controller_error, controller_warn)}"
            )
    except Exception as exc:
        print(f"[xarm] warning: failed to read controller status: {exc}")

    try:
        clean_code = get_code(arm.clean_error())
        if clean_code != 0:
            print(f"[xarm] warning: clean_error() returned code={clean_code}")
    except Exception as exc:
        print(f"[xarm] warning: clean_error() raised {exc}")

    return False


def read_pose(arm: XArmAPI) -> tuple[list[float], list[float]] | None:
    result = arm.get_position(is_radian=False)
    code = get_code(result)
    if code != 0:
        handle_xarm_result(arm, "get_position", result)
        return None
    pose = result[1]
    if pose is None or len(pose) < 6:
        return None
    return (
        [float(pose[0]), float(pose[1]), float(pose[2])],
        [float(pose[3]), float(pose[4]), float(pose[5])],
    )


def read_joint_state(arm: XArmAPI) -> np.ndarray | None:
    try:
        result = arm.get_joint_states()
    except Exception as exc:
        print(f"[xarm] warning: get_joint_states failed: {exc}")
        return None

    code = get_code(result)
    if code != 0:
        handle_xarm_result(arm, "get_joint_states", result)
        return None

    payload = result[1]
    if payload is None or len(payload) == 0:
        return None
    joints = payload[0]
    if joints is None or len(joints) < 7:
        return None
    return np.asarray(joints[:7], dtype=np.float32)


def read_gripper_position(arm: XArmAPI) -> int | None:
    result = arm.get_gripper_position()
    code = get_code(result)
    if code != 0:
        handle_xarm_result(arm, "get_gripper_position", result)
        return None
    position = result[1]
    return int(position) if position is not None else None


def stop_robot(arm: XArmAPI) -> None:
    try:
        handle_xarm_result(
            arm,
            "vc_set_cartesian_velocity",
            arm.vc_set_cartesian_velocity([0, 0, 0, 0, 0, 0]),
        )
    except Exception:
        pass
    handle_xarm_result(arm, "set_state", arm.set_state(4))
    handle_xarm_result(arm, "set_gripper_enable", arm.set_gripper_enable(False))


def apply_reduced_safety_boundary(arm: XArmAPI, cfg: CollectionConfig) -> None:
    if cfg.reduced_tcp_boundary is not None:
        print(f"[xarm] applying reduced tcp boundary {cfg.reduced_tcp_boundary}")
        handle_xarm_result(
            arm,
            "set_reduced_tcp_boundary",
            arm.set_reduced_tcp_boundary(cfg.reduced_tcp_boundary),
        )
        if not cfg.reduced_mode:
            print(
                "[xarm] warning: reduced tcp boundary configured but reduced mode is not enabled; "
                "boundary will take effect only when reduced mode is enabled."
            )

    if cfg.reduced_mode:
        print("[xarm] enabling reduced safety boundary mode")
        handle_xarm_result(arm, "set_reduced_mode", arm.set_reduced_mode(True))


def send_velocity_command(arm: XArmAPI, velocity_cmd: dict[str, float]) -> bool:
    if not any(abs(velocity_cmd[key]) > 1e-9 for key in ("vx_mm_s", "vy_mm_s", "vz_mm_s", "vroll_deg_s", "vpitch_deg_s", "vyaw_deg_s")):
        return handle_xarm_result(
            arm,
            "vc_set_cartesian_velocity",
            arm.vc_set_cartesian_velocity([0, 0, 0, 0, 0, 0]),
        )

    result = arm.vc_set_cartesian_velocity(
        [
            velocity_cmd["vx_mm_s"],
            velocity_cmd["vy_mm_s"],
            velocity_cmd["vz_mm_s"],
            velocity_cmd["vroll_deg_s"],
            velocity_cmd["vpitch_deg_s"],
            velocity_cmd["vyaw_deg_s"],
        ]
    )
    return handle_xarm_result(arm, "vc_set_cartesian_velocity", result)


def crop_and_resize(frame: np.ndarray, crop: list[int]) -> np.ndarray:
    width, height, x_offset, y_offset = crop
    cropped = frame[y_offset : y_offset + height, x_offset : x_offset + width]
    return cv2.resize(cropped, CROP_TARGET_SIZE, interpolation=cv2.INTER_LINEAR)


def init_cameras(primary_id: int, wrist_id: int):
    primary_cap = cv2.VideoCapture(primary_id, cv2.CAP_V4L2)
    wrist_cap = cv2.VideoCapture(wrist_id, cv2.CAP_V4L2)
    for cap in (primary_cap, wrist_cap):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_FRAME_SIZE[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_FRAME_SIZE[1])
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return primary_cap, wrist_cap


def warmup_cameras(primary_cap, wrist_cap, num_frames: int) -> None:
    for _ in range(num_frames):
        primary_cap.read()
        wrist_cap.read()


def read_observation_images(primary_cap, wrist_cap) -> tuple[np.ndarray, np.ndarray]:
    ret_primary, primary = primary_cap.read()
    ret_wrist, wrist = wrist_cap.read()
    if not ret_primary or not ret_wrist:
        raise RuntimeError("Camera read failed")
    primary = crop_and_resize(primary, EXO1_CROP)
    wrist = crop_and_resize(wrist, WRISTCAM_CROP)
    primary = cv2.cvtColor(primary, cv2.COLOR_BGR2RGB)
    wrist = cv2.cvtColor(wrist, cv2.COLOR_BGR2RGB)
    return (
        np.asarray(primary, dtype=np.uint8),
        np.asarray(wrist, dtype=np.uint8),
    )


def apply_dead_zone(value: float, threshold: float) -> float:
    return 0.0 if abs(value) < threshold else float(value)


def normalize_trigger(value: float) -> float:
    return float(np.clip((value + 1.0) * 0.5, 0.0, 1.0))


def wrapped_angle_delta_deg(current_deg: float, previous_deg: float) -> float:
    delta = current_deg - previous_deg
    return ((delta + 180.0) % 360.0) - 180.0


def jpeg_encode_image(image: np.ndarray, quality: int = 95) -> np.ndarray:
    if image.dtype != np.uint8:
        raise ValueError("Image array must be uint8 for JPEG encoding.")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Image must have shape (H, W, 3).")
    bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    ok, buffer = cv2.imencode(".jpg", bgr_image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return np.frombuffer(buffer.tobytes(), dtype=np.uint8)


def pose_mm_deg_to_m_rad(position_mm: list[float], rpy_deg: list[float]) -> np.ndarray:
    return np.asarray(
        [
            position_mm[0] / 1000.0,
            position_mm[1] / 1000.0,
            position_mm[2] / 1000.0,
            math.radians(rpy_deg[0]),
            math.radians(rpy_deg[1]),
            math.radians(rpy_deg[2]),
        ],
        dtype=np.float32,
    )


def gripper_raw_to_dataset_state(raw_value: float) -> float:
    midpoint = 0.5 * (GRIPPER_OPEN_POS + GRIPPER_CLOSED_POS)
    return -1.0 if float(raw_value) >= midpoint else 1.0


def build_absolute_actions(actions: np.ndarray, ee_states: np.ndarray, gripper_states: np.ndarray) -> np.ndarray:
    if actions.ndim != 2 or actions.shape[1] != 7:
        raise ValueError(f"actions must have shape (T, 7), got {actions.shape}")
    if ee_states.ndim != 2 or ee_states.shape[1] != 6:
        raise ValueError(f"ee_states must have shape (T, 6), got {ee_states.shape}")
    if len(actions) != len(ee_states):
        raise ValueError(f"actions and ee_states lengths differ: {len(actions)} != {len(ee_states)}")
    if len(gripper_states) != len(ee_states):
        raise ValueError(f"gripper_states and ee_states lengths differ: {len(gripper_states)} != {len(ee_states)}")
    if len(actions) == 0:
        return np.zeros((0, 7), dtype=np.float32)

    absolute_actions = np.zeros(actions.shape, dtype=np.float32)
    absolute_actions[:, :6] = ee_states.astype(np.float32)
    absolute_actions[:, 6] = np.asarray(gripper_states, dtype=np.float32).reshape(-1)

    return absolute_actions


class PS5Controller:
    def __init__(self, cfg: CollectionConfig):
        pygame.init()
        pygame.joystick.init()

        if pygame.joystick.get_count() < 1:
            raise RuntimeError("No PS5 controller detected.")

        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        self.cfg = cfg

        print(
            f"[controller] connected: {self.joystick.get_name()} | "
            f"axes={self.joystick.get_numaxes()} buttons={self.joystick.get_numbuttons()}"
        )

    def poll(self) -> tuple[dict[str, float], str | None, bool]:
        quit_requested = False
        gripper_action = None

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit_requested = True
            elif event.type == pygame.JOYBUTTONDOWN:
                if event.button == BUTTON_CROSS:
                    gripper_action = "close"
                elif event.button == BUTTON_CIRCLE:
                    gripper_action = "open"
                elif event.button == BUTTON_OPTIONS:
                    quit_requested = True

        left_x = apply_dead_zone(self.joystick.get_axis(AXIS_LEFT_X), self.cfg.joystick_dead_zone)
        left_y = apply_dead_zone(self.joystick.get_axis(AXIS_LEFT_Y), self.cfg.joystick_dead_zone)
        right_x = apply_dead_zone(self.joystick.get_axis(AXIS_RIGHT_X), self.cfg.joystick_dead_zone)
        right_y = apply_dead_zone(self.joystick.get_axis(AXIS_RIGHT_Y), self.cfg.joystick_dead_zone)
        l2 = normalize_trigger(self.joystick.get_axis(AXIS_L2))
        r2 = normalize_trigger(self.joystick.get_axis(AXIS_R2))
        l1 = bool(self.joystick.get_button(BUTTON_L1))
        r1 = bool(self.joystick.get_button(BUTTON_R1))

        vx_mm_s = -left_x * self.cfg.max_linear_speed_mm_s
        vy_mm_s = left_y * self.cfg.max_linear_speed_mm_s
        vz_mm_s = (l2 - r2) * self.cfg.max_linear_speed_mm_s
        vroll_deg_s = (float(r1) - float(l1)) * self.cfg.max_rotation_speed_deg_s
        vpitch_deg_s = -right_y * self.cfg.max_rotation_speed_deg_s
        vyaw_deg_s = right_x * self.cfg.max_rotation_speed_deg_s

        return (
            {
                "vx_mm_s": vx_mm_s,
                "vy_mm_s": vy_mm_s,
                "vz_mm_s": vz_mm_s,
                "vroll_deg_s": vroll_deg_s,
                "vpitch_deg_s": vpitch_deg_s,
                "vyaw_deg_s": vyaw_deg_s,
            },
            gripper_action,
            quit_requested,
        )

    def shutdown(self) -> None:
        pygame.quit()


class TerminalKeyReader:
    def __init__(self):
        self.enabled = sys.stdin.isatty()
        self.fd = None
        self.old_term = None

    def __enter__(self):
        if self.enabled:
            self.fd = sys.stdin.fileno()
            self.old_term = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
        else:
            print("[keyboard] stdin is not a TTY; keyboard shortcuts disabled")
        return self

    def read_keys(self) -> list[str]:
        keys: list[str] = []
        if not self.enabled:
            return keys
        while True:
            ready, _, _ = select.select([sys.stdin], [], [], 0.0)
            if not ready:
                break
            char = sys.stdin.read(1)
            if not char:
                break
            keys.append(char.lower())
        return keys

    def __exit__(self, exc_type, exc, tb):
        if self.enabled and self.fd is not None and self.old_term is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_term)


class TrajectoryRecorder:
    def __init__(self, cfg: CollectionConfig):
        self.cfg = cfg
        self.task_slug = safe_task_name(cfg.task_name)
        self.resume_mode = cfg.resume_path is not None
        if self.resume_mode:
            self.dataset_path = cfg.resume_path or ""
            self.session_dir = os.path.dirname(self.dataset_path)
            self.trajectory_index = self._prepare_resume_dataset()
        else:
            session_stamp = time.strftime("%Y_%m_%d-%H_%M_%S")
            self.session_dir = os.path.join(cfg.save_root, self.task_slug, f"session_{session_stamp}")
            os.makedirs(self.session_dir, exist_ok=True)
            self.dataset_path = os.path.join(self.session_dir, f"{self.task_slug}_{session_stamp}.hdf5")
            with h5py.File(self.dataset_path, "a") as root:
                root.require_group("data")
            self.trajectory_index = 0
        self.recording = False
        self.start_time = None
        self.last_saved_demo_key: str | None = None
        self._reset_buffers()

    def _prepare_resume_dataset(self) -> int:
        with h5py.File(self.dataset_path, "a") as root:
            data_grp = root.require_group("data")
            self._sync_task_metadata_from_existing_file(data_grp)
            next_index = self._next_trajectory_index(data_grp)
        return next_index

    def _sync_task_metadata_from_existing_file(self, data_grp: h5py.Group) -> None:
        indexed_demo_keys: list[tuple[int, str]] = []
        for key in data_grp.keys():
            demo_index = demo_index_from_key(key)
            if demo_index is not None:
                indexed_demo_keys.append((demo_index, key))
        if not indexed_demo_keys:
            return

        _, first_demo_key = min(indexed_demo_keys)
        first_demo_attrs = data_grp[first_demo_key].attrs

        existing_task_name = decode_hdf5_string(first_demo_attrs.get("task_description"))
        if existing_task_name is not None and existing_task_name != self.cfg.task_name:
            print(
                "[resume] task name mismatch; "
                f"using existing dataset task name {existing_task_name!r} instead of {self.cfg.task_name!r}"
            )
            self.cfg.task_name = existing_task_name

        existing_task_slug = decode_hdf5_string(first_demo_attrs.get("subtask_key"))
        if existing_task_slug is not None and existing_task_slug != self.task_slug:
            print(
                "[resume] subtask key mismatch; "
                f"using existing dataset subtask key {existing_task_slug!r} instead of {self.task_slug!r}"
            )
            self.task_slug = existing_task_slug

    def _next_trajectory_index(self, data_grp: h5py.Group) -> int:
        next_index = 0
        for key in data_grp.keys():
            demo_index = demo_index_from_key(key)
            if demo_index is not None:
                next_index = max(next_index, demo_index + 1)
        return next_index

    def _reset_buffers(self) -> None:
        self.primary_images: list[np.ndarray] = []
        self.wrist_images: list[np.ndarray] = []
        self.joint_states: list[np.ndarray] = []
        self.ee_states: list[np.ndarray] = []
        self.gripper_states: list[float] = []
        self.actions: list[np.ndarray] = []

    def start(self) -> None:
        if self.recording:
            return
        self._reset_buffers()
        self.start_time = time.time()
        self.recording = True
        print(f"[recording] started trajectory {self.trajectory_index:04d}")

    def append_step(
        self,
        primary_image: np.ndarray,
        wrist_image: np.ndarray,
        joint_state: np.ndarray,
        ee_state: np.ndarray,
        gripper_state: float,
        action: np.ndarray,
    ) -> None:
        if not self.recording:
            return
        self.primary_images.append(primary_image.copy())
        self.wrist_images.append(wrist_image.copy())
        self.joint_states.append(joint_state.astype(np.float32).copy())
        self.ee_states.append(ee_state.astype(np.float32).copy())
        self.gripper_states.append(float(gripper_state))
        self.actions.append(action.astype(np.float32).copy())

    def discard(self) -> None:
        if self.recording:
            self.recording = False
            self._reset_buffers()
            print("[recording] discarded current trajectory")
            return
        if self.last_saved_demo_key is not None:
            self._discard_saved_demo(self.last_saved_demo_key)
            return
        print("[recording] nothing to discard")

    def _discard_saved_demo(self, demo_key: str) -> None:
        with h5py.File(self.dataset_path, "a") as root:
            data_grp = root.require_group("data")
            if demo_key in data_grp:
                del data_grp[demo_key]
        demo_index = demo_index_from_key(demo_key)
        if demo_index is not None:
            self.trajectory_index = demo_index
        print(f"[recording] discarded saved {demo_key} from {self.dataset_path}")
        self.last_saved_demo_key = None

    def _write_jpeg_image_dataset(self, root: h5py.Group, key: str, frames: list[np.ndarray]) -> None:
        jpeg_list = [jpeg_encode_image(frame, quality=self.cfg.jpeg_quality) for frame in frames]
        dt = h5py.vlen_dtype(np.dtype("uint8"))
        root.create_dataset(key, data=jpeg_list, dtype=dt)

    def stop_and_save(self) -> str | None:
        if not self.recording:
            print("[recording] nothing to save")
            return None
        self.recording = False

        if len(self.actions) == 0:
            self._reset_buffers()
            print("[recording] trajectory had no steps; nothing saved")
            return None

        end_time = time.time()
        demo_key = f"demo_{self.trajectory_index}"
        num_steps = len(self.actions)
        actions = np.stack(self.actions, axis=0).astype(np.float32)
        joint_states = np.stack(self.joint_states, axis=0).astype(np.float32)
        ee_states = np.stack(self.ee_states, axis=0).astype(np.float32)
        gripper_states = np.asarray(self.gripper_states, dtype=np.float32).reshape(-1, 1)
        absolute_actions = build_absolute_actions(actions, ee_states, gripper_states)
        dones = np.zeros(num_steps, dtype=np.uint8)
        dones[-1] = 1
        rewards = np.zeros(num_steps, dtype=np.uint8)
        states = np.zeros((num_steps, 1), dtype=np.float32)

        with h5py.File(self.dataset_path, "a") as root:
            data_grp = root.require_group("data")
            if demo_key in data_grp:
                del data_grp[demo_key]
            demo_grp = data_grp.create_group(demo_key)
            obs_grp = demo_grp.create_group("obs")

            self._write_jpeg_image_dataset(obs_grp, "agentview_rgb_jpeg", self.primary_images)
            self._write_jpeg_image_dataset(obs_grp, "eye_in_hand_rgb_jpeg", self.wrist_images)
            obs_grp.create_dataset("ee_states", data=ee_states, dtype=np.float32)
            obs_grp.create_dataset("ee_pos", data=ee_states[:, :3], dtype=np.float32)
            obs_grp.create_dataset("ee_ori", data=ee_states[:, 3:], dtype=np.float32)
            obs_grp.create_dataset("gripper_states", data=gripper_states, dtype=np.float32)

            demo_grp.create_dataset("actions", data=actions, dtype=np.float32)
            demo_grp.create_dataset("absolute_actions", data=absolute_actions, dtype=np.float32)
            demo_grp.create_dataset("joint_states", data=joint_states, dtype=np.float32)
            demo_grp.create_dataset("states", data=states, dtype=np.float32)
            demo_grp.create_dataset("rewards", data=rewards, dtype=np.uint8)
            demo_grp.create_dataset("dones", data=dones, dtype=np.uint8)

            demo_grp.attrs["num_samples"] = int(num_steps)
            demo_grp.attrs["source_real_world_episode"] = True
            demo_grp.attrs["subtask_key"] = self.task_slug
            demo_grp.attrs["success"] = False
            demo_grp.attrs["task_description"] = self.cfg.task_name
            demo_grp.attrs["recording_start_time"] = float(self.start_time)
            demo_grp.attrs["recording_end_time"] = float(end_time)
            demo_grp.attrs["control_hz"] = float(self.cfg.control_hz)
            demo_grp.attrs["absolute_actions_added"] = True

            root.attrs["primary_camera_id"] = int(self.cfg.primary_cam_id)
            root.attrs["wrist_camera_id"] = int(self.cfg.wrist_cam_id)
            root.attrs["primary_crop"] = np.asarray(EXO1_CROP, dtype=np.int32)
            root.attrs["wrist_crop"] = np.asarray(WRISTCAM_CROP, dtype=np.int32)
            root.attrs["crop_target_size"] = np.asarray(CROP_TARGET_SIZE, dtype=np.int32)
            root.attrs["capture_frame_size"] = np.asarray(CAPTURE_FRAME_SIZE, dtype=np.int32)
            root.attrs["tcp_offset"] = np.asarray(TCP_OFFSET, dtype=np.float32)
            root.attrs["home_pose_mm_deg"] = np.asarray(HOME_POSE_MM_DEG, dtype=np.float32)
            root.attrs["control_mode"] = "mode_5_vc_set_cartesian_velocity"
            root.attrs["actions_order"] = "dx_m,dy_m,dz_m,droll_rad,dpitch_rad,dyaw_rad,gripper_cmd"
            root.attrs["actions_source"] = "observed_ee_state_delta_over_one_control_step + gripper_command"
            root.attrs["absolute_actions_order"] = "x_m,y_m,z_m,roll_rad,pitch_rad,yaw_rad,gripper_state"
            root.attrs["absolute_actions_source"] = "obs/ee_states + obs/gripper_states"
            root.attrs["joint_states_order"] = "joint_1,...,joint_7_degrees"
            root.attrs["ee_states_order"] = "x_m,y_m,z_m,roll_rad,pitch_rad,yaw_rad"
            root.attrs["gripper_states_order"] = "(-1=open, +1=closed)"

        print(f"[recording] saved {demo_key} to {self.dataset_path}")
        self.last_saved_demo_key = demo_key
        self.trajectory_index += 1
        self._reset_buffers()
        return self.dataset_path


def move_home(arm: XArmAPI, cfg: CollectionConfig) -> tuple[list[float], list[float], int]:
    handle_xarm_result(arm, "set_mode", arm.set_mode(0))
    handle_xarm_result(arm, "set_state", arm.set_state(0))
    handle_xarm_result(
        arm,
        "set_position",
        arm.set_position(
            HOME_POSE_MM_DEG[0],
            HOME_POSE_MM_DEG[1],
            HOME_POSE_MM_DEG[2],
            HOME_POSE_MM_DEG[3],
            HOME_POSE_MM_DEG[4],
            HOME_POSE_MM_DEG[5],
            speed=cfg.home_speed_mm_s,
            mvacc=cfg.home_accel_mm_s2,
            wait=True,
        ),
    )
    handle_xarm_result(arm, "set_gripper_mode", arm.set_gripper_mode(0))
    handle_xarm_result(arm, "set_gripper_enable", arm.set_gripper_enable(True))
    handle_xarm_result(arm, "set_gripper_speed", arm.set_gripper_speed(cfg.gripper_speed))
    handle_xarm_result(
        arm,
        "set_gripper_position",
        arm.set_gripper_position(GRIPPER_OPEN_POS, wait=True),
    )

    pose = read_pose(arm)
    if pose is None:
        position = list(HOME_POSE_MM_DEG[:3])
        rpy = list(HOME_POSE_MM_DEG[3:])
    else:
        position, rpy = pose
    gripper = read_gripper_position(arm)
    if gripper is None:
        gripper = GRIPPER_OPEN_POS
    return position, rpy, gripper


def init_arm(cfg: CollectionConfig) -> tuple[XArmAPI, list[float], list[float], int]:
    print(f"[xarm] connecting to xArm7 at {cfg.robot_ip}...")
    arm = XArmAPI(cfg.robot_ip)
    time.sleep(0.5)
    handle_xarm_result(arm, "motion_enable", arm.motion_enable(True))
    handle_xarm_result(arm, "set_tcp_offset", arm.set_tcp_offset(TCP_OFFSET))
    apply_reduced_safety_boundary(arm, cfg)
    position, rpy, gripper = move_home(arm, cfg)
    set_velocity_mode(arm)
    print("[xarm] ready in velocity teleop mode")
    return arm, position, rpy, gripper


def print_controls() -> None:
    print(
        "[controls]\n"
        "  keyboard: r=start/stop save, d=discard (current, or last saved if not recording), h=home, q=quit\n"
        "  controller: left stick=xy, L2/R2=z up/down, right stick=yaw/pitch, R1/L1=roll,\n"
        "              Cross=close gripper, Circle=open gripper, Options=quit\n"
    )


def set_velocity_mode(arm: XArmAPI) -> None:
    handle_xarm_result(arm, "set_mode", arm.set_mode(5))
    handle_xarm_result(arm, "set_state", arm.set_state(0))


def apply_gripper_target(arm: XArmAPI, target_hw: int | None, current_raw: float, tolerance: float = 5.0) -> int | None:
    if target_hw is None:
        return None

    if handle_xarm_result(
        arm,
        "set_gripper_position",
        arm.set_gripper_position(target_hw, wait=False),
    ) and abs(float(current_raw) - float(target_hw)) <= tolerance:
        return None

    return target_hw


def gripper_command_value(gripper_action: str | None) -> float:
    gripper_cmd = 0.0
    if gripper_action == "close":
        gripper_cmd = 1.0
    elif gripper_action == "open":
        gripper_cmd = -1.0

    return gripper_cmd


def build_observed_relative_action(
    previous_pos: list[float],
    previous_rpy: list[float],
    current_pos: list[float],
    current_rpy: list[float],
    gripper_action: str | None,
) -> np.ndarray:
    gripper_cmd = gripper_command_value(gripper_action)

    return np.asarray(
        [
            (current_pos[0] - previous_pos[0]) / 1000.0,
            (current_pos[1] - previous_pos[1]) / 1000.0,
            (current_pos[2] - previous_pos[2]) / 1000.0,
            math.radians(wrapped_angle_delta_deg(current_rpy[0], previous_rpy[0])),
            math.radians(wrapped_angle_delta_deg(current_rpy[1], previous_rpy[1])),
            math.radians(wrapped_angle_delta_deg(current_rpy[2], previous_rpy[2])),
            gripper_cmd,
        ],
        dtype=np.float32,
    )


def parse_args() -> CollectionConfig:
    parser = argparse.ArgumentParser(description="PS5-only xArm Cosmos data collection")
    parser.add_argument("--task-name", type=str, default=CollectionConfig.task_name)
    parser.add_argument("--save-root", type=str, default=CollectionConfig.save_root)
    parser.add_argument(
        "--resume-path",
        type=str,
        default=CollectionConfig.resume_path,
        help="Append new demos to an existing HDF5 file instead of creating a new session file.",
    )
    parser.add_argument("--robot-ip", type=str, default=CollectionConfig.robot_ip)
    parser.add_argument("--primary-cam-id", type=int, default=CollectionConfig.primary_cam_id)
    parser.add_argument("--wrist-cam-id", type=int, default=CollectionConfig.wrist_cam_id)
    parser.add_argument("--warmup-frames", type=int, default=CollectionConfig.warmup_frames)
    parser.add_argument("--control-hz", type=float, default=CollectionConfig.control_hz)
    parser.add_argument("--max-linear-speed-mm-s", type=float, default=CollectionConfig.max_linear_speed_mm_s)
    parser.add_argument("--max-rotation-speed-deg-s", type=float, default=CollectionConfig.max_rotation_speed_deg_s)
    parser.add_argument("--joystick-dead-zone", type=float, default=CollectionConfig.joystick_dead_zone)
    parser.add_argument("--home-speed-mm-s", type=float, default=CollectionConfig.home_speed_mm_s)
    parser.add_argument("--home-accel-mm-s2", type=float, default=CollectionConfig.home_accel_mm_s2)
    parser.add_argument("--gripper-speed", type=int, default=CollectionConfig.gripper_speed)
    parser.add_argument(
        "--reduced-mode",
        action="store_true",
        help="Enable xArm reduced safety-boundary mode.",
    )
    parser.add_argument(
        "--reduced-tcp-boundary",
        type=float,
        nargs=6,
        metavar=("X_MAX", "X_MIN", "Y_MAX", "Y_MIN", "Z_MAX", "Z_MIN"),
        help="Set the xArm reduced TCP safety boundary.",
    )
    args = parser.parse_args()

    resume_path = None
    if args.resume_path is not None:
        resume_path = os.path.abspath(os.path.expanduser(args.resume_path))
        if not os.path.isfile(resume_path):
            parser.error(f"--resume-path must point to an existing HDF5 file: {resume_path}")

    return CollectionConfig(
        task_name=args.task_name,
        save_root=args.save_root,
        resume_path=resume_path,
        robot_ip=args.robot_ip,
        primary_cam_id=args.primary_cam_id,
        wrist_cam_id=args.wrist_cam_id,
        warmup_frames=args.warmup_frames,
        control_hz=args.control_hz,
        max_linear_speed_mm_s=args.max_linear_speed_mm_s,
        max_rotation_speed_deg_s=args.max_rotation_speed_deg_s,
        joystick_dead_zone=args.joystick_dead_zone,
        home_speed_mm_s=args.home_speed_mm_s,
        home_accel_mm_s2=args.home_accel_mm_s2,
        gripper_speed=args.gripper_speed,
        reduced_mode=args.reduced_mode,
        reduced_tcp_boundary=list(args.reduced_tcp_boundary) if args.reduced_tcp_boundary is not None else None,
    )


def main() -> None:
    cfg = parse_args()
    controller: PS5Controller | None = None
    arm: XArmAPI | None = None
    primary_cap = None
    wrist_cap = None

    recorder = TrajectoryRecorder(cfg)
    last_observed_pos = list(HOME_POSE_MM_DEG[:3])
    last_observed_rpy = list(HOME_POSE_MM_DEG[3:])
    last_joint_state = np.zeros(7, dtype=np.float32)
    last_gripper_raw = float(GRIPPER_OPEN_POS)

    gripper_target = GRIPPER_OPEN_POS
    pending_step: dict[str, object] | None = None

    print_controls()
    if recorder.resume_mode:
        print(f"[session] resuming trajectories in {recorder.dataset_path}")
    else:
        print(f"[session] saving trajectories under {recorder.session_dir}")
    print(f"[session] next trajectory index {recorder.trajectory_index:04d}")

    try:
        controller = PS5Controller(cfg)
        arm, _, _, gripper_target = init_arm(cfg)
        primary_cap, wrist_cap = init_cameras(cfg.primary_cam_id, cfg.wrist_cam_id)
        warmup_cameras(primary_cap, wrist_cap, cfg.warmup_frames)
        last_gripper_raw = float(gripper_target)

        period_s = 1.0 / cfg.control_hz
        status_interval_s = 1.0
        last_status_print = 0.0

        with TerminalKeyReader() as key_reader:
            while True:
                loop_start = time.time()

                for key in key_reader.read_keys():
                    if key == "r":
                        if recorder.recording:
                            recorder.stop_and_save()
                            pending_step = None
                        else:
                            recorder.start()
                            pending_step = None
                    elif key == "d":
                        recorder.discard()
                        pending_step = None
                    elif key == "h":
                        if recorder.recording:
                            print("[recording] stop or discard before homing")
                        else:
                            print("[xarm] returning home...")
                            _, _, gripper_target = move_home(arm, cfg)
                            set_velocity_mode(arm)
                            last_gripper_raw = float(gripper_target)
                            pending_step = None
                    elif key == "q":
                        print("[session] keyboard quit requested")
                        if recorder.recording:
                            recorder.stop_and_save()
                        pending_step = None
                        return

                velocity_cmd, gripper_action, quit_requested = controller.poll()
                if quit_requested:
                    print("[session] controller quit requested")
                    if recorder.recording:
                        recorder.stop_and_save()
                    pending_step = None
                    return

                pose = read_pose(arm)
                if pose is not None:
                    last_observed_pos, last_observed_rpy = pose

                joint_state = read_joint_state(arm)
                if joint_state is not None:
                    last_joint_state = joint_state

                gripper_raw = read_gripper_position(arm)
                if gripper_raw is not None:
                    last_gripper_raw = float(gripper_raw)

                primary_rgb = None
                wrist_rgb = None
                try:
                    primary_rgb, wrist_rgb = read_observation_images(primary_cap, wrist_cap)
                except Exception as exc:
                    print(f"[camera] warning: {exc}")

                if recorder.recording and pending_step is not None:
                    observed_relative_action = build_observed_relative_action(
                        previous_pos=pending_step["pos"],
                        previous_rpy=pending_step["rpy"],
                        current_pos=last_observed_pos,
                        current_rpy=last_observed_rpy,
                        gripper_action=pending_step["gripper_action"],
                    )
                    recorder.append_step(
                        primary_image=pending_step["primary_image"],
                        wrist_image=pending_step["wrist_image"],
                        joint_state=pending_step["joint_state"],
                        ee_state=pending_step["ee_state"],
                        gripper_state=float(pending_step["gripper_state"]),
                        action=observed_relative_action,
                    )
                    pending_step = None

                if gripper_action == "close":
                    gripper_target = GRIPPER_CLOSED_POS
                elif gripper_action == "open":
                    gripper_target = GRIPPER_OPEN_POS

                gripper_target = apply_gripper_target(arm, gripper_target, last_gripper_raw)

                if recorder.recording and primary_rgb is not None and wrist_rgb is not None:
                    ee_state = pose_mm_deg_to_m_rad(last_observed_pos, last_observed_rpy)
                    pending_step = {
                        "primary_image": primary_rgb.copy(),
                        "wrist_image": wrist_rgb.copy(),
                        "joint_state": last_joint_state.astype(np.float32).copy(),
                        "ee_state": ee_state,
                        "gripper_state": gripper_raw_to_dataset_state(last_gripper_raw),
                        "pos": list(last_observed_pos),
                        "rpy": list(last_observed_rpy),
                        "gripper_action": gripper_action,
                    }
                elif not recorder.recording:
                    pending_step = None

                send_velocity_command(arm, velocity_cmd)

                now = time.time()
                if now - last_status_print >= status_interval_s:
                    recording_state = "ON" if recorder.recording else "OFF"
                    print(
                        "[status] "
                        f"recording={recording_state} "
                        f"pose_mm=({last_observed_pos[0]:.1f}, {last_observed_pos[1]:.1f}, {last_observed_pos[2]:.1f}) "
                        f"rpy_deg=({last_observed_rpy[0]:.1f}, {last_observed_rpy[1]:.1f}, {last_observed_rpy[2]:.1f}) "
                        f"gripper={int(last_gripper_raw)}"
                    )
                    last_status_print = now

                elapsed = time.time() - loop_start
                sleep_time = period_s - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[session] KeyboardInterrupt")
        if recorder.recording:
            recorder.stop_and_save()

    finally:
        if primary_cap is not None:
            primary_cap.release()
        if wrist_cap is not None:
            wrist_cap.release()
        if arm is not None:
            stop_robot(arm)
            arm.disconnect()
        if controller is not None:
            controller.shutdown()


if __name__ == "__main__":
    main()
