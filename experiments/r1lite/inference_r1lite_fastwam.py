#!/usr/bin/env python3
"""Run a workstation-hosted FastWAM policy from a Galaxea R1 Lite ROS 2 PC.

The client supports absolute or relative arm outputs, publishes absolute
gripper targets, and holds the last target while the next chunk arrives.
The selected mode must match the action contract reported by the server.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import queue
import re
import signal
import subprocess
import threading
import time
import traceback
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import cv2
import h5py
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rollout_hdf5 import HDF5RolloutRecorder
from sensor_msgs.msg import CompressedImage, Image, JointState

ARM_FEEDBACK_TOPICS = {"left": "/hdas/feedback_arm_left", "right": "/hdas/feedback_arm_right"}
GRIPPER_FEEDBACK_TOPICS = {"left": "/hdas/feedback_gripper_left", "right": "/hdas/feedback_gripper_right"}
ARM_TARGET_TOPICS = {
    "left": "/motion_target/target_joint_state_arm_left",
    "right": "/motion_target/target_joint_state_arm_right",
}
GRIPPER_TARGET_TOPICS = {
    "left": "/motion_target/target_position_gripper_left",
    "right": "/motion_target/target_position_gripper_right",
}
HEAD_CAMERA_TOPIC = "/hdas/camera_head/left_raw/image_raw_color/compressed"
RAW_WRIST_TOPICS = {
    "left_wrist": "/hdas/camera_wrist_left/color/image_raw",
    "right_wrist": "/hdas/camera_wrist_right/color/image_raw",
}
COMPRESSED_WRIST_TOPICS = {
    "left_wrist": "/hdas/camera_wrist_left/color/image_raw/compressed",
    "right_wrist": "/hdas/camera_wrist_right/color/image_raw/compressed",
}
JOINTS_PER_ARM, ACTION_DIM = 6, 14
RESET_SCRIPT = Path(__file__).resolve().parent / "reset_to_initial_position.sh"
JOINT_LIMITS_RAD = np.asarray(
    [(-1.20, 0.70), (-0.40, 2.80), (-2.35, 0.40), (-0.80, 1.20), (-0.70, 0.70), (-1.10, 1.30)],
    dtype=np.float32,
)


def _safe_path_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    if not cleaned:
        raise ValueError("policy config must contain at least one letter or number")
    return cleaned


@dataclass(frozen=True)
class RolloutSession:
    root: Path

    @classmethod
    def create(cls, rollouts_dir: Path, policy_config: str) -> RolloutSession:
        now = datetime.now().astimezone()
        root = (
            rollouts_dir.expanduser()
            / _safe_path_component(policy_config)
            / now.strftime("%Y%m%d")
            / f"run_{now.strftime('%H%M%S_%f')}"
        )
        root.mkdir(parents=True, exist_ok=False)
        return cls(root=root)

@dataclass(frozen=True)
class ObservationBundle:
    """One temporally aligned robot observation selected from ROS message stamps."""

    timestamp_s: float
    state: np.ndarray
    images: dict[str, np.ndarray]
    sample_timestamps_s: dict[str, float]


class RobotIO(Node):
    def __init__(
        self,
        wrist_transport: str,
        hold_hz: float,
        observation_buffer_s: float,
        rollout_recorder: HDF5RolloutRecorder,
        max_observation_skew_s: float,
        max_observation_age_s: float,
    ) -> None:
        super().__init__("r1lite_fastwam_inference")
        self.bridge = CvBridge()
        self.arm: dict[str, np.ndarray | None] = {"left": None, "right": None}
        self.arm_names: dict[str, list[str] | None] = {"left": None, "right": None}
        self.gripper: dict[str, float | None] = {"left": None, "right": None}
        self.images: dict[str, np.ndarray | None] = {"head": None, "left_wrist": None, "right_wrist": None}
        self._observation_buffer_s = observation_buffer_s
        self._last_observation_stamp = float('-inf')
        self._invalid_stamp_warned: set[str] = set()
        self.rollout_recorder = rollout_recorder
        self._capture_skew_s = max_observation_skew_s
        self._capture_age_s = max_observation_age_s
        self._last_rollout_stamp = float("-inf")
        self._samples: dict[str, deque[tuple[float, np.ndarray]]] = {
            "head": deque(),
            "left_wrist": deque(),
            "right_wrist": deque(),
            "left_arm": deque(),
            "right_arm": deque(),
            "left_gripper": deque(),
            "right_gripper": deque(),
        }
        self._command_lock = threading.Lock()
        self._last_targets: np.ndarray | None = None
        self._hold_enabled = threading.Event()
        self._hold_closed = threading.Event()
        self._hold_hz = hold_hz
        self._active_demo_id: str | None = None
        self._last_target_chunk_id: int | None = None
        self._last_target_action_step: int | None = None
        self._arm_feedback_sequence = {"left": 0, "right": 0}

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
        for side, topic in ARM_FEEDBACK_TOPICS.items():
            self.create_subscription(JointState, topic, self._arm_callback(side), qos)
        for side, topic in GRIPPER_FEEDBACK_TOPICS.items():
            self.create_subscription(JointState, topic, self._gripper_callback(side), qos)
        self.create_subscription(CompressedImage, HEAD_CAMERA_TOPIC, self._image_callback("head"), qos)
        wrist_topics = RAW_WRIST_TOPICS if wrist_transport == "raw" else COMPRESSED_WRIST_TOPICS
        wrist_type = Image if wrist_transport == "raw" else CompressedImage
        for name, topic in wrist_topics.items():
            self.create_subscription(wrist_type, topic, self._image_callback(name), qos)
        self.arm_publishers = {side: self.create_publisher(JointState, topic, 10) for side, topic in ARM_TARGET_TOPICS.items()}
        self.gripper_publishers = {side: self.create_publisher(JointState, topic, 10) for side, topic in GRIPPER_TARGET_TOPICS.items()}
        self._hold_thread = threading.Thread(target=self._hold_loop, daemon=True)
        self._hold_thread.start()

    @staticmethod
    def _message_timestamp_s(message: Image | CompressedImage | JointState) -> float:
        """Preserve publisher stamps; receipt times cannot establish capture alignment."""
        stamp = message.header.stamp
        timestamp_s = float(stamp.sec) + float(stamp.nanosec) * 1e-9
        return timestamp_s

    def _append_sample(self, name: str, timestamp_s: float, value: np.ndarray) -> None:
        if timestamp_s <= 0.0:
            if name not in self._invalid_stamp_warned:
                self.get_logger().warning(f"{name}: missing ROS timestamp; excluded from synchronized observations")
                self._invalid_stamp_warned.add(name)
            return
        samples = self._samples[name]
        if samples and timestamp_s <= samples[-1][0]:
            return
        samples.append((timestamp_s, value.copy()))
        cutoff = timestamp_s - self._observation_buffer_s
        while samples and samples[0][0] < cutoff:
            samples.popleft()

    def _arm_callback(self, side: str) -> Callable[[JointState], None]:
        def callback(message: JointState) -> None:
            if len(message.position) >= JOINTS_PER_ARM:
                self.arm[side] = np.asarray(message.position[:JOINTS_PER_ARM], dtype=np.float32)
                self.arm_names[side] = list(message.name[:JOINTS_PER_ARM])
                self._arm_feedback_sequence[side] += 1
                self._append_sample(f"{side}_arm", self._message_timestamp_s(message), self.arm[side])
        return callback

    def _gripper_callback(self, side: str) -> Callable[[JointState], None]:
        def callback(message: JointState) -> None:
            if message.position:
                self.gripper[side] = float(message.position[0])
                self._append_sample(
                    f"{side}_gripper",
                    self._message_timestamp_s(message),
                    np.asarray([self.gripper[side]], dtype=np.float32),
                )
        return callback

    def _image_callback(self, name: str) -> Callable[[Image | CompressedImage], None]:
        def callback(message: Image | CompressedImage) -> None:
            try:
                if isinstance(message, CompressedImage):
                    # Decode the compressed payload directly. This avoids the
                    # cv_bridge compressed path, which can reject valid JPEG
                    # format strings published by the head camera.
                    encoded = np.frombuffer(bytes(message.data), dtype=np.uint8)
                    bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                    if bgr is None:
                        raise ValueError("OpenCV could not decode compressed image bytes")
                else:
                    bgr = self.bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
            except Exception as exc:  # noqa: BLE001
                self.get_logger().warning(f"Could not decode {name} image: {exc}")
                return

            self.images[name] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            timestamp_s = self._message_timestamp_s(message)
            self._append_sample(name, timestamp_s, self.images[name])
        return callback

    def spin_for(self, seconds: float) -> None:
        deadline = time.monotonic() + seconds
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=min(0.01, deadline - time.monotonic()))
            self.capture_rollout_frame()

    def capture_rollout_frame(self, observation: ObservationBundle | None = None) -> None:
        if not self.rollout_recorder.recording_frames:
            return
        if observation is None:
            observation = self.synchronized_observation(
                self._capture_skew_s, self._capture_age_s,
                after_stamp=self._last_rollout_stamp,
            )
        if observation is None or observation.timestamp_s <= self._last_rollout_stamp:
            return
        if observation.timestamp_s - self._last_rollout_stamp < 1.0 / 15.0:
            return
        self._last_rollout_stamp = observation.timestamp_s
        held = self.held_targets()
        target = observation.state if held is None else held
        self.rollout_recorder.submit_frame(
            observation.timestamp_s, observation.state, observation.images,
            observation.sample_timestamps_s, target, held is not None,
        )

    def reset_rollout_stamp(self) -> None:
        self._last_rollout_stamp = float("-inf")

    def local_state(self) -> np.ndarray:
        if not all(x is not None for x in self.arm.values()) or not all(x is not None for x in self.gripper.values()):
            raise RuntimeError("Robot arm/gripper feedback is incomplete")
        return np.concatenate((self.arm["left"], self.arm["right"], np.asarray([self.gripper["left"], self.gripper["right"]], dtype=np.float32)))

    def synchronized_observation(
        self, max_skew_s: float, max_age_s: float, after_stamp: float | None = None,
    ) -> ObservationBundle | None:
        """Return the newest observation whose full timestamp span is ``max_skew_s``.

        A head-camera timestamp is used as the candidate capture time. For every
        other stream, the nearest buffered ROS-stamped sample is selected. The
        earliest and latest selected samples must be no more than ``max_skew_s``
        apart. This makes the state and all three images refer to one bounded
        time interval, rather than independently reading whatever callback
        arrived most recently.
        """
        if max_skew_s <= 0.0 or any(not samples for samples in self._samples.values()):
            return None

        now = self.get_clock().now().nanoseconds / 1e9
        cutoff_stamp = self._last_observation_stamp if after_stamp is None else after_stamp
        for target_time, head in reversed(self._samples["head"]):
            if target_time <= cutoff_stamp:
                continue
            selected: dict[str, tuple[float, np.ndarray]] = {"head": (target_time, head)}
            for name, samples in self._samples.items():
                if name == "head":
                    continue
                nearest = min(samples, key=lambda item: abs(item[0] - target_time))
                selected[name] = nearest
            selected_times = [sample[0] for sample in selected.values()]
            if max(selected_times) - min(selected_times) > max_skew_s:
                continue
            if min(selected_times) < now - max_age_s or max(selected_times) > now:
                continue

            state = np.concatenate(
                (
                    selected["left_arm"][1],
                    selected["right_arm"][1],
                    selected["left_gripper"][1],
                    selected["right_gripper"][1],
                )
            ).astype(np.float32, copy=False)
            return ObservationBundle(
                timestamp_s=target_time,
                state=state.copy(),
                images={name: selected[name][1].copy() for name in ("head", "left_wrist", "right_wrist")},
                sample_timestamps_s={name: value[0] for name, value in selected.items()},
            )
        return None

    @staticmethod
    def _encode_jpeg(image: np.ndarray) -> str:
        success, encoded = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR), (cv2.IMWRITE_JPEG_QUALITY, 90))
        if not success:
            raise RuntimeError("Failed to encode camera image")
        return base64.b64encode(encoded.tobytes()).decode("ascii")

    def make_payload(
        self,
        prompt: str,
        run_id: str,
        demo_id: str,
        chunk_id: int,
        observation: ObservationBundle,
    ) -> dict[str, object]:
        return {
            "prompt": prompt,
            "state": observation.state.tolist(),
            "head_image": self._encode_jpeg(observation.images["head"]),
            "left_wrist_image": self._encode_jpeg(observation.images["left_wrist"]),
            "right_wrist_image": self._encode_jpeg(observation.images["right_wrist"]),
            "run_id": run_id,
            "demo_id": demo_id,
            "chunk_id": chunk_id,
        }

    def prepare_targets(
        self,
        action: np.ndarray,
        gripper_min: float,
        gripper_max: float,
        current_state: np.ndarray | None = None,
    ) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (ACTION_DIM,) or not np.isfinite(action).all():
            raise ValueError(f"Expected one finite {ACTION_DIM}-D action, got {action.shape}")
        target = self.local_state() if current_state is None else np.asarray(current_state, dtype=np.float32).copy()
        target[:12] += action[:12]
        target[:6] = np.clip(target[:6], JOINT_LIMITS_RAD[:, 0], JOINT_LIMITS_RAD[:, 1])
        target[6:12] = np.clip(target[6:12], JOINT_LIMITS_RAD[:, 0], JOINT_LIMITS_RAD[:, 1])
        target[12:14] = np.clip(action[12:14], gripper_min, gripper_max)
        return target

    def prepare_absolute_targets(
        self,
        target: np.ndarray,
        gripper_min: float,
        gripper_max: float,
    ) -> np.ndarray:
        """Clip an already-absolute 14-D target to safety limits, with no add."""
        target = np.asarray(target, dtype=np.float32).copy()
        if target.shape != (ACTION_DIM,) or not np.isfinite(target).all():
            raise ValueError(f"Expected one finite {ACTION_DIM}-D target, got {target.shape}")
        target[:6] = np.clip(target[:6], JOINT_LIMITS_RAD[:, 0], JOINT_LIMITS_RAD[:, 1])
        target[6:12] = np.clip(target[6:12], JOINT_LIMITS_RAD[:, 0], JOINT_LIMITS_RAD[:, 1])
        target[12:14] = np.clip(target[12:14], gripper_min, gripper_max)
        return target

    def set_active_demo(self, demo_id: str | None) -> None:
        with self._command_lock:
            self._active_demo_id = demo_id

    def wait_for_right_arm_convergence(
        self,
        target: np.ndarray,
        enabled: threading.Event,
        minimum_wait_s: float,
        timeout_s: float,
        tolerance_rad: float,
        consecutive_samples: int,
    ) -> tuple[bool, float, float]:
        """Wait for fresh right-arm feedback to converge on one published target."""
        started = time.monotonic()
        deadline = started + max(minimum_wait_s, timeout_s)
        last_sequence = self._arm_feedback_sequence["right"]
        consecutive = 0
        max_error = float("inf")

        while rclpy.ok() and enabled.is_set():
            now = time.monotonic()
            if now >= deadline:
                break
            self.spin_for(min(0.01, deadline - now))
            if time.monotonic() - started < minimum_wait_s:
                continue
            sequence = self._arm_feedback_sequence["right"]
            if sequence == last_sequence:
                continue
            last_sequence = sequence
            measured = self.local_state()[6:12]
            max_error = float(np.max(np.abs(measured - target[6:12])))
            consecutive = consecutive + 1 if max_error <= tolerance_rad else 0
            if consecutive >= consecutive_samples:
                return True, time.monotonic() - started, max_error

        if np.isinf(max_error):
            max_error = float(np.max(np.abs(self.local_state()[6:12] - target[6:12])))
        return False, time.monotonic() - started, max_error

    def _publish_targets_locked(self, target: np.ndarray, source: str) -> None:
        stamp = self.get_clock().now().to_msg()
        for arm_index, side in enumerate(("left", "right")):
            start = arm_index * JOINTS_PER_ARM
            arm = JointState()
            arm.header.stamp = stamp
            arm.name = self.arm_names[side] or []
            arm.position = target[start:start + JOINTS_PER_ARM].tolist()
            self.arm_publishers[side].publish(arm)
            gripper = JointState()
            gripper.header.stamp = stamp
            gripper.position = [float(target[12 + arm_index])]
            self.gripper_publishers[side].publish(gripper)
        self.rollout_recorder.submit_publication({
            "monotonic_ns": time.monotonic_ns(),
            "wall_time": wall_time(),
            "source": source,
            "demo_id": self._active_demo_id,
            "chunk_id": self._last_target_chunk_id,
            "action_step": self._last_target_action_step,
            "target": target.tolist(),
        })

    def publish_targets(self, target: np.ndarray) -> None:
        target = np.asarray(target, dtype=np.float32).copy()
        with self._command_lock:
            self._last_targets = target
            self._last_target_chunk_id = None
            self._last_target_action_step = None
            self._publish_targets_locked(target, "direct")

    def publish_targets_if_enabled(
        self,
        target: np.ndarray,
        enabled: threading.Event,
        chunk_id: int | None = None,
        action_step: int | None = None,
    ) -> bool:
        target = np.asarray(target, dtype=np.float32).copy()
        with self._command_lock:
            if not enabled.is_set():
                return False
            self._last_targets = target
            self._last_target_chunk_id = chunk_id
            self._last_target_action_step = action_step
            self._publish_targets_locked(target, "action_step")
        return True

    def start_hold(self) -> bool:
        with self._command_lock:
            if self._last_targets is None:
                try:
                    self._last_targets = self.local_state().copy()
                except RuntimeError:
                    return False
            self._publish_targets_locked(self._last_targets, "hold_start")
        self._hold_enabled.set()
        return True

    def hold_current_feedback(self) -> bool:
        with self._command_lock:
            try:
                self._last_targets = self.local_state().copy()
            except RuntimeError:
                return False
            self._last_target_chunk_id = None
            self._last_target_action_step = None
            self._publish_targets_locked(self._last_targets, "hold_current_feedback")
        self._hold_enabled.set()
        return True

    def stop_hold(self) -> None:
        self._hold_enabled.clear()
        with self._command_lock:
            pass

    def held_targets(self) -> np.ndarray | None:
        with self._command_lock:
            return None if self._last_targets is None else self._last_targets.copy()

    def _hold_loop(self) -> None:
        period = 1.0 / self._hold_hz
        while not self._hold_closed.is_set():
            if not self._hold_enabled.wait(timeout=period):
                continue
            tick_started = time.monotonic()
            with self._command_lock:
                if self._last_targets is not None:
                    self._publish_targets_locked(self._last_targets, "hold_thread")
            remaining = period - (time.monotonic() - tick_started)
            if remaining > 0:
                self._hold_closed.wait(remaining)

    def close(self) -> None:
        self._hold_closed.set()
        self._hold_enabled.set()
        self._hold_thread.join()


class OperatorInput:
    """Sole stdin reader for demo toggles and scores."""

    def __init__(self, on_stop: Callable[[], object] | None = None) -> None:
        self.on_stop = on_stop
        self.running = threading.Event()
        self.closed = threading.Event()
        self.stop_handled = threading.Event()
        self._lock = threading.Lock()
        self._mode = "idle"
        self._scores: queue.Queue[int] = queue.Queue()
        self.worker = threading.Thread(target=self._run, daemon=True)
        self.worker.start()

    def _run(self) -> None:
        while not self.closed.is_set():
            try:
                line = input()
            except EOFError:
                self.closed.set()
                return
            if self.closed.is_set():
                return
            stop = False
            with self._lock:
                if self._mode == "idle":
                    self._mode = "arming"
                    self.running.set()
                    self.stop_handled.clear()
                    print("\nSTART requested.")
                elif self._mode == "arming":
                    self._mode = "idle"
                    self.running.clear()
                    print("\nStart cancelled before a synchronized frame was available.")
                elif self._mode == "recording":
                    self._mode = "resetting"
                    self.running.clear()
                    stop = True
                elif self._mode == "resetting":
                    print("Reset in progress; enter the score after it finishes.")
                elif self._mode == "scoring":
                    value = line.strip()
                    if value.isdecimal() and 1 <= int(value) <= 100:
                        self._scores.put(int(value))
                        self._mode = "busy"
                        print(f"Accuracy score: {value}/100")
                    else:
                        print("Enter a whole-number accuracy score from 1 to 100: ", end="", flush=True)
                else:
                    print("Finishing reset and rollout checkpoint; input ignored.")
            if stop:
                self._finish_stop()

    def _finish_stop(self) -> None:
        try:
            if self.on_stop is not None:
                self.on_stop()
        finally:
            self.stop_handled.set()
        print("\nSTOP requested. Holding latest feedback, then resetting.")

    def request_score(self) -> None:
        """Open score entry once reset output is done, so the prompt stays visible."""
        with self._lock:
            if self._mode != "resetting":
                raise RuntimeError(f"Cannot request a score while input is {self._mode}")
            self._mode = "scoring"
            print("Enter a whole-number accuracy score from 1 to 100: ", end="", flush=True)

    def request_stop(self) -> None:
        with self._lock:
            if self._mode != "recording":
                return
            self._mode = "resetting"
            self.running.clear()
        self._finish_stop()

    def mark_recording(self, on_start: Callable[[], None]) -> bool:
        with self._lock:
            if self._mode != "arming" or not self.running.is_set():
                return False
            on_start()
            self._mode = "recording"
            return True

    def enable_start(self) -> None:
        with self._lock:
            if self._mode not in ("busy", "idle"):
                raise RuntimeError("Cannot accept a new demo before scoring")
            self._mode = "idle"

    def wait_for_score(self) -> int:
        while not self.closed.is_set():
            try:
                return self._scores.get(timeout=0.1)
            except queue.Empty:
                continue
        raise RuntimeError("Score input closed")

    def close(self) -> None:
        self.closed.set()


def reset_to_initial_position(robot: RobotIO, execute: bool) -> None:
    """Run the shared arm, gripper, and torso reset script."""
    if not RESET_SCRIPT.is_file():
        raise FileNotFoundError(f"Reset script not found: {RESET_SCRIPT}")

    target_file = Path(os.environ.get(
        "R1LITE_INITIAL_POSITION_FILE", str(RESET_SCRIPT.parent / "initial_robot_position.json")
    )).expanduser()
    reset_target = np.asarray(json.loads(target_file.read_text(encoding="utf-8"))["position"], dtype=np.float32)
    if reset_target.shape != (18,) or not np.isfinite(reset_target).all():
        raise ValueError(f"Expected an 18-D finite reset target in {target_file}")

    command = ["bash", str(RESET_SCRIPT)]
    print(f"Running reset script: {RESET_SCRIPT}")
    if not execute:
        subprocess.run(command, check=True)
        print("Reset finished.")
        return

    handoff_pose = robot.held_targets()
    if handoff_pose is None:
        raise RuntimeError("No held feedback pose is available for reset handoff")
    command.extend(
        [
            "--execute",
            "--wait-for-takeover",
            "--hold-final",
            "--start-arm-gripper",
            *(f"{value:.9g}" for value in handoff_pose),
        ]
    )
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    inference_has_control = True

    def wait_for_marker(marker: str) -> None:
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            if line.strip() == marker:
                return
        return_code = process.wait()
        raise RuntimeError(f"Reset process exited with code {return_code} before reporting {marker}")

    try:
        wait_for_marker("RESET_READY")
        process.send_signal(signal.SIGUSR1)
        wait_for_marker("RESET_TOOK_OVER")
        robot.stop_hold()
        inference_has_control = False

        wait_for_marker("RESET_HOLDING_FINAL")
        robot.spin_for(0.2)
        robot.publish_targets(reset_target[:ACTION_DIM])
        robot.start_hold()
        inference_has_control = True
        process.send_signal(signal.SIGINT)
        process.wait(timeout=5.0)
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        if not inference_has_control:
            robot.spin_for(0.05)
            robot.publish_targets(robot.local_state())
            robot.start_hold()
        if process.returncode not in (0, -signal.SIGINT):
            raise subprocess.CalledProcessError(process.returncode, command)

    print("Reset finished.")


def wall_time() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def post_json(endpoint: str, payload: dict[str, object], timeout_s: float) -> dict[str, object]:
    request = Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise RuntimeError(f"Policy server HTTP {exc.code}: {exc.read().decode(errors='replace')}") from exc
    except URLError as exc:
        raise RuntimeError(f"Could not reach policy server: {exc.reason}") from exc


def ask_num_demos() -> int:
    while True:
        try:
            value = int(input("How many demos do you want to run? ").strip())
        except ValueError:
            print("Enter a positive whole number.")
            continue
        if value > 0:
            return value
        print("Enter a positive whole number.")


def request_actions(
    endpoint: str,
    payload: dict[str, object],
    timeout_s: float,
    expected_action_mode: str,
) -> np.ndarray:
    result = post_json(endpoint, payload, timeout_s)
    server_action_mode = str(result.get("action_semantics", "relative"))
    if server_action_mode != expected_action_mode:
        raise RuntimeError(
            "Client/server action-mode mismatch: "
            f"client={expected_action_mode!r}, server={server_action_mode!r}"
        )
    actions = np.asarray(result.get("actions"), dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != ACTION_DIM or not np.isfinite(actions).all():
        raise RuntimeError(f"Policy returned invalid actions: {actions.shape}")
    return actions


def request_actions_while_spinning(
    robot: RobotIO,
    endpoint: str,
    payload: dict[str, object],
    timeout_s: float,
    expected_action_mode: str,
) -> np.ndarray:
    """Run the blocking HTTP request without stopping ROS feedback callbacks.

    Target holding remains owned by ``RobotIO._hold_loop``. The main thread
    keeps spinning this node until the inference worker finishes, so camera and
    joint feedback continue to update throughout the server request.
    """
    result_queue: queue.Queue[tuple[np.ndarray | None, BaseException | None]] = queue.Queue(maxsize=1)

    def infer() -> None:
        try:
            result_queue.put(
                (request_actions(endpoint, payload, timeout_s, expected_action_mode), None)
            )
        except BaseException as exc:  # noqa: BLE001 - propagate worker failure on the main thread
            result_queue.put((None, exc))

    worker = threading.Thread(target=infer, name="fastwam-inference-request", daemon=True)
    worker.start()
    while worker.is_alive():
        robot.spin_for(0.01)
        if not rclpy.ok():
            time.sleep(0.01)
    worker.join()

    actions, error = result_queue.get_nowait()
    if error is not None:
        raise error
    assert actions is not None
    return actions


def load_replay_actions(path: Path, demo_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Load one HDF5 demo's actions in two forms:

    - relative: arm deltas from ``actions``, absolute gripper targets from
      ``absolute_actions`` (the model's own convention; see
      serve_fastwam_r1lite.py's action_semantics).
    - absolute: the recorded ``absolute_actions`` targets exactly as
      commanded during collection, for all 14 dims.
    """
    demo_name = demo_name if demo_name.startswith("demo_") else f"demo_{demo_name}"
    with h5py.File(path, "r") as file:
        if file.attrs.get("schema_kind") == "fastwam_live_rollout_v1":
            raise ValueError(
                "Live rollout HDF5 files use viewer target-minus-state actions and cannot be "
                "replayed as collection demonstrations"
            )
        key = f"data/{demo_name}"
        if key not in file:
            available = sorted(file.get("data", {}).keys())
            raise KeyError(f"{demo_name!r} is not in {path}. Available demos: {available}")
        demo = file[key]
        if "actions" not in demo or "absolute_actions" not in demo or "joint_states" not in demo:
            raise KeyError(f"{key} must contain actions, absolute_actions, and joint_states")
        actions = np.asarray(demo["actions"], dtype=np.float32)
        absolute_actions = np.asarray(demo["absolute_actions"], dtype=np.float32)
        initial_state = np.asarray(demo["joint_states"][0], dtype=np.float32)
        task = str(demo.attrs.get("task_description", demo.attrs.get("subtask_key", "unknown")))

    if actions.shape != absolute_actions.shape or actions.ndim != 2 or actions.shape[1] < ACTION_DIM:
        raise ValueError(f"Expected actions/absolute_actions shaped [T, >={ACTION_DIM}], got {actions.shape}")
    if initial_state.shape != (ACTION_DIM,):
        raise ValueError(f"Expected a {ACTION_DIM}-D initial joint state, got {initial_state.shape}")

    absolute_targets = absolute_actions[:, :ACTION_DIM].copy()
    relative_actions = actions[:, :ACTION_DIM].copy()
    relative_actions[:, 12:14] = absolute_targets[:, 12:14]
    if not np.isfinite(relative_actions).all() or not np.isfinite(absolute_targets).all() or not np.isfinite(initial_state).all():
        raise ValueError("The selected demo contains non-finite state or action values")
    return relative_actions, absolute_targets, initial_state, task


def move_to_replay_start(
    robot: RobotIO,
    hotkey: OperatorInput,
    target: np.ndarray,
    args: argparse.Namespace,
) -> bool:
    start = robot.local_state().copy()
    print("Replay initial state:", np.array2string(target, precision=4, suppress_small=True))
    print(
        f"Moving to it over {args.replay_initial_move_s:.1f}s; "
        f"maximum arm travel={np.max(np.abs(target[:12] - start[:12])):.4f} rad"
    )
    if not args.execute:
        print("DRY RUN: no initial-position targets published.")
        return True

    steps = max(1, int(np.ceil(args.replay_initial_move_s * args.control_hz)))
    period = 1.0 / args.control_hz
    for step in range(1, steps + 1):
        if not hotkey.running.is_set():
            return False
        tick_started = time.monotonic()
        command = start + (step / steps) * (target - start)
        if not robot.publish_targets_if_enabled(command, hotkey.running):
            return False
        robot.start_hold()
        remaining = period - (time.monotonic() - tick_started)
        if remaining > 0:
            robot.spin_for(remaining)

    robot.spin_for(0.3)
    measured = robot.local_state().copy()
    arm_error = float(np.max(np.abs(measured[:12] - target[:12])))
    gripper_error = float(np.max(np.abs(measured[12:14] - target[12:14])))
    print(f"Replay initial-pose error: arms={arm_error:.5f} rad, grippers={gripper_error:.3f}")
    if arm_error > args.replay_initial_arm_tolerance or gripper_error > args.replay_initial_gripper_tolerance:
        raise RuntimeError(
            "Robot did not reach the replay demo's initial pose: limits are "
            f"{args.replay_initial_arm_tolerance:.3f} rad and "
            f"{args.replay_initial_gripper_tolerance:.3f} gripper units"
        )
    return hotkey.running.is_set()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--server", default=None, help="e.g. http://192.168.1.10:8000 (required unless --replay-hdf5 is set)"
    )
    parser.add_argument("--prompt", required=True)
    parser.add_argument(
        "--replay-hdf5",
        type=Path,
        default=None,
        help="Replay one HDF5 demo's stored actions instead of querying a FastWAM server",
    )
    parser.add_argument(
        "--replay-demo", default="demo_0", help="Demo to replay when --replay-hdf5 is set, e.g. demo_12 or 12"
    )
    parser.add_argument(
        "--replay-mode",
        choices=("relative", "absolute"),
        default="relative",
        help=(
            "relative (default): apply the model's own convention (arm deltas onto live "
            "feedback, absolute gripper) each step. absolute: republish the recorded "
            "absolute_actions target exactly, ignoring live feedback for target computation."
        ),
    )
    parser.add_argument("--replay-initial-move-s", type=float, default=5.0)
    parser.add_argument("--replay-initial-arm-tolerance", type=float, default=0.12)
    parser.add_argument("--replay-initial-gripper-tolerance", type=float, default=15.0)
    parser.add_argument("--execute", action="store_true", help="Publish targets; default is dry-run")
    parser.add_argument(
        "--action-mode",
        choices=("relative", "absolute"),
        default="absolute",
        help=(
            "How to interpret live model arm outputs. relative adds the first 12 values "
            "to live feedback; absolute publishes all 14 values as absolute targets."
        ),
    )
    parser.add_argument(
        "--control-hz",
        type=float,
        default=20.0,
        help="Action execution rate (default: the 20 Hz rate used by the processed R1 Lite training data)",
    )
    parser.add_argument(
        "--replan-steps",
        type=int,
        default=None,
        help="Actions to execute per query; default executes the complete horizon returned by the trained server",
    )
    parser.add_argument(
        "--wait-for-convergence",
        action="store_true",
        help="Wait after each action until the right arm reaches its target or the settle timeout expires",
    )
    parser.add_argument(
        "--settle-tolerance-deg",
        type=float,
        default=2.0,
        help="Maximum right-arm joint error considered settled (default: 2 degrees)",
    )
    parser.add_argument(
        "--settle-timeout-s",
        type=float,
        default=0.25,
        help="Maximum total time allowed for each action step in convergence mode (default: 0.25 s)",
    )
    parser.add_argument(
        "--settle-consecutive-samples",
        type=int,
        default=2,
        help="Fresh in-tolerance feedback samples required before advancing (default: 2)",
    )
    parser.add_argument("--request-timeout", type=float, default=30.0)
    parser.add_argument(
        "--rollouts-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "rollouts" / "r1lite",
        help="Robot-local root for dated FastWAM rollout sessions",
    )
    parser.add_argument(
        "--policy-config",
        default="fastwam",
        help="Directory name identifying the served FastWAM policy/config",
    )
    parser.add_argument("--gripper-min", type=float, default=0.0)
    parser.add_argument("--gripper-max", type=float, default=100.0)
    parser.add_argument("--wrist-transport", choices=("raw", "compressed"), default="raw")
    parser.add_argument(
        "--max-observation-skew-ms",
        type=float,
        default=50.0,
        help="Maximum earliest-to-latest ROS-stamp span within one model observation (default: 50 ms)",
    )
    parser.add_argument(
        "--max-observation-age-ms", type=float, default=250.0,
        help="Oldest permitted sample age at selection, using the ROS clock (default: 250 ms)",
    )
    parser.add_argument(
        "--observation-buffer-s",
        type=float,
        default=3.0,
        help="Timestamped ROS history retained while finding a synchronized observation (default: 3 s)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (
        args.control_hz <= 0
        or (args.replan_steps is not None and args.replan_steps <= 0)
        or args.gripper_min > args.gripper_max
        or args.max_observation_skew_ms <= 0
        or args.observation_buffer_s <= 0
        or args.max_observation_age_ms <= 0
        or args.settle_tolerance_deg <= 0
        or args.settle_timeout_s <= 0
        or args.settle_consecutive_samples <= 0
    ):
        raise ValueError("Invalid control, replan, delta, or gripper limits")
    if args.replay_hdf5 is None and args.server is None:
        raise ValueError("--server is required unless --replay-hdf5 is set")
    server = args.server.rstrip("/") if args.server else None
    endpoint = server + "/infer" if server else None
    period = 1.0 / args.control_hz
    replay_actions = replay_initial_state = replay_task = None
    if args.replay_hdf5 is not None:
        relative_actions, absolute_targets, replay_initial_state, replay_task = load_replay_actions(
            args.replay_hdf5, args.replay_demo
        )
        replay_actions = relative_actions if args.replay_mode == "relative" else absolute_targets
        print(
            f"Replay source: {args.replay_hdf5} [{args.replay_demo}] ({replay_task}); "
            f"{len(replay_actions)} actions; mode={args.replay_mode}"
        )
    num_demos = ask_num_demos()
    run_metadata = {
        "policy_config": args.policy_config,
        "prompt": args.prompt,
        "num_demos": num_demos,
        "execute": args.execute,
        "control_hz": args.control_hz,
        "wait_for_convergence": args.wait_for_convergence,
        "settle_tolerance_deg": args.settle_tolerance_deg,
        "settle_timeout_s": args.settle_timeout_s,
        "settle_consecutive_samples": args.settle_consecutive_samples,
        "max_observation_skew_ms": args.max_observation_skew_ms,
        "max_observation_age_ms": args.max_observation_age_ms,
        "created_at": wall_time(),
        "host": os.uname().nodename,
        "replay_hdf5": str(args.replay_hdf5) if args.replay_hdf5 else None,
        "replay_demo": args.replay_demo if args.replay_hdf5 else None,
    }
    session = RolloutSession.create(args.rollouts_dir, args.policy_config)
    if args.replay_hdf5 is not None:
        run_id = f"replay_{_safe_path_component(args.replay_demo)}_{datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')}"
    else:
        run_response = post_json(
            server + "/runs/start",
            {
                "prompt": args.prompt,
                "num_demos": num_demos,
                "execute": args.execute,
                "control_hz": args.control_hz,
            },
            args.request_timeout,
        )
        run_id = str(run_response["run_id"])
    run_metadata["run_id"] = run_id
    action_queue: deque[tuple[int, int, np.ndarray]] = deque()
    rclpy.init()
    recorder = HDF5RolloutRecorder(session.root, run_metadata, num_demos)
    try:
        robot = RobotIO(
            args.wrist_transport,
            max(args.control_hz, 20.0),
            args.observation_buffer_s,
            recorder,
            args.max_observation_skew_ms / 1000.0,
            args.max_observation_age_ms / 1000.0,
        )
    except Exception:
        recorder.close()
        rclpy.shutdown()
        raise
    print("EXECUTE ENABLED" if args.execute else "DRY RUN: add --execute to move the robot")
    replan_description = args.replan_steps if args.replan_steps is not None else "full trained horizon"
    if args.replay_hdf5 is not None:
        print(f"REPLAY MODE: {args.replay_hdf5} [{args.replay_demo}]; replan steps: {replan_description}")
    else:
        print(
            f"FastWAM server: {endpoint}; replan steps: {replan_description}; "
            f"max observation skew: {args.max_observation_skew_ms:.1f} ms"
        )
    if args.wait_for_convergence:
        print(
            "CONVERGENCE MODE: right-arm error <= "
            f"{args.settle_tolerance_deg:.2f} deg for {args.settle_consecutive_samples} fresh samples; "
            f"per-step timeout {args.settle_timeout_s:.3f}s."
        )
    print(f"Recording {num_demos} demos under {session.root}.")
    print("Press Enter to start demo_1. Press Enter again to stop and reset.")

    def on_stop() -> None:
        recorder.stop_frames()
        robot.set_active_demo(None)
        if args.execute:
            robot.hold_current_feedback()

    hotkey = OperatorInput(on_stop=on_stop)
    inference_started = False
    completed_demos = 0
    success_count = 0
    chunk_id = 0
    replay_index = 0
    first_checkpoint = None

    def current_demo_id() -> str:
        return f"demo_{completed_demos + 1}"

    def log_event(event: str, details: dict[str, object] | None = None) -> None:
        recorder.submit_event(current_demo_id(), {
            "event": event, "client_time": wall_time(), "details": details or {},
        })

    try:
        while rclpy.ok():
            if hotkey.closed.is_set():
                raise EOFError("Operator input closed before the run completed")
            if not hotkey.running.is_set():
                action_queue.clear()
                if inference_started:
                    hotkey.stop_handled.wait()
                    log_event("demo_stopped", {"chunks_queried": chunk_id})
                    demo_id = current_demo_id()
                    first_checkpoint = recorder.seal_demo(demo_id, time.time())
                    reset_ok = False
                    for attempt in (1, 2):
                        try:
                            reset_to_initial_position(robot, args.execute)
                            reset_ok = True
                            break
                        except Exception as exc:  # noqa: BLE001
                            print(f"Reset attempt {attempt}/2 failed: {exc}")
                            if args.execute:
                                robot.hold_current_feedback()
                            if attempt == 1:
                                robot.spin_for(1.0)
                    if not reset_ok:
                        print(
                            "Automatic reset failed twice; holding last feedback instead of "
                            "resetting. Scoring and checkpointing this demo before another start."
                        )
                    recorder.stop_publications()
                    recorder.submit_post_event(demo_id, {
                        "event": "reset_finished" if reset_ok else "reset_failed",
                        "client_time": wall_time(), "details": {},
                    })
                    hotkey.request_score()
                    score = hotkey.wait_for_score()
                    first_checkpoint.wait()
                    final_checkpoint = recorder.finalize_demo(demo_id, score, reset_ok)
                    final_checkpoint.wait()
                    first_checkpoint = None
                    success_count += int(score == 100)
                    inference_started = False
                    completed_demos += 1
                    if completed_demos >= num_demos:
                        print(f"Completed requested {num_demos} demos in {run_id}.")
                        print(f"Success rate: {success_count}/{num_demos} = {100.0 * success_count / num_demos:.1f}%")
                        break
                    hotkey.enable_start()
                    print(f"Press Enter to start {current_demo_id()}.")
                robot.spin_for(0.05)
                continue

            if not inference_started:
                robot.spin_for(0.05)
                initial_observation = robot.synchronized_observation(
                    args.max_observation_skew_ms / 1000.0,
                    args.max_observation_age_ms / 1000.0,
                    after_stamp=float("-inf"),
                )
                if initial_observation is None:
                    continue

                def start_demo(observation: ObservationBundle = initial_observation) -> None:
                    robot.set_active_demo(current_demo_id())
                    recorder.begin_demo(current_demo_id(), args.prompt, args.control_hz, time.time())
                    robot.reset_rollout_stamp()
                    robot.capture_rollout_frame(observation)
                    log_event("demo_started")

                if not hotkey.mark_recording(start_demo):
                    continue
                inference_started = True
                chunk_id = 0
                print(f"{current_demo_id()} inference started.")
                if args.replay_hdf5 is not None:
                    replay_index = 0
                    if not move_to_replay_start(robot, hotkey, replay_initial_state, args):
                        continue

            if not action_queue and args.replay_hdf5 is not None:
                n_execute = len(replay_actions) - replay_index
                if args.replan_steps is not None:
                    n_execute = min(n_execute, args.replan_steps)
                if n_execute <= 0:
                    print("Replay demo exhausted; stopping.")
                    if args.execute:
                        robot.hold_current_feedback()
                    hotkey.request_stop()
                    continue
                chunk_id += 1
                chunk = replay_actions[replay_index : replay_index + n_execute]
                replay_index += n_execute
                action_queue.extend(
                    (chunk_id, action_step, action.copy())
                    for action_step, action in enumerate(chunk, start=1)
                )
                print(f"Replaying actions {replay_index - n_execute + 1}-{replay_index}/{len(replay_actions)}")
                robot.spin_for(0.05)

            if not action_queue:
                robot.spin_for(0.10)
                observation = robot.synchronized_observation(
                    args.max_observation_skew_ms / 1000.0, args.max_observation_age_ms / 1000.0
                )
                if observation is None:
                    print(
                        "Waiting for a temporally synchronized state + three-camera observation "
                        f"within {args.max_observation_skew_ms:.1f} ms, "
                        f"no older than {args.max_observation_age_ms:.1f} ms..."
                    )
                    continue
                chunk_id += 1
                started = time.monotonic()
                payload = robot.make_payload(args.prompt, run_id, current_demo_id(), chunk_id, observation)
                if not hotkey.running.is_set():
                    continue
                recorder.submit_query(
                    current_demo_id(), chunk_id, payload, observation.sample_timestamps_s,
                )
                robot._last_observation_stamp = observation.timestamp_s
                log_event("observation_selected", {
                    "chunk_id": chunk_id, "timestamps_s": observation.sample_timestamps_s,
                    "span_ms": 1000.0 * (max(observation.sample_timestamps_s.values()) - min(observation.sample_timestamps_s.values())),
                })
                if args.execute and robot.held_targets() is None and not robot.hold_current_feedback():
                    raise RuntimeError("Could not establish the initial hold target before inference")
                if not hotkey.running.is_set():
                    continue
                actions = request_actions_while_spinning(
                    robot,
                    endpoint,
                    payload,
                    args.request_timeout,
                    args.action_mode,
                )
                n_execute = len(actions) if args.replan_steps is None else min(args.replan_steps, len(actions))
                if not hotkey.running.is_set():
                    print("Action chunk returned after STOP; discarding it.")
                    continue
                action_queue.extend(
                    (chunk_id, action_step, action.copy())
                    for action_step, action in enumerate(actions[:n_execute], start=1)
                )
                print(f"Received {len(actions)} actions in {time.monotonic() - started:.3f}s; executing {n_execute}.")

                # Allow any callbacks queued at request completion to run.
                robot.spin_for(0.05)

            if not hotkey.running.is_set():
                action_queue.clear()
                continue

            tick_started = time.monotonic()
            queued_chunk_id, action_step, action = action_queue.popleft()
            measured_before = robot.local_state().copy()
            if (
                (args.replay_hdf5 is not None and args.replay_mode == "absolute")
                or (args.replay_hdf5 is None and args.action_mode == "absolute")
            ):
                target = robot.prepare_absolute_targets(action, args.gripper_min, args.gripper_max)
            else:
                target = robot.prepare_targets(
                    action,
                    args.gripper_min,
                    args.gripper_max,
                    current_state=measured_before,
                )
            published = False
            if args.execute:
                if not robot.publish_targets_if_enabled(
                    target,
                    hotkey.running,
                    chunk_id=queued_chunk_id,
                    action_step=action_step,
                ):
                    action_queue.clear()
                    continue
                robot.start_hold()
                published = True
            else:
                print("[dry-run] target=", np.array2string(target, precision=4))
            settled: bool | None = None
            settle_elapsed_s = 0.0
            right_arm_max_error_rad: float | None = None
            if args.execute and args.wait_for_convergence:
                settled, settle_elapsed_s, right_arm_max_error_rad = robot.wait_for_right_arm_convergence(
                    target,
                    hotkey.running,
                    minimum_wait_s=period,
                    timeout_s=args.settle_timeout_s,
                    tolerance_rad=np.deg2rad(args.settle_tolerance_deg),
                    consecutive_samples=args.settle_consecutive_samples,
                )
            else:
                remaining = period - (time.monotonic() - tick_started)
                if remaining > 0:
                    robot.spin_for(remaining)
            measured_after = robot.local_state().copy()
            recorder.submit_step(
                current_demo_id(),
                {
                    "run_id": run_id,
                    "demo_id": current_demo_id(),
                    "chunk_id": queued_chunk_id,
                    "action_step": action_step,
                    "client_time": wall_time(),
                    "model_action": action.tolist(),
                    "action_mode": args.replay_mode if args.replay_hdf5 is not None else args.action_mode,
                    "measured_before": measured_before.tolist(),
                    "commanded_target": target.tolist(),
                    "measured_after": measured_after.tolist(),
                    "published": published,
                    "tick_duration_s": time.monotonic() - tick_started,
                    "convergence_enabled": args.wait_for_convergence,
                    "settled": settled,
                    "settle_elapsed_s": settle_elapsed_s,
                    "right_arm_max_error_rad": right_arm_max_error_rad,
                }
            )
    except KeyboardInterrupt:
        print("\nEmergency exit requested; automatic reset is skipped.")
        if inference_started:
            try:
                recorder.stop_frames()
                robot.set_active_demo(None)
                if args.execute:
                    robot.hold_current_feedback()
                if recorder.phase in ("recording", "stopping"):
                    log_event("emergency_exit", {"chunks_queried": chunk_id})
                    interrupted_checkpoint = recorder.seal_demo(current_demo_id(), time.time())
                    interrupted_checkpoint.wait()
                if recorder.phase == "sealing":
                    if first_checkpoint is not None:
                        first_checkpoint.wait()
                    recorder.finalize_demo(current_demo_id(), None, None).wait()
            except Exception as exc:  # noqa: BLE001
                print(f"INTERRUPTED_ROLLOUT_CHECKPOINT_FAILED: {exc}")
        failed = False
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        print("\nUnhandled error in the control loop; exiting.")
        if inference_started:
            try:
                recorder.stop_frames()
                robot.set_active_demo(None)
                if args.execute:
                    robot.hold_current_feedback()
                if recorder.phase in ("recording", "stopping"):
                    log_event("interrupted_by_error", {"chunks_queried": chunk_id})
                    recorder.seal_demo(current_demo_id(), time.time()).wait()
                if recorder.phase == "sealing":
                    if first_checkpoint is not None:
                        first_checkpoint.wait()
                    recorder.finalize_demo(current_demo_id(), None, None).wait()
            except Exception as exc:  # noqa: BLE001
                print(f"INTERRUPTED_ROLLOUT_CHECKPOINT_FAILED: {exc}")
        failed = True
    else:
        failed = False
    finally:
        hotkey.close()
        robot.close()
        try:
            recorder.close(completed=completed_demos == num_demos and not failed)
        except Exception as exc:  # noqa: BLE001
            print(f"ROLLOUT_CLOSE_FAILED: {exc}")
            failed = True
        robot.destroy_node()
        rclpy.shutdown()

    # OperatorInput's daemon may still be blocked in input() while holding
    # stdin's internal lock, so exit directly after all cleanup is complete.
    os._exit(1 if failed else 0)


if __name__ == "__main__":
    main()
