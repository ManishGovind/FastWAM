"""
pi-WAM (router) inference client for xArm7 (talks to serve_piwam_xarm7.py).

Copied from xarm_fastwam_inference.py: robot interaction, cameras/crops, control
rate, Z floor and chunk execution are unchanged. Differences: each query also
sends episode_id + chunk_id (the server keeps the router's K=32 action history
per episode), and the router's per-chunk stream choice (rgb | depth | flow) and
probabilities are logged and saved per trial in the HDF5:
    router_modality [Q] (bytes), router_probs [Q, S] (float, NaN if unavailable),
    router_query_step [Q] (step index where that chunk starts), attr router_streams.
Rollouts go to experiments/xarm7_piwam/data/rollouts_piwam/ (separate from ../xarm7/data/rollouts/).

- Sends three 480x480 RGB views (primary/exo, wrist, ego) + 7D joint-state
  proprio + task prompt to the server's /infer endpoint.
- The server returns an (H, 7) chunk of ABSOLUTE targets at 15 Hz:
  x, y, z (m), roll, pitch, yaw (rad), gripper (-1 open, +1 closed).
  The first CHUNK_EXECUTE_STEPS are queued and executed at CONTROL_HZ.
"""

import argparse
import base64
import math
import os
import queue
import re
import select
import shutil
import signal
import sys
import termios
import threading
import time
import tty
import uuid
from collections import deque
from dataclasses import dataclass

import cv2
import h5py
import imageio
import json_numpy
import numpy as np
import requests
from xarm.wrapper import XArmAPI
from PIL import Image

# Patch json AFTER all third-party imports finish — patching earlier breaks
# numpy/scipy/transformers' own json.loads calls during their startup.
json_numpy.patch()

DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")


def log_message(message, log_file=None):
    print(message)
    if log_file:
        log_file.write(message + "\n")
        log_file.flush()


def setup_logging(cfg, task_identifier, log_dir, run_id_note=None):
    """Open a per-run log file. Returns (log_file, log_path, run_id)."""
    run_id = f"ENV_EVAL-{task_identifier}-piwam-{DATE_TIME}"
    if run_id_note is not None:
        run_id += f"--{run_id_note}"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, run_id + ".txt")
    return open(log_path, "w"), log_path, run_id


def _text_attr(value):
    if value is None:
        return ""
    return value.decode() if isinstance(value, bytes) else str(value)


def jpeg_encode_image(image, quality=95):
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return np.frombuffer(buf.tobytes(), dtype=np.uint8)


def save_rollout_video(images, idx, success, task_description, log_file=None, fps=24,
                       out_dir=None):
    """Write an MP4 of an episode; the caller moves it into the test dir."""
    out_dir = out_dir or ROLLOUTS_DIR
    os.makedirs(out_dir, exist_ok=True)
    task_tag = task_description.lower().replace(" ", "_")[:50]
    mp4_path = os.path.join(
        out_dir, f"{DATE_TIME}--episode={idx}--success={success}--task={task_tag}.mp4"
    )
    writer = imageio.get_writer(mp4_path, fps=max(fps, 1), codec="libx264",
                                format="FFMPEG", pixelformat="yuv420p")
    for img in images:
        writer.append_data(np.asarray(img))
    writer.close()
    log_message(f"Saved rollout MP4 at path {mp4_path}", log_file)
    return mp4_path


# Exact collection-time crop settings (from xarm.py)
# crop format: [width, height, x_offset, y_offset]
WRISTCAM_CROP = [480, 480, 80, 0]
EXO1_CROP = [248, 258, 194, 104]
EGO_CROP = [696, 696, 323, 0]  # same as xarm_data_collection.py
CROP_TARGET_SIZE = (480, 480)
CAPTURE_FRAME_SIZE = (640, 480)
EGO_CAPTURE_FRAME_SIZE = (1280, 720)
WARMUP_FRAMES = 60
# Training data is 15 Hz and the server's chunk is 32 steps at 15 Hz.
CONTROL_HZ = 15
CHUNK_EXECUTE_STEPS = 16  # first N actions of each chunk are executed before re-querying
# Minimum commanded z (mm), from the latest cosmos-xarm deploy script
# (training_demo_online_inference_rollouts_added.py: max(z, 185)). Training data
# reaches z=174 mm, so this also trims the lowest part of the approach.
Z_FLOOR_MM = 185.0
# The client owns the cameras (a /dev/video device can only be opened once), so it
# publishes the latest model-view frames (480x480 JPEG, exactly what the model sees)
# here for experiments/xarm7/live_overlay.py. "" disables publishing.
LIVE_FRAME_DIR = "/tmp/xarm_live"
LIVE_FRAME_HZ = 10.0
DEFAULT_CKPT_TAG = "piwam_xarm7_20k"

# z held at 300, matching the training distribution: across 130 demos in the
# collected datasets every episode starts at 296.8-301.1mm (std 0.1-0.4mm), so
# anything lower is an out-of-distribution starting state. Smit's original list
# spanned z=240-300, and over 50 episodes the models could not acquire the
# target from below 280mm at all (7/30 touches vs 20/20 above it), which capped
# any achievable score near 40%.
#
# All three axes vary, where Smit's list left y fixed at 0 and duplicated x
# (6 unique values across 10 poses):
#   x: 10 distinct values over 280-330, as in Smit's range
#   y: 10 distinct values over +/-5mm, symmetric (mean exactly 0)
#   z: 10 distinct values over 285-300, evenly spaced
#
# z stays at or above 285. Training demos all start at 296.8-301.1mm, and
# measured over 50 episodes the models touched 20/20 from z >= 280 but only
# 7/30 from below it, so this range sits entirely inside the working band --
# unlike Smit's list, which reached down to 240 and capped scores near 40%.
#
# y is kept small for the same reason: in the carrot demos the end-effector
# sweeps y from 0 to -153mm on its way to the goal, so y carries task-progress
# information much as z does, and a large offset risks the same phase confusion
# that made low-z starts fail. 5mm is ~3% of that sweep.
#
# The three axes are assigned so they are mutually decorrelated (|r| < 0.001)
# and none is monotonic in trial order, so no effect can be confounded with
# another axis or with session drift.
#
# SUPERSEDED — see EXPERIMENT_STARTING_JOINT_POSITIONS below. xArm7 is
# kinematically redundant at a fixed 6-DOF Cartesian pose (one free
# "elbow" DOF), and arm.set_position() resolves that redundancy via
# IK-nearest-solution to wherever the arm's joints happen to be *before*
# the move — not a fixed canonical solution. Measured live: homing to the
# exact (300,0,300,180,0,0) pose these tuples target can land 5-8 degrees
# away (12-30 sigma) from every collected session's actual starting joint
# distribution, even though the Cartesian pose itself matches to <0.1mm.
# Proprio fed to the model is raw joint angles, so a Cartesian-only home is
# not sufficient to reproduce an in-distribution starting state.
# EXPERIMENT_STARTING_POSITIONS = (
#     (300, 0, 300.0, 180, 0, 0),
#     (300, 0, 295, 180, 0, 0),
#     (295, -5, 290.0, 180, 0, 0),
#     (310, 3, 296.7, 180, 0, 0),
#     (290, 1, 293.3, 180, 0, 0),
#     (305, -4, 291.7, 180, 0, 0),
#     (280, 5, 285.0, 180, 0, 0),
#     (320, -1, 286.7, 180, 0, 0),
#     (285, -3, 295.0, 180, 0, 0),
#     (330, 2, 288.3, 180, 0, 0),
# )

# Joint-space starting positions (degrees, 7D: joint_1..joint_7), used with
# arm.set_servo_angle() instead of Cartesian set_position() so each trial's
# proprio is actually drawn from an in-distribution joint configuration,
# not just an in-distribution Cartesian pose.
#
# One shared list of 20 is used across t1/t2/t3 (paper trials), so the same
# 20 configurations are compared across tasks. Reference point = pooled mean
# of the real collected starting joint_states[0] from t1 (put_carrot_in_bowl,
# n=50) + t3 (put_eggplant_in_bowl, n=50): these two sessions' starting
# configs agree to within 0.01-0.04 degrees per joint (same physical elbow
# solution), so pooling them (n=100) gives a solid in-distribution anchor.
# t2 (put_carrot_in_plate) recorded a different elbow solution for the same
# Cartesian pose (joints 1/3/5/7 offset ~1.2-2.6 degrees from the t1/t3
# cluster) — an artifact of IK-nearest-solution history-dependence at
# collection time, not a task-semantic difference (all three tasks share the
# identical (300,0,300,180,0,0) start pose). Centering on the 2-of-3-task
# cluster was chosen over the alternative (splitting into 3 per-task lists)
# so all tasks are evaluated at the same 20 positions.
#
# Each of the 20 trials perturbs every joint by +/-1.5x that joint's own
# empirical std from the t1+t3 pooled data (joints 2/4/6 barely move at all
# in the real data -- std 0.005-0.02 degrees -- so they're held effectively
# fixed; joints 1/3/5/7, the redundant-DOF-affected ones, get std
# 0.19-0.40 degrees of real spread to draw from). The 20 per-joint offsets
# are independently permuted (max pairwise cross-joint correlation 0.01, so
# no two joints move together) and checked against trial order (no joint's
# offset trends monotonically across the 20 trials), mirroring the
# decorrelation goals of the original Cartesian list above.
EXPERIMENT_STARTING_JOINT_POSITIONS = (
    (0.998, -28.246, -2.806, 23.497, -0.225, 51.737, 0.353),
    (1.566, -28.237, -2.551, 23.496, -0.800, 51.744, 0.505),
    (0.871, -28.247, -1.656, 23.490, -0.186, 51.750, 0.626),
    (1.756, -28.236, -2.231, 23.504, -0.416, 51.708, 0.898),
    (1.503, -28.230, -2.487, 23.486, -0.378, 51.763, 0.444),
    (1.377, -28.249, -2.295, 23.500, -0.723, 51.705, 0.686),
    (1.819, -28.239, -2.167, 23.503, -0.340, 51.741, 0.565),
    (1.313, -28.242, -1.912, 23.491, -0.148, 51.724, 0.837),
    (1.124, -28.234, -1.720, 23.495, -0.761, 51.754, 0.595),
    (1.692, -28.235, -1.976, 23.492, -0.685, 51.747, 0.807),
    (1.882, -28.238, -2.614, 23.485, -0.263, 51.734, 0.716),
    (1.945, -28.243, -1.848, 23.501, -0.493, 51.731, 0.414),
    (1.629, -28.244, -2.040, 23.484, -0.877, 51.711, 0.323),
    (0.808, -28.232, -2.423, 23.489, -0.570, 51.715, 0.868),
    (0.934, -28.241, -2.742, 23.502, -0.838, 51.757, 0.777),
    (1.061, -28.231, -2.678, 23.498, -0.301, 51.702, 0.384),
    (1.250, -28.245, -2.359, 23.507, -0.455, 51.760, 0.656),
    (1.187, -28.233, -2.103, 23.506, -0.531, 51.721, 0.474),
    (0.745, -28.240, -1.784, 23.494, -0.608, 51.718, 0.535),
    (1.440, -28.250, -2.870, 23.488, -0.646, 51.728, 0.747),
)

# The one fixed home / starting configuration: the arm starts here, 'h' returns
# here, and it parks here on exit. The rest of the list above is no longer used.
HOME_JOINT_POSITION = EXPERIMENT_STARTING_JOINT_POSITIONS[0]

# Task instructions. These strings must match the prompts the server has cached
# T5 embeddings for (models/xarm7_3task/*.pt); any other string makes
# the server reject the request.
TASKS = {
    "insert_bottle": "insert the bottle",
    "stack_cubes": "stack the cubes",
    "stack_cups": "stack the cups",
}

TCP_OFFSET = [0, 0, 0, 0, 0, 0]

# All rollout outputs live under data/rollouts/ next to this script:
#   runs/          per-run, per-trial videos and episode HDF5s
#   observations/  per-query model-input PNGs
#   *.txt          run logs
ROLLOUTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "rollouts_piwam")
OBSERVATIONS_DIR = os.path.join(ROLLOUTS_DIR, "observations")
# One HDF5 per --name (new file per launch by default; reusing a name appends);
# this is the viewer's default root.
COMBINED_DIR = os.path.join(ROLLOUTS_DIR, "combined")


def next_test_dir():
    os.makedirs(OBSERVATIONS_DIR, exist_ok=True)
    existing = [d for d in os.listdir(OBSERVATIONS_DIR)
                if d.startswith("test") and d[4:].isdigit()]
    n = max((int(d[4:]) for d in existing), default=0) + 1
    test_dir = os.path.join(OBSERVATIONS_DIR, f"test{n}")
    os.makedirs(os.path.join(test_dir, "primary"))
    os.makedirs(os.path.join(test_dir, "wrist"))
    return test_dir


@dataclass
class PolicyEvalConfig:
    suite: str = "libero"

    model_family: str = "cosmos"
    config: str = "cosmos_predict2_2b_480p_libero__inference_only"
    config_file: str = "cosmos_policy/config/config.py"
    ckpt_path: str = ""
    dataset_stats_path: str = ""
    t5_text_embeddings_path: str = ""

    use_third_person_image: bool = True
    num_third_person_images: int = 1
    use_wrist_image: bool = True
    num_wrist_images: int = 1
    use_proprio: bool = True

    flip_images: bool = False
    use_jpeg_compression: bool = True
    trained_with_image_aug: bool = True
    use_variance_scale: bool = False

    normalize_proprio: bool = True
    unnormalize_actions: bool = True

    chunk_size: int = 16
    num_open_loop_steps: int = CHUNK_EXECUTE_STEPS
    num_denoising_steps_action: int = 5
    num_denoising_steps_future_state: int = 5
    num_denoising_steps_value: int = 3

    deterministic: bool = True
    randomize_seed: bool = False
    seed: int = 20
    data_collection: bool = True
    jpeg_compress: bool = True
    local_log_dir: str = ROLLOUTS_DIR
    run_id_note: str = "xarm"

    ar_future_prediction: bool = False
    ar_value_prediction: bool = False
    ar_qvalue_prediction: bool = False

    use_ensemble_future_state_predictions: bool = False
    num_future_state_predictions_in_ensemble: int = 3
    future_state_ensemble_aggregation_scheme: str = "average"

    use_ensemble_value_predictions: bool = False
    num_value_predictions_in_ensemble: int = 5
    value_ensemble_aggregation_scheme: str = "average"

    search_depth: int = 1
    mask_current_state_action_for_value_prediction: bool = False
    mask_future_state_for_qvalue_prediction: bool = False


def _encode_jpeg_b64(img_rgb):
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr)
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def get_action_from_piwam(obs_snapshot, server_endpoint):
    """POST to a pi-WAM /infer server; returns {"actions": (H, 7), "active_future", "router_probs"}."""
    resp = requests.post(server_endpoint, json=obs_snapshot, timeout=120)
    resp.raise_for_status()
    actions = np.asarray(resp.json()["actions"], dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 7:
        raise RuntimeError(f"Unexpected pi-WAM action shape {actions.shape}, expected (H, 7)")
    body = resp.json()
    return {"actions": actions, "active_future": body.get("active_future"),
            "router_probs": body.get("router_probs"), "history_len": body.get("history_len")}


def crop_and_resize(frame, crop):
    w, h, x, y = crop
    cropped = frame[y:y + h, x:x + w]
    return cv2.resize(cropped, CROP_TARGET_SIZE, interpolation=cv2.INTER_LINEAR)


class CameraStream:
    """Reads a camera continuously in a background thread so frames are always fresh
    (no stale V4L2 buffer) and publishes the cropped model view for the live overlay.
    Drop-in for cv2.VideoCapture as used here: read(), grab(), release(), isOpened()."""

    def __init__(self, cap, view, crop, publish_dir=None, publish_hz=LIVE_FRAME_HZ):
        self.cap = cap
        self.view = view
        self.crop = crop
        self.publish_dir = publish_dir or None
        self.min_publish_interval = 1.0 / publish_hz
        self._lock = threading.Lock()
        self._frame = None
        self._stop = threading.Event()
        if self.publish_dir:
            os.makedirs(self.publish_dir, exist_ok=True)
        self._thread = threading.Thread(target=self._loop, name=f"cam-{view}", daemon=True)
        self._thread.start()

    def _publish(self, frame):
        view = crop_and_resize(frame, self.crop)
        ok, buf = cv2.imencode(".jpg", view, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if not ok:
            return
        tmp = os.path.join(self.publish_dir, f".{self.view}.tmp.jpg")
        with open(tmp, "wb") as f:
            f.write(buf.tobytes())
        os.replace(tmp, os.path.join(self.publish_dir, f"{self.view}.jpg"))  # atomic

    def _loop(self):
        last_publish = 0.0
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self._lock:
                self._frame = frame
            now = time.monotonic()
            if self.publish_dir and now - last_publish >= self.min_publish_interval:
                last_publish = now
                try:
                    self._publish(frame)
                except Exception as exc:  # noqa: BLE001 - never let publishing break inference
                    print(f"[camera] {self.view}: live-frame publish failed: {exc}")
                    self.publish_dir = None

    def read(self, timeout=5.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if self._frame is not None:
                    return True, self._frame.copy()
            time.sleep(0.005)
        return False, None

    def grab(self):  # frames are already drained continuously
        return True

    def isOpened(self):
        return self.cap.isOpened()

    def release(self):
        self._stop.set()
        self._thread.join(timeout=2.0)
        self.cap.release()


def init_cameras(primary_id, wrist_id, ego_id, publish_dir=LIVE_FRAME_DIR):
    # Explicit V4L2 backend: OpenCV's default backend (GStreamer on this
    # box) fails to negotiate CAPTURE_FRAME_SIZE (1280x800) on at least one
    # of the two RealSense color streams and warmup_cameras() never gets a
    # frame from it, even though the same device opens and streams fine at
    # 1280x800 under V4L2.
    primary_cap = cv2.VideoCapture(primary_id, cv2.CAP_V4L2)
    wrist_cap = cv2.VideoCapture(wrist_id, cv2.CAP_V4L2)
    ego_cap = cv2.VideoCapture(ego_id, cv2.CAP_V4L2)
    cameras = (
        ("primary", primary_id, primary_cap),
        ("wrist", wrist_id, wrist_cap),
        ("ego", ego_id, ego_cap),
    )
    failed = [f"{name} camera id {cam_id}" for name, cam_id, cap in cameras if not cap.isOpened()]
    if failed:
        for _, _, cap in cameras:
            cap.release()
        raise RuntimeError(
            "Could not open " + " and ".join(failed) +
            ". Check /dev/video* and restart with --primary-cam-id/--wrist-cam-id/--ego-cam-id."
        )

    for name, _, cap in cameras:
        size = EGO_CAPTURE_FRAME_SIZE if name == "ego" else CAPTURE_FRAME_SIZE
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return (
        CameraStream(primary_cap, "exo", EXO1_CROP, publish_dir),
        CameraStream(wrist_cap, "wrist", WRISTCAM_CROP, publish_dir),
        CameraStream(ego_cap, "ego", EGO_CROP, publish_dir),
    )


def warmup_cameras(primary_cap, wrist_cap, ego_cap, num_frames=WARMUP_FRAMES):
    ret1 = ret2 = ret3 = False
    time.sleep(2.0)  # let auto-exposure settle; the streams keep reading in the background
    for _ in range(3):
        ret1, _ = primary_cap.read()
        ret2, _ = wrist_cap.read()
        ret3, _ = ego_cap.read()
    if not ret1 or not ret2 or not ret3:
        raise RuntimeError(
            "Camera warmup failed. At least one configured camera opened but did not return frames."
        )


def init_arm(ip, starting_joint_position):
    arm = XArmAPI(ip)
    arm.motion_enable(True)
    arm.set_tcp_offset(TCP_OFFSET)
    arm.set_mode(0)
    arm.set_state(0)
    arm.set_gripper_mode(0)
    arm.set_gripper_enable(True)
    arm.set_gripper_speed(5000)

    # Joint-space move (not Cartesian set_position): set_position()'s IK
    # picks the nearest solution to the arm's current joints, which is
    # history-dependent and does not reliably land on the same redundant
    # elbow configuration the training data was collected with. Commanding
    # joint angles directly makes the starting proprio actually match.
    arm.set_servo_angle(angle=list(starting_joint_position), is_radian=False,
                         wait=True, speed=20)
    arm.set_gripper_position(850, wait=True)
    return arm


def get_gripper_value(arm):
    _, pos = arm.get_gripper_position()
    pos = max(0, min(pos, 850))
    return pos / 850.0


# Last gripper position commanded (0 closed, 850 open, None unset), so identical
# commands are not re-sent every step. Reset per episode, since home_arm() reopens
# the gripper physically and the cached value would otherwise be stale.
_last_gripper_target = None


def get_proprio(arm):
    """State fed to the model: [x, y, z (m), roll, pitch, yaw (rad), gripper (-1 open, +1 closed)].

    The 30k checkpoint was trained on end-effector pose + gripper state, NOT joint
    angles (see dataset_stats.json state: x mean 0.416 m, roll mean 3.141 rad).
    Roll is wrapped to [0, 2*pi) like the training data, so roll -180 deg -> pi.
    Gripper comes from the last commanded target (no Modbus read in the loop);
    it starts open because home_arm() reopens the gripper.
    """
    pose = arm.get_position(is_radian=False)[1]
    x, y, z = (float(v) / 1000.0 for v in pose[:3])
    roll = math.radians(float(pose[3])) % (2.0 * math.pi)
    pitch = math.radians(float(pose[4]))
    yaw = math.radians(float(pose[5]))
    gripper = 1.0 if _last_gripper_target == 0 else -1.0
    return np.array([x, y, z, roll, pitch, yaw, gripper], dtype=np.float32)


def read_observation_images(primary_cap, wrist_cap, ego_cap):
    """Returns RGB uint8 480x480 (primary, wrist, ego), cropped exactly as at collection time."""
    ret1, primary = primary_cap.read()
    ret2, wrist = wrist_cap.read()
    ret3, ego = ego_cap.read()
    if not ret1 or not ret2 or not ret3:
        raise RuntimeError("Camera read failed")
    views = []
    for frame, crop in ((primary, EXO1_CROP), (wrist, WRISTCAM_CROP), (ego, EGO_CROP)):
        rgb = cv2.cvtColor(crop_and_resize(frame, crop), cv2.COLOR_BGR2RGB)
        views.append(np.ascontiguousarray(rgb, dtype=np.uint8))
    return tuple(views)


def reset_gripper_command_cache():
    global _last_gripper_target
    _last_gripper_target = None


def _wrap_deg(angle_deg):
    """Wrap to [-180, 180) as the xArm reports roll/pitch/yaw."""
    return (angle_deg + 180.0) % 360.0 - 180.0


def execute_action_cartesian_position(arm, action, speed=100):
    """Send one ABSOLUTE target [x,y,z (m), roll,pitch,yaw (rad), gripper (-1 open/+1 closed)]."""
    global _last_gripper_target
    x, y, z = (float(v) * 1000.0 for v in action[:3])
    roll, pitch, yaw = (_wrap_deg(math.degrees(float(v))) for v in action[3:6])
    if z < Z_FLOOR_MM:
        z = Z_FLOOR_MM

    arm.set_position(
        x=x, y=y, z=z, roll=roll, pitch=pitch, yaw=yaw,
        relative=False, wait=False, speed=speed,
    )
    executed_action = [x, y, z, roll, pitch, yaw, -1.0]

    gripper = float(action[6])
    if gripper > 0:
        target_gripper_pos = 0
    elif gripper < 0:
        target_gripper_pos = 850
    else:
        target_gripper_pos = None

    # Only command the gripper when the target changes: re-sending every step
    # saturates the gripper's Modbus channel and wedges the SDK lock.
    if target_gripper_pos is not None:
        if target_gripper_pos != _last_gripper_target:
            arm.set_gripper_position(target_gripper_pos, wait=False)
            _last_gripper_target = target_gripper_pos
        executed_action[6] = float(target_gripper_pos)

    return executed_action


def session_hdf5_path(name):
    """This session's HDF5 under COMBINED_DIR (read by rollout_viewer.py)."""
    return os.path.join(COMBINED_DIR, f"{_safe_run_name(name)}.hdf5")


def save_episode_data(cfg, collected_data, task_description, h5_path,
                      model_tag, log_file=None):
    """Append one trial as the next data/trial_<N> group of the named HDF5.

    The file is opened in append mode and closed again per trial, so earlier
    trials survive a later crash. N continues past the highest existing trial
    number, so repeated runs never overwrite each other.
    """
    os.makedirs(os.path.dirname(h5_path), exist_ok=True)
    with h5py.File(h5_path, "a") as f:
        models = {m for m in _text_attr(f.attrs.get("model_name")).split(",") if m}
        models.add(model_tag)
        f.attrs["model_name"] = ",".join(sorted(models))
        data = f.require_group("data")
        used = [int(k.split("_")[1]) for k in data if re.fullmatch(r"trial_\d+", k)]
        trial_name = f"trial_{max(used, default=0) + 1}"
        g = data.create_group(trial_name)
        g.attrs["model_name"] = model_tag
        g.attrs["run_dir_time"] = DATE_TIME
        for k, v in collected_data.items():
            if isinstance(v, np.ndarray):
                is_image = v.ndim == 4 and v.shape[-1] == 3 and v.dtype == np.uint8
                if is_image and cfg.jpeg_compress:
                    jpeg_list = [jpeg_encode_image(frame, quality=95) for frame in v]
                    dt = h5py.vlen_dtype(np.dtype("uint8"))
                    # Build the object array explicitly. Passing the list
                    # straight to create_dataset lets numpy collapse
                    # equal-length buffers (identical frames, or a
                    # single-frame list) into a 2-D array, which the vlen
                    # converter then rejects with "'int' object has no
                    # attribute 'dtype'".
                    jpeg_arr = np.empty(len(jpeg_list), dtype=object)
                    jpeg_arr[:] = jpeg_list
                    g.create_dataset(k + "_jpeg", data=jpeg_arr, dtype=dt)
                else:
                    g.create_dataset(k, data=v)
            else:
                g.attrs[k] = v
        g.attrs["task_description"] = task_description
    log_message(f"[episode] saved {trial_name} to {h5_path}", log_file)
    return trial_name


def delete_trial_from_hdf5(h5_path, trial_name):
    with h5py.File(h5_path, "r+") as f:
        if trial_name in f["data"]:
            del f["data"][trial_name]


def run(cfg, model, dataset_stats, arm, primary_cap, wrist_cap, ego_cap, task,
        save_debug_image=False, debug_image_dir=None, obs_dir=None, log_file=None,
        server_endpoint=None, keys=None):
    """Run live inference until the operator presses Enter (stop) or d (discard).

    Returns (..., step_count, stop_reason) with stop_reason "stop" or "discard".
    """
    stop_reason = "stop"
    action_queue = deque(maxlen=cfg.num_open_loop_steps)
    rollout_primary_images = []
    rollout_wrist_images = []
    future_image_predictions_list = []
    primary_images_list = []
    wrist_images_list = []
    ego_images_list = []
    proprio_list = []
    actions_list = []
    recording_start_time = time.time()
    debug_saved = False
    query_idx = 0
    step_count = 0
    # Router bookkeeping: one entry per chunk (= per server query).
    episode_id = f"{DATE_TIME}-{uuid.uuid4().hex[:8]}"
    chunk_id = 0
    router_modality, router_probs, router_query_step = [], [], []
    router_streams = None
    # home_arm() has just physically reopened the gripper, so forget whatever
    # was last commanded and let the first gripper action re-send.
    reset_gripper_command_cache()

    try:
        while True:
            # Keys are only checked here, between whole steps, so the recorded
            # images / proprio / actions always stay the same length.
            if keys is not None:
                reason = keys.poll_stop()
                if reason:
                    stop_reason = reason
                    log_message(f"[episode] operator {reason}", log_file)
                    break
            # V4L2 ignores CAP_PROP_BUFFERSIZE=1. After the multi-second
            # get_action call, the buffer is full of stale frames; drain
            # them so the read that actually feeds the model is fresh.
            if len(action_queue) == 0:
                for _ in range(5):
                    primary_cap.grab()
                    wrist_cap.grab()
                    ego_cap.grab()
            primary, wrist, ego = read_observation_images(primary_cap, wrist_cap, ego_cap)
            rollout_primary_images.append(primary.copy())
            rollout_wrist_images.append(wrist.copy())

            if save_debug_image and not debug_saved:
                out_dir = debug_image_dir or "."
                os.makedirs(out_dir, exist_ok=True)
                cv2.imwrite(os.path.join(out_dir, "debug_primary.png"),
                            cv2.cvtColor(primary, cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(out_dir, "debug_wrist.png"),
                            cv2.cvtColor(wrist, cv2.COLOR_RGB2BGR))
                debug_saved = True

            proprio = get_proprio(arm)
            if cfg.data_collection:
                primary_images_list.append(primary.copy())
                wrist_images_list.append(wrist.copy())
                ego_images_list.append(ego.copy())
                proprio_list.append(proprio.copy())

            if len(action_queue) == 0:
                if obs_dir is not None:
                    os.makedirs(obs_dir, exist_ok=True)
                    cv2.imwrite(os.path.join(obs_dir, "primary", f"{query_idx:05d}.png"),
                                cv2.cvtColor(primary, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(os.path.join(obs_dir, "wrist", f"{query_idx:05d}.png"),
                                cv2.cvtColor(wrist, cv2.COLOR_RGB2BGR))
                    os.makedirs(os.path.join(obs_dir, "ego"), exist_ok=True)
                    cv2.imwrite(os.path.join(obs_dir, "ego", f"{query_idx:05d}.png"),
                                cv2.cvtColor(ego, cv2.COLOR_RGB2BGR))
                    query_idx += 1

                print(primary.shape, wrist.shape, proprio.shape)
                obs_snapshot = {
                    "primary_image": _encode_jpeg_b64(primary),
                    "wrist_image": _encode_jpeg_b64(wrist),
                    "ego_image": _encode_jpeg_b64(ego),
                    "state": proprio.tolist(),
                    "prompt": task,
                    "episode_id": episode_id,
                    "chunk_id": chunk_id + 1,
                }
                out = get_action_from_piwam(obs_snapshot, server_endpoint)
                chunk_id += 1
                acts = out["actions"]
                probs = out.get("router_probs") or {}
                if router_streams is None and probs:
                    router_streams = list(probs)
                router_modality.append(str(out.get("active_future")))
                router_probs.append([float(probs.get(k, np.nan)) for k in (router_streams or [])])
                router_query_step.append(step_count)
                log_message(
                    f"[router] chunk {chunk_id} step {step_count}: active_future={out.get('active_future')} "
                    f"probs={ {k: round(v, 3) for k, v in probs.items()} } history_len={out.get('history_len')}",
                    log_file,
                )
                log_message(
                    f"[query {query_idx}] state={np.round(proprio, 4).tolist()} "
                    f"first_action={np.round(acts[0], 4).tolist()} "
                    f"last_action={np.round(acts[-1], 4).tolist()}",
                    log_file,
                )
                if "future_image_predictions" in out:
                    future_image_predictions_list.append(out["future_image_predictions"])
                # maxlen would silently keep the LAST N of a longer chunk; take the first N.
                action_queue.extend(out["actions"][:cfg.num_open_loop_steps])
                log_message(f"[episode] queued new chunk (size={len(action_queue)} of {len(out['actions'])})", log_file)
            
            # predicted_actions = np.asarray(list(action_queue), dtype=np.float32)
            # summed_action = np.sum(predicted_actions, axis=0, dtype=np.float32)


            action = action_queue.popleft()
            log_message(f"[step {step_count + 1}] target={np.round(np.asarray(action), 4).tolist()}", log_file)

            # Closer-loop variant: clearing the queue here forces a re-query
            # every step (only out["actions"][0] used).
            # action_queue = deque(maxlen=cfg.num_open_loop_steps)
            print("Action queue: " + str(action))
            if cfg.data_collection:
                actions_list.append(np.asarray(action, dtype=np.float32).copy())
            tick_start = time.time()
            execute_action_cartesian_position(
                arm, action,
            )
            sleep_time = (1.0 / CONTROL_HZ) - (time.time() - tick_start)
            if sleep_time > 0:
                time.sleep(sleep_time)
            step_count += 1
            log_message(f"[episode] step {step_count}", log_file)
            # breakpoint()

    except KeyboardInterrupt:
        log_message("[episode] stopped inference loop", log_file)

    collected_data = None
    if cfg.data_collection and primary_images_list:
        collected_data = {
            "primary_images": np.stack(primary_images_list, axis=0),
            "wrist_images": np.stack(wrist_images_list, axis=0),
            "ego_images": np.stack(ego_images_list, axis=0),
            # Attrs read by experiments/xarm7/rollout_viewer.py (playback rate + duration).
            "control_hz": float(CONTROL_HZ),
            "recording_start_time": recording_start_time,
            "recording_end_time": time.time(),
            "proprio": np.stack(proprio_list, axis=0),
            "actions": np.stack(actions_list, axis=0) if actions_list else np.zeros((0, 7), dtype=np.float32),
            "success": False,
            "router_modality": np.array(router_modality, dtype="S8"),
            "router_probs": (np.array(router_probs, dtype=np.float32) if router_streams
                             else np.full((len(router_modality), 0), np.nan, dtype=np.float32)),
            "router_query_step": np.array(router_query_step, dtype=np.int32),
            "router_streams": ",".join(router_streams or []),
        }
        if len(future_image_predictions_list) > 0:
            future_primary_images = [
                x["future_image"] for x in future_image_predictions_list
                if isinstance(x, dict) and x.get("future_image") is not None
            ]
            if future_primary_images:
                collected_data["future_primary_images"] = np.stack(future_primary_images, axis=0)
            future_wrist_images = [
                x["future_wrist_image"] for x in future_image_predictions_list
                if isinstance(x, dict) and x.get("future_wrist_image") is not None
            ]
            if future_wrist_images:
                collected_data["future_wrist_images"] = np.stack(future_wrist_images, axis=0)

    return (rollout_primary_images, rollout_wrist_images, future_image_predictions_list,
            collected_data, step_count, stop_reason)


def _model_tag(ckpt_path):
    base = os.path.splitext(os.path.basename(ckpt_path))[0]
    if base in ("model", "checkpoint", "ckpt"):
        parent = os.path.basename(os.path.dirname(ckpt_path))
        if parent:
            return parent
    return base


def _tag_rollout_path(mp4_path, model_tag):
    if not mp4_path or not os.path.exists(mp4_path):
        return mp4_path
    directory, filename = os.path.split(mp4_path)
    new_filename = filename.replace("--episode=", f"--model={model_tag}--episode=", 1)
    if new_filename == filename:
        return mp4_path
    new_path = os.path.join(directory, new_filename)
    os.rename(mp4_path, new_path)
    return new_path


def _safe_run_name(name):
    """Convert a model/checkpoint tag into a readable directory name."""
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    safe = re.sub(r"_+", "_", safe).strip("._-")
    return safe or "untagged"


def next_run_dir(runs_root, name_hint=None):
    """Create data/rollouts/runs/<model_tag>/, suffixing repeat names as needed."""
    os.makedirs(runs_root, exist_ok=True)
    if name_hint:
        base = _safe_run_name(name_hint)
        run_dir = os.path.join(runs_root, base)
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)
            return run_dir

        existing = [
            d for d in os.listdir(runs_root)
            if d.startswith(f"{base}_run_") and d[len(f"{base}_run_"):].isdigit()
        ]
        n = max(
            (int(d[len(f"{base}_run_"):]) for d in existing),
            default=1,
        ) + 1
        run_dir = os.path.join(runs_root, f"{base}_run_{n:03d}")
        os.makedirs(run_dir)
        return run_dir

    existing = [d for d in os.listdir(runs_root)
                if d.startswith("run_") and d[4:].isdigit()]
    n = max((int(d[4:]) for d in existing), default=0) + 1
    run_dir = os.path.join(runs_root, f"run_{n:03d}")
    os.makedirs(run_dir)
    return run_dir


def home_arm(arm, starting_joint_position):
    """Move to a requested experiment start joint configuration and open the gripper."""
    arm.set_mode(0)
    arm.set_state(0)
    arm.set_servo_angle(angle=list(starting_joint_position), is_radian=False,
                         wait=True, speed=20)
    arm.set_gripper_position(850, wait=True)


# Trial scoring rubric: after stopping a trial the operator presses one digit
# key to record how far it got (edit the labels here if the meaning changes).
SCORE_LEVELS = {
    0: "no grasp",
    1: "grasp",
    2: "grasp + place",
}
MAX_TRIAL_SCORE = max(SCORE_LEVELS)
GOAL_SCORE = MAX_TRIAL_SCORE  # full-credit outcome (grasp + place), recorded as `success`


class KeyReader:
    """Single-keypress stdin reader for the whole session.

    A background thread (terminal in cbreak mode, so no Enter is needed and
    Ctrl+C still raises SIGINT) pushes every key onto a queue. Enter arrives as
    "\n"; letters are lower-cased."""

    def __init__(self):
        self._q = queue.Queue()
        self._stop = threading.Event()
        self._thread = None
        self._fd = None
        self._old_settings = None

    def __enter__(self):
        self._fd = sys.stdin.fileno()
        if sys.stdin.isatty():
            try:
                self._old_settings = termios.tcgetattr(self._fd)
                tty.setcbreak(self._fd)
            except (termios.error, OSError):
                self._old_settings = None
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._old_settings is not None:
            try:
                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)
            except (termios.error, OSError):
                pass
        return False

    def _loop(self):
        while not self._stop.is_set():
            try:
                r, _, _ = select.select([self._fd], [], [], 0.1)
                if not r:
                    continue
                data = os.read(self._fd, 1)
            except (OSError, ValueError):
                return
            if not data:
                return
            ch = data.decode("utf-8", "ignore")
            self._q.put("\n" if ch == "\r" else ch.lower())

    def flush(self):
        while True:
            try:
                self._q.get_nowait()
            except queue.Empty:
                return

    def get(self, valid=None):
        """Block until a key in `valid` (all keys if None) is pressed."""
        while True:
            ch = self._q.get()
            if valid is None or ch in valid:
                return ch

    def poll_stop(self):
        """Non-blocking: "stop" on Enter, "discard" on d, else None. Other keys are dropped."""
        while True:
            try:
                ch = self._q.get_nowait()
            except queue.Empty:
                return None
            if ch == "\n":
                return "stop"
            if ch == "d":
                return "discard"


def _move_to(src_path, dst_dir, dst_basename):
    if not src_path or not os.path.exists(src_path):
        return None
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, dst_basename)
    os.replace(src_path, dst)
    return dst


def _resize_images_to_shape(images, target_shape):
    if images is None:
        return None
    target_h, target_w = target_shape[:2]
    resized = []
    for img in images:
        if img.shape[:2] == (target_h, target_w):
            resized.append(img)
            continue
        resized.append(
            np.asarray(
                Image.fromarray(img).resize((target_w, target_h), Image.LANCZOS),
                dtype=np.uint8,
            )
        )
    return resized


def save_test_artifacts(cfg, test_dir, h5_path, model_tag, success, task, log_file,
                        rollout_primary_images, rollout_wrist_images,
                        future_image_predictions_list, collected_data, step_count,
                        score=0):
    """Save the rollout video into test_dir (rubric score + model tag in the
    filename) and append the trial to the HDF5 at h5_path as
    the next group data/trial_<N>. `success` is goal-reached, i.e. the
    full-credit outcome. Returns the new trial's group name (None if no data)."""
    trial_name = None
    suffix = f"--score={score}--success={success}--model={model_tag}"

    if rollout_primary_images:
        mp4_path = save_rollout_video(
            rollout_primary_images,
            idx=0,
            success=success,
            task_description=task,
            log_file=None,
            fps=CONTROL_HZ,
        )
        _move_to(mp4_path, test_dir, f"rollout{suffix}.mp4")

    if cfg.data_collection and collected_data is not None:
        # Non-array values land in the hdf5 attrs, so the rubric components
        # are recoverable from the episode file alone.
        collected_data["success"] = success
        collected_data["goal_reached"] = bool(success)
        collected_data["score"] = int(score)
        trial_name = save_episode_data(
            cfg, collected_data, task_description=task, h5_path=h5_path,
            model_tag=model_tag, log_file=log_file,
        )
        log_message(f"[test] saved data for {step_count} steps", log_file)
    return trial_name


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--server_endpoint", default="http://127.0.0.1:8780/infer",
                   help="pi-WAM server /infer endpoint (FastWAM baseline uses 8777)")
    p.add_argument("--ckpt_tag", default=None,
                   help="Override the model tag used in log/file names (label for the "
                        "checkpoint the pi-WAM server is serving).")
    p.add_argument(
        "--task",
        type=str.lower,
        choices=tuple(TASKS),
        required=True,
        help="Which task to evaluate: " + ", ".join(
            f"{k}={v!r}" for k, v in TASKS.items()
        ),
    )
    p.add_argument("--primary-cam-id", type=int, default=18,
                   help="OpenCV camera index for exo1 / primary_image.")
    p.add_argument("--wrist-cam-id", type=int, default=4,
                   help="OpenCV camera index for wristcam / wrist_image.")
    p.add_argument("--live-frame-dir", default=LIVE_FRAME_DIR,
                   help="Directory where the latest exo/wrist/ego model-view JPEGs are published "
                        "for experiments/xarm7/live_overlay.py ('' disables).")
    p.add_argument("--name", default=None,
                   help="Name of the HDF5 to write (combined/<name>.hdf5). A new name "
                        "creates a new file; an existing name appends new trials to "
                        "it. Default: <task>_<ckpt_tag>_<timestamp> (always new).")
    p.add_argument("--ego-cam-id", type=int, default=10,
                   help="OpenCV camera index for the ego camera / ego_image.")
    return p.parse_args()


def main():
    args = _parse_args()
    deploy_dir = os.path.dirname(os.path.abspath(__file__))

    cfg = PolicyEvalConfig()
    model_tag = args.ckpt_tag if args.ckpt_tag else DEFAULT_CKPT_TAG

    task = TASKS[args.task]
    h5_path = session_hdf5_path(args.name or f"{args.task}_{model_tag}_{DATE_TIME}")
    robot_ip = "192.168.1.198"
    primary_cam_id = args.primary_cam_id    # exo1 -> primary_image
    wrist_cam_id = args.wrist_cam_id        # wristcam -> wrist_image
    ego_cam_id = args.ego_cam_id            # ego -> ego_image
    save_debug_image = False


    runs_root = os.path.join(deploy_dir, cfg.local_log_dir, "runs")
    # Task goes in the run dir name too — the same model gets evaluated on
    # several tasks, and without it those runs are only distinguishable by
    # an auto-incremented _run_NNN suffix.
    run_dir = next_run_dir(runs_root, f"{model_tag}_{args.task}")
    print(f"\n[run] saving rollouts to {run_dir}")
    print(f"[run] model tag: {model_tag}")
    print(f"[run] task: {args.task} = {task!r}")
    print(f"[run] HDF5: {h5_path}" + (" (exists: appending new trials)" if os.path.exists(h5_path) else " (new file)"))

    log_file, _, run_id = setup_logging(
        cfg,
        task_identifier="xarm",
        log_dir=cfg.local_log_dir,
        run_id_note=cfg.run_id_note,
    )
    log_message(
        f"[run] run_dir={run_dir} model={model_tag} task={args.task} ({task!r})",
        log_file,
    )

    log_message(f"[run] querying pi-WAM server {args.server_endpoint}", log_file)
    model = None
    dataset_stats = None

    primary_cap, wrist_cap, ego_cap = init_cameras(primary_cam_id, wrist_cam_id, ego_cam_id, args.live_frame_dir)
    warmup_cameras(primary_cap, wrist_cap, ego_cap)
    arm = init_arm(robot_ip, HOME_JOINT_POSITION)

    # Capture the original terminal mode so the SIGINT handler can restore
    # it before hard-exit. Without this, Ctrl+C while KeyReader has stdin in
    # cbreak mode leaves the terminal unable to echo typed input.
    orig_term_settings = None
    if sys.stdin.isatty():
        try:
            orig_term_settings = termios.tcgetattr(sys.stdin.fileno())
        except (termios.error, OSError):
            orig_term_settings = None

    def _restore_terminal():
        if orig_term_settings is not None:
            try:
                termios.tcsetattr(
                    sys.stdin.fileno(), termios.TCSADRAIN, orig_term_settings,
                )
            except (termios.error, OSError):
                pass

    # Ctrl+C is the safety override: halt the arm and hard-exit immediately,
    # no labeling, no save. Every other control is a normal keypress read by
    # KeyReader.
    def _sigint_handler(signum, frame):
        log_message("[run] SIGINT (Ctrl+C) — emergency stop, hard exit", log_file)
        try:
            arm.set_state(4)
        except Exception:
            pass
        _restore_terminal()
        os._exit(1)

    signal.signal(signal.SIGINT, _sigint_handler)

    records = []  # one dict per trial saved this session; the last is what 'd' discards
    at_home = True  # init_arm() has just moved to HOME_JOINT_POSITION

    def discard_record(rec):
        if rec["trial"] is not None:
            delete_trial_from_hdf5(rec["h5_path"], rec["trial"])
        shutil.rmtree(rec["test_dir"], ignore_errors=True)
        shutil.rmtree(rec["obs_dir"], ignore_errors=True)
        log_message(f"[run] discarded {rec['trial']} (score {rec['score']}) from {rec['h5_path']}", log_file)
        print(f"Discarded {rec['trial']} (score {rec['score']}).")

    def print_controls():
        print("\nControls:")
        print("  Enter  start live inference (press Enter again to stop it)")
        print("  0-2    after a stop, score the trial: " + ", ".join(
            f"{k}={v}" for k, v in SCORE_LEVELS.items()))
        print("  d      discard: during inference, abort and drop the current trial;")
        print("         otherwise drop the most recent saved trial")
        print("  h      move the arm to the home position")
        print("  q      quit (arm must be at home)")
        print("  Ctrl+C EMERGENCY STOP: halts the arm and exits immediately")

    try:
        with KeyReader() as keys:
            print_controls()
            print(f"\nReady at home. {len(records)} trial(s) saved this session.")
            while True:
                key = keys.get(valid={"\n", "h", "d", "q"})

                if key == "h":
                    try:
                        home_arm(arm, HOME_JOINT_POSITION)
                        at_home = True
                        log_message("[run] moved to home position", log_file)
                        print("Arm at home.")
                    except Exception as e:
                        log_message(f"[run] home error: {e}", log_file)
                    continue

                if key == "d":
                    if records:
                        discard_record(records.pop())
                    else:
                        print("Nothing to discard.")
                    continue

                if key == "q":
                    if not at_home:
                        print("Arm is not at home — press h first (clear the workspace).")
                        continue
                    break

                # Enter: run one live-inference trial.
                if not at_home:
                    print("Arm is not at home — press h first (clear the workspace).")
                    continue
                obs_dir = next_test_dir()  # primary/ wrist/ subdirs already created
                keys.flush()
                print("\nLive inference. Enter = stop, d = discard, Ctrl+C = EMERGENCY STOP.")
                log_message("[run] starting live inference", log_file)
                (rollout_primary_images, rollout_wrist_images,
                 future_image_predictions_list, collected_data, step_count,
                 stop_reason) = run(
                    cfg, model, dataset_stats, arm, primary_cap, wrist_cap, ego_cap, task,
                    save_debug_image=save_debug_image,
                    debug_image_dir=os.path.join(deploy_dir, "rollouts"),
                    obs_dir=obs_dir,
                    log_file=log_file,
                    server_endpoint=args.server_endpoint,
                    keys=keys,
                )
                # Halt any queued arm motion immediately while we score / save.
                try:
                    arm.set_state(4)
                except Exception:
                    pass
                at_home = False

                if stop_reason == "discard" or step_count == 0:
                    shutil.rmtree(obs_dir, ignore_errors=True)
                    log_message("[run] trial discarded before saving", log_file)
                    print("Trial discarded. Press h to go home.")
                    continue

                print(f"\nStopped after {step_count} steps. Score it: "
                      + ", ".join(f"{k}={v}" for k, v in SCORE_LEVELS.items())
                      + "  (d = discard)")
                scored = keys.get(valid=set("".join(map(str, SCORE_LEVELS)) + "d"))
                if scored == "d":
                    shutil.rmtree(obs_dir, ignore_errors=True)
                    log_message("[run] trial discarded at scoring prompt", log_file)
                    print("Trial discarded. Press h to go home.")
                    continue

                score = int(scored)
                success = score == GOAL_SCORE
                n = 1 + len([d for d in os.listdir(run_dir) if d.startswith("test_")])
                test_dir = os.path.join(run_dir, f"test_{n}")
                trial_name = save_test_artifacts(
                    cfg, test_dir, h5_path, model_tag, success, task, log_file,
                    rollout_primary_images, rollout_wrist_images,
                    future_image_predictions_list, collected_data, step_count,
                    score=score,
                )
                records.append({"trial": trial_name, "score": score, "test_dir": test_dir,
                                "obs_dir": obs_dir, "h5_path": h5_path})
                log_message(f"[run] saved {trial_name} score={score} to {test_dir}", log_file)
                print(f"Saved {trial_name}: score {score}/{MAX_TRIAL_SCORE} ({SCORE_LEVELS[score]}); "
                      f"running {sum(r['score'] for r in records)}/{MAX_TRIAL_SCORE * len(records)}. "
                      "Press h to go home.")
    finally:
        _restore_terminal()
        try:
            arm.set_state(4)
            arm.set_mode(0)
            arm.set_state(0)
            home_arm(arm, HOME_JOINT_POSITION)
            log_message("[run] home reached", log_file)
            arm.set_gripper_enable(False)
            arm.disconnect()
        except Exception as e:
            log_message(f"[run] cleanup error: {e}", log_file)
        primary_cap.release()
        wrist_cap.release()
        ego_cap.release()

    if records:
        total = sum(r["score"] for r in records)
        possible = MAX_TRIAL_SCORE * len(records)
        completions = sum(1 for r in records if r["score"] == GOAL_SCORE)
        summary = (
            f"[run] {args.task} {model_tag}: "
            f"success rate {100.0 * completions / len(records):.1f}% "
            f"({completions} completions), "
            f"accuracy {100.0 * total / possible:.1f}%"
        )
        print("\n" + "=" * len(summary))
        print(summary)
        for r in records:
            print(f"    {r['trial']}: score={r['score']}/{MAX_TRIAL_SCORE} ({SCORE_LEVELS[r['score']]})")
        print("=" * len(summary))
        log_message(summary, log_file)

    log_message(
        f"[run] finished — {run_dir}",
        log_file,
    )
    if log_file:
        log_file.close()


if __name__ == "__main__":
    main()
