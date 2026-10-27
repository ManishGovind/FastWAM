"""
Cosmos Policy inference on xArm7, aligned with the official LIBERO/Cosmos
training-time preprocessing and chunk execution pattern.

- Predict a chunk, queue it, execute one action at a time.
- Reuse cosmos_utils.get_action for shared preprocessing (JPEG compression,
  center crop, resize). We only do the collection-time crop + resize + BGR->RGB.
- 7D joint-state proprio (matches training data).
- Action: dx/dy/dz in meters, droll/dpitch/dyaw in radians, gripper in [-1,1];
  applied as Cartesian pose delta.
"""

import argparse
import base64
import math
import os
import re
import select
import signal
import sys
import termios
import threading
import time
import tty
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
    run_id = f"ENV_EVAL-{task_identifier}-fastwam-{DATE_TIME}"
    if run_id_note is not None:
        run_id += f"--{run_id_note}"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, run_id + ".txt")
    return open(log_path, "w"), log_path, run_id


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
CROP_TARGET_SIZE = (480, 480)
CAPTURE_FRAME_SIZE = (640, 480)
WARMUP_FRAMES = 60
CONTROL_HZ = 5

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

# Return here after the final experiment.
PARK_POSITION = EXPERIMENT_STARTING_JOINT_POSITIONS[0]

# Task instructions, keyed by the sheet's T1/T2/T3 labels. These strings must
# match the keys in the t5 embeddings pickle exactly — lookup is by exact dict
# key, so a near-miss silently conditions the model on the wrong instruction.
TASKS = {
    "t1": "put the carrot in the bowl",
    "t2": "put the carrot on the plate",
    "t3": "put the eggplant in the bowl",
}

TCP_OFFSET = [0, 0, 0, 0, 0, 0]

# All rollout outputs live under data/rollouts/ next to this script:
#   runs/          per-run, per-trial videos and episode HDF5s
#   observations/  per-query model-input PNGs
#   *.txt          run logs
ROLLOUTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "rollouts")
OBSERVATIONS_DIR = os.path.join(ROLLOUTS_DIR, "observations")


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
    num_open_loop_steps: int = 16
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


def get_action_from_fastwam(obs_snapshot, server_endpoint):
    """POST to a FastWAM /infer server; returns {"actions": (H, 7) array}."""
    resp = requests.post(server_endpoint, json=obs_snapshot, timeout=120)
    resp.raise_for_status()
    actions = np.asarray(resp.json()["actions"], dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 7:
        raise RuntimeError(f"Unexpected FastWAM action shape {actions.shape}, expected (H, 7)")
    return {"actions": actions}


def crop_and_resize(frame, crop):
    w, h, x, y = crop
    cropped = frame[y:y + h, x:x + w]
    return cv2.resize(cropped, CROP_TARGET_SIZE, interpolation=cv2.INTER_LINEAR)


def init_cameras(primary_id, wrist_id):
    # Explicit V4L2 backend: OpenCV's default backend (GStreamer on this
    # box) fails to negotiate CAPTURE_FRAME_SIZE (1280x800) on at least one
    # of the two RealSense color streams and warmup_cameras() never gets a
    # frame from it, even though the same device opens and streams fine at
    # 1280x800 under V4L2.
    primary_cap = cv2.VideoCapture(primary_id, cv2.CAP_V4L2)
    wrist_cap = cv2.VideoCapture(wrist_id, cv2.CAP_V4L2)
    cameras = (
        ("primary", primary_id, primary_cap),
        ("wrist", wrist_id, wrist_cap),
    )
    failed = [f"{name} camera id {cam_id}" for name, cam_id, cap in cameras if not cap.isOpened()]
    if failed:
        for _, _, cap in cameras:
            cap.release()
        raise RuntimeError(
            "Could not open " + " and ".join(failed) +
            ". Check /dev/video* and restart with --primary-cam-id/--wrist-cam-id."
        )

    for _, _, cap in cameras:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_FRAME_SIZE[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_FRAME_SIZE[1])
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return primary_cap, wrist_cap


def warmup_cameras(primary_cap, wrist_cap, num_frames=WARMUP_FRAMES):
    ret1 = ret2 = False
    for _ in range(num_frames):
        ret1, _ = primary_cap.read()
        ret2, _ = wrist_cap.read()
    if not ret1 or not ret2:
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


def get_proprio(arm):
    # Only joint angles are returned. The previous get_position() and
    # get_gripper_value() calls here computed `pose` and `gripper` and then
    # discarded both -- get_gripper_position() in particular is a Modbus
    # round-trip (robot port 502) that ran at the 5Hz control rate for no
    # effect, and gripper Modbus traffic is what the run loop hangs on.
    abs_joint_state = arm.get_joint_states()[1][0]

    return np.array([
        abs_joint_state[0],
        abs_joint_state[1],
        abs_joint_state[2],
        abs_joint_state[3],
        abs_joint_state[4],
        abs_joint_state[5],
        abs_joint_state[6]
    ], dtype=np.float32)


def read_observation_images(primary_cap, wrist_cap):
    ret1, primary = primary_cap.read()
    ret2, wrist = wrist_cap.read()
    if not ret1 or not ret2:
        raise RuntimeError("Camera read failed")
    primary = crop_and_resize(primary, EXO1_CROP)
    wrist = crop_and_resize(wrist, WRISTCAM_CROP)
    primary = cv2.cvtColor(primary, cv2.COLOR_BGR2RGB)
    wrist = cv2.cvtColor(wrist, cv2.COLOR_BGR2RGB)
    # video_primary/video_wrist must be numpy arrays (imageio's
    # append_data requires ndarray), so copy them BEFORE primary/wrist get
    # reassigned to PIL Images below for the 256x256 model-input resize.
    video_primary = primary.copy()
    video_wrist = wrist.copy()
    primary = Image.fromarray(primary).resize((256, 256), resample=0)
    wrist = Image.fromarray(wrist).resize((256, 256), resample=0)

    return (
        np.asarray(primary, dtype=np.uint8),
        np.asarray(wrist, dtype=np.uint8),
        video_primary,
        video_wrist,
    )


# Last gripper position commanded, so identical commands are not re-sent every
# step. Reset at the start of each episode, since home_arm() reopens the
# gripper physically and the cached value would otherwise be stale.
_last_gripper_target = None


def reset_gripper_command_cache():
    global _last_gripper_target
    _last_gripper_target = None


def execute_action_cartesian_position(arm, action, speed=15, radius=5.0):
    global _last_gripper_target
    current_position = arm.get_position(is_radian=False)[1]

    lin_mm = np.asarray(action[:3], dtype=float) * 1000.0
    _rot_deg = np.asarray(action[3:6], dtype=float) * 180.0 / math.pi

    current_position[0] += lin_mm[0]
    current_position[1] += lin_mm[1]
    current_position[2] += lin_mm[2]

    

    print(f"Executing action with linear (mm) before: {current_position[:3]}, {action[6]}")

    gripper = float(action[6])
    # if _last_gripper_target != 0 and gripper > 0 :
    #     # Open/unset -> close transition: this is the grasp step, so nudge
    #     # down 25mm before closing so the gripper actually descends onto the
    #     # object instead of closing 25mm short of it. _last_gripper_target
    #     # is left alone here — the send/cache-update logic further down
    #     # handles that; mutating it here would make it equal to
    #     # target_gripper_pos early and silently skip the actual close command.
    #     current_position[2] = current_position[2] - 25
    #     print(f"Grasp z offset (25mm down): {current_position[:3]}")
    # # if current_position[2] <= 215:    # if current_position[2] <= 195:
    # #     print(f"Executing action with offset  : {current_position[:3]}")
    # #     current_position[2] = current_position[2] - 5
    # #     current_position[2] = max(current_position[2], 180)  # enforce a minimum height of 150mm to avoid collisions
        
    # print(f"Executing action with linear (mm) after : {current_position[:3]}")
    #     current_position[2] = current_position[2] - 5
    #     current_position[2] = max(current_position[2], 175)  # enforce a minimum height of 150mm to avoid collisions
    # print(f"Executing action with linear (mm) after : {current_position[:3]}")
    

    arm.set_position(
        x=current_position[0], 
        y=current_position[1],
        z=current_position[2],
        roll=current_position[3],
        pitch=current_position[4],
        yaw=current_position[5],
        relative=False,
        wait=False,
        speed=100,
    )

    x, y, z, roll, pitch, yaw = current_position
    executed_action = [
        float(x),
        float(y),
        float(z),
        float(roll),
        float(pitch),
        float(yaw),
        -1.0,
    ]

    gripper = float(action[6])

    if gripper > 0:
        target_gripper_pos = 0
    elif gripper < 0:
        target_gripper_pos = 850
    else:
        target_gripper_pos = None

    

    if target_gripper_pos is not None:
        # Only command the gripper when the target actually changes. This used
        # to fire every step: action[6] is a continuous float so it is almost
        # never exactly 0, meaning a Modbus write went out at 5Hz for the whole
        # episode, re-sending a position the gripper was already holding. That
        # saturates the gripper's Modbus channel and is what wedges the SDK
        # lock. replay_inference.py already guards the same way.
        
        if target_gripper_pos != _last_gripper_target:
            arm.set_gripper_position(target_gripper_pos, wait=False)
            _last_gripper_target = target_gripper_pos
        executed_action[6] = float(target_gripper_pos)

    return executed_action


def save_episode_data(cfg, collected_data, task_description, success=False, episode_idx=0, log_file=None):
    rollout_data_dir = os.path.join(cfg.local_log_dir, "rollout_data")
    os.makedirs(rollout_data_dir, exist_ok=True)
    ep_filename = (
        f"episode_data--suite=xarm--{DATE_TIME}--task=0--ep={episode_idx}"
        f"--success={success}--{cfg.run_id_note}.hdf5"
    )
    ep_filepath = os.path.join(rollout_data_dir, ep_filename)
    with h5py.File(ep_filepath, "w") as f:
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
                    f.create_dataset(k + "_jpeg", data=jpeg_arr, dtype=dt)
                else:
                    f.create_dataset(k, data=v)
            else:
                f.attrs[k] = v
        f.attrs["task_description"] = task_description
    log_message(f"[episode] saved {ep_filepath}", log_file)
    return ep_filepath


def run(cfg, model, dataset_stats, arm, primary_cap, wrist_cap, task,
        save_debug_image=False, debug_image_dir=None, obs_dir=None, log_file=None,
        server_endpoint=None):
    action_queue = deque(maxlen=cfg.num_open_loop_steps)
    rollout_primary_images = []
    rollout_wrist_images = []
    future_image_predictions_list = []
    primary_images_list = []
    wrist_images_list = []
    proprio_list = []
    actions_list = []
    debug_saved = False
    query_idx = 0
    step_count = 0
    # home_arm() has just physically reopened the gripper, so forget whatever
    # was last commanded and let the first gripper action re-send.
    reset_gripper_command_cache()

    try:
        while True:
            # V4L2 ignores CAP_PROP_BUFFERSIZE=1. After the multi-second
            # get_action call, the buffer is full of stale frames; drain
            # them so the read that actually feeds the model is fresh.
            if len(action_queue) == 0:
                for _ in range(5):
                    primary_cap.grab()
                    wrist_cap.grab()
            primary, wrist, video_primary, video_wrist = read_observation_images(primary_cap, wrist_cap)
            rollout_primary_images.append(video_primary.copy())
            rollout_wrist_images.append(video_wrist.copy())

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
                proprio_list.append(proprio.copy())

            if len(action_queue) == 0:
                if obs_dir is not None:
                    os.makedirs(obs_dir, exist_ok=True)
                    cv2.imwrite(os.path.join(obs_dir, "primary", f"{query_idx:05d}.png"),
                                cv2.cvtColor(primary, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(os.path.join(obs_dir, "wrist", f"{query_idx:05d}.png"),
                                cv2.cvtColor(wrist, cv2.COLOR_RGB2BGR))
                    query_idx += 1

                print(primary.shape, wrist.shape, proprio.shape)
                obs_snapshot = {
                    "primary_image": _encode_jpeg_b64(primary),
                    "wrist_image": _encode_jpeg_b64(wrist),
                    "state": proprio.tolist(),
                    "prompt": task,
                }
                out = get_action_from_fastwam(obs_snapshot, server_endpoint)
                if "future_image_predictions" in out:
                    future_image_predictions_list.append(out["future_image_predictions"])
                action_queue.extend(out["actions"])
                log_message(f"[episode] queued new chunk (size={len(out['actions'])})", log_file)
            
            # predicted_actions = np.asarray(list(action_queue), dtype=np.float32)
            # summed_action = np.sum(predicted_actions, axis=0, dtype=np.float32)


            action = action_queue.popleft()

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
            "proprio": np.stack(proprio_list, axis=0),
            "actions": np.stack(actions_list, axis=0) if actions_list else np.zeros((0, 7), dtype=np.float32),
            "success": False,
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

    return rollout_primary_images, rollout_wrist_images, future_image_predictions_list, collected_data, step_count


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


# Partial-scoring rubric: 50 points for touching the target object, 50 for
# reaching the goal. Keys are deliberately letters rather than the old 1/0 —
# '1' used to mean "success", and reusing it would silently record 50 points
# where the operator meant 100.
SCORE_KEYS = {
    "n": (False, False),   #   0 — no contact with the target
    "t": (True, False),    #  50 — touched the target, goal not reached
    "g": (True, True),     # 100 — touched and reached the goal
}
TOUCH_POINTS = 50
GOAL_POINTS = 50
MAX_TRIAL_SCORE = TOUCH_POINTS + GOAL_POINTS


def trial_score(touched, goal):
    return TOUCH_POINTS * bool(touched) + GOAL_POINTS * bool(goal)


class TestEndKeyListener:
    """Background stdin watcher for the rubric keys in SCORE_KEYS ('n', 't',
    'g'). On a recognised key, records touched/goal and sends SIGUSR1 to the
    main process so the SIGUSR1 handler halts the arm and breaks run()'s
    loop. Unrecognised keys are ignored, so an old-habit '1' does nothing
    visible rather than scoring the trial wrongly. SIGINT is reserved for
    true emergency stop (Ctrl+C) — see the handlers in main(). Restores
    terminal on exit. Falls back to a no-op if stdin isn't a tty."""

    def __init__(self):
        self.result = {"decided": False, "touched": False, "goal": False}
        self._stop = threading.Event()
        self._thread = None
        self._fd = None
        self._old_settings = None

    def __enter__(self):
        if sys.stdin.isatty():
            try:
                self._fd = sys.stdin.fileno()
                self._old_settings = termios.tcgetattr(self._fd)
                tty.setcbreak(self._fd)
            except (termios.error, OSError):
                self._fd = None
                self._old_settings = None
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._fd is not None and self._old_settings is not None:
            try:
                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)
            except (termios.error, OSError):
                pass
        return False

    def _loop(self):
        if self._fd is None:
            return
        while not self._stop.is_set():
            try:
                r, _, _ = select.select([self._fd], [], [], 0.1)
            except (OSError, ValueError):
                return
            if not r:
                continue
            try:
                ch = sys.stdin.read(1)
            except (OSError, ValueError):
                return
            verdict = SCORE_KEYS.get(ch.lower() if ch else "")
            if verdict is not None:
                self.result["decided"] = True
                self.result["touched"], self.result["goal"] = verdict
                os.kill(os.getpid(), signal.SIGUSR1)
                return


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


def save_test_artifacts(cfg, test_dir, model_tag, success, task, log_file,
                        rollout_primary_images, rollout_wrist_images,
                        future_image_predictions_list, collected_data, step_count,
                        touched=False, score=0):
    """Save rollout videos + episode HDF5 into test_dir, embedding the rubric
    score + model tag in filenames so a single ls makes the run readable.
    `success` is goal-reached, i.e. the full-credit outcome."""
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
        collected_data["touched"] = bool(touched)
        collected_data["goal_reached"] = bool(success)
        collected_data["score"] = int(score)
        ep_path = save_episode_data(
            cfg, collected_data, task_description=task,
            success=success, episode_idx=0, log_file=log_file,
        )
        _move_to(ep_path, test_dir, f"episode_data{suffix}.hdf5")
        log_message(f"[test] saved data for {step_count} steps", log_file)


def _prompt_num_tests(max_tests):
    while True:
        raw = input(
            f"How many test runs would you like to perform? (1-{max_tests}) "
        ).strip()
        try:
            n = int(raw)
            if 0 < n <= max_tests:
                return n
        except ValueError:
            pass
        print(f"Please enter an integer from 1 to {max_tests}.")


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--server_endpoint", default="http://0.0.0.0:8777/infer",
                   help="FastWAM server /infer endpoint")
    p.add_argument("--ckpt_tag", default=None,
                   help="Override the model tag used in log/file names (label for the "
                        "checkpoint the FastWAM server is serving).")
    p.add_argument(
        "--task",
        type=str.lower,
        choices=tuple(TASKS),
        required=True,
        help="Which task to evaluate: " + ", ".join(
            f"{k}={v!r}" for k, v in TASKS.items()
        ),
    )
    p.add_argument(
        "--start_position",
        type=int,
        choices=range(1, len(EXPERIMENT_STARTING_JOINT_POSITIONS) + 1),
        default=1,
        metavar=f"1-{len(EXPERIMENT_STARTING_JOINT_POSITIONS)}",
        help="One-based experiment position to start from (default: 1).",
    )
    p.add_argument("--primary-cam-id", type=int, default=18,
                   help="OpenCV camera index for exo1 / primary_image.")
    p.add_argument("--wrist-cam-id", type=int, default=4,
                   help="OpenCV camera index for wristcam / wrist_image.")
    return p.parse_args()


def main():
    args = _parse_args()
    deploy_dir = os.path.dirname(os.path.abspath(__file__))

    cfg = PolicyEvalConfig(
        ckpt_path=os.path.join(deploy_dir, "/home/skapse/workspace/cosmos-xarm/cosmos-deploy/real-world-model-weights/cosmos_predict2_2b_480p_real_world_xarm_put_carrot_in_bowl_5hz/checkpoints/iter_000008000/model.pt"),
        dataset_stats_path=os.path.join(deploy_dir, "/home/skapse/workspace/cosmos-xarm/cosmos-deploy/real-world-model-weights/cosmos_predict2_2b_480p_real_world_xarm_put_carrot_in_bowl_5hz/dataset_statistics.json"),
        t5_text_embeddings_path=os.path.join(deploy_dir, "t5_embeddings.pkl"),
    )
    model_tag = args.ckpt_tag if args.ckpt_tag else _model_tag(cfg.ckpt_path)

    task = TASKS[args.task]
    robot_ip = "192.168.1.198"
    primary_cam_id = args.primary_cam_id    # exo1 -> primary_image
    wrist_cam_id = args.wrist_cam_id        # wristcam -> wrist_image
    save_debug_image = False


    start_position_index = args.start_position - 1
    remaining_positions = len(EXPERIMENT_STARTING_JOINT_POSITIONS) - start_position_index
    num_tests = _prompt_num_tests(remaining_positions)

    runs_root = os.path.join(deploy_dir, cfg.local_log_dir, "runs")
    # Task goes in the run dir name too — the same model gets evaluated on
    # several tasks, and without it those runs are only distinguishable by
    # an auto-incremented _run_NNN suffix.
    run_dir = next_run_dir(runs_root, f"{model_tag}_{args.task}")
    print(f"\n[run] saving rollouts to {run_dir}")
    print(f"[run] model tag: {model_tag}")
    print(f"[run] task: {args.task} = {task!r}")
    print(f"[run] performing {num_tests} test(s)\n")

    log_file, _, run_id = setup_logging(
        cfg,
        task_identifier="xarm",
        log_dir=cfg.local_log_dir,
        run_id_note=cfg.run_id_note,
    )
    log_message(
        f"[run] run_dir={run_dir} model={model_tag} num_tests={num_tests} "
        f"start_position={args.start_position} task={args.task} ({task!r})",
        log_file,
    )

    log_message(f"[run] querying FastWAM server {args.server_endpoint}", log_file)
    model = None
    dataset_stats = None

    primary_cap, wrist_cap = init_cameras(primary_cam_id, wrist_cam_id)
    warmup_cameras(primary_cap, wrist_cap)
    arm = init_arm(
        robot_ip,
        EXPERIMENT_STARTING_JOINT_POSITIONS[start_position_index],
    )

    # Capture the original terminal mode so the SIGINT handler can restore
    # it before hard-exit. Without this, Ctrl+C while TestEndKeyListener has
    # stdin in cbreak mode leaves the terminal unable to echo typed input.
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
    # no labeling, no save. SIGUSR1 is the normal end-of-test signal sent by
    # the keypress listener — it raises KeyboardInterrupt which run()'s
    # internal try/except unwinds gracefully so we can label and save.
    def _sigint_handler(signum, frame):
        log_message("[run] SIGINT (Ctrl+C) — emergency stop, hard exit", log_file)
        try:
            arm.set_state(4)
        except Exception:
            pass
        _restore_terminal()
        os._exit(1)

    def _sigusr1_handler(signum, frame):
        try:
            arm.set_state(4)
        except Exception:
            pass
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _sigint_handler)
    signal.signal(signal.SIGUSR1, _sigusr1_handler)

    trial_scores = []  # (position_number, touched, goal, score) per trial

    try:
        for test_num in range(1, num_tests + 1):
            position_index = start_position_index + test_num - 1
            position_number = position_index + 1
            starting_joint_position = EXPERIMENT_STARTING_JOINT_POSITIONS[position_index]
            print(f"\n========== Test {test_num}/{num_tests} ==========")
            print(f"Experiment position {position_number} (joint deg): {starting_joint_position}")
            print("Set up the scene. During inference, press one of:")
            print("    n = no contact with target      (  0)")
            print("    t = touched target, no goal     ( 50)")
            print("    g = touched AND reached goal    (100)")
            print("Ctrl+C is an EMERGENCY STOP — arm halts and the script exits immediately.")
            input("Press Enter to start inference...")

            test_dir = os.path.join(run_dir, f"test_{position_number}")
            os.makedirs(test_dir, exist_ok=True)

            # Auto-incremented per-test obs dir with primary/ and wrist/ subdirs
            # already created. Without this, cv2.imwrite silently no-ops because
            # the subdirectories don't exist.
            obs_dir = next_test_dir()

            log_message(
                f"[test {position_number}] starting joint position={starting_joint_position}",
                log_file,
            )

            rollout_primary_images = []
            rollout_wrist_images = []
            future_image_predictions_list = []
            collected_data = None
            step_count = 0

            with TestEndKeyListener() as listener:
                try:
                    (rollout_primary_images, rollout_wrist_images,
                     future_image_predictions_list, collected_data, step_count) = run(
                        cfg, model, dataset_stats, arm, primary_cap, wrist_cap, task,
                        save_debug_image=save_debug_image,
                        debug_image_dir=os.path.join(deploy_dir, "rollouts"),
                        obs_dir=obs_dir,
                        log_file=log_file,
                        server_endpoint=args.server_endpoint,
                    )
                except KeyboardInterrupt:
                    pass

            # Halt any queued arm motion immediately while we save / prompt.
            try:
                arm.set_state(4)
            except Exception:
                pass

            if listener.result["decided"]:
                touched = listener.result["touched"]
                success = listener.result["goal"]
                score = trial_score(touched, success)
                log_message(
                    f"[test {position_number}] keypress label touched={touched} "
                    f"goal={success} score={score}",
                    log_file,
                )
            else:
                # run() returned without a rubric keypress. Ctrl+C would have
                # hard-exited before reaching here, so this is the rare
                # "loop ended naturally" case — score 0 and keep going.
                touched = False
                success = False
                score = 0
                log_message(
                    f"[test {position_number}] no label, defaulting to score=0",
                    log_file,
                )

            trial_scores.append((position_number, touched, success, score))
            print(f"[test {position_number}] score {score}/{MAX_TRIAL_SCORE}  "
                  f"(running {sum(s for *_, s in trial_scores)}/"
                  f"{MAX_TRIAL_SCORE * len(trial_scores)})")

            save_test_artifacts(
                cfg, test_dir, model_tag, success, task, log_file,
                rollout_primary_images, rollout_wrist_images,
                future_image_predictions_list, collected_data, step_count,
                touched=touched, score=score,
            )
            log_message(f"[test {position_number}] saved to {test_dir}", log_file)

            if test_num == num_tests:
                break

            # Re-home for the next test so the arm is parked while the
            # user re-stages the scene.
            try:
                next_starting_joint_position = EXPERIMENT_STARTING_JOINT_POSITIONS[position_index + 1]
                home_arm(arm, next_starting_joint_position)
                log_message(
                    f"[test {position_number}] moved to next starting joint position="
                    f"{next_starting_joint_position}",
                    log_file,
                )
            except Exception as e:
                log_message(f"[test {test_num}] re-home error: {e}", log_file)
    finally:
        _restore_terminal()
        try:
            arm.set_state(4)
            arm.set_mode(0)
            arm.set_state(0)
            home_arm(arm, PARK_POSITION)
            log_message("[run] home reached", log_file)
            arm.set_gripper_enable(False)
            arm.disconnect()
        except Exception as e:
            log_message(f"[run] cleanup error: {e}", log_file)
        primary_cap.release()
        wrist_cap.release()

    if trial_scores:
        total = sum(s for *_, s in trial_scores)
        possible = MAX_TRIAL_SCORE * len(trial_scores)
        touched_n = sum(1 for _, t, _, _ in trial_scores if t)
        goal_n = sum(1 for _, _, g, _ in trial_scores if g)
        summary = (
            f"[run] {args.task} {model_tag}: {total}/{possible} = "
            f"{100.0 * total / possible:.1f}%  "
            f"(touched {touched_n}/{len(trial_scores)}, "
            f"goal {goal_n}/{len(trial_scores)})"
        )
        print("\n" + "=" * len(summary))
        print(summary)
        for pos, t, g, s in trial_scores:
            print(f"    position {pos:2d}: touched={int(t)} goal={int(g)} score={s:3d}")
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
