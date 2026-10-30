"""xArm7 + XHand1 hardware layer for the FastWAM robot client.

Vendored from Dev_dex_hand_tactile vpp_real/scripts/deploy_policy.py (commit
b84379f) so the client needs neither a vpp_real checkout nor torch. Only the
definitions the client uses are kept, copied verbatim except where marked
"FastWAM:". Dropped: the VPP model, replay mode, the ZMQ inference protocol,
and the VPP run loops.

Open-Teach is still required on the robot workstation: it provides the XHand
driver (XHandLink), the per-command pose clamp (clamp_pose_step), the XHand
limit constants checked at startup, and the camera publisher this module
subscribes to.
"""

import base64
import json
import os
import pickle
import select
import socket
import sys
import termios
import threading
import time
import tty
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


ROOT = os.path.dirname(os.path.abspath(__file__))  # FastWAM: experiments/xhand1/
CHUNK = 8
# Chunk cadence, NOT the recording rate: the dataset is stored at 30 Hz but
# policy_index_10hz/ is built with --sequence-interval 3, so the 8 predicted
# targets are 0.1 s apart and the chunk covers 0.8 s. Checked against the
# loaded checkpoint by _check_policy_hz() -- a mismatch would silently run
# the arm at the wrong speed, so it is a hard error rather than an inference.
# FastWAM: CHUNK/POLICY_HZ are only InferenceRecorder defaults here; the client
# overwrites the per-demo attrs with the server's policy_hz and --execute-steps.
POLICY_HZ = 10.0
EXECUTION_HZ = 30.0
INTERPOLATION_SUBSTEPS = max(1, int(round(EXECUTION_HZ / POLICY_HZ)))
EE_DIM, HAND_DIM = 6, 12
STATIC_SERIAL = "341222300710"   # ego cam  -> rgb_static
WRIST_SERIAL = "334622071624"    # wrist cam -> rgb_gripper
EXO_SERIAL = "341522301282"      # exo cam  -> recording only
CAMERA_SOURCE_RGB_SHAPES = {
    "static": (720, 1280),
    "wrist": (480, 640),
    "exo": (480, 640),
}
CAMERA_TARGET_SIZE = 480
CAMERA_JPEG_QUALITY = 95
CAMERA_ROIS_PATH = os.path.join(ROOT, "camera_rois.json")  # FastWAM: copied from vpp_real/assets/
DIRECT_RESIZE_CAMERAS = frozenset(("cam1",))
TACTILE_FINGER_ORDER = ("thumb", "index", "middle", "ring", "little")
TACTILE_TAXEL_ROWS = 10
TACTILE_TAXEL_COLS = 12
TACTILE_TAXELS = TACTILE_TAXEL_ROWS * TACTILE_TAXEL_COLS
TACTILE_TEMPERATURE_VALUES = 20
TACTILE_FORCE_CLIP = 20.0
TACTILE_HEATMAP_SCALE = 10
MAX_JPEG_BYTES = 8 * 1024 * 1024
MAX_TASK_TEXT_BYTES = 4096


def _validate_task_text(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("task text must be a non-empty string")
    text = value.strip()
    if len(text.encode("utf-8")) > MAX_TASK_TEXT_BYTES:
        raise ValueError(f"task text exceeds {MAX_TASK_TEXT_BYTES} UTF-8 bytes")
    return text


def interpolate_targets(state18, targets, substeps=INTERPOLATION_SUBSTEPS):
    """Interpolate absolute targets; rotations follow the SO(3) geodesic.

    The 10 Hz policy and 30 Hz executor use three substeps, i.e. 24 commands
    sent at 30 Hz over the chunk's 0.8 s. Any (POLICY_HZ, substeps) pair
    works as long as commands go out at POLICY_HZ * substeps, which is what
    the robot loop computes -- otherwise the physical horizon is distorted.
    """
    if int(substeps) != substeps or substeps < 1:
        raise ValueError("substeps must be a positive integer")
    substeps = int(substeps)
    keys = np.concatenate([state18[None], targets], axis=0)    # (9,18)
    out = []
    for i in range(len(targets)):
        rotations = Rotation.from_rotvec(keys[i:i + 2, 3:6])
        slerp = Slerp([0.0, 1.0], rotations)
        for k in range(1, substeps + 1):
            fraction = k / substeps
            command = keys[i] + (keys[i + 1] - keys[i]) * fraction
            command[3:6] = slerp([fraction]).as_rotvec()[0]
            out.append(command)
    return np.array(out)


# Workspace, arm motion, XHand ranges, and XHand gains below mirror the current
# Open-Teach implementation so they remain visible in this one-file entrypoint.
# The full xArm joint ranges use UFACTORY's published ranges with 1 degree of
# endpoint headroom; J1 keeps Open-Teach's much narrower +/-90 degree range.
TELEOP_COMMAND_PORTS = (8118, 8119, 8122, 8123)
WORKSPACE_MM = {
    "x": (100.0, 700.0),
    "y": (-400.0, 400.0),
    "z": (50.0, 800.0),
}
XARM_JOINT_LOWER_DEG = np.array([-90.0, -116.0, -359.0, -5.0, -359.0, -96.0, -359.0])
XARM_JOINT_UPPER_DEG = np.array([200.0, 115.0, 359.0, 224.0, 359.0, 179.0, 359.0])
XHAND_SDK_LOWER = np.array(
    [0.0, -1.05, 0.0, -0.087, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
XHAND_SDK_UPPER = np.array(
    [1.57, 1.57, 1.57, 0.2974, 1.92, 1.92, 1.92, 1.92, 1.92, 1.92, 1.92, 1.92])
XHAND_COMMAND_LOWER = np.array(
    [0.0, -1.049, 0.0, -0.086, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
XHAND_COMMAND_UPPER = np.array(
    [1.569, 1.569, 1.569, 0.2964, 1.890, 1.890, 1.890, 1.890,
     1.890, 1.890, 1.890, 1.890])
XHAND_HOME = np.array(
    [0.0, 1.1, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
XHAND_KP, XHAND_KI, XHAND_KD = 100, 0, 50
XHAND_TOR_MAX = np.array([100, 100, 100, 150, 100, 100, 100, 100, 100, 100, 100, 100])
XHAND_CONNECT_TIMEOUT_SECONDS = 10.0
XHAND_READY_TIMEOUT_SECONDS = 15.0
XHAND_STARTUP_POLL_SECONDS = 0.25
XHAND_READY_CONSECUTIVE_READS = 2


class SafetyViolation(RuntimeError):
    """Fail-closed observation, limit, IK, or SDK failure."""


def _finite_vector(value, length, name):
    value = np.asarray(value, dtype=float)
    if value.shape != (length,):
        raise SafetyViolation(f"{name}: expected ({length},), got {value.shape}")
    if not np.all(np.isfinite(value)):
        raise SafetyViolation(f"{name}: non-finite values at {np.flatnonzero(~np.isfinite(value)).tolist()}")
    return value.copy()


def _validate_model_state(value):
    """Validate the exact 18-D observation contract at both network ends."""
    state = _finite_vector(value, EE_DIM + HAND_DIM, "model state")
    pose_mm = state[:EE_DIM].copy()
    pose_mm[:3] *= 1000.0
    if not _inside_workspace(pose_mm, WORKSPACE_MM):
        raise SafetyViolation(
            f"model state xArm pose outside workspace: {pose_mm[:3].tolist()}mm")
    bad_hand = np.flatnonzero(
        (state[EE_DIM:] < XHAND_SDK_LOWER)
        | (state[EE_DIM:] > XHAND_SDK_UPPER)
    ).tolist()
    if bad_hand:
        raise SafetyViolation(f"model state XHand joints outside SDK limits: {bad_hand}")
    return state


def _clip_xhand_command(hand_target, label):
    """Clip a finite 12-D hand target and warn instead of stopping rollout."""
    hand_target = np.asarray(hand_target, dtype=float)
    clipped = np.clip(hand_target, XHAND_COMMAND_LOWER, XHAND_COMMAND_UPPER)
    changed = np.flatnonzero(clipped != hand_target).tolist()
    if changed:
        print(
            f"[warning] {label} XHand joints reached command limits: {changed}; "
            f"requested={hand_target[changed].tolist()} "
            f"deployed={clipped[changed].tolist()}",
            file=sys.stderr,
        )
    return clipped


def _check_xarm(operation, code):
    if not isinstance(code, (int, np.integer)) or isinstance(code, (bool, np.bool_)) or int(code) != 0:
        raise SafetyViolation(f"xArm {operation} failed with SDK status {code!r}")


def _check_xhand(operation, status):
    if status is None or status is True:
        return
    if status is False:
        raise SafetyViolation(f"XHand {operation} failed")
    code = getattr(status, "error_code", getattr(status, "value", status))
    if isinstance(code, (int, np.integer)) and not isinstance(code, (bool, np.bool_)):
        if int(code) == 0:
            return
        raise SafetyViolation(f"XHand {operation} failed with SDK code {int(code)}")
    raise SafetyViolation(f"XHand {operation} returned unknown status {status!r}")


def _extract_tactile_arrays(hand_state):
    """Match OpenTeach's fixed-shape five-finger tactile representation."""
    calc_force = np.full((len(TACTILE_FINGER_ORDER), 3), np.nan, dtype=np.float32)
    raw_force = np.full(
        (len(TACTILE_FINGER_ORDER), TACTILE_TAXELS, 3),
        np.nan,
        dtype=np.float32,
    )
    temperature = np.full(
        (len(TACTILE_FINGER_ORDER), TACTILE_TEMPERATURE_VALUES),
        np.nan,
        dtype=np.float32,
    )
    calc_temperature = np.full(
        (len(TACTILE_FINGER_ORDER),), np.nan, dtype=np.float32)
    tactile = hand_state.get("tactile", {}) if isinstance(hand_state, dict) else {}
    if not isinstance(tactile, dict):
        tactile = {}
    for index, finger in enumerate(TACTILE_FINGER_ORDER):
        data = tactile.get(finger)
        if not isinstance(data, dict):
            continue
        calc = np.asarray(data.get("calc_force", ()), dtype=np.float32)
        if calc.shape == (3,):
            calc_force[index] = calc
        raw = np.asarray(data.get("raw_force", ()), dtype=np.float32)
        if raw.shape == (TACTILE_TAXELS, 3):
            raw_force[index] = raw
        temp = np.asarray(data.get("temperature", ()), dtype=np.float32)
        if temp.shape == (TACTILE_TEMPERATURE_VALUES,):
            temperature[index] = temp
        try:
            calc_temperature[index] = np.float32(data["calc_temperature"])
        except (KeyError, TypeError, ValueError):
            pass
    return calc_force, raw_force, temperature, calc_temperature


def _validated_xhand_tactile(hand_state):
    """Require complete finite five-finger tactile data before recording."""
    arrays = _extract_tactile_arrays(hand_state)
    invalid = []
    for index, finger in enumerate(TACTILE_FINGER_ORDER):
        if any(not np.all(np.isfinite(values[index])) for values in arrays):
            invalid.append(finger)
    if invalid:
        raise SafetyViolation(
            "XHand tactile state missing or invalid for: " + ", ".join(invalid))
    return arrays


def _normalize_hand_state_sample(hand_state):
    """Keep tactile values and their SDK-read timestamps on the 30 Hz grid."""
    calc, raw, temperature, calc_temperature = _validated_xhand_tactile(
        hand_state)
    return {
        "tactile_calc_force": calc,
        "tactile_raw_force": raw,
        "tactile_temperature": temperature,
        "tactile_calc_temperature": calc_temperature,
        "hand_state_monotonic_ns": int(hand_state.get("sample_monotonic_ns", -1)),
        "hand_state_read_start_monotonic_ns": int(
            hand_state.get("read_start_monotonic_ns", -1)),
        "hand_state_read_end_monotonic_ns": int(
            hand_state.get("read_end_monotonic_ns", -1)),
        "hand_state_seq": int(hand_state.get("sample_seq", -1)),
    }


def _tactile_heatmap_jpeg(raw_force):
    """Encode OpenTeach's 10x12 RGB force grid, enlarged 10x for viewing."""
    raw = np.asarray(raw_force, dtype=np.float32)
    if raw.shape != (TACTILE_TAXELS, 3):
        grid = np.full(
            (TACTILE_TAXEL_ROWS, TACTILE_TAXEL_COLS, 3), 128, dtype=np.uint8)
    else:
        nan_mask = np.isnan(raw).any(axis=-1)
        clipped = np.clip(raw, -TACTILE_FORCE_CLIP, TACTILE_FORCE_CLIP)
        scaled = (
            (clipped + TACTILE_FORCE_CLIP)
            / (2 * TACTILE_FORCE_CLIP)
            * 255.0
        )
        scaled[nan_mask] = 128.0
        grid = scaled.reshape(
            TACTILE_TAXEL_ROWS, TACTILE_TAXEL_COLS, 3).astype(np.uint8)
    image = cv2.resize(
        grid,
        (
            TACTILE_TAXEL_COLS * TACTILE_HEATMAP_SCALE,
            TACTILE_TAXEL_ROWS * TACTILE_HEATMAP_SCALE,
        ),
        interpolation=cv2.INTER_NEAREST,
    )
    ok, encoded = cv2.imencode(
        ".jpg",
        image[:, :, ::-1],
        [int(cv2.IMWRITE_JPEG_QUALITY), 95],
    )
    if not ok:
        raise RuntimeError("XHand tactile heatmap JPEG encoding failed")
    return np.asarray(encoded, dtype=np.uint8).reshape(-1)


def _inside_workspace(pose_mm_rad, workspace):
    pose = _finite_vector(pose_mm_rad, EE_DIM, "xArm pose")
    return (workspace["x"][0] <= pose[0] <= workspace["x"][1]
            and workspace["y"][0] <= pose[1] <= workspace["y"][1]
            and workspace["z"][0] <= pose[2] <= workspace["z"][1])


def _decode_camera_jpeg(
    name, jpeg, expected_shapes=CAMERA_SOURCE_RGB_SHAPES, *, rgb=True
):
    if not isinstance(jpeg, bytes) or not jpeg or len(jpeg) > MAX_JPEG_BYTES:
        size = len(jpeg) if isinstance(jpeg, bytes) else None
        raise SafetyViolation(f"{name} camera JPEG has invalid size: {size}")
    bgr = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise SafetyViolation(f"{name} camera JPEG decode failed")
    if bgr.shape[:2] != expected_shapes[name]:
        raise SafetyViolation(
            f"{name} camera resolution {bgr.shape[1]}x{bgr.shape[0]} does not match "
            f"expected {expected_shapes[name][1]}x{expected_shapes[name][0]}")
    return bgr[:, :, ::-1].copy() if rgb else bgr


def _load_camera_transforms(path, include_exo=False):
    """Load the fixed Open-Teach camera transforms used to rewrite training data."""
    try:
        roi_path = Path(path).expanduser().resolve()
        payload = json.loads(roi_path.read_text(encoding="utf-8"))
        if int(payload["target_size"]) != CAMERA_TARGET_SIZE:
            raise ValueError(
                f"target_size={payload['target_size']}, expected {CAMERA_TARGET_SIZE}")
        cameras = payload["cameras"]
        transforms = {}
        role_serials = [
            ("static", STATIC_SERIAL),
            ("wrist", WRIST_SERIAL),
        ]
        if include_exo:
            role_serials.append(("exo", EXO_SERIAL))
        for role, serial in role_serials:
            matches = [
                (camera, item)
                for camera, item in cameras.items()
                if str(item.get("camera_serial")) == serial
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"expected one ROI for {role} serial {serial}, got {len(matches)}")
            camera, item = matches[0]
            mode = (
                "full_frame_resize"
                if camera in DIRECT_RESIZE_CAMERAS
                else "square_crop_resize"
            )
            if item.get("mode", mode) != mode:
                raise ValueError(
                    f"{camera} mode={item.get('mode')!r}, expected {mode!r}")
            transform = {
                "camera": camera,
                "mode": mode,
                "x": int(item["x"]),
                "y": int(item["y"]),
                "size": int(item["size"]),
                "source_width": int(item["source_width"]),
                "source_height": int(item["source_height"]),
            }
            if transform["source_width"] <= 0 or transform["source_height"] <= 0:
                raise ValueError(f"{camera} source dimensions must be positive")
            if mode == "square_crop_resize" and (
                transform["x"] < 0
                or transform["y"] < 0
                or transform["size"] <= 0
                or transform["x"] + transform["size"] > transform["source_width"]
                or transform["y"] + transform["size"] > transform["source_height"]
            ):
                raise ValueError(f"{camera} square ROI is outside the source image")
            transforms[role] = transform
        return transforms
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise SafetyViolation(f"invalid camera ROI file {path}: {exc}") from exc


def _transform_camera_for_model(name, source_bgr, transform):
    """Apply the same crop/resize/JPEG settings used by the HDF5 rewrite."""
    height, width = source_bgr.shape[:2]
    expected = (transform["source_width"], transform["source_height"])
    if (width, height) != expected:
        raise SafetyViolation(
            f"{name} camera source resolution {width}x{height} does not match "
            f"ROI source {expected[0]}x{expected[1]}")
    if transform["mode"] == "full_frame_resize":
        cropped = source_bgr
        resize_source = min(width, height)
    else:
        x, y, size = transform["x"], transform["y"], transform["size"]
        cropped = source_bgr[y:y + size, x:x + size]
        if cropped.shape[:2] != (size, size):
            raise SafetyViolation(f"{name} camera ROI produced {cropped.shape[:2]}")
        resize_source = size
    interpolation = (
        cv2.INTER_AREA if resize_source >= CAMERA_TARGET_SIZE else cv2.INTER_LINEAR)
    resized = cv2.resize(
        cropped,
        (CAMERA_TARGET_SIZE, CAMERA_TARGET_SIZE),
        interpolation=interpolation,
    )
    ok, encoded = cv2.imencode(
        ".jpg",
        resized,
        [int(cv2.IMWRITE_JPEG_QUALITY), CAMERA_JPEG_QUALITY],
    )
    if not ok:
        raise SafetyViolation(f"{name} camera JPEG encoding failed")
    jpeg = np.asarray(encoded, dtype=np.uint8).reshape(-1).tobytes()
    if not jpeg or len(jpeg) > MAX_JPEG_BYTES:
        raise SafetyViolation(f"{name} cropped camera JPEG has invalid size")
    return resized[:, :, ::-1].copy(), jpeg


def _validate_xarm_joints(joints_rad, name="xArm joints"):
    joints = _finite_vector(joints_rad, 7, name)
    lower, upper = np.deg2rad(XARM_JOINT_LOWER_DEG), np.deg2rad(XARM_JOINT_UPPER_DEG)
    bad = np.flatnonzero((joints < lower) | (joints > upper)).tolist()
    if bad:
        details = [f"J{i + 1}={np.rad2deg(joints[i]):.2f}deg" for i in bad]
        raise SafetyViolation(f"{name} outside operational range: {', '.join(details)}")
    return joints


def _active_teleop_ports():
    active = []
    for port in TELEOP_COMMAND_PORTS:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.05):
                active.append(port)
        except OSError:
            pass
    return active


class OpenTeachCameraStream:
    """Threaded latest-frame subscriber adapted from Open-Teach's collector."""

    TOPIC = b"rgb_image "

    def __init__(self, name, host, port, expected_serial, transform):
        import zmq

        self.zmq = zmq
        self.name = name
        self.host = host
        self.port = int(port)
        self.expected_serial = str(expected_serial)
        self.transform = transform
        self._condition = threading.Condition()
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()
        self._latest = None
        self._error = None
        self._prepare_lock = threading.Lock()
        self._prepared_source_ns = None
        self._prepared_packet = None
        self._thread = threading.Thread(
            target=self._receive_loop,
            name=f"vpp-camera-zmq-{self.port}",
            daemon=True,
        )
        self._thread.start()
        self._ready_event.wait(timeout=1.0)

    def decode_message(self, raw, received_monotonic_ns):
        if not raw.startswith(self.TOPIC):
            raise SafetyViolation(f"{self.name} camera sent an unexpected topic")
        try:
            packet = pickle.loads(raw[len(self.TOPIC):])
            serial = str(packet["camera_serial"])
            source_ns = int(packet["host_monotonic_ns"])
            device_timestamp_ms = float(packet["device_timestamp_ms"])
            frame_number = int(packet["frame_number"])
            jpeg = base64.b64decode(packet["rgb_image"])
        except Exception as exc:
            raise SafetyViolation(
                f"{self.name} camera packet is missing required metadata") from exc
        if not jpeg or len(jpeg) > MAX_JPEG_BYTES:
            raise SafetyViolation(
                f"{self.name} camera JPEG has invalid size: {len(jpeg)}")
        if serial != self.expected_serial:
            raise SafetyViolation(
                f"{self.name} camera serial mismatch: expected "
                f"{self.expected_serial}, got {serial}")
        return {
            "source_jpeg": jpeg,
            "camera_serial": serial,
            "host_monotonic_ns": source_ns,
            "device_timestamp_ms": device_timestamp_ms,
            "frame_number": frame_number,
            "received_monotonic_ns": int(received_monotonic_ns),
        }

    def prepare(self, packet):
        """Apply the workstation ROI once for each newly received source frame."""
        source_ns = int(packet["host_monotonic_ns"])
        with self._prepare_lock:
            if self._prepared_source_ns == source_ns:
                return self._prepared_packet
            if self.transform is None:
                model_rgb = None
                model_jpeg = packet["source_jpeg"]
            else:
                source_bgr = _decode_camera_jpeg(
                    self.name, packet["source_jpeg"], rgb=False)
                model_rgb, model_jpeg = _transform_camera_for_model(
                    self.name, source_bgr, self.transform)
            prepared = {
                "rgb": model_rgb,
                "jpeg": model_jpeg,
                "camera_serial": packet["camera_serial"],
                "host_monotonic_ns": source_ns,
                "device_timestamp_ms": packet["device_timestamp_ms"],
                "frame_number": packet["frame_number"],
                "received_monotonic_ns": packet["received_monotonic_ns"],
            }
            self._prepared_source_ns = source_ns
            self._prepared_packet = prepared
            return prepared

    def _receive_loop(self):
        context = self.zmq.Context()
        stream = context.socket(self.zmq.SUB)
        stream.setsockopt(self.zmq.LINGER, 0)
        stream.setsockopt(self.zmq.CONFLATE, 1)
        stream.setsockopt(self.zmq.SUBSCRIBE, b"rgb_image")
        stream.connect(f"tcp://{self.host}:{self.port}")
        self._ready_event.set()
        try:
            while not self._stop_event.is_set():
                if not stream.poll(100, self.zmq.POLLIN):
                    continue
                packet = self.decode_message(stream.recv(), time.monotonic_ns())
                self._accept_packet(packet)
        except BaseException as exc:
            with self._condition:
                self._error = exc
                self._condition.notify_all()
        finally:
            stream.close(linger=0)
            context.term()

    def _accept_packet(self, packet):
        with self._condition:
            if (self._latest is not None
                    and packet["host_monotonic_ns"]
                    <= self._latest["host_monotonic_ns"]):
                raise SafetyViolation(
                    f"{self.name} camera timestamp did not advance")
            self._latest = packet
            self._condition.notify_all()

    def latest(self):
        with self._condition:
            if self._error is not None:
                raise SafetyViolation(
                    f"{self.name} camera stream failed") from self._error
            return self._latest

    def wait_for_new(self, previous_source_ns, timeout_ms):
        deadline = time.monotonic() + timeout_ms / 1000.0
        with self._condition:
            while True:
                if self._error is not None:
                    raise SafetyViolation(
                        f"{self.name} camera stream failed") from self._error
                if (self._latest is not None
                        and (previous_source_ns is None
                             or self._latest["host_monotonic_ns"]
                             > previous_source_ns)):
                    return self._latest
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise SafetyViolation(
                        f"{self.name} camera timeout after {timeout_ms}ms")
                self._condition.wait(timeout=remaining)

    def close(self):
        self._stop_event.set()
        with self._condition:
            self._condition.notify_all()
        self._thread.join(timeout=1.0)


class OpenTeachCameraPair:
    """Share two model cameras plus the optional recording-only exo camera."""

    def __init__(
        self, host, static_port, wrist_port, timeout_ms, max_age_ms, camera_rois,
        exo_port=None,
    ):
        self.host = host
        self.timeout_ms = timeout_ms
        self.max_age_ns = int(max_age_ms * 1e6)
        self.same_host_clock = host in ("127.0.0.1", "localhost", "::1")
        serials = {"static": STATIC_SERIAL, "wrist": WRIST_SERIAL}
        transforms = _load_camera_transforms(
            camera_rois, include_exo=exo_port is not None)
        ports = {"static": static_port, "wrist": wrist_port}
        if exo_port is not None:
            serials["exo"] = EXO_SERIAL
            ports["exo"] = exo_port
        self.streams = {}
        try:
            for name in ports:
                self.streams[name] = OpenTeachCameraStream(
                    name, host, ports[name], serials[name], transforms[name])
        except Exception:
            self.close()
            raise
        self.last_read_source_ns = {"static": None, "wrist": None}

    def _validate_age(self, name, packet):
        if self.same_host_clock:
            age_ns = time.monotonic_ns() - packet["host_monotonic_ns"]
            if age_ns < -50_000_000 or age_ns > self.max_age_ns:
                raise SafetyViolation(
                    f"{name} camera frame age is {age_ns / 1e6:.1f}ms")

    def read_packets(self):
        packets = []
        for name in ("static", "wrist"):
            packet = self.streams[name].wait_for_new(
                self.last_read_source_ns[name], self.timeout_ms)
            self._validate_age(name, packet)
            self.last_read_source_ns[name] = packet["host_monotonic_ns"]
            packets.append(self.streams[name].prepare(packet))
        return tuple(packets)

    def latest_packets(self):
        """Return the two post-ROI frames consumed by the policy."""
        packets = tuple(
            self.streams[name].latest() for name in ("static", "wrist"))
        if any(packet is None for packet in packets):
            return None
        return tuple(
            self.streams[name].prepare(packet)
            for name, packet in zip(("static", "wrist"), packets)
        )

    def latest_recording_packets(self):
        """Return all three post-ROI 480x480 camera JPEGs for recording."""
        names = ("static", "wrist", "exo")
        if "exo" not in self.streams:
            raise SafetyViolation("exo camera was not opened for recording")
        packets = tuple(self.streams[name].latest() for name in names)
        if any(packet is None for packet in packets):
            return None
        return tuple(
            self.streams[name].prepare(packet)
            for name, packet in zip(names, packets)
        )

    def wait_for_recording_packets(self):
        """Require all three camera streams before the rollout can start."""
        names = ("static", "wrist", "exo")
        if "exo" not in self.streams:
            raise SafetyViolation("exo camera was not opened for recording")
        packets = []
        for name in names:
            packet = self.streams[name].wait_for_new(None, self.timeout_ms)
            self._validate_age(name, packet)
            packets.append(self.streams[name].prepare(packet))
        return tuple(packets)

    def close(self):
        for stream in getattr(self, "streams", {}).values():
            stream.close()


class InferenceRecorder:
    """Record a workstation rollout on one fixed-rate HDF5 timeline.

    All three cameras retain their post-ROI 480x480 JPEGs. Robot, tactile, and
    camera values are sampled on the same 30 Hz timeline.
    """

    DATASET_TYPE = "vpp_workstation_inference_rollout_v1"
    CAMERA_NAMES = ("cam0", "cam1", "cam2")
    CAMERA_SERIALS = (STATIC_SERIAL, WRIST_SERIAL, EXO_SERIAL)
    COMMANDS_PER_CHUNK = CHUNK * INTERPOLATION_SUBSTEPS

    def __init__(self, path, args):
        try:
            import h5py
        except ImportError as exc:
            raise RuntimeError(
                "--record-h5 requires h5py in the workstation environment"
            ) from exc

        self.h5py = h5py
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = None
        self.demo = None
        self.datasets = {}
        self._h5_lock = threading.Lock()
        self._source_lock = threading.Lock()
        self._fixed_stop_event = threading.Event()
        self._fixed_thread = None
        self._fixed_error = None
        self._fixed_cameras = None
        self._latest_robot_sample = None
        self._state_revision = 0
        self._arm_revision = 0
        self._hand_state_revision = 0
        self._target_revision = 0
        self._last_fixed_revisions = {
            "state": -1, "arm": -1, "hand": -1, "target": -1}
        self._last_fixed_frame_numbers = {
            name: None for name in self.CAMERA_NAMES}
        self._fixed_missed_deadlines = 0
        self._last_fixed_flush_ns = 0
        self.start_wall_time_ns = time.time_ns()
        self.start_monotonic_ns = time.monotonic_ns()
        try:
            self.file = h5py.File(self.path, "a")
            existing_type = self.file.attrs.get("dataset_type")
            if isinstance(existing_type, bytes):
                existing_type = existing_type.decode("utf-8")
            data = self.file.require_group("data")
            if len(data) and existing_type != self.DATASET_TYPE:
                raise ValueError(
                    f"refusing to mix inference rollouts with existing HDF5 data in "
                    f"{self.path}; use a dedicated --record-h5 path"
                )
            if existing_type not in (None, self.DATASET_TYPE):
                raise ValueError(
                    f"unsupported dataset_type={existing_type!r} in {self.path}"
                )
            indices = [
                int(key[5:]) for key in data
                if key.startswith("demo_") and key[5:].isdigit()
            ]
            self.demo_key = f"demo_{max(indices, default=-1) + 1}"
            self.demo = data.create_group(self.demo_key)
            self._create_layout()
            self._set_start_metadata(args)
            self.file.flush()
        except Exception:
            if self.file is not None:
                self.file.close()
                self.file = None
            raise
        print(f"[recording] writing {self.demo_key} to {self.path}")

    def _numeric_dataset(self, group, name, tail_shape, dtype, fillvalue):
        dataset = group.create_dataset(
            name,
            shape=(0, *tail_shape),
            maxshape=(None, *tail_shape),
            chunks=(1, *tail_shape) if tail_shape else (64,),
            dtype=dtype,
            fillvalue=fillvalue,
        )
        self.datasets[f"{group.name}/{name}"] = dataset
        return dataset

    def _create_layout(self):
        obs = self.demo.create_group("obs")
        timestamps = self.demo.create_group("timestamps")
        alignment = self.demo.create_group("alignment")
        jpeg_dtype = self.h5py.vlen_dtype(np.dtype("uint8"))
        for name in self.CAMERA_NAMES:
            dataset = obs.create_dataset(
                f"{name}_rgb_jpeg",
                shape=(0,), maxshape=(None,), chunks=(64,), dtype=jpeg_dtype,
            )
            self.datasets[f"{obs.name}/{name}_rgb_jpeg"] = dataset
        self._numeric_dataset(obs, "model_state", (EE_DIM + HAND_DIM,), np.float64, np.nan)
        self._numeric_dataset(obs, "arm_ee_state", (EE_DIM,), np.float64, np.nan)
        self._numeric_dataset(
            obs, "hand_joint_states_measured", (HAND_DIM,), np.float64, np.nan)
        self._numeric_dataset(
            obs, "arm_joint_states", (7,), np.float64, np.nan)
        self._numeric_dataset(
            obs, "tactile_calc_force",
            (len(TACTILE_FINGER_ORDER), 3), np.float32, np.nan)
        self._numeric_dataset(
            obs, "tactile_raw_force",
            (len(TACTILE_FINGER_ORDER), TACTILE_TAXELS, 3), np.float32, np.nan)
        self._numeric_dataset(
            obs, "tactile_temperature",
            (len(TACTILE_FINGER_ORDER), TACTILE_TEMPERATURE_VALUES),
            np.float32, np.nan)
        self._numeric_dataset(
            obs, "tactile_calc_temperature",
            (len(TACTILE_FINGER_ORDER),), np.float32, np.nan)
        self._numeric_dataset(
            self.demo, "executor_targets", (EE_DIM + HAND_DIM,),
            np.float64, np.nan)
        for name in (
            "state_policy_chunk_index", "state_command_index",
            "executor_target_policy_chunk_index", "executor_target_command_index",
        ):
            self._numeric_dataset(self.demo, name, (), np.int64, -1)
        for name in (
            "step_monotonic_ns", "state_source_monotonic_ns",
            "state_available_monotonic_ns", "executor_target_monotonic_ns",
            "hand_state_monotonic_ns", "hand_state_read_start_monotonic_ns",
            "hand_state_read_end_monotonic_ns", "hand_state_seq",
        ):
            self._numeric_dataset(timestamps, name, (), np.int64, -1)
        self._numeric_dataset(
            timestamps, "hand_state_age_ms", (), np.float64, np.nan)
        for name in self.CAMERA_NAMES:
            for suffix in ("monotonic_ns", "received_monotonic_ns", "frame_number"):
                self._numeric_dataset(
                    timestamps, f"{name}_{suffix}", (), np.int64, -1)
            self._numeric_dataset(
                timestamps, f"{name}_device_timestamp_ms", (), np.float64, np.nan)
        for name in (
            *(f"{camera}_is_new" for camera in self.CAMERA_NAMES),
            "model_state_is_new", "arm_joint_state_is_new",
            "xhand_state_is_new", "executor_target_is_new",
        ):
            self._numeric_dataset(alignment, name, (), np.uint8, 0)
        self.demo.attrs["target_hz"] = EXECUTION_HZ
        self.demo.attrs["num_samples"] = 0
        self.demo.attrs["missed_deadlines"] = 0

    def _set_start_metadata(self, args):
        record_task = getattr(args, "record_task", None)
        task = _validate_task_text(
            record_task if record_task is not None else args.task_text
        )
        mode = str(getattr(args, "mode", "robot-client"))
        execution = str(getattr(args, "execution", "shadow"))
        self.file.attrs["dataset_type"] = self.DATASET_TYPE
        self.file.attrs["camera_names"] = list(self.CAMERA_NAMES)
        self.file.attrs["camera_serials"] = list(self.CAMERA_SERIALS)
        self.file.attrs["camera_order_note"] = (
            "cam0=static/ego, cam1=wrist, cam2=exo; per-demo metadata is authoritative"
        )
        self.file.attrs["model_state_order"] = (
            "x_m,y_m,z_m,rx_rad,ry_rad,rz_rad,12 measured XHand joints"
        )
        self.file.attrs["arm_ee_state_order"] = (
            "x_m,y_m,z_m,rx_rad,ry_rad,rz_rad (axis-angle, not RPY)"
        )
        self.file.attrs["arm_joint_states_order"] = "joint_1..joint_7 (radians)"
        self.file.attrs["camera_image_note"] = (
            "obs/cam0_rgb_jpeg, obs/cam1_rgb_jpeg, and obs/cam2_rgb_jpeg store "
            "post-ROI 480x480 JPEGs; cam0/cam1 are used for inference and cam2 "
            "is recording-only, all sampled on the fixed-rate 30Hz timeline"
        )
        self.file.attrs["tactile_finger_order"] = list(TACTILE_FINGER_ORDER)
        self.file.attrs["tactile_heatmap_note"] = (
            "obs/tactile_heatmap_{finger}_jpeg is OpenTeach's lossy RGB force "
            "visualization of the 10x12 raw taxel grid, enlarged 10x; "
            "obs/tactile_raw_force is authoritative"
        )
        self.file.attrs["monotonic_clock_note"] = (
            "Monotonic timestamps are comparable only on this workstation "
            "during the same boot; RealSense device timestamps use camera clocks"
        )

        self.demo.attrs["source_live_robot_session"] = True
        self.demo.attrs["source_real_world_episode"] = execution == "physical"
        self.demo.attrs["task_description"] = task
        self.demo.attrs["mode"] = mode
        self.demo.attrs["execution"] = execution
        self.demo.attrs["status"] = "running"
        self.demo.attrs["task_score"] = -1
        self.demo.attrs["termination_reason"] = "running"
        self.demo.attrs["recording_start_time"] = self.start_wall_time_ns / 1e9
        self.demo.attrs["recording_start_wall_time_ns"] = self.start_wall_time_ns
        self.demo.attrs["recording_start_monotonic_ns"] = self.start_monotonic_ns
        self.demo.attrs["policy_hz"] = POLICY_HZ
        self.demo.attrs["execution_hz"] = EXECUTION_HZ
        self.demo.attrs["record_hz"] = EXECUTION_HZ
        self.demo.attrs["chunk_size"] = CHUNK
        self.demo.attrs["commands_per_chunk"] = self.COMMANDS_PER_CHUNK
        self.demo.attrs["camera_names"] = list(self.CAMERA_NAMES)
        self.demo.attrs["camera_serials"] = list(self.CAMERA_SERIALS)
        self.demo.attrs["camera_rois"] = str(
            Path(getattr(args, "camera_rois", CAMERA_ROIS_PATH)).expanduser())
        self.demo.attrs["executor_targets_note"] = (
            "Targets after arm step clamping and XHand command-limit clipping. "
            "Whether SDK writes occurred is determined by the execution attr: "
            "shadow=none, controller-sim=xArm only, physical=xArm and XHand."
        )
        self.demo.attrs["recording_timeline_note"] = (
            "The demo datasets use one absolute-monotonic 30Hz timeline. "
            "Each row snapshots the newest cached camera, robot, tactile, and "
            "executor-target values already obtained by the main control thread; "
            "it never performs concurrent robot SDK reads. Reused values are "
            "identified under alignment."
        )

    @staticmethod
    def _array_or_none(value, shape):
        if value is None:
            return None
        try:
            array = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError):
            return None
        return array if array.shape == shape else None

    def update_latest_robot_sample(
        self, *, state, state_source_ns, state_available_ns, arm_joints=None,
        hand_state_sample=None, executor_target=None, executor_target_ns=None,
        policy_chunk_index=-1, command_index=-1,
    ):
        """Publish main-thread SDK results for nonblocking 30 Hz snapshots."""
        state = _validate_model_state(state).copy()
        arm = self._array_or_none(arm_joints, (7,))
        target = self._array_or_none(
            executor_target, (EE_DIM + HAND_DIM,))
        hand_sample = (
            _normalize_hand_state_sample(hand_state_sample)
            if isinstance(hand_state_sample, dict) else None
        )
        with self._source_lock:
            previous = self._latest_robot_sample or {
                "arm_joints": np.full(7, np.nan),
                "tactile_calc_force": np.full(
                    (len(TACTILE_FINGER_ORDER), 3), np.nan, dtype=np.float32),
                "tactile_raw_force": np.full(
                    (len(TACTILE_FINGER_ORDER), TACTILE_TAXELS, 3),
                    np.nan, dtype=np.float32),
                "tactile_temperature": np.full(
                    (len(TACTILE_FINGER_ORDER), TACTILE_TEMPERATURE_VALUES),
                    np.nan, dtype=np.float32),
                "tactile_calc_temperature": np.full(
                    (len(TACTILE_FINGER_ORDER),), np.nan, dtype=np.float32),
                "hand_state_monotonic_ns": -1,
                "hand_state_read_start_monotonic_ns": -1,
                "hand_state_read_end_monotonic_ns": -1,
                "hand_state_seq": -1,
                "executor_target": np.full(EE_DIM + HAND_DIM, np.nan),
                "executor_target_ns": -1,
                "executor_target_policy_chunk_index": -1,
                "executor_target_command_index": -1,
            }
            self._state_revision += 1
            sample = {
                **previous,
                "state": state,
                "state_source_ns": int(state_source_ns),
                "state_available_ns": int(state_available_ns),
                "state_policy_chunk_index": int(policy_chunk_index),
                "state_command_index": int(command_index),
                "state_revision": self._state_revision,
                "arm_revision": self._arm_revision,
                "hand_state_revision": self._hand_state_revision,
                "target_revision": self._target_revision,
            }
            if arm is not None:
                self._arm_revision += 1
                sample["arm_joints"] = arm.copy()
                sample["arm_revision"] = self._arm_revision
            if hand_sample is not None:
                self._hand_state_revision += 1
                sample.update(hand_sample)
                sample["hand_state_revision"] = self._hand_state_revision
            if target is not None:
                self._target_revision += 1
                sample["executor_target"] = target.copy()
                sample["executor_target_ns"] = int(executor_target_ns)
                sample["executor_target_policy_chunk_index"] = int(
                    policy_chunk_index)
                sample["executor_target_command_index"] = int(command_index)
                sample["target_revision"] = self._target_revision
            self._latest_robot_sample = sample

    def _snapshot_latest_robot_sample(self):
        with self._source_lock:
            if self._latest_robot_sample is None:
                return None
            return {
                key: value.copy() if isinstance(value, np.ndarray) else value
                for key, value in self._latest_robot_sample.items()
            }

    def _append_fixed_rate_sample(self, packets, robot, step_monotonic_ns):
        with self._h5_lock:
            if self.file is None:
                return
            index = int(self.demo.attrs.get("num_samples", 0))
            for dataset in self.datasets.values():
                dataset.resize(index + 1, axis=0)

            obs = self.demo["obs"]
            timestamps = self.demo["timestamps"]
            alignment = self.demo["alignment"]
            for name, packet in zip(self.CAMERA_NAMES, packets):
                obs[f"{name}_rgb_jpeg"][index] = np.frombuffer(
                    bytes(packet["jpeg"]), dtype=np.uint8)
                timestamps[f"{name}_monotonic_ns"][index] = int(
                    packet["host_monotonic_ns"])
                timestamps[f"{name}_received_monotonic_ns"][index] = int(
                    packet.get("received_monotonic_ns", -1))
                timestamps[f"{name}_device_timestamp_ms"][index] = float(
                    packet.get("device_timestamp_ms", np.nan))
                frame_number = int(packet.get("frame_number", -1))
                timestamps[f"{name}_frame_number"][index] = frame_number
                previous_frame = self._last_fixed_frame_numbers[name]
                alignment[f"{name}_is_new"][index] = int(
                    previous_frame is None or frame_number != previous_frame)
                self._last_fixed_frame_numbers[name] = frame_number

            state = robot["state"]
            obs["model_state"][index] = state
            obs["arm_ee_state"][index] = state[:EE_DIM]
            obs["hand_joint_states_measured"][index] = state[EE_DIM:]
            obs["arm_joint_states"][index] = robot["arm_joints"]
            obs["tactile_calc_force"][index] = robot["tactile_calc_force"]
            obs["tactile_raw_force"][index] = robot["tactile_raw_force"]
            obs["tactile_temperature"][index] = robot["tactile_temperature"]
            obs["tactile_calc_temperature"][index] = robot[
                "tactile_calc_temperature"]
            self.demo["executor_targets"][index] = robot["executor_target"]
            self.demo["state_policy_chunk_index"][index] = robot[
                "state_policy_chunk_index"]
            self.demo["state_command_index"][index] = robot[
                "state_command_index"]
            self.demo["executor_target_policy_chunk_index"][index] = robot[
                "executor_target_policy_chunk_index"]
            self.demo["executor_target_command_index"][index] = robot[
                "executor_target_command_index"]
            timestamps["step_monotonic_ns"][index] = int(step_monotonic_ns)
            timestamps["state_source_monotonic_ns"][index] = robot[
                "state_source_ns"]
            timestamps["state_available_monotonic_ns"][index] = robot[
                "state_available_ns"]
            timestamps["executor_target_monotonic_ns"][index] = robot[
                "executor_target_ns"]
            for name in (
                "hand_state_monotonic_ns", "hand_state_read_start_monotonic_ns",
                "hand_state_read_end_monotonic_ns", "hand_state_seq",
            ):
                timestamps[name][index] = robot[name]
            hand_state_ns = int(robot["hand_state_monotonic_ns"])
            if hand_state_ns >= 0:
                timestamps["hand_state_age_ms"][index] = (
                    step_monotonic_ns - hand_state_ns) / 1e6

            revisions = {
                "state": int(robot["state_revision"]),
                "arm": int(robot["arm_revision"]),
                "hand": int(robot["hand_state_revision"]),
                "target": int(robot["target_revision"]),
            }
            alignment["model_state_is_new"][index] = int(
                revisions["state"] != self._last_fixed_revisions["state"])
            alignment["arm_joint_state_is_new"][index] = int(
                revisions["arm"] > 0
                and revisions["arm"] != self._last_fixed_revisions["arm"])
            alignment["xhand_state_is_new"][index] = int(
                revisions["hand"] > 0
                and revisions["hand"] != self._last_fixed_revisions["hand"])
            alignment["executor_target_is_new"][index] = int(
                revisions["target"] > 0
                and revisions["target"] != self._last_fixed_revisions["target"])
            self._last_fixed_revisions = revisions
            self.demo.attrs["num_samples"] = index + 1
            now_ns = time.monotonic_ns()
            if now_ns - self._last_fixed_flush_ns >= 1_000_000_000:
                self.file.flush()
                self._last_fixed_flush_ns = now_ns

    def _sample_fixed_rate_once(self):
        packets = self._fixed_cameras.latest_recording_packets()
        robot = self._snapshot_latest_robot_sample()
        if packets is None or robot is None:
            return False
        self._append_fixed_rate_sample(packets, robot, time.monotonic_ns())
        return True

    def _fixed_rate_loop(self, next_deadline_ns):
        period_ns = int(round(1e9 / EXECUTION_HZ))
        try:
            while not self._fixed_stop_event.is_set():
                remaining_ns = next_deadline_ns - time.monotonic_ns()
                if remaining_ns > 0 and self._fixed_stop_event.wait(
                        remaining_ns / 1e9):
                    break
                now_ns = time.monotonic_ns()
                if now_ns - next_deadline_ns >= period_ns:
                    missed = (now_ns - next_deadline_ns) // period_ns
                    self._fixed_missed_deadlines += int(missed)
                    next_deadline_ns += int(missed) * period_ns
                self._sample_fixed_rate_once()
                next_deadline_ns += period_ns
        except BaseException as exc:
            with self._source_lock:
                self._fixed_error = exc
            self._fixed_stop_event.set()

    def start_fixed_rate(
        self, cameras, *, state, state_monotonic_ns, chunk_index,
        arm_joints=None, hand_state_sample=None,
    ):
        if self._fixed_thread is not None:
            return
        self._fixed_cameras = cameras
        self.update_latest_robot_sample(
            state=state,
            arm_joints=arm_joints,
            hand_state_sample=hand_state_sample,
            state_source_ns=state_monotonic_ns,
            state_available_ns=state_monotonic_ns,
            policy_chunk_index=chunk_index,
            command_index=-1,
        )
        self._sample_fixed_rate_once()
        period_ns = int(round(1e9 / EXECUTION_HZ))
        self._fixed_thread = threading.Thread(
            target=self._fixed_rate_loop,
            args=(time.monotonic_ns() + period_ns,),
            name="vpp-rollout-30hz",
            daemon=True,
        )
        self._fixed_thread.start()

    @property
    def is_running(self):
        return self._fixed_thread is not None

    def stop_fixed_rate(self):
        thread = self._fixed_thread
        if thread is None:
            return
        self._fixed_stop_event.set()
        thread.join(timeout=2.0)
        if thread.is_alive():
            raise RuntimeError("30 Hz recorder thread did not stop")
        self._fixed_thread = None
        if self.demo is not None:
            self.demo.attrs["missed_deadlines"] = self._fixed_missed_deadlines

    def raise_if_fixed_rate_failed(self):
        with self._source_lock:
            error = self._fixed_error
        if error is not None:
            raise SafetyViolation(f"30 Hz recorder failed: {error}") from error

    def _write_tactile_heatmaps(self):
        """Generate the five OpenTeach heatmap streams only for a kept demo."""
        obs = self.demo["obs"]
        raw_force = obs["tactile_raw_force"]
        jpeg_dtype = self.h5py.vlen_dtype(np.dtype("uint8"))
        for finger_index, finger in enumerate(TACTILE_FINGER_ORDER):
            dataset = obs.create_dataset(
                f"tactile_heatmap_{finger}_jpeg",
                shape=(len(raw_force),),
                maxshape=(None,),
                chunks=(64,),
                dtype=jpeg_dtype,
            )
            for row in range(len(raw_force)):
                dataset[row] = _tactile_heatmap_jpeg(
                    raw_force[row, finger_index])

    def close(self, error=None, task_score=None, termination_reason=None):
        self.stop_fixed_rate()
        if self.file is None:
            return
        with self._source_lock:
            fixed_error = self._fixed_error
        raise_after_close = error is None and fixed_error is not None
        if raise_after_close:
            error = SafetyViolation(f"30 Hz recorder failed: {fixed_error}")
        end_wall_time_ns = time.time_ns()
        end_monotonic_ns = time.monotonic_ns()
        if error is None:
            status = "completed"
        elif isinstance(error, KeyboardInterrupt):
            status = "interrupted"
        else:
            status = "failed"
        with self._h5_lock:
            self._write_tactile_heatmaps()
            self.demo.attrs["status"] = status
            self.demo.attrs["completed_without_error"] = error is None
            if task_score is not None:
                task_score = int(task_score)
                if task_score not in range(4):
                    raise ValueError(
                        f"task score must be 0, 1, 2, or 3; got {task_score}")
                self.demo.attrs["task_score"] = task_score
            if termination_reason is None:
                if isinstance(error, KeyboardInterrupt):
                    termination_reason = "keyboard_interrupt"
                elif error is not None:
                    termination_reason = "failure"
                else:
                    termination_reason = "max_chunks"
            self.demo.attrs["termination_reason"] = termination_reason
            self.demo.attrs["terminal_error"] = (
                "" if error is None else f"{type(error).__name__}: {error}"
            )
            self.demo.attrs["recording_end_time"] = end_wall_time_ns / 1e9
            self.demo.attrs["recording_end_wall_time_ns"] = end_wall_time_ns
            self.demo.attrs["recording_end_monotonic_ns"] = end_monotonic_ns
            num_samples = int(self.demo.attrs.get("num_samples", 0))
            if num_samples > 1:
                step_ns = self.demo["timestamps/step_monotonic_ns"]
                duration_ns = int(step_ns[-1]) - int(step_ns[0])
                if duration_ns > 0:
                    self.demo.attrs["effective_hz"] = (
                        (num_samples - 1) * 1e9 / duration_ns)
            alignment = self.demo["alignment"]
            for name in (
                *self.CAMERA_NAMES, "model_state", "arm_joint_state",
                "xhand_state", "executor_target",
            ):
                flags = alignment[f"{name}_is_new"][:]
                alignment.attrs[f"{name}_reused_count"] = int(
                    len(flags) - np.count_nonzero(flags))
            self.file.flush()
            self.file.close()
            self.file = None
        print(
            f"[recording] saved {self.demo_key}: "
            f"{num_samples} 30Hz rows to {self.path} ({status})"
        )
        if raise_after_close:
            raise error

    def discard(self):
        """Delete only the trajectory group created by this invocation."""
        self.stop_fixed_rate()
        if self.file is None:
            return
        demo_key = self.demo_key
        try:
            with self._h5_lock:
                self.datasets.clear()
                self.demo = None
                del self.file["data"][demo_key]
                self.file.flush()
        finally:
            self.file.close()
            self.file = None
        print(f"[recording] discarded {demo_key} from {self.path}")


class TerminalTaskScoreReader:
    """Read a human score or discard decision without blocking the loop."""

    VALID_SCORES = frozenset("0123")
    DISCARD_DECISION = "discard"

    def __init__(self):
        self.enabled = sys.stdin.isatty()
        self.fd = None
        self.old_term = None

    def __enter__(self):
        if self.enabled:
            self.fd = sys.stdin.fileno()
            self.old_term = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
            print(
                "[score] press 0, 1, 2, or 3 when the task is complete; "
                "press d to end and discard this rollout"
            )
        else:
            print(
                "[score] stdin is not a TTY; live scoring/discard is unavailable",
                file=sys.stderr,
            )
        return self

    def read_decision(self):
        if not self.enabled:
            return None
        while True:
            ready, _, _ = select.select([sys.stdin], [], [], 0.0)
            if not ready:
                return None
            char = sys.stdin.read(1)
            if not char:
                return None
            if char in self.VALID_SCORES:
                return int(char)
            if char.lower() == "d":
                return self.DISCARD_DECISION

    def __exit__(self, exc_type, exc, tb):
        if self.enabled and self.fd is not None and self.old_term is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_term)


def _configure_openteach_import(args):
    openteach_root = Path(args.openteach_root).expanduser().resolve()
    if not (openteach_root / "openteach/ros_links/xhand.py").is_file():
        raise SafetyViolation(f"invalid --openteach-root: {openteach_root}")
    if str(openteach_root) not in sys.path:
        sys.path.insert(0, str(openteach_root))


def _open_xhand(args):
    _configure_openteach_import(args)
    try:
        from openteach.constants import (
            XHAND_JOINT_LOWER, XHAND_JOINT_UPPER,
            XHAND_SDK_JOINT_LOWER, XHAND_SDK_JOINT_UPPER,
        )
        from openteach.ros_links.xhand import (
            DEFAULT_KD, DEFAULT_KI, DEFAULT_KP, DEFAULT_TOR_MAX,
            TOR_MAX_OVERRIDES, XHandLink,
        )
    except Exception as exc:
        raise SafetyViolation("Open-Teach XHand dependencies are unavailable") from exc

    # # Abort before motion on silent drift between this entrypoint and Open-Teach.
    for label, local, source in (
        ("XHand command lower", XHAND_COMMAND_LOWER, XHAND_JOINT_LOWER),
        ("XHand command upper", XHAND_COMMAND_UPPER, XHAND_JOINT_UPPER),
        ("XHand SDK lower", XHAND_SDK_LOWER, XHAND_SDK_JOINT_LOWER),
        ("XHand SDK upper", XHAND_SDK_UPPER, XHAND_SDK_JOINT_UPPER),
    ):
        if not np.allclose(local, np.asarray(source, dtype=float), atol=0, rtol=0):
            raise SafetyViolation(f"{label} changed in Open-Teach; review before deployment")
    source_torque = np.full(HAND_DIM, DEFAULT_TOR_MAX, dtype=int)
    for joint_id, limit in TOR_MAX_OVERRIDES.items():
        source_torque[joint_id] = limit
    if ((DEFAULT_KP, DEFAULT_KI, DEFAULT_KD) != (XHAND_KP, XHAND_KI, XHAND_KD)
            or not np.array_equal(source_torque, XHAND_TOR_MAX)):
        raise SafetyViolation("XHand gains/torque limits changed in Open-Teach; review first")

    deadline = time.monotonic() + XHAND_CONNECT_TIMEOUT_SECONDS
    last_error = None
    while True:
        try:
            return XHandLink(
                port=args.xhand_port,
                baudrate=args.xhand_baudrate,
                device_id=args.xhand_device_id,
                comm_type=args.xhand_comm_type,
                ifname=args.xhand_ifname,
                kp=XHAND_KP,
                ki=XHAND_KI,
                kd=XHAND_KD,
                tor_max=XHAND_TOR_MAX,
                zero_tactile_on_start=False,
                strict_connect=True,
            )
        except Exception as exc:
            last_error = exc
            if time.monotonic() >= deadline:
                raise SafetyViolation(
                    f"XHand connection did not become ready within "
                    f"{XHAND_CONNECT_TIMEOUT_SECONDS:.1f}s: {last_error}"
                ) from exc
            print(f"[robot] XHand connection not ready: {exc}; retrying", file=sys.stderr)
            time.sleep(XHAND_STARTUP_POLL_SECONDS)


def _validated_xhand_positions(state):
    if not isinstance(state, dict):
        raise SafetyViolation("XHand state read failed")
    status = state.get("sdk_status", {})
    if status.get("code") not in (None, 0):
        raise SafetyViolation(f"XHand read failed: {status}")
    if status.get("code") is None and status.get("raw") not in (None, "None", "True"):
        raise SafetyViolation(f"XHand read returned unknown/failing status: {status}")
    if state.get("hardware_errors"):
        raise SafetyViolation(f"XHand hardware errors: {state['hardware_errors']}")
    hand = _finite_vector(state.get("joint_positions"), HAND_DIM, "measured XHand joints")
    bad = np.flatnonzero((hand < XHAND_SDK_LOWER) | (hand > XHAND_SDK_UPPER)).tolist()
    if bad:
        raise SafetyViolation(f"measured XHand joints outside SDK limits: {bad}")
    return hand


def _wait_for_xhand_ready(hand):
    print(
        "[robot] waiting for XHand joint and tactile state to become ready "
        f"({XHAND_READY_CONSECUTIVE_READS} consecutive healthy reads)"
    )
    deadline = time.monotonic() + XHAND_READY_TIMEOUT_SECONDS
    healthy_reads = 0
    last_error = None
    last_reported = None
    while True:
        try:
            state = hand.read_hand_state()
            _validated_xhand_positions(state)
            _validated_xhand_tactile(state)
            healthy_reads += 1
            last_error = None
            if healthy_reads >= XHAND_READY_CONSECUTIVE_READS:
                print("[robot] XHand state is ready")
                return
        except SafetyViolation as exc:
            healthy_reads = 0
            last_error = exc
            message = str(exc)
            if message != last_reported:
                print(f"[robot] XHand not ready: {message}; waiting", file=sys.stderr)
                last_reported = message

        if time.monotonic() >= deadline:
            detail = str(last_error) if last_error is not None else "state was not stable"
            raise SafetyViolation(
                f"XHand state did not become ready within "
                f"{XHAND_READY_TIMEOUT_SECONDS:.1f}s: {detail}"
            )
        time.sleep(XHAND_STARTUP_POLL_SECONDS)


def _reset_xhand_before_confirmation(args):
    if args.execution != "physical":
        return None
    active = _active_teleop_ports()
    if active:
        raise SafetyViolation(
            f"teleop publishers are active on {active}; stop teleop/run_teleop_camera first")
    hand = None
    try:
        hand = _open_xhand(args)
        _wait_for_xhand_ready(hand)
        print("[robot] resetting XHand joints to the open home pose before confirmation")
        # _check_xhand("startup home", hand.home())
        _check_xhand("startup home", hand.send_positions(XHAND_HOME))
        _wait_for_xhand_ready(hand)
    except Exception:
        if hand is not None:
            hand.close()
        raise
    print("[robot] XHand home command accepted; verify the hand before continuing")
    return hand


class SafeRobotExecutor:
    """Open-Teach-compatible state reader and guarded xArm/XHand command sink."""

    def __init__(self, args, preopened_hand=None):
        self.args = args
        self.clamp_pose_step = None
        self.arm = None
        self.hand = preopened_hand
        self.last_arm_joint_state = None
        self.last_hand_state_sample = None
        self.command_mode = False
        try:
            active = _active_teleop_ports()
            if active:
                raise SafetyViolation(
                    f"teleop publishers are active on {active}; "
                    "stop teleop/run_teleop_camera first")
            _configure_openteach_import(args)
            try:
                from openteach.utils.xarm7_mapping import clamp_pose_step
                from xarm.wrapper import XArmAPI
            except Exception as exc:
                raise SafetyViolation(
                    "Open-Teach xArm dependencies are unavailable") from exc
            self.clamp_pose_step = clamp_pose_step
            self.arm = XArmAPI(args.robot_ip, is_radian=True, is_tool_coord=False)
            if not self.arm.connected:
                raise SafetyViolation(f"cannot connect to xArm at {args.robot_ip}")
            if self.hand is None:
                self.hand = _open_xhand(args)
                _wait_for_xhand_ready(self.hand)
            self.read_state()  # validate measured state before changing arm mode
            self._configure_arm()
            self.read_state()  # validate again after controller mode selection
        except Exception:
            self.close()
            raise

    def _arm_healthy(self):
        if self.arm is None or not self.arm.connected:
            raise SafetyViolation("xArm disconnected")
        if self.arm.has_error or self.arm.has_warn:
            raise SafetyViolation(
                f"xArm fault/warning: error={self.arm.error_code}, warn={self.arm.warn_code}")

    def _configure_arm(self):
        self._arm_healthy()
        if self.args.execution == "shadow":
            code, moving = self.arm.get_is_moving()
            _check_xarm("get_is_moving", code)
            if moving:
                raise SafetyViolation("xArm is already moving; shadow mode will not take control")
            print("[robot] SHADOW: state + safety checks only; no position commands")
            return
        # From the first state-changing call onward, every failure must run the
        # active-arm stop path, including a nonzero preflight-stop response.
        self.command_mode = True
        _check_xarm("preflight stop", self.arm.set_state(4))
        controller_sim = self.args.execution == "controller-sim"
        _check_xarm("set_simulation_robot", self.arm.set_simulation_robot(controller_sim))
        _check_xarm("motion_enable", self.arm.motion_enable(enable=True))
        _check_xarm("set_mode(7)", self.arm.set_mode(7))
        _check_xarm("set_state(0)", self.arm.set_state(0))
        time.sleep(0.2)
        self._arm_healthy()

    def _read_arm(self):
        self._arm_healthy()
        code, pose = self.arm.get_position_aa()
        _check_xarm("get_position_aa", code)
        pose = _finite_vector(pose, EE_DIM, "measured xArm pose")
        if not _inside_workspace(pose, WORKSPACE_MM):
            raise SafetyViolation(f"measured xArm pose outside workspace: {pose[:3].tolist()}mm")
        code, joints = self.arm.get_servo_angle()
        _check_xarm("get_servo_angle", code)
        return pose, _validate_xarm_joints(joints, "measured xArm joints")

    def _read_hand(self):
        sample = self.hand.read_hand_state()
        positions = _validated_xhand_positions(sample)
        self.last_hand_state_sample = sample
        return positions

    def read_state(self):
        pose_mm, joints = self._read_arm()
        self.last_arm_joint_state = joints.copy()
        pose_model = pose_mm.copy()
        pose_model[:3] /= 1000.0
        return np.concatenate([pose_model, self._read_hand()])

    def send(self, target18):
        target = _finite_vector(target18, EE_DIM + HAND_DIM, "policy target")
        current_pose, current_joints = self._read_arm()
        current_hand = self._read_hand()  # fail closed before every send
        measured_pose = current_pose.copy()
        measured_pose[:3] /= 1000.0
        measured_state = np.concatenate([measured_pose, current_hand])

        arm_target = target[:EE_DIM].copy()
        arm_target[:3] *= 1000.0
        arm_target[3:] = Rotation.from_rotvec(arm_target[3:]).as_rotvec()
        if not _inside_workspace(arm_target, WORKSPACE_MM):
            raise SafetyViolation(f"policy arm target outside workspace: {arm_target[:3].tolist()}mm")
        arm_target = self.clamp_pose_step(
            current_pose, arm_target,
            self.args.max_arm_step_mm, self.args.max_arm_step_rad)
        arm_target = np.asarray(arm_target, dtype=float)
        if not _inside_workspace(arm_target, WORKSPACE_MM):
            raise SafetyViolation("clamped arm target outside workspace")

        rpy = Rotation.from_rotvec(arm_target[3:]).as_euler("xyz", degrees=False)
        ik_pose = np.concatenate([arm_target[:3], rpy])
        try:
            # code, ik_joints = self.arm.get_inverse_kinematics(
            #     ik_pose.tolist(), input_is_radian=True, return_is_radian=True)
            code, ik_joints = self.arm.get_inverse_kinematics(
                ik_pose.tolist(),
                input_is_radian=True,
                return_is_radian=True,
                limited=False,
                ref_angles=current_joints.tolist(),
            )
        except Exception as exc:
            raise SafetyViolation("xArm IK raised an exception") from exc
        _check_xarm("get_inverse_kinematics", code)
        ik_joints = _validate_xarm_joints(ik_joints, "IK xArm joints")
        joint_delta = (ik_joints - current_joints + np.pi) % (2 * np.pi) - np.pi
        bad_delta = np.flatnonzero(np.abs(joint_delta) > self.args.max_ik_joint_step_rad).tolist()
        if bad_delta:
            raise SafetyViolation(f"IK branch/joint jump too large at joints {bad_delta}")

        hand_target = _clip_xhand_command(target[EE_DIM:], "executor target")
        effective_arm_target = arm_target.copy()
        effective_arm_target[:3] /= 1000.0
        effective_target = np.concatenate([effective_arm_target, hand_target])

        if self.args.execution == "shadow":
            return {
                "measured_state": measured_state,
                "arm_joint_state": current_joints.copy(),
                "effective_target": effective_target,
            }
        self._arm_healthy()
        _check_xarm(
            "set_position_aa",
            self.arm.set_position_aa(
                arm_target.tolist(), wait=False, relative=False,
                speed=self.args.xarm_speed_mm_s,
                mvacc=self.args.xarm_accel_mm_s2))
        if self.args.execution == "physical":
            _check_xhand("send_positions", self.hand.send_positions(hand_target))
        return {
            "measured_state": measured_state,
            "arm_joint_state": current_joints.copy(),
            "effective_target": effective_target,
        }

    def close(self):
        if self.arm is not None:
            if self.arm.connected and self.command_mode:
                # Open-Teach ordering, with a stricter dependency: never put
                # the arm back in ready state if stop or mode reset failed.
                stopped = False
                mode_reset = False
                try:
                    _check_xarm("shutdown set_state(4)", self.arm.set_state(4))
                    stopped = True
                except Exception as exc:
                    print(f"[shutdown] xArm set_state(4) failed: {exc}", file=sys.stderr)
                if stopped:
                    try:
                        _check_xarm("shutdown set_mode(0)", self.arm.set_mode(0))
                        mode_reset = True
                    except Exception as exc:
                        print(f"[shutdown] xArm set_mode(0) failed: {exc}", file=sys.stderr)
                if stopped and mode_reset:
                    try:
                        _check_xarm("shutdown set_state(0)", self.arm.set_state(0))
                    except Exception as exc:
                        print(f"[shutdown] xArm set_state(0) failed: {exc}", file=sys.stderr)
            try:
                if self.arm.connected:
                    self.arm.disconnect()
            except Exception as exc:
                print(f"[shutdown] xArm disconnect failed: {exc}", file=sys.stderr)
            self.arm = None
        if self.hand is not None:
            try:
                self.hand.close()  # Open-Teach: zero torque, then close device.
            except Exception as exc:
                print(f"[shutdown] XHand close failed: {exc}", file=sys.stderr)
            self.hand = None


def print_safety_constraints(args):
    # FastWAM: banner title describes the FastWAM client.
    inference = "local CUDA model"
    if args.mode == "robot-client":
        inference = (
            f"remote {args.inference_endpoint}; fail closed after "
            f"{args.inference_timeout_ms} ms")
    print(f"""
=== FastWAM xArm7 + XHand1 hardware constraints ===
execution                    {args.execution} (default shadow)
inference                    {inference}
physical enable              --execution physical + type MOVE
exclusive ownership          ports {list(TELEOP_COMMAND_PORTS)} must be inactive
XHand startup readiness      connect retry {XHAND_CONNECT_TIMEOUT_SECONDS:.0f}s; {XHAND_READY_CONSECUTIVE_READS} healthy full-state reads within {XHAND_READY_TIMEOUT_SECONDS:.0f}s
XHand startup reset          open-hand home before MOVE (physical only)
automatic recovery           disabled (no xArm home/clean_error/clean_warn/retry)
xArm workspace [mm]          x={WORKSPACE_MM['x']} y={WORKSPACE_MM['y']} z={WORKSPACE_MM['z']}
xArm command                 mode 7, absolute base-frame axis-angle
xArm per-command cap         {args.max_arm_step_mm} mm / {args.max_arm_step_rad} rad
xArm speed/acceleration      {args.xarm_speed_mm_s} mm/s / {args.xarm_accel_mm_s2} mm/s^2
xArm IK joint range [deg]    lower={XARM_JOINT_LOWER_DEG.tolist()}
                             upper={XARM_JOINT_UPPER_DEG.tolist()}
xArm max IK joint jump       {args.max_ik_joint_step_rad} rad
XHand SDK lower/upper [rad]  {XHAND_SDK_LOWER.tolist()}
                             {XHAND_SDK_UPPER.tolist()}
XHand cmd lower/upper [rad]  {XHAND_COMMAND_LOWER.tolist()}
                             {XHAND_COMMAND_UPPER.tolist()}
XHand target saturation      clip to cmd limits + warning; rollout continues
XHand PD gains               kp={XHAND_KP}, ki={XHAND_KI}, kd={XHAND_KD}
XHand torque caps [SDK unit] {XHAND_TOR_MAX.tolist()}
XHand extra slew smoothing   none (range + PD/torque guards remain active)
camera serials               ego={STATIC_SERIAL}, wrist={WRIST_SERIAL}, exo={EXO_SERIAL}
camera RGB source            ego=1280x720, wrist=640x480, exo=OpenTeach native
model/network RGB            ego=480x480, wrist=480x480
camera ROI file              {args.camera_rois}
camera timeout/max age       {args.camera_timeout_ms} ms / {args.camera_max_age_ms} ms
workstation HDF5 recording   {getattr(args, 'record_h5', None) or 'disabled'}
recording timeline           {EXECUTION_HZ:g} Hz latest-frame snapshots
failure behavior             any read/status/IK/limit/send failure stops rollout
shutdown                     xArm stop/mode0; XHand zero torque + close
xArm Studio responsibility   payload/COM, TCP, collision sensitivity/model are not overridden
================================
""".strip())


def _confirm_physical_execution(args):
    if args.execution == "physical":
        print("Clear the workspace, keep the emergency stop ready, and stop teleop first.")
        if input("Type MOVE to continue: ").strip() != "MOVE":
            print("[robot] cancelled before xArm command mode; releasing XHand")
            return False
    return True
