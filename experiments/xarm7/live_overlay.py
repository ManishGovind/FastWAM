#!/usr/bin/env python3
"""Overlay a first-frame reference PNG on a live Open-Teach camera.

References are the exported XHand demo first frames in
data/xhand_demo_first_frame_data/frames/, named
<task>__episode_<N>__<ego|exo|wrist>__frame_<K>.png (already 480x480). The live
frame goes through the same ROI crop/resize as deploy; references that are
already 480x480 are used as-is, so both sides line up in the model's view.

The browser UI picks task and demo (episode), changes opacity, and selects
blend, outline, or side-by-side display.

Usage (with Open-Teach cameras publishing):
  python experiments/xarm7/live_overlay.py --camera cam0 [--task clean_the_whiteboard --demo 1]
xArm7 (the inference client owns the cameras and publishes frames to /tmp/xarm_live,
so start the client first):
  python experiments/xarm7/live_overlay.py --camera xarm_exo --task stack_cubes --demo 1
Match a recorded rollout (e.g. restage the object positions of trial 4 of an earlier
eval): pass its HDF5; --demo N is trial N and frame 0 is the start of the trial:
  python experiments/xarm7/live_overlay.py --camera xarm_exo \
      --hdf5 experiments/xarm7/data/rollouts/combined/stack_cubes_30k.hdf5 --demo 4
Without a client running you can add --direct to open /dev/video yourself.
then open http://<this-host>:8765
"""

import argparse
import base64
import json
import pickle
import re
import signal
import socket
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]  # FastWAM/
DEFAULT_FRAMES = ROOT / "data" / "xhand_demo_first_frame_data" / "frames"

# Camera constants and ROIs mirror the Open-Teach deploy setup (camera_rois.json
# in vpp_real); keep them in sync if the deploy-time transform changes.
CAMERA_TARGET_SIZE = 480
CAMERA_JPEG_QUALITY = 95

# cam{i} is published on port 10005+i (Open-Teach configs/network.yaml).
# "view" is the camera name used in the reference frame file names.
CAMERAS = {
    "cam0": {"serial": "341222300710", "port": 10005, "view": "ego",
             "roi": {"x": 323, "y": 0, "size": 696, "source_width": 1280, "source_height": 720,
                     "mode": "square_crop_resize"}},
    "cam1": {"serial": "334622071624", "port": 10006, "view": "wrist",
             "roi": {"x": 115, "y": 0, "size": 480, "source_width": 640, "source_height": 480,
                     "mode": "full_frame_resize"}},
    "cam2": {"serial": "341522301282", "port": 10007, "view": "exo",
             "roi": {"x": 194, "y": 104, "size": 258, "source_width": 640, "source_height": 480,
                     "mode": "square_crop_resize"}},
}
# xArm7 cameras read directly from /dev/video* with the collection-time crops from
# experiments/xarm7/xarm_data_collection.py (crop = [width, height, x, y]).
# Use with references exported by scripts/export_xarm_first_frames.py.
XARM_CAMERAS = {
    "xarm_exo": {"video_id": 18, "view": "exo", "capture": (640, 480), "crop": [248, 258, 194, 104]},
    "xarm_wrist": {"video_id": 4, "view": "wrist", "capture": (640, 480), "crop": [480, 480, 80, 0]},
    "xarm_ego": {"video_id": 10, "view": "ego", "capture": (1280, 720), "crop": [696, 696, 323, 0]},
}
# The inference client (xarm_fastwam_inference.py) owns the cameras while it runs and
# publishes the latest model-view frames here as <exo|wrist|ego>.jpg.
DEFAULT_LIVE_DIR = Path("/tmp/xarm_live")
DEFAULT_XARM_FRAMES = ROOT / "data" / "xarm7_first_frame_data" / "frames"
TOPIC = b"rgb_image "
FRAME_RE = re.compile(r"^(?P<task>.+)__episode_(?P<episode>\d+)__(?P<view>[a-z]+)__frame_(?P<frame>\d+)\.png$")


def _transform_camera_for_model(name, source_bgr, transform):
    """Same crop/resize/JPEG round trip as deploy_policy; returns BGR."""
    if transform["mode"] == "full_frame_resize":
        cropped = source_bgr
        resize_source = min(source_bgr.shape[:2])
    else:
        x, y, size = transform["x"], transform["y"], transform["size"]
        cropped = source_bgr[y:y + size, x:x + size]
        if cropped.shape[:2] != (size, size):
            raise ValueError(f"{name} ROI produced {cropped.shape[:2]}")
        resize_source = size
    interpolation = (
        cv2.INTER_AREA if resize_source >= CAMERA_TARGET_SIZE else cv2.INTER_LINEAR)
    resized = cv2.resize(
        cropped, (CAMERA_TARGET_SIZE, CAMERA_TARGET_SIZE), interpolation=interpolation)
    ok, encoded = cv2.imencode(
        ".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), CAMERA_JPEG_QUALITY])
    if not ok:
        raise ValueError(f"{name} JPEG encoding failed")
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def _demo_sort_key(name):
    task, _, episode = name.rpartition("__episode_")
    return (task, int(episode)) if episode.isdigit() else (name, 0)


def _to_model_view(name, bgr, transform):
    """Crop/resize to the 480x480 model view; returns (bgr, note)."""
    height, width = bgr.shape[:2]
    if (width, height) == (transform["source_width"], transform["source_height"]):
        return _transform_camera_for_model(name, bgr, transform), None
    if (height, width) == (CAMERA_TARGET_SIZE, CAMERA_TARGET_SIZE):
        return bgr, None
    side = min(height, width)
    x, y = (width - side) // 2, (height - side) // 2
    resized = cv2.resize(
        bgr[y:y + side, x:x + side], (CAMERA_TARGET_SIZE, CAMERA_TARGET_SIZE),
        interpolation=cv2.INTER_AREA)
    note = (f"{name} is {width}x{height}, not the ROI source "
            f"{transform['source_width']}x{transform['source_height']} or "
            f"{CAMERA_TARGET_SIZE}x{CAMERA_TARGET_SIZE}; center-cropped instead")
    return resized, note


def _reference_edges(bgr, min_edge_px):
    """Outline mask for the reference: blur suppresses carpet/clutter texture,
    and edge fragments shorter than min_edge_px are dropped."""
    gray = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    edges = cv2.Canny(gray, 50, 140)
    if min_edge_px > 0:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(edges, connectivity=8)
        keep = stats[:, cv2.CC_STAT_AREA] >= min_edge_px
        keep[0] = False
        edges = np.where(keep[labels], 255, 0).astype(np.uint8)
    return cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1) > 0


class FrameStore:
    """First-frame PNGs for one camera view, grouped into demos named
    <task>__episode_<N>; each demo holds its frames sorted by frame index."""

    def __init__(self, frames_dir, view, transform):
        self.path = Path(frames_dir).expanduser().resolve()
        if not self.path.is_dir():
            raise FileNotFoundError(f"frames directory does not exist: {self.path}")
        self.view = view
        self.transform = transform
        self.frames = {}
        for image in self.path.glob("*.png"):
            match = FRAME_RE.match(image.name)
            if match is None or match["view"] != view:
                continue
            demo = f"{match['task']}__episode_{int(match['episode']):06d}"
            self.frames.setdefault(demo, []).append((int(match["frame"]), image))
        if not self.frames:
            raise ValueError(f"no *__{view}__frame_*.png files in {self.path}")
        for items in self.frames.values():
            items.sort()
        self.demos = sorted(self.frames, key=_demo_sort_key)
        self.reference_note = None

    def resolve_demo(self, requested):
        if requested in self.frames:
            return requested
        raise ValueError(
            f"unknown demo {requested!r}; e.g. {self.demos[0]} ... {self.demos[-1]} "
            f"({len(self.demos)} total)")

    def frame_count(self, demo):
        return len(self.frames[demo])

    def demo_number(self, demo):
        """1-based position of the demo within its task."""
        task = demo.rsplit("__episode_", 1)[0]
        return [d for d in self.demos if d.startswith(f"{task}__episode_")].index(demo) + 1

    def task(self, demo):
        return demo.rsplit("__episode_", 1)[0].replace("_", " ")

    def load(self, demo, frame_index):
        count = self.frame_count(demo)
        if not 0 <= frame_index < count:
            raise IndexError(f"frame {frame_index} is outside [0, {count - 1}] for {demo}")
        image_path = self.frames[demo][frame_index][1]
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"could not decode {image_path}")
        image, self.reference_note = _to_model_view("reference", bgr, self.transform)
        return image

    def close(self):
        pass


# Reference-image datasets per view, rollout/inference layout first, then collected demos.
HDF5_VIEW_KEYS = {
    "exo": ("primary_images_jpeg", "obs/agentview_rgb_jpeg"),
    "wrist": ("wrist_images_jpeg", "obs/eye_in_hand_rgb_jpeg"),
    "ego": ("ego_images_jpeg", "obs/ego_rgb_jpeg"),
}


class HDF5FrameStore:
    """Reference frames read straight from a rollout / demo HDF5 (--hdf5).

    Every trial (data/trial_N, or data/demo_N for collected demos) becomes one
    demo named <file-stem>__episode_<N>, so the UI works as with PNG frames. The
    stored JPEGs are already the 480x480 model view. Frame 0 is the scene as the
    trial started, which is what you match when staging the same positions."""

    def __init__(self, hdf5_path, view, transform):
        import h5py
        self._h5py = h5py
        self.path = Path(hdf5_path).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"HDF5 does not exist: {self.path}")
        if view not in HDF5_VIEW_KEYS:
            raise ValueError(f"no {view!r} view in HDF5 references")
        self.view = view
        self.transform = transform
        self.stem = re.sub(r"[^A-Za-z0-9]+", "_", self.path.stem).strip("_")
        self.groups = {}   # demo name -> (group path inside file, dataset key)
        self._counts = {}
        with h5py.File(self.path, "r", locking=False) as f:
            data = f.get("data")
            groups = ({k: v for k, v in data.items() if isinstance(v, h5py.Group)}
                      if isinstance(data, h5py.Group) else {"episode": f})
            for name, group in groups.items():
                key = next((k for k in HDF5_VIEW_KEYS[view]
                            if isinstance(group.get(k), h5py.Dataset)), None)
                if key is None or group[key].shape[0] == 0:
                    continue
                digits = re.search(r"(\d+)$", name)
                number = int(digits.group(1)) if digits else 0
                demo = f"{self.stem}__episode_{number:06d}"
                self.groups[demo] = (group.name, key)
                self._counts[demo] = int(group[key].shape[0])
        if not self.groups:
            raise ValueError(f"no {view} images (tried {HDF5_VIEW_KEYS[view]}) in {self.path}")
        self.demos = sorted(self.groups, key=_demo_sort_key)
        self.reference_note = None

    def resolve_demo(self, requested):
        if requested in self.groups:
            return requested
        raise ValueError(
            f"unknown trial {requested!r}; e.g. {self.demos[0]} ... {self.demos[-1]} "
            f"({len(self.demos)} total)")

    def frame_count(self, demo):
        return self._counts[demo]

    def demo_number(self, demo):
        """Trial number (the N in trial_N)."""
        return int(demo.rsplit("__episode_", 1)[1])

    def task(self, demo):
        return self.stem.replace("_", " ")

    def load(self, demo, frame_index):
        count = self.frame_count(demo)
        if not 0 <= frame_index < count:
            raise IndexError(f"frame {frame_index} is outside [0, {count - 1}] for {demo}")
        group_path, key = self.groups[demo]
        with self._h5py.File(self.path, "r", locking=False) as f:
            row = np.asarray(f[group_path][key][frame_index], dtype=np.uint8)
        bgr = cv2.imdecode(row, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"could not decode frame {frame_index} of {demo} in {self.path}")
        image, self.reference_note = _to_model_view("reference", bgr, self.transform)
        return image

    def close(self):
        pass


class OverlayState:
    MODES = ("blend", "outline", "side-by-side")

    def __init__(self, store, camera, demo, frame_index, alpha, stale_s, min_edge_px):
        self.store = store
        self.camera = camera
        self.stale_s = stale_s
        self.min_edge_px = min_edge_px
        self.lock = threading.Lock()
        self.live_frame = None
        self.live_time = 0.0
        self.live_error = None
        self.alpha = float(np.clip(alpha, 0.0, 1.0))
        self.mode = "blend"
        self.demo = store.resolve_demo(demo)
        self.frame_index = frame_index
        self.reference = store.load(self.demo, self.frame_index)
        self.reference_edges = _reference_edges(self.reference, min_edge_px)

    def set_live(self, frame):
        """Accept a decoded BGR frame already in the 480x480 model view."""
        with self.lock:
            self.live_frame = frame
            self.live_time = time.monotonic()
            self.live_error = None

    def set_live_error(self, message):
        with self.lock:
            self.live_error = message

    def configure(self, demo=None, frame_index=None, alpha=None, mode=None):
        with self.lock:
            next_demo = self.demo if demo is None else self.store.resolve_demo(demo)
            next_frame = self.frame_index if frame_index is None else int(frame_index)
            if next_demo != self.demo and frame_index is None:
                next_frame = 0
            count = self.store.frame_count(next_demo)
            next_frame = int(np.clip(next_frame, 0, count - 1))
            if next_demo != self.demo or next_frame != self.frame_index:
                reference = self.store.load(next_demo, next_frame)
                reference_edges = _reference_edges(reference, self.min_edge_px)
            else:
                reference = self.reference
                reference_edges = self.reference_edges
            if mode is not None and mode not in self.MODES:
                raise ValueError(f"mode must be one of {self.MODES}")
            self.demo = next_demo
            self.frame_index = next_frame
            self.reference = reference
            self.reference_edges = reference_edges
            if alpha is not None:
                self.alpha = float(np.clip(alpha, 0.0, 1.0))
            if mode is not None:
                self.mode = mode

    def _live_age(self):
        return None if self.live_time == 0 else time.monotonic() - self.live_time

    def status(self):
        with self.lock:
            age = self._live_age()
            if age is None:
                live = "waiting"
            elif age > self.stale_s:
                live = "stale"
            else:
                live = "live"
            store = self.store
            return {
                "frames_dir": str(store.path),
                "view": store.view,
                "camera": self.camera,
                "demos": store.demos,
                "demo": self.demo,
                "task": store.task(self.demo),
                "demo_number": store.demo_number(self.demo),
                "frame": self.frame_index,
                "frame_count": store.frame_count(self.demo),
                "alpha": self.alpha,
                "mode": self.mode,
                "modes": self.MODES,
                "live": live,
                "live_age_s": age,
                "live_error": self.live_error,
                "reference_note": store.reference_note,
            }

    def render(self):
        with self.lock:
            live = None if self.live_frame is None else self.live_frame.copy()
            reference = None if self.reference is None else self.reference.copy()
            reference_edges = self.reference_edges
            demo = self.demo
            frame_index = self.frame_index
            alpha = self.alpha
            mode = self.mode
            live_age = self._live_age()
            live_error = self.live_error

        blank = np.zeros((CAMERA_TARGET_SIZE, CAMERA_TARGET_SIZE, 3), np.uint8)
        if live is None:
            live = blank.copy()
            cv2.putText(live, "Waiting for live camera...", (30, live.shape[0] // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

        if mode == "blend":
            output = cv2.addWeighted(live, 1.0 - alpha, reference, alpha, 0.0)
        elif mode == "outline":
            output = live
            output[reference_edges] = (0, 255, 255)
        else:
            output = np.concatenate((reference, live), axis=1)
            cv2.putText(output, "DEMO", (12, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(output, "LIVE", (reference.shape[1] + 12, 62),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        if live_age is None:
            live_text = "waiting"
        elif live_age > self.stale_s:
            live_text = f"STALE {live_age:.1f}s"
        else:
            live_text = f"live {live_age:.2f}s"
        reference_text = f"{self.store.task(demo)} #{self.store.demo_number(demo)} f{frame_index}"
        label = f"{self.camera} {reference_text} {mode} a{alpha:.2f} {live_text}"
        cv2.rectangle(output, (0, 0), (output.shape[1], 38), (0, 0, 0), -1)
        stale = live_age is not None and live_age > self.stale_s
        cv2.putText(output, label, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                    (0, 0, 255) if stale or live_error else (255, 255, 255), 1, cv2.LINE_AA)
        return output


class OpenTeachSubscriber:
    """Latest-frame Open-Teach camera subscriber that survives camera restarts.

    Unlike deploy_policy.OpenTeachCameraStream, bad packets are reported in the
    UI and skipped instead of latching a SafetyViolation: nothing here commands
    the robot.
    """

    def __init__(self, name, host, port, transform, state, max_fps):
        import zmq

        self.zmq = zmq
        self.name = name
        self.expected_serial = CAMERAS[name]["serial"]
        self.endpoint = f"tcp://{host}:{port}"
        self.transform = transform
        self.state = state
        self.min_interval_s = 1.0 / max_fps
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._receive_loop, name=f"overlay-camera-{port}", daemon=True)
        self._thread.start()

    def _handle(self, raw):
        if not raw.startswith(TOPIC):
            raise ValueError("unexpected topic")
        packet = pickle.loads(raw[len(TOPIC):])
        serial = str(packet["camera_serial"])
        if serial != self.expected_serial:
            raise ValueError(
                f"serial {serial} on {self.endpoint}, expected {self.expected_serial}")
        jpeg = base64.b64decode(packet["rgb_image"])
        bgr = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("JPEG decode failed")
        frame, note = _to_model_view(self.name, bgr, self.transform)
        if note is not None:
            raise ValueError(note)
        self.state.set_live(frame)

    def _receive_loop(self):
        context = self.zmq.Context()
        stream = context.socket(self.zmq.SUB)
        stream.setsockopt(self.zmq.LINGER, 0)
        stream.setsockopt(self.zmq.CONFLATE, 1)
        stream.setsockopt(self.zmq.SUBSCRIBE, b"rgb_image")
        stream.connect(self.endpoint)
        try:
            while not self._stop_event.is_set():
                if not stream.poll(100, self.zmq.POLLIN):
                    continue
                try:
                    self._handle(stream.recv())
                except Exception as exc:
                    self.state.set_live_error(f"{self.name}: {exc}")
                # CONFLATE keeps only the newest frame, so decoding at the
                # display rate instead of the camera's 30 Hz loses nothing and
                # keeps CPU free for a concurrently running deploy_policy.py.
                self._stop_event.wait(self.min_interval_s)
        finally:
            stream.close(linger=0)
            context.term()

    def close(self):
        self._stop_event.set()
        self._thread.join(timeout=1.0)


class FileSubscriber:
    """Reads the 480x480 model-view JPEG the inference client publishes, so the overlay
    never opens the camera device itself and can run alongside live inference."""

    def __init__(self, name, view, live_dir, state, max_fps):
        self.name = name
        self.path = Path(live_dir) / f"{view}.jpg"
        self.state = state
        self.min_interval_s = 1.0 / max_fps
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._loop, name=f"overlay-{name}", daemon=True)
        self._thread.start()

    def _loop(self):
        last_mtime = None
        while not self._stop_event.is_set():
            try:
                mtime = self.path.stat().st_mtime_ns
                if mtime != last_mtime:
                    bgr = cv2.imread(str(self.path), cv2.IMREAD_COLOR)
                    if bgr is None:
                        raise ValueError("JPEG decode failed")
                    if bgr.shape[:2] != (CAMERA_TARGET_SIZE, CAMERA_TARGET_SIZE):
                        raise ValueError(f"expected 480x480, got {bgr.shape[1]}x{bgr.shape[0]}")
                    last_mtime = mtime
                    self.state.set_live(bgr)
            except FileNotFoundError:
                self.state.set_live_error(
                    f"{self.name}: no frames at {self.path}; start the inference client "
                    "(it publishes them) or run this overlay with --direct")
            except Exception as exc:
                self.state.set_live_error(f"{self.name}: {exc}")
            self._stop_event.wait(self.min_interval_s)

    def close(self):
        self._stop_event.set()
        self._thread.join(timeout=2.0)


class V4L2Subscriber:
    """Reads a local /dev/video camera, applies the xArm collection crop, and feeds
    OverlayState with 480x480 BGR frames (same JPEG round trip as deploy)."""

    def __init__(self, name, spec, state, max_fps):
        self.name = name
        self.spec = spec
        self.state = state
        self.min_interval_s = 1.0 / max_fps
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._loop, name=f"overlay-{name}", daemon=True)
        self._thread.start()

    def _loop(self):
        cap = cv2.VideoCapture(self.spec["video_id"], cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.spec["capture"][0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.spec["capture"][1])
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        w, h, x, y = self.spec["crop"]
        try:
            if not cap.isOpened():
                self.state.set_live_error(f"{self.name}: cannot open /dev/video{self.spec['video_id']}")
            while not self._stop_event.is_set():
                if not cap.isOpened():
                    self._stop_event.wait(1.0)
                    cap.open(self.spec["video_id"], cv2.CAP_V4L2)
                    continue
                ok, bgr = cap.read()
                if not ok:
                    self.state.set_live_error(f"{self.name}: camera read failed")
                    self._stop_event.wait(0.5)
                    continue
                try:
                    cropped = bgr[y:y + h, x:x + w]
                    if cropped.shape[:2] != (h, w):
                        raise ValueError(f"crop produced {cropped.shape[:2]}, camera frame is {bgr.shape[:2]}")
                    resized = cv2.resize(cropped, (CAMERA_TARGET_SIZE, CAMERA_TARGET_SIZE),
                                         interpolation=cv2.INTER_LINEAR)
                    ok, enc = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), CAMERA_JPEG_QUALITY])
                    self.state.set_live(cv2.imdecode(enc, cv2.IMREAD_COLOR))
                except Exception as exc:
                    self.state.set_live_error(f"{self.name}: {exc}")
                self._stop_event.wait(self.min_interval_s)
        finally:
            cap.release()

    def close(self):
        self._stop_event.set()
        self._thread.join(timeout=2.0)


INDEX_HTML = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>Live Overlay</title>
<style>
body{margin:0;background:#111;color:#eee;font:15px sans-serif}main{max-width:1100px;margin:auto;padding:16px}
.controls{display:flex;flex-wrap:wrap;gap:14px;align-items:center;background:#222;padding:12px;border-radius:8px;margin-bottom:10px}
label{display:flex;gap:7px;align-items:center}label.grow{flex:1 1 360px;min-width:0}label.grow select{flex:1;min-width:0;width:0}
[hidden]{display:none!important}select,input,button{font-size:15px}
button{background:#333;color:#eee;border:1px solid #555;border-radius:6px;padding:4px 10px;cursor:pointer}button:hover{border-color:#8ab4f8}
img{display:block;max-width:100%;margin-top:4px;border-radius:8px;background:#000}
#status{color:#aaa;margin-top:9px;word-break:break-all}.bad{color:#f66}.wide{width:260px}
#msg{min-height:18px;margin-top:6px}
</style></head><body><main><h2>First-frame overlay</h2>
<div class="controls">
<label>Task <select id="task"></select></label>
<label>Demo <select id="demo"></select></label>
<label>Frame <input id="frame" type="range" min="0" value="0" class="wide"><span id="frameValue"></span></label>
<label>Opacity <input id="alpha" type="range" min="0" max="1" step="0.05" value="0.45"><span id="alphaValue"></span></label>
<label>Mode <select id="mode"></select></label>
</div><img id="stream" src="stream.mjpg"><div id="status"></div><div id="msg"></div>
<script>
const ids=n=>document.getElementById(n); let busy=false, taskDemos=null, lastDir;
const msg=(text,bad)=>{ids('msg').textContent=text||'';ids('msg').className=bad?'bad':'';};

/* demos are named <task>__episode_<N>; the Demo dropdown shows the 1-based position within the task */
function showDemos(task, selected){
 const demo=ids('demo'); demo.replaceChildren();
 taskDemos[task].forEach((d,i)=>demo.add(new Option(String(i+1),d)));
 demo.value=selected && taskDemos[task].includes(selected) ? selected : taskDemos[task][0];
}
function rebuildDemos(s){
 ids('demo').replaceChildren(); ids('task').replaceChildren(); taskDemos={};
 s.demos.forEach(d=>(taskDemos[d.replace(/__episode_\d+$/,'')]??=[]).push(d));
 Object.keys(taskDemos).sort().forEach(t=>ids('task').add(new Option(t.replaceAll('_',' '),t)));
}
async function loadStatus(){
 const s=await (await fetch('api/status')).json();
 if(!ids('mode').options.length)s.modes.forEach(m=>ids('mode').add(new Option(m,m)));
 if(s.frames_dir!==lastDir){lastDir=s.frames_dir; rebuildDemos(s);}
 const task=s.demo.replace(/__episode_\d+$/,'');
 if(ids('task').value!==task || !ids('demo').options.length){ids('task').value=task;showDemos(task,s.demo);}
 ids('demo').value=s.demo; ids('frame').max=s.frame_count-1; ids('frame').value=s.frame;
 ids('frame').disabled=s.frame_count<2;
 ids('frameValue').textContent=`${s.frame}/${s.frame_count-1}`;
 ids('alpha').value=s.alpha; ids('alphaValue').textContent=Number(s.alpha).toFixed(2); ids('mode').value=s.mode;
 const live={live:'LIVE',stale:`STALE (${Number(s.live_age_s).toFixed(1)}s)`,waiting:'WAITING FOR CAMERA'}[s.live];
 const st=ids('status'); st.replaceChildren();
 const line=(text,bad)=>{const d=document.createElement('div'); d.textContent=text; if(bad)d.className='bad'; st.append(d);};
 line(`${live} | ${s.camera} (${s.view}) | ${s.task} | ${s.frames_dir}`, s.live!=='live');
 if(s.live_error) line(s.live_error, true);
 if(s.reference_note) line(s.reference_note, true);
}
async function update(params){if(busy)return;busy=true;try{await fetch('api/config?'+new URLSearchParams(params));await loadStatus();}finally{busy=false;}}
ids('task').onchange=e=>{showDemos(e.target.value);update({demo:ids('demo').value});};
ids('demo').onchange=e=>update({demo:e.target.value}); ids('frame').oninput=e=>{ids('frameValue').textContent=e.target.value}; ids('frame').onchange=e=>update({frame:e.target.value});
ids('alpha').oninput=e=>{ids('alphaValue').textContent=Number(e.target.value).toFixed(2)}; ids('alpha').onchange=e=>update({alpha:e.target.value}); ids('mode').onchange=e=>update({mode:e.target.value});
loadStatus(); setInterval(loadStatus,1000);
</script></main></body></html>"""


class OverlayHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, state, stream_fps):
        super().__init__(address, OverlayHandler)
        self.state = state
        self.stream_fps = stream_fps


class OverlayHandler(BaseHTTPRequestHandler):
    def _send_json(self, value, status=HTTPStatus.OK):
        payload = json.dumps(value).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        request = urlparse(self.path)
        if request.path == "/":
            payload = INDEX_HTML.encode()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        elif request.path == "/api/status":
            self._send_json(self.server.state.status())
        elif request.path == "/api/config":
            try:
                values = parse_qs(request.query)
                self.server.state.configure(
                    demo=values.get("demo", [None])[0],
                    frame_index=(int(values["frame"][0]) if "frame" in values else None),
                    alpha=(float(values["alpha"][0]) if "alpha" in values else None),
                    mode=values.get("mode", [None])[0],
                )
                self._send_json(self.server.state.status())
            except Exception as exc:
                self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        elif request.path == "/snapshot.jpg":
            self._send_jpeg(self.server.state.render())
        elif request.path == "/stream.mjpg":
            self._stream()
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def _send_jpeg(self, frame):
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if not ok:
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, "JPEG encoding failed")
            return
        payload = encoded.tobytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def _stream(self):
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        delay = 1.0 / self.server.stream_fps
        try:
            while True:
                ok, encoded = cv2.imencode(
                    ".jpg", self.server.state.render(), [cv2.IMWRITE_JPEG_QUALITY, 82])
                if ok:
                    payload = encoded.tobytes()
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(payload)}\r\n\r\n".encode())
                    self.wfile.write(payload + b"\r\n")
                time.sleep(delay)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, format, *args):
        return


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frames-dir", type=Path, default=DEFAULT_FRAMES,
                        help="directory of first-frame PNGs")
    parser.add_argument("--camera", choices=sorted(CAMERAS) + sorted(XARM_CAMERAS), default="cam0",
                        help="cam0=ego (over-the-shoulder), cam1=wrist, cam2=exo (Open-Teach ZMQ); "
                             "xarm_exo/xarm_wrist/xarm_ego read /dev/video18/4/10 directly "
                             "(xArm7; use --frames-dir data/xarm7_first_frame_data/frames). "
                             "Selects the reference view too")
    parser.add_argument("--video-id", type=int, help="override /dev/video index for xarm_* cameras (--direct only)")
    parser.add_argument("--direct", action="store_true",
                        help="xarm_* cameras: open /dev/video directly instead of reading the frames the "
                             "inference client publishes. Do NOT use while the client is running "
                             "(a camera can only be opened once).")
    parser.add_argument("--live-dir", type=Path, default=DEFAULT_LIVE_DIR,
                        help="xarm_* cameras: directory the inference client publishes frames to")
    parser.add_argument("--hdf5", type=Path,
                        help="use frames from this rollout/demo HDF5 as the reference instead of "
                             "--frames-dir PNGs; each trial (trial_N) is one demo, so --demo N "
                             "picks trial N and --frame 0 is the scene at the start of the trial")
    parser.add_argument("--task", help="initial task, e.g. clean_the_whiteboard (default: first)")
    parser.add_argument("--demo", default="1",
                        help="1-based demo number within --task, or a full group name "
                             "like clean_the_whiteboard__episode_000100 (default: first)")
    parser.add_argument("--frame", type=int, default=0, help="reference frame within the demo")
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--camera-host", default="127.0.0.1")
    parser.add_argument("--camera-port", type=int,
                        help="Open-Teach camera port; default 10005+i for cam{i}")
    parser.add_argument("--stale-s", type=float, default=1.0,
                        help="mark the live feed stale after this many seconds")
    parser.add_argument("--min-edge-px", type=int, default=25,
                        help="outline mode: drop reference edge fragments smaller "
                             "than this many pixels (0 keeps all)")
    parser.add_argument("--http-host", default="0.0.0.0")
    parser.add_argument("--http-port", type=int, default=8765)
    parser.add_argument("--stream-fps", type=float, default=10.0)
    args = parser.parse_args()
    if args.stream_fps <= 0 or args.stale_s <= 0:
        parser.error("--stream-fps and --stale-s must be positive")

    is_xarm = args.camera in XARM_CAMERAS
    camera = XARM_CAMERAS[args.camera] if is_xarm else CAMERAS[args.camera]
    # xArm references are already the 480x480 post-crop view; the roi is only
    # used for source-size matching, so a 480x480 "source" makes it a passthrough.
    transform = ({"mode": "full_frame_resize", "source_width": 0, "source_height": 0}
                 if is_xarm else camera["roi"])
    if is_xarm and args.frames_dir == DEFAULT_FRAMES:
        args.frames_dir = DEFAULT_XARM_FRAMES
    if args.hdf5 is not None:
        if not is_xarm:
            parser.error("--hdf5 needs an xarm_* camera")
        store = HDF5FrameStore(args.hdf5, camera["view"], transform)
    else:
        store = FrameStore(args.frames_dir, camera["view"], transform)
    task = args.task or store.demos[0].rsplit("__episode_", 1)[0]
    task_demos = [d for d in store.demos if d.startswith(f"{task}__episode_")]
    if not task_demos:
        parser.error(f"no demos for task {task!r}")
    if args.demo.isdigit() and args.hdf5 is not None:
        # trial number, e.g. --demo 4 = trial_4
        matches = [d for d in task_demos if store.demo_number(d) == int(args.demo)]
        if not matches:
            parser.error(f"no trial {args.demo} in {args.hdf5}; have "
                         f"{[store.demo_number(d) for d in task_demos]}")
        demo = matches[0]
    elif args.demo.isdigit():
        if not 1 <= int(args.demo) <= len(task_demos):
            parser.error(f"--demo must be in [1, {len(task_demos)}] for {task}")
        demo = task_demos[int(args.demo) - 1]
    else:
        demo = args.demo
    state = OverlayState(store, args.camera, demo, args.frame, args.alpha,
                         args.stale_s, args.min_edge_px)
    if is_xarm and not args.direct:
        subscriber = FileSubscriber(args.camera, camera["view"], args.live_dir, state, args.stream_fps)
    elif is_xarm:
        spec = dict(camera)
        if args.video_id is not None:
            spec["video_id"] = args.video_id
        subscriber = V4L2Subscriber(args.camera, spec, state, args.stream_fps)
    else:
        port = args.camera_port or camera["port"]
        subscriber = OpenTeachSubscriber(
            args.camera, args.camera_host, port, transform, state, args.stream_fps)
    server = OverlayHTTPServer((args.http_host, args.http_port), state, args.stream_fps)

    def stop(_signum, _frame):
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    print(f"Frames: {store.path} ({camera['view']} view)")
    print(f"Demos: {store.demos[0]} ... {store.demos[-1]} ({len(store.demos)} total)")
    if is_xarm and not args.direct:
        print(f"Live: {args.camera} from {args.live_dir}/{camera['view']}.jpg (published by the inference client)")
    elif is_xarm:
        print(f"Live: {args.camera} DIRECT on /dev/video{spec['video_id']} crop {spec['crop']}")
    else:
        print(f"Live: {args.camera} serial {camera['serial']} on tcp://{args.camera_host}:{port}")
    print(f"Open http://{socket.gethostname()}:{args.http_port} in a browser")
    try:
        server.serve_forever()
    finally:
        server.server_close()
        subscriber.close()


if __name__ == "__main__":
    main()
