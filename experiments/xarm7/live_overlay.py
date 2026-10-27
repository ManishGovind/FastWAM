#!/usr/bin/env python3
"""Overlay a recorded HDF5 frame on a live xArm7 camera.

Point --hdf5 at a recording with exo/wrist images: an inference episode from
xarm_fastwam_inference.py or a demo file from xarm_data_collection.py. The live
frame goes through the same crop/resize/JPEG round trip as the collection and
inference scripts (crops imported from xarm_data_collection.py), so both sides
line up in the 480x480 view the model sees.

The browser UI picks the reference file (Model and File dropdowns over the
recordings under --root), switches demos and frames, changes opacity, and
selects blend, outline, or side-by-side display.

The camera is opened directly (V4L2), so stop data collection, inference and
any other camera user first.

Usage (fastwam_xarm7 env):
  python live_overlay.py --camera exo [--hdf5 <file>.hdf5]
then open http://<this-host>:8765
"""

import argparse
import json
import signal
import socket
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import h5py
import numpy as np

from rollout_viewer import (  # same model naming and file layouts as the rollout viewer
    CAMERAS as STREAM_PATHS,
    _demo_groups,
    _model_label,
)
from xarm_data_collection import (
    CAPTURE_FRAME_SIZE,
    CROP_TARGET_SIZE,
    EXO1_CROP,
    WRISTCAM_CROP,
    crop_and_resize,
)

ROOT = Path(__file__).resolve().parent  # experiments/xarm7/

CAMERA_TARGET_SIZE = CROP_TARGET_SIZE[0]
CAMERA_JPEG_QUALITY = 95

# --camera -> /dev/video index and crop, the same defaults as the collection
# and inference scripts.
CAMERAS = {
    "exo": {"device": 18, "crop": EXO1_CROP},
    "wrist": {"device": 4, "crop": WRISTCAM_CROP},
}


def _camera_transform(camera):
    return {
        "crop": CAMERAS[camera]["crop"],
        "source_width": CAPTURE_FRAME_SIZE[0],
        "source_height": CAPTURE_FRAME_SIZE[1],
    }


def _transform_camera_for_model(name, source_bgr, transform):
    """Same crop/resize/JPEG round trip as the data scripts; returns BGR."""
    resized = crop_and_resize(source_bgr, transform["crop"])
    ok, encoded = cv2.imencode(
        ".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), CAMERA_JPEG_QUALITY])
    if not ok:
        raise ValueError(f"{name} JPEG encoding failed")
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def _demo_sort_key(name):
    suffix = name.rsplit("_", 1)[-1]
    return (0, int(suffix)) if suffix.isdigit() else (1, name)


def _to_model_view(name, bgr, transform):
    """Crop/resize to the 480x480 model view; returns (bgr, note)."""
    height, width = bgr.shape[:2]
    if (width, height) == (transform["source_width"], transform["source_height"]):
        return _transform_camera_for_model(name, bgr, transform), None
    if height == width:  # collected demos are 480x480, inference episodes 256x256
        if height != CAMERA_TARGET_SIZE:
            bgr = cv2.resize(bgr, (CAMERA_TARGET_SIZE, CAMERA_TARGET_SIZE),
                             interpolation=cv2.INTER_LINEAR)
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


class DemoStore:
    def __init__(self, path, camera, transform):
        self.path = Path(path).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"HDF5 file does not exist: {self.path}")
        # locking=False: a default read lock held for the overlay's lifetime
        # would stop a running script from appending to this file.
        self.file = h5py.File(self.path, "r", locking=False)
        self.transform = transform
        self.lock = threading.Lock()
        groups = _demo_groups(self.file)
        self.groups = groups
        # Per demo, the first of this camera's dataset paths that exists.
        self.paths = {}
        for name, group in groups.items():
            for candidate in STREAM_PATHS[camera]:
                if isinstance(group.get(candidate), h5py.Dataset):
                    self.paths[name] = candidate
                    break
        self.demos = sorted(self.paths, key=_demo_sort_key)
        if not self.demos:
            raise ValueError(
                f"no {camera} images ({' or '.join(STREAM_PATHS[camera])}) in {self.path}")
        self.image_key = self.paths[self.demos[0]]
        self.reference_note = None

    def resolve_demo(self, requested):
        candidates = [requested]
        if requested.isdigit():
            candidates.insert(0, f"demo_{int(requested)}")
            candidates.append(self.demos[0])  # inference episodes have one demo, "episode"
        for candidate in candidates:
            if candidate in self.demos:
                return candidate
        raise ValueError(f"unknown demo {requested!r}; choices: {', '.join(self.demos)}")

    def frame_count(self, demo):
        return int(self.groups[demo][self.paths[demo]].shape[0])

    def task(self, demo):
        value = self.groups[demo].attrs.get("task_description", "")
        return value.decode() if isinstance(value, bytes) else str(value)

    def load(self, demo, frame_index):
        count = self.frame_count(demo)
        if not 0 <= frame_index < count:
            raise IndexError(f"frame {frame_index} is outside [0, {count - 1}] for {demo}")
        with self.lock:
            value = np.asarray(self.groups[demo][self.paths[demo]][frame_index])
        bgr = cv2.imdecode(value.astype(np.uint8, copy=False), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"could not decode {demo}/{self.paths[demo]}[{frame_index}]")
        image, self.reference_note = _to_model_view("reference", bgr, self.transform)
        return image

    def close(self):
        self.file.close()


class ReferenceFiles:
    """Reference recordings offered in the UI: the *.hdf5 files under --root,
    labelled by model. Only these can be opened from the browser."""

    def __init__(self, root):
        self.root = Path(root).expanduser().resolve()

    def recordings(self):
        if not self.root.is_dir():
            return []
        paths = sorted(self.root.rglob("*.hdf5"), key=lambda p: p.stat().st_mtime, reverse=True)
        return [{"path": str(p), "name": str(p.relative_to(self.root)),
                 "model": _model_label(p), "size_bytes": p.stat().st_size} for p in paths]

    def check(self, path):
        path = Path(path).expanduser().resolve()
        if path.suffix != ".hdf5" or not path.is_file() or not path.is_relative_to(self.root):
            raise ValueError(f"not a recording under {self.root}: {path.name}")
        return path


class OverlayState:
    MODES = ("blend", "outline", "side-by-side")

    def __init__(self, store, camera, demo, frame_index, alpha, stale_s, min_edge_px,
                 transform):
        self.store = store
        self.camera = camera
        self.stale_s = stale_s
        self.min_edge_px = min_edge_px
        self.transform = transform
        self.lock = threading.Lock()
        self.live_frame = None
        self.live_time = 0.0
        self.live_error = None
        self.alpha = float(np.clip(alpha, 0.0, 1.0))
        self.mode = "blend"
        self.demo = None
        self.frame_index = 0
        self.reference = None
        self.reference_edges = None
        if store is not None:
            self.demo = store.resolve_demo(demo)
            self.frame_index = frame_index
            self.reference = store.load(self.demo, self.frame_index)
            self.reference_edges = _reference_edges(self.reference, min_edge_px)

    def open_file(self, path):
        """Switch the reference recording; the first demo's frame 0 is shown."""
        store = DemoStore(path, self.camera, self.transform)
        try:
            demo = store.demos[0]
            reference = store.load(demo, 0)
            edges = _reference_edges(reference, self.min_edge_px)
        except Exception:
            store.close()
            raise
        with self.lock:
            previous = self.store
            self.store, self.demo, self.frame_index = store, demo, 0
            self.reference, self.reference_edges = reference, edges
        if previous is not None:
            previous.close()

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
            if self.store is None:
                if mode is not None and mode not in self.MODES:
                    raise ValueError(f"mode must be one of {self.MODES}")
                if alpha is not None:
                    self.alpha = float(np.clip(alpha, 0.0, 1.0))
                if mode is not None:
                    self.mode = mode
                return
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
                "hdf5": None if store is None else str(store.path),
                "image_key": None if store is None else store.paths[self.demo],
                "camera": self.camera,
                "demos": [] if store is None else store.demos,
                "demo": self.demo,
                "task": "" if store is None else store.task(self.demo),
                "frame": self.frame_index,
                "frame_count": 0 if store is None else store.frame_count(self.demo),
                "alpha": self.alpha,
                "mode": self.mode,
                "modes": self.MODES,
                "live": live,
                "live_age_s": age,
                "live_error": self.live_error,
                "reference_note": None if store is None else store.reference_note,
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

        if reference is None:
            # No reference chosen yet: show the live view alone.
            output = live
            if mode == "side-by-side":
                reference = blank.copy()
                cv2.putText(reference, "Choose a reference file", (40, reference.shape[0] // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
                output = np.concatenate((reference, live), axis=1)
        elif mode == "blend":
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
        reference_text = "no reference" if demo is None else f"{demo} f{frame_index}"
        label = f"{self.camera} {reference_text} {mode} a{alpha:.2f} {live_text}"
        cv2.rectangle(output, (0, 0), (output.shape[1], 38), (0, 0, 0), -1)
        stale = live_age is not None and live_age > self.stale_s
        cv2.putText(output, label, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                    (0, 0, 255) if stale or live_error else (255, 255, 255), 1, cv2.LINE_AA)
        return output


class CameraReader:
    """Latest-frame V4L2 camera reader.

    Reads the camera directly with the data scripts' crop. Bad reads are
    reported in the UI and retried; nothing here commands the robot.
    """

    def __init__(self, name, device, transform, state, max_fps):
        self.name = name
        self.device = device
        self.transform = transform
        self.state = state
        self.min_interval_s = 1.0 / max_fps
        self.cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"could not open /dev/video{device} for {name}; is another process "
                "(data collection, inference, robot_camera.py) using it?")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_FRAME_SIZE[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_FRAME_SIZE[1])
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._read_loop, name=f"overlay-camera-{device}", daemon=True)
        self._thread.start()

    def _read_loop(self):
        last = 0.0
        while not self._stop_event.is_set():
            # grab() every pass drains V4L2's buffer (it ignores BUFFERSIZE=1);
            # only decode and transform at the display rate.
            if not self.cap.grab():
                self.state.set_live_error(f"{self.name}: camera read failed")
                self._stop_event.wait(0.2)
                continue
            now = time.monotonic()
            if now - last < self.min_interval_s:
                continue
            last = now
            try:
                ok, bgr = self.cap.retrieve()
                if not ok:
                    raise ValueError("camera read failed")
                frame, note = _to_model_view(self.name, bgr, self.transform)
                if note is not None:
                    raise ValueError(note)
                self.state.set_live(frame)
            except Exception as exc:
                self.state.set_live_error(f"{self.name}: {exc}")

    def close(self):
        self._stop_event.set()
        self._thread.join(timeout=1.0)
        self.cap.release()


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
</style></head><body><main><h2>xArm7 demonstration overlay</h2>
<div class="controls">
<label>Model <select id="model"></select></label>
<label class="grow">File <select id="file"></select></label>
</div>
<div class="controls">
<label id="taskLabel" hidden>Task <select id="task"></select></label>
<label id="demoLabel">Demo <select id="demo"></select></label>
<label>Frame <input id="frame" type="range" min="0" value="0" class="wide"><span id="frameValue"></span></label>
<label>Opacity <input id="alpha" type="range" min="0" max="1" step="0.05" value="0.45"><span id="alphaValue"></span></label>
<label>Mode <select id="mode"></select></label>
</div><img id="stream" src="stream.mjpg"><div id="status"></div><div id="msg"></div>
<script>
const ids=n=>document.getElementById(n); let busy=false, taskTrials=null, lastHdf5, files=[], model='';
// File name only, without .hdf5; long names keep their start and distinguishing end.
const shortName=(p,n=44)=>{const b=p.split('/').pop().replace(/\.hdf5$/,'');return b.length>n?b.slice(0,n-17)+'…'+b.slice(-16):b;};
const fmtB=b=>b>1e9?(b/1e9).toFixed(2)+' GB':(b/1e6).toFixed(0)+' MB';
const msg=(text,bad)=>{ids('msg').textContent=text||'';ids('msg').className=bad?'bad':'';};

/* reference file: model + file dropdowns */
async function loadFiles(){
 files=(await (await fetch('api/files')).json()).files||[];
 const counts={}; files.forEach(f=>counts[f.model]=(counts[f.model]||0)+1);
 const models=Object.keys(counts).sort((a,b)=>(a.startsWith('ICL')-b.startsWith('ICL'))||a.localeCompare(b));
 const m=ids('model'); m.replaceChildren(new Option(`All models (${files.length})`,''));
 models.forEach(x=>m.add(new Option(`${x} (${counts[x]})`,x))); m.value=model; fillFiles();
}
function fillFiles(current){
 const sel=ids('file'), shown=files.filter(f=>!model||f.model===model);
 sel.replaceChildren(new Option(`— choose a reference file (${shown.length}) —`,''));
 shown.forEach(f=>{const o=new Option(`${shortName(f.name)} — ${model?'':f.model+' · '}${fmtB(f.size_bytes)}`,f.path);o.title=f.name;sel.add(o);});
 const cur=current!==undefined?current:lastHdf5;
 if(cur&&!shown.some(f=>f.path===cur)){
  // An open file outside the model filter: switch the filter to its model,
  // or to all models for a --hdf5 file outside --root.
  const known=files.find(f=>f.path===cur); const next=known?known.model:'';
  if(next!==model){model=next;ids('model').value=model;return fillFiles(cur);}
  const o=new Option(shortName(cur),cur); o.title=cur.split('/').pop(); sel.add(o);
 }
 sel.value=cur||''; sel.title=cur?cur.split('/').pop():'';
}
ids('model').onchange=e=>{model=e.target.value;fillFiles();};
ids('file').onchange=e=>{if(e.target.value)openFile(e.target.value);};
async function openFile(path){
 msg('Opening…');
 const r=await fetch('api/open?'+new URLSearchParams({path})); const s=await r.json();
 if(!r.ok){msg(s.error,true);fillFiles();return;}
 msg(''); await loadStatus(); fillFiles(s.hdf5);
}

/* demo / frame / opacity / mode */
function showTrials(task, selected){
 const demo=ids('demo'); demo.replaceChildren();
 taskTrials[task].forEach(d=>demo.add(new Option(d.match(/_trial_(\d+)$/)[1],d)));
 demo.value=selected && taskTrials[task].includes(selected) ? selected : taskTrials[task][0];
}
function rebuildDemos(s){
 ids('demo').replaceChildren(); ids('task').replaceChildren(); taskTrials=null;
 ids('taskLabel').hidden=true; ids('demoLabel').firstChild.textContent='Demo ';
 const matches=s.demos.map(d=>d.match(/^(.+)_trial_(\d+)$/));
 if(s.demos.length&&matches.every(Boolean)){
  taskTrials={}; s.demos.forEach((d,i)=>(taskTrials[matches[i][1]]??=[]).push(d));
  Object.keys(taskTrials).sort().forEach(t=>ids('task').add(new Option(t.replaceAll('_',' '),t)));
  ids('taskLabel').hidden=false; ids('demoLabel').firstChild.textContent='Trial ';
 }else{s.demos.forEach(d=>ids('demo').add(new Option(d,d)));}
}
async function loadStatus(){
 const s=await (await fetch('api/status')).json();
 if(!ids('mode').options.length)s.modes.forEach(m=>ids('mode').add(new Option(m,m)));
 if(s.hdf5!==lastHdf5){lastHdf5=s.hdf5; rebuildDemos(s); fillFiles(s.hdf5);}
 const has=!!s.hdf5; ['demo','task','frame'].forEach(n=>ids(n).disabled=!has);
 if(has){
  if(taskTrials){const task=s.demo.match(/^(.+)_trial_(\d+)$/)[1]; if(ids('task').value!==task || !ids('demo').options.length){ids('task').value=task;showTrials(task,s.demo);}}
  ids('demo').value=s.demo; ids('frame').max=s.frame_count-1; ids('frame').value=s.frame;
  ids('frameValue').textContent=`${s.frame}/${s.frame_count-1}`;
 }else{ids('frameValue').textContent='';}
 ids('alpha').value=s.alpha; ids('alphaValue').textContent=Number(s.alpha).toFixed(2); ids('mode').value=s.mode;
 const live={live:'LIVE',stale:`STALE (${Number(s.live_age_s).toFixed(1)}s)`,waiting:'WAITING FOR CAMERA'}[s.live];
 const st=ids('status'); st.replaceChildren();
 const line=(text,bad)=>{const d=document.createElement('div'); d.textContent=text; if(bad)d.className='bad'; st.append(d);};
 line(`${live} | ${s.camera} | ${has?(s.task ? s.task+' | ' : '')+s.hdf5.split('/').pop()+' : '+s.image_key:'no reference file: choose one above'}`, s.live!=='live'||!has);
 if(s.live_error) line(s.live_error, true);
 if(s.reference_note) line(s.reference_note, true);
}
async function update(params){if(busy)return;busy=true;try{await fetch('api/config?'+new URLSearchParams(params));await loadStatus();}finally{busy=false;}}
ids('task').onchange=e=>{showTrials(e.target.value);update({demo:ids('demo').value});};
ids('demo').onchange=e=>update({demo:e.target.value}); ids('frame').oninput=e=>{ids('frameValue').textContent=e.target.value}; ids('frame').onchange=e=>update({frame:e.target.value});
ids('alpha').oninput=e=>{ids('alphaValue').textContent=Number(e.target.value).toFixed(2)}; ids('alpha').onchange=e=>update({alpha:e.target.value}); ids('mode').onchange=e=>update({mode:e.target.value});
loadFiles().then(loadStatus); setInterval(loadStatus,1000); setInterval(loadFiles,15000);
</script></main></body></html>"""


class OverlayHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, state, stream_fps, files):
        super().__init__(address, OverlayHandler)
        self.state = state
        self.stream_fps = stream_fps
        self.files = files


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
        elif request.path in ("/api/files", "/api/open"):
            try:
                values = {key: items[0] for key, items in parse_qs(request.query).items()}
                files = self.server.files
                if request.path == "/api/files":
                    self._send_json({"root": str(files.root), "files": files.recordings()})
                else:
                    self.server.state.open_file(files.check(values["path"]))
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
    parser.add_argument("--hdf5", type=Path,
                        help="initial reference recording (rollout or teleop HDF5); "
                             "otherwise choose one in the browser")
    parser.add_argument("--root", type=Path, default=ROOT / "data" / "rollouts" / "runs",
                        help="recordings listed in the Model/File dropdowns")
    parser.add_argument("--camera", choices=sorted(CAMERAS), default="exo",
                        help="exo=overhead (/dev/video18), wrist=gripper (/dev/video4)")
    parser.add_argument("--demo", default="0", help="demo number or group name (inference episodes: 'episode')")
    parser.add_argument("--frame", type=int, default=0, help="reference frame within the demo")
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--camera-device", type=int,
                        help="/dev/video index; default per --camera")
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

    device = args.camera_device if args.camera_device is not None else CAMERAS[args.camera]["device"]
    transform = _camera_transform(args.camera)
    files = ReferenceFiles(args.root)
    store = None if args.hdf5 is None else DemoStore(args.hdf5, args.camera, transform)
    state = OverlayState(store, args.camera, args.demo, args.frame, args.alpha,
                         args.stale_s, args.min_edge_px, transform)
    subscriber = CameraReader(args.camera, device, transform, state, args.stream_fps)
    server = OverlayHTTPServer(
        (args.http_host, args.http_port), state, args.stream_fps, files)

    def stop(_signum, _frame):
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    if store is None:
        print("HDF5: none yet; choose a reference file in the browser")
    else:
        print(f"HDF5: {store.path} ({store.image_key})")
        print(f"Demos: {store.demos[0]} ... {store.demos[-1]} ({len(store.demos)} total)")
    print(f"Live: {args.camera} on /dev/video{device}")
    print(f"Open http://{socket.gethostname()}:{args.http_port} in a browser")
    try:
        server.serve_forever()
    finally:
        server.server_close()
        subscriber.close()
        if state.store is not None:
            state.store.close()


if __name__ == "__main__":
    main()
