#!/usr/bin/env python3
"""Browse, play back, and score recorded xArm7 rollouts and demonstrations.

Lists every *.hdf5 under --root (default xarm_logs/runs, where
xarm_fastwam_inference.py saves its episodes), shows each demo's metadata,
plays the exo and wrist cameras (buffered in the browser and drawn every
display frame), and edits the 0/50/100 trial score of inference episodes.

Two file layouts are read:
  - inference episodes (xarm_fastwam_inference.py): datasets at the file root
    (primary_images_jpeg, wrist_images_jpeg), shown as one demo, "episode".
  - collected demos (xarm_data_collection.py): data/demo_N/obs/
    {agentview,eye_in_hand}_rgb_jpeg. These are read-only here, since they
    carry no trial score.

Reads never take an HDF5 file lock. Score edits take the normal write lock.

Usage (fastwam_xarm7 env):
  python rollout_viewer.py [--root xarm_logs/runs] [--http-port 8766]
then open http://<this-host>:8766
"""

import argparse
import json
import socket
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent  # experiments/xarm7/
# Trial rubric from xarm_fastwam_inference.py (SCORE_KEYS).
SCORE_LEVELS = {0: "no contact", 50: "touched", 100: "goal reached"}
# Stream id -> dataset paths to try, collected-demo layout first.
CAMERAS = {
    "exo": ("obs/agentview_rgb_jpeg", "primary_images_jpeg"),
    "wrist": ("obs/eye_in_hand_rgb_jpeg", "wrist_images_jpeg"),
}
DEFAULT_HZ = 5.0  # inference control rate; collected demos store control_hz
MAX_BATCH_FRAMES = 120
MAX_STRIDE = 64


class Conflict(Exception):
    """The file is locked by another writer (normally the live recorder)."""


def _text(value, default=""):
    if value is None:
        return default
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return str(value)


def _number(value, default=None):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default


def _model_label(path, root_attrs=None):
    """Policy that produced a recording: a model_name file attribute when one
    exists, otherwise the file naming convention (*_icl_<variant> = ICL head,
    anything else = the original VPP head)."""
    name = _text((root_attrs or {}).get("model_name"))
    if name:
        return name
    stem = Path(path).stem
    if "--model=" in stem:  # episode_data--score=..--success=..--model=<tag>
        return stem.split("--model=", 1)[1]
    return "collected demos"


def _demo_groups(root):
    """Demos in a file: data/demo_N groups, or the file itself as one
    'episode' for an inference episode with datasets at the root."""
    data = root.get("data")
    if isinstance(data, h5py.Group):
        return {k: v for k, v in data.items() if isinstance(v, h5py.Group)}
    return {"episode": root}


def _demo_sort_key(name):
    suffix = name.rsplit("_", 1)[-1]
    return (0, int(suffix)) if suffix.isdigit() else (1, name)


def _streams(group):
    """Camera streams present in a demo, keyed by the id the UI requests."""
    streams = {}
    for camera, paths in CAMERAS.items():
        for path in paths:
            if isinstance(group.get(path), h5py.Dataset):
                streams[camera] = path
                break
    return streams


def _demo_summary(key, group):
    attrs = group.attrs
    streams = _streams(group)
    lengths = [int(group[path].shape[0]) for path in streams.values()]
    # Play only the range every camera has: a demo still being recorded can
    # have unequal stream lengths.
    frames = min(lengths) if lengths else 0
    start = _number(attrs.get("recording_start_time"))
    end = _number(attrs.get("recording_end_time"))
    score = int(attrs.get("score", -1))
    hz = _number(attrs.get("control_hz"), DEFAULT_HZ) or DEFAULT_HZ
    return {
        "key": key,
        "task": _text(attrs.get("task_description")),
        "status": "completed",
        "score": score if score in SCORE_LEVELS else -1,
        "success": bool(attrs.get("success", False)),
        "frames": frames,
        "start_time": start,
        "duration_s": (end - start) if start is not None and end is not None else frames / hz,
        "record_hz": hz,
        "streams": list(streams),
    }


class RecordingStore:
    def __init__(self, root):
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            print(f"[viewer] recording root does not exist yet: {self.root}")
        self._summaries = {}

    def resolve(self, relative):
        path = (self.root / str(relative)).resolve()
        if (path.suffix != ".hdf5" or not path.is_file()
                or not path.is_relative_to(self.root)):
            raise FileNotFoundError(f"no recording {relative!r} under {self.root}")
        return path

    def _open(self, path):
        return h5py.File(path, "r", locking=False)

    @staticmethod
    def _version(path):
        stat = path.stat()
        return f"{stat.st_mtime_ns}-{stat.st_size}"

    def list_files(self):
        files = []
        if not self.root.is_dir():
            return files
        paths = sorted(self.root.rglob("*.hdf5"), key=lambda p: p.stat().st_mtime, reverse=True)
        for path in paths:
            version = self._version(path)
            cached = self._summaries.get(path)
            if cached is None or cached["version"] != version:
                cached = self._summarize_file(path, version)
                self._summaries[path] = cached
            files.append(cached)
        return files

    def _summarize_file(self, path, version):
        summary = {
            "path": str(path.relative_to(self.root)),
            "model": _model_label(path),
            "version": version,
            "size_bytes": path.stat().st_size,
            "modified": path.stat().st_mtime,
        }
        try:
            with self._open(path) as root:
                summary["model"] = _model_label(path, root.attrs)
                demos = _demo_groups(root)
                scores, tasks = [], set()
                for group in demos.values():
                    attrs = group.attrs
                    score = int(attrs.get("score", -1))
                    if score in SCORE_LEVELS:
                        scores.append(score)
                    task = _text(attrs.get("task_description"))
                    if task:
                        tasks.add(task)
                summary.update(
                    demos=len(demos),
                    scored=len(scores),
                    mean_score=float(np.mean(scores)) if scores else None,
                    running=0,
                    tasks=sorted(tasks),
                )
        except Exception as exc:
            summary["read_error"] = str(exc)
        return summary

    def file_detail(self, relative):
        path = self.resolve(relative)
        with self._open(path) as root:
            groups = _demo_groups(root)
            demos = [_demo_summary(key, groups[key]) for key in sorted(groups, key=_demo_sort_key)]
            editable = "score" in root.attrs  # inference episodes carry a trial score
        return {
            "path": str(path.relative_to(self.root)),
            "version": self._version(path),
            "editable": editable,
            "score_levels": SCORE_LEVELS,
            "demos": demos,
        }

    def _demo(self, root, demo):
        group = _demo_groups(root).get(str(demo))
        if group is None:
            raise FileNotFoundError(f"no demo {demo!r}")
        return group

    def timeline(self, relative, demo):
        """Seconds from the demo's first row, from the recorded step clock."""
        with self._open(self.resolve(relative)) as root:
            group = self._demo(root, demo)
            frames = _demo_summary(demo, group)["frames"]
            hz = _number(group.attrs.get("control_hz"), DEFAULT_HZ) or DEFAULT_HZ
            return (np.arange(frames) / hz).round(4).tolist()

    def frames(self, relative, demo, stream, start, count, stride=1):
        """Every stride-th JPEG from start, as one body: u32 n, n x u32
        lengths, then the bytes.

        Batching keeps playback to a few requests per second per camera, so the
        browser can buffer ahead and draw every display frame; the stride lets
        fast playback fetch only the frames a 60 fps display can show.
        """
        count = max(0, min(int(count), MAX_BATCH_FRAMES))
        stride = max(1, min(int(stride), MAX_STRIDE))
        with self._open(self.resolve(relative)) as root:
            group = self._demo(root, demo)
            path = _streams(group).get(stream)
            if path is None:
                raise FileNotFoundError(f"{demo} has no stream {stream!r}")
            dataset = group[path]
            if not 0 <= start < dataset.shape[0]:
                raise FileNotFoundError(
                    f"frame {start} outside {stream} [0, {dataset.shape[0]})")
            rows = dataset[start:min(start + count * stride, dataset.shape[0]):stride]
        jpegs = [np.asarray(row, dtype=np.uint8).tobytes() for row in rows]
        header = np.array([len(jpegs), *map(len, jpegs)], dtype="<u4").tobytes()
        return header + b"".join(jpegs)

    def set_score(self, relative, demo, score):
        if score != -1 and score not in SCORE_LEVELS:
            raise ValueError("score must be 0, 50, 100, or -1 (unscored)")
        path = self.resolve(relative)
        try:
            root = h5py.File(path, "r+")
        except OSError as exc:
            raise Conflict(
                "file is open for writing (a rollout is probably being recorded "
                "to it); score it after the session ends") from exc
        with root:
            if "score" not in root.attrs:
                raise ValueError("only inference episodes can be scored here")
            # Keep the rubric components consistent with the score.
            root.attrs["score"] = int(score)
            if score != -1:
                root.attrs["touched"] = score >= 50
                root.attrs["goal_reached"] = score == 100
                root.attrs["success"] = score == 100
            root.attrs["score_edited_time"] = time.time()
        return self.file_detail(relative)


INDEX_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Rollout Viewer</title>
<style>
:root{--bg:#111315;--panel:#1b1e21;--panel2:#23272b;--line:#30353a;--text:#e8eaed;--muted:#9aa0a6;
--accent:#8ab4f8;--warn:#fdd663;--err:#f28b82;--ok:#81c995}
*{box-sizing:border-box}[hidden]{display:none!important}
body{margin:0;background:var(--bg);color:var(--text);font:14px/1.4 system-ui,sans-serif}
header{display:flex;flex-wrap:wrap;align-items:center;gap:10px 16px;padding:10px 16px;border-bottom:1px solid var(--line)}
header h1{font-size:16px;margin:0 6px 0 0;white-space:nowrap}
label.pick{display:flex;align-items:center;gap:6px;color:var(--muted);min-width:0;flex:1 1 300px}
label.pick select{flex:1;min-width:0}
label.pick.model{flex:0 1 260px}
select,button{font:inherit;color:var(--text);background:var(--panel2);border:1px solid var(--line);border-radius:6px;padding:5px 9px}
button{cursor:pointer}button:hover:not(:disabled){border-color:var(--accent)}button:disabled{opacity:.45;cursor:default}
button.primary{background:#1a3a5c;border-color:#2f5f8f}
main{padding:12px 16px;max-width:1500px;margin:0 auto}
.empty{color:var(--muted);padding:48px 0;text-align:center}
.info{display:flex;flex-wrap:wrap;gap:4px 18px;background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:9px 12px}
.info b{color:var(--muted);font-weight:500;margin-right:5px}
.cams{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:10px;margin-top:12px}
figure{margin:0;background:#000;border-radius:8px;overflow:hidden;position:relative}
figure canvas{display:block;width:100%;aspect-ratio:1}
figcaption{position:absolute;left:8px;top:6px;font-size:12px;background:#000a;padding:1px 6px;border-radius:4px}
.bar{display:flex;flex-wrap:wrap;align-items:center;gap:10px;margin-top:12px;background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:9px 12px}
.bar input[type=range]{flex:1;min-width:160px}
.mono{font-variant-numeric:tabular-nums;color:var(--muted)}
#time{min-width:200px}#fps{min-width:150px;text-align:right}
.tag{color:var(--muted)}.warn{color:var(--warn)}.err{color:var(--err)}.ok{color:var(--ok)}
.confirm{display:flex;flex-wrap:wrap;align-items:center;gap:10px;background:#2a2410;border-color:#5c4d1a}
</style></head><body>
<header>
<h1>Rollout viewer</h1>
<label class="pick model">Model <select id="model"></select></label>
<label class="pick">File <select id="file"></select></label>
<label class="pick">Demo <select id="demo" disabled></select></label>
</header>
<main id="main"><div class="empty">Choose a file.</div></main>
<template id="playerTpl">
<div class="info" id="info"></div>
<div class="cams" id="cams"></div>
<div class="bar">
<button id="play">Play</button>
<select id="speed" title="playback speed"><option value="0.25">0.25x</option><option value="0.5">0.5x</option><option value="1" selected>1x</option><option value="2">2x</option><option value="4">4x</option><option value="8">8x</option><option value="16">16x</option></select>
<input type="range" id="scrub" min="0" value="0" step="1" aria-label="frame">
<span class="mono" id="time"></span><span class="mono" id="fps"></span>
</div>
<div class="bar">
<span class="tag">Score</span>
<select id="scoreSel" aria-label="score"></select>
<button id="scoreSave" class="primary">Save score</button>
<span id="scoreNote" class="tag"></span>
</div>
<div class="bar confirm" id="confirm" hidden>
<span id="confirmText"></span>
<button id="confirmYes" class="primary">Confirm</button><button id="confirmNo">Cancel</button>
</div>
<div id="msg" style="min-height:20px;margin-top:8px"></div>
</template>
<script>
const $=id=>document.getElementById(id), enc=encodeURIComponent;
const LEVELS={'-1':'unscored',0:'0 · no contact',50:'50 · touched',100:'100 · goal reached'};
const NAMES={exo:'exo',wrist:'wrist'};
const CHUNK=60, AHEAD_S=3, KEEP=300;   // frames per request, read-ahead, frames kept around the playhead
const S={files:[],filesKey:'',model:'',file:'',detail:null,demo:'',times:[],cams:[],ctx:{},frame:0,drawn:-1,
 playing:false,speed:1,clock:0,last:0,buffering:false,token:0,bitmaps:new Map(),pending:new Set(),ticks:[]};
async function api(url,opts){const r=await fetch(url,opts);const j=await r.json().catch(()=>({error:r.statusText}));if(!r.ok)throw new Error(j.error||r.statusText);return j;}
const esc=s=>String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const fmtT=t=>t==null?'–':(t>=60?`${Math.floor(t/60)}m${String(Math.round(t%60)).padStart(2,'0')}s`:`${t.toFixed(1)}s`);
const fmtB=b=>b>1e9?(b/1e9).toFixed(2)+' GB':(b/1e6).toFixed(0)+' MB';
const demoInfo=()=>S.detail&&S.detail.demos.find(d=>d.key===S.demo);

/* ---------- file and demo dropdowns ---------- */
async function loadFiles(){
 const r=await api('api/files'); const key=JSON.stringify(r.files.map(f=>[f.path,f.version,f.scored,f.model]));
 if(key===S.filesKey)return; S.filesKey=key; S.files=r.files; $('file').title=r.root;
 fillModels(); fillFiles();
}
function fillModels(){
 // VPP heads first, then ICL variants; each with its file count.
 const counts={}; for(const f of S.files)counts[f.model]=(counts[f.model]||0)+1;
 const models=Object.keys(counts).sort((a,b)=>(a.startsWith('ICL')-b.startsWith('ICL'))||a.localeCompare(b));
 const sel=$('model'); sel.replaceChildren(new Option(`All models (${S.files.length} files)`,''));
 for(const m of models)sel.add(new Option(`${m} (${counts[m]} file${counts[m]>1?'s':''})`,m));
 if(S.model&&!counts[S.model])S.model=''; sel.value=S.model;
}
function fillFiles(){
 const sel=$('file'), shown=S.files.filter(f=>!S.model||f.model===S.model);
 sel.replaceChildren(new Option(`— choose a file (${shown.length}) —`,''));
 for(const f of shown){
  const meta=f.read_error?'unreadable':`${f.demos} demos, ${f.scored} scored${f.mean_score!=null?', mean '+f.mean_score.toFixed(2):''}, ${fmtB(f.size_bytes)}${f.running?', RECORDING':''}`;
  sel.add(new Option(`${f.path} — ${S.model?'':f.model+' · '}${meta}`,f.path));
 }
 sel.value=shown.some(f=>f.path===S.file)?S.file:'';
}
$('model').onchange=e=>{
 S.model=e.target.value; fillFiles();
 // Leave the open demo only if it belongs to the chosen model.
 if(S.file&&$('file').value!==S.file)selectFile('');
};
$('file').onchange=e=>selectFile(e.target.value);
$('demo').onchange=e=>selectDemo(e.target.value);
async function selectFile(path,keepDemo){
 const token=++S.token; stop(); resetFrames(); S.file=path; if(!keepDemo)S.demo='';
 const dsel=$('demo'); dsel.disabled=true;
 if(!path){dsel.replaceChildren();$('main').innerHTML='<div class="empty">Choose a file.</div>';return;}
 try{
  const d=await api('api/file?path='+enc(path)); if(token!==S.token)return; S.detail=d; fillDemos();
  const pick=keepDemo&&S.demo?S.demo:(d.demos[0]&&d.demos[0].key);
  if(pick)await selectDemo(pick,keepDemo); else $('main').innerHTML='<div class="empty">This file has no demos.</div>';
 }catch(e){if(token===S.token)$('main').innerHTML=`<div class="empty err">${esc(e.message)}</div>`;}
}
function fillDemos(){
 const sel=$('demo'); sel.replaceChildren();
 for(const d of S.detail.demos){
  const st=d.status==='running'?' · RECORDING':d.status!=='completed'?' · '+d.status:'';
  sel.add(new Option(`${d.key} — ${LEVELS[d.score]} · ${fmtT(d.duration_s)}${st}`,d.key));
 }
 sel.disabled=!S.detail.demos.length; sel.value=S.demo;
}
async function selectDemo(key,keepFrame){
 const token=++S.token; stop(); resetFrames(); S.demo=key; $('demo').value=key;
 try{
  const t=await api(`api/timeline?path=${enc(S.file)}&demo=${enc(key)}`); if(token!==S.token)return;
  S.times=t.times; if(!keepFrame)S.frame=0; S.frame=Math.min(S.frame,Math.max(0,S.times.length-1));
  S.clock=S.times[S.frame]||0; renderPlayer();
 }catch(e){if(token===S.token)$('main').innerHTML=`<div class="empty err">${esc(e.message)}</div>`;}
}

/* ---------- player ---------- */
function renderPlayer(){
 const d=demoInfo(); if(!d)return; $('main').replaceChildren($('playerTpl').content.cloneNode(true));
 const hz=d.effective_hz?`${d.effective_hz.toFixed(1)} Hz`:(d.record_hz?`${d.record_hz} Hz`:'–');
 const items=[['task',d.task||'–'],['status',d.status],['success',d.success],
  ['duration',fmtT(d.duration_s)],['frames',d.frames],['recorded at',hz]];
 $('info').innerHTML=items.map(([k,v])=>`<span><b>${k}</b>${esc(v)}</span>`).join('');
 S.cams=d.streams; S.ctx={};
 for(const c of S.cams){const f=document.createElement('figure');f.innerHTML=`<canvas width="480" height="480"></canvas><figcaption>${NAMES[c]||c}</figcaption>`;$('cams').append(f);S.ctx[c]=f.querySelector('canvas').getContext('2d');}
 $('scrub').max=Math.max(0,S.times.length-1); $('scrub').value=S.frame;
 $('scrub').oninput=e=>{stop();seek(+e.target.value);};
 $('play').onclick=()=>S.playing?stop():play();
 $('speed').value=String(S.speed); $('speed').onchange=e=>{S.speed=+e.target.value;};
 renderScore(); S.drawn=-1; seek(S.frame);
 if(!d.frames)$('msg').innerHTML='<span class="warn">This demo has no recorded frames.</span>';
}
function resetFrames(){for(const b of S.bitmaps.values())b.close();S.bitmaps.clear();S.pending.clear();S.drawn=-1;}
const fkey=(c,i)=>c+':'+i;
function stride(){
 // While playing fast, fetch and draw only every k-th recorded frame: about
 // what a 60 fps display can show. Paused or slow playback uses every frame.
 const n=S.times.length; if(!S.playing||n<2)return 1;
 const hz=(n-1)/Math.max(S.times[n-1],1e-3);
 return Math.max(1,Math.floor(S.speed*hz/60));
}
const onGrid=(i,k)=>i-(i%k);
async function fetchChunk(cam,chunk,stepK){
 const k=cam+':c'+chunk+':s'+stepK; if(S.pending.has(k))return; S.pending.add(k); const token=S.token;
 const start=chunk*CHUNK*stepK;
 try{
  const r=await fetch(`api/frames?path=${enc(S.file)}&demo=${enc(S.demo)}&stream=${cam}&start=${start}&count=${CHUNK}&stride=${stepK}&v=${S.detail.version}`);
  if(!r.ok)throw new Error((await r.json().catch(()=>({}))).error||r.statusText);
  const buf=await r.arrayBuffer(); if(token!==S.token)return;
  const dv=new DataView(buf), n=dv.getUint32(0,true); let off=4+4*n; const parts=[];
  for(let j=0;j<n;j++){const len=dv.getUint32(4+4*j,true);parts.push(new Blob([new Uint8Array(buf,off,len)],{type:'image/jpeg'}));off+=len;}
  const bitmaps=await Promise.all(parts.map(b=>createImageBitmap(b)));
  if(token!==S.token){bitmaps.forEach(b=>b.close());return;}
  bitmaps.forEach((b,j)=>{const key=fkey(cam,start+j*stepK);const old=S.bitmaps.get(key);if(old)old.close();S.bitmaps.set(key,b);});
  S.pending.delete(k);
 }catch(e){
  // Leave k pending so a failing chunk is reported once, not refetched every frame.
  if(token===S.token)$('msg').innerHTML=`<span class="err">frames: ${esc(e.message)}</span>`;}
}
function ensure(i){
 // Read ahead AHEAD_S seconds of playback (recorded time x speed) from the playhead.
 const n=S.times.length; if(!n)return; const k=stride(), g=onGrid(i,k), span=CHUNK*k;
 let j=i; const until=S.times[i]+(S.playing?AHEAD_S*S.speed:0);
 while(j<n-1&&S.times[j+1]<=until)j++;
 for(let c=Math.floor(g/span);c<=Math.floor(j/span);c++)
  for(const cam of S.cams)if(!S.bitmaps.has(fkey(cam,Math.max(c*span,g))))fetchChunk(cam,c,k);
 if(S.bitmaps.size>KEEP*S.cams.length*1.5)for(const [key,b] of S.bitmaps){const idx=+key.split(':')[1];if(idx<i-KEEP/3||idx>j+KEEP){b.close();S.bitmaps.delete(key);}}
}
const ready=i=>S.cams.every(c=>S.bitmaps.has(fkey(c,i)));
function draw(i){
 for(const c of S.cams){const b=S.bitmaps.get(fkey(c,i));if(b)S.ctx[c].drawImage(b,0,0,480,480);}
 S.drawn=i; const n=S.times.length; $('scrub').value=i;
 $('time').textContent=`${fmtT(S.times[i])} / ${fmtT(S.times[n-1])} · frame ${i+1}/${n}`;
}
function seek(i){const n=S.times.length;if(!n){$('time').textContent='no frames';return;}S.frame=Math.max(0,Math.min(n-1,i));S.clock=S.times[S.frame];ensure(S.frame);}
function play(){const n=S.times.length;if(n<2)return;if(S.frame>=n-1)seek(0);S.playing=true;S.buffering=false;S.ticks=[];$('play').textContent='Pause';}
function stop(){S.playing=false;const p=$('play');if(p)p.textContent='Play';const f=$('fps');if(f)f.textContent='';}
function loop(now){
 const dt=S.last?(now-S.last)/1000:0; S.last=now; const n=S.times.length;
 if(n&&$('time')){
  if(S.playing){
   if(!S.buffering)S.clock+=Math.min(dt,0.1)*S.speed;
   let i=S.frame; while(i<n-1&&S.times[i+1]<=S.clock)i++;
   ensure(i); const shown=i>=n-1?n-1:onGrid(i,stride());
   if(ready(shown)){S.buffering=false;S.frame=i;if(shown!==S.drawn)draw(shown);if(i>=n-1)stop();}
   else if(i>=n-1){S.frame=i;stop();}
   else{S.buffering=true;}
   S.ticks.push(now); while(S.ticks.length&&now-S.ticks[0]>1000)S.ticks.shift();
   $('fps').textContent=S.buffering?'buffering…':`${S.ticks.length} fps display`;
  }else if(S.drawn!==S.frame&&ready(S.frame)){draw(S.frame);}
 }
 requestAnimationFrame(loop);
}
requestAnimationFrame(loop);

/* ---------- scoring with confirmation ---------- */
function renderScore(){
 const d=demoInfo(), sel=$('scoreSel'); sel.replaceChildren();
 for(const s of [-1,0,50,100])sel.add(new Option(LEVELS[s],String(s)));
 sel.value=String(d.score); $('confirm').hidden=true;
 const editable=S.detail.editable&&d.status!=='running';
 sel.disabled=!editable; $('scoreSave').disabled=true;
 $('scoreNote').innerHTML=!S.detail.editable?'read-only: no trial score (collected demo)':d.status==='running'?'<span class="warn">still recording</span>':'';
 sel.onchange=()=>{$('scoreSave').disabled=+sel.value===d.score;$('confirm').hidden=true;};
 $('scoreSave').onclick=()=>{
  $('confirmText').textContent=`Change ${S.file} / ${d.key} from "${LEVELS[d.score]}" to "${LEVELS[sel.value]}"?`;
  $('confirm').hidden=false; $('confirmYes').focus();
 };
 $('confirmNo').onclick=()=>{$('confirm').hidden=true;sel.value=String(d.score);$('scoreSave').disabled=true;};
 $('confirmYes').onclick=()=>saveScore(+sel.value);
}
async function saveScore(score){
 const token=S.token; $('confirm').hidden=true; $('msg').textContent='Saving…';
 try{
  const detail=await api('api/score',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({path:S.file,demo:S.demo,score})});
  if(token!==S.token)return; S.detail=detail; fillDemos(); renderScore();
  $('msg').innerHTML=`<span class="ok">Saved ${esc(S.demo)}: ${esc(LEVELS[score])}</span>`; S.filesKey=''; loadFiles();
 }catch(e){$('msg').innerHTML=`<span class="err">${esc(e.message)}</span>`;renderScore();}
}

document.addEventListener('keydown',e=>{
 if(!S.demo||!$('time')||['SELECT','BUTTON'].includes(e.target.tagName))return;
 if(e.code==='Space'){e.preventDefault();S.playing?stop():play();}
 else if(e.code==='ArrowRight'||e.code==='ArrowLeft'){e.preventDefault();stop();seek(S.frame+(e.code==='ArrowRight'?1:-1)*(e.shiftKey?30:1));}
});
loadFiles().catch(e=>{$('main').innerHTML=`<div class="empty err">${esc(e.message)}</div>`;});
setInterval(()=>{if(!S.playing)loadFiles().catch(()=>{});},10000);
</script></body></html>"""


class ViewerServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, store):
        super().__init__(address, ViewerHandler)
        self.store = store


class ViewerHandler(BaseHTTPRequestHandler):
    server: ViewerServer

    def _send(self, status, body, content_type, cache=False):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        # Frame URLs carry the file version, so they are safe to cache.
        self.send_header("Cache-Control", "max-age=3600" if cache else "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, value, status=HTTPStatus.OK):
        self._send(status, json.dumps(value).encode(), "application/json")

    def _error(self, exc):
        if isinstance(exc, Conflict):
            status = HTTPStatus.CONFLICT
        elif isinstance(exc, FileNotFoundError):
            status = HTTPStatus.NOT_FOUND
        elif isinstance(exc, (ValueError, KeyError)):
            status = HTTPStatus.BAD_REQUEST
        else:
            status = HTTPStatus.INTERNAL_SERVER_ERROR
        self._json({"error": str(exc)}, status)

    def do_GET(self):
        request = urlparse(self.path)
        query = {key: values[0] for key, values in parse_qs(request.query).items()}
        store = self.server.store
        try:
            if request.path == "/":
                self._send(HTTPStatus.OK, INDEX_HTML.encode(), "text/html; charset=utf-8")
            elif request.path == "/api/files":
                self._json({"root": str(store.root), "files": store.list_files()})
            elif request.path == "/api/file":
                self._json(store.file_detail(query["path"]))
            elif request.path == "/api/timeline":
                self._json({"times": store.timeline(query["path"], query["demo"])})
            elif request.path == "/api/frames":
                body = store.frames(query["path"], query["demo"], query["stream"],
                                    int(query["start"]), int(query["count"]),
                                    int(query.get("stride", 1)))
                self._send(HTTPStatus.OK, body, "application/octet-stream", cache=True)
            else:
                self.send_error(HTTPStatus.NOT_FOUND)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:
            self._error(exc)

    def do_POST(self):
        if urlparse(self.path).path != "/api/score":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(min(length, 4096)) or b"{}")
            self._json(self.server.store.set_score(
                body["path"], body["demo"], int(body["score"])))
        except Exception as exc:
            self._error(exc)

    def log_message(self, format, *args):
        return


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=ROOT / "xarm_logs" / "runs",
                        help="directory searched recursively for *.hdf5 recordings")
    parser.add_argument("--http-host", default="0.0.0.0")
    parser.add_argument("--http-port", type=int, default=8766)
    args = parser.parse_args()

    store = RecordingStore(args.root)
    server = ViewerServer((args.http_host, args.http_port), store)
    print(f"Recordings: {store.root} ({len(store.list_files())} files)")
    print(f"Open http://{socket.gethostname()}:{args.http_port} in a browser")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
