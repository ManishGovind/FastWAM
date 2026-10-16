"""Single-writer, checkpointed HDF5 storage for R1 Lite inference runs."""
from __future__ import annotations

import base64
import json
import os
import queue
import re
import shutil
import threading
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np

CAMERAS = {
    "head": "agentview_rgb_jpeg",
    "left_wrist": "wrist_left_rgb_jpeg",
    "right_wrist": "wrist_right_rgb_jpeg",
}
SAMPLE_NAMES = (
    "head", "left_wrist", "right_wrist", "left_arm", "right_arm",
    "left_gripper", "right_gripper",
)
JPEG_DTYPE = h5py.vlen_dtype(np.dtype("uint8"))
TEXT_DTYPE = h5py.string_dtype(encoding="utf-8")


def _append(dataset: h5py.Dataset, value: Any) -> None:
    index = len(dataset)
    dataset.resize(index + 1, axis=0)
    dataset[index] = value


def _json(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), allow_nan=False)


def _jpeg(image: np.ndarray) -> np.ndarray:
    ok, encoded = cv2.imencode(
        ".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
        (cv2.IMWRITE_JPEG_QUALITY, 90),
    )
    if not ok:
        raise RuntimeError("Could not encode rollout camera frame")
    return encoded.reshape(-1)


def _array(group: h5py.Group, name: str, tail: tuple[int, ...], dtype: Any) -> h5py.Dataset:
    return group.create_dataset(
        name, shape=(0, *tail), maxshape=(None, *tail),
        chunks=(1 if dtype == JPEG_DTYPE else 64, *tail), dtype=dtype,
    )


def _validate(path: Path, expected_demo: str) -> None:
    """Check the copy before it can replace the viewer's last good file."""
    with h5py.File(path, "r") as file:
        if file.attrs.get("schema_kind") != "fastwam_live_rollout_v1":
            raise RuntimeError("Checkpoint has the wrong schema")
        if f"data/{expected_demo}" not in file:
            raise RuntimeError(f"Checkpoint is missing {expected_demo}")
        for demo in file["data"].values():
            lengths = [len(demo[name]) for name in (
                "joint_states", "absolute_actions", "actions", "frame_timestamps_s",
            )]
            lengths.extend(len(demo["obs"][name]) for name in CAMERAS.values())
            if not lengths[0] or len(set(lengths)) != 1:
                raise RuntimeError(f"Unequal or empty viewer arrays in {demo.name}: {lengths}")
            if bytes(demo["obs"]["agentview_rgb_jpeg"][0][:2]) != b"\xff\xd8":
                raise RuntimeError(f"Invalid first JPEG in {demo.name}")
            steps = demo["rollout/steps"]
            if len({len(value) for value in steps.values()}) != 1:
                raise RuntimeError(f"Unequal step arrays in {demo.name}")


class Checkpoint:
    def __init__(self) -> None:
        self.done = threading.Event()
        self.error: Exception | None = None
        self.thread: threading.Thread | None = None

    def wait(self) -> None:
        self.done.wait()
        assert self.thread is not None
        self.thread.join()
        if self.error is not None:
            raise RuntimeError(f"Rollout checkpoint failed: {self.error}") from self.error


class HDF5RolloutRecorder:
    """One HDF5 owner thread; checkpoint copies run only while its file is closed."""

    def __init__(
        self,
        root: Path,
        metadata: dict[str, object],
        requested_demos: int,
        resume_from: Path | None = None,
    ) -> None:
        self.root = root
        policy_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(metadata.get("policy_config", "fastwam"))).strip("._")
        policy_name = policy_name or "fastwam"
        self.output = resume_from if resume_from is not None else root / f"{policy_name}.hdf5"
        stem = self.output.stem
        self.working = root / f"{stem}.inprogress"
        self.temporary = root / f"{stem}.checkpoint.tmp"
        self._resuming = resume_from is not None
        if self._resuming:
            if self.output.parent != root or not self.output.is_file():
                raise FileNotFoundError(f"Cannot resume missing rollout checkpoint: {self.output}")
            # Recover from the last validated atomic checkpoint, discarding
            # any partial active-demo tail left in the working file.
            shutil.copyfile(self.output, self.working)
        self._pending: queue.Queue[tuple[str, str | None, Any, threading.Event | None]] = queue.Queue()
        self._frame_slots = threading.BoundedSemaphore(30)
        self._lock = threading.Lock()
        self._phase = "idle"
        self._demo_id: str | None = None
        self._capture_publications = False
        self._dropped_frames = 0
        self._error: Exception | None = None
        self._ready = threading.Event()
        self._checkpoint: Checkpoint | None = None
        self._worker = threading.Thread(
            target=self._run, args=(metadata, requested_demos),
            name="rollout-hdf5-writer", daemon=False,
        )
        self._worker.start()
        self._ready.wait()
        self._check_error()

    def _check_error(self) -> None:
        if self._error is not None:
            raise RuntimeError(f"Rollout HDF5 writer failed: {self._error}") from self._error

    @property
    def recording_frames(self) -> bool:
        with self._lock:
            return self._phase == "recording"

    @property
    def phase(self) -> str:
        with self._lock:
            return self._phase

    def begin_demo(self, demo_id: str, task: str, control_hz: float, start_time_s: float) -> None:
        with self._lock:
            self._check_error()
            if self._phase != "idle":
                raise RuntimeError(f"Cannot start {demo_id} while rollout is {self._phase}")
            self._phase = "recording"
            self._demo_id = demo_id
            self._capture_publications = True
            self._dropped_frames = 0
            self._pending.put(("start", demo_id, (task, control_hz, start_time_s), None))

    def stop_frames(self) -> None:
        with self._lock:
            if self._phase == "recording":
                self._phase = "stopping"

    def submit_frame(
        self, timestamp_s: float, state: np.ndarray, images: dict[str, np.ndarray],
        sample_timestamps_s: dict[str, float], target: np.ndarray, has_target: bool,
    ) -> bool:
        if not self._frame_slots.acquire(blocking=False):
            with self._lock:
                self._dropped_frames += 1
            return False
        try:
            payload = (
                float(timestamp_s), np.asarray(state, dtype=np.float32).copy(),
                {name: image.copy() for name, image in images.items()},
                [float(sample_timestamps_s[name]) for name in SAMPLE_NAMES],
                np.asarray(target, dtype=np.float32).copy(), bool(has_target),
            )
        except Exception:
            self._frame_slots.release()
            raise
        with self._lock:
            if self._phase != "recording" or self._demo_id is None:
                self._frame_slots.release()
                return False
            self._pending.put(("frame", self._demo_id, payload, None))
        return True

    def submit_event(self, demo_id: str, record: dict[str, object]) -> None:
        with self._lock:
            if self._phase not in ("recording", "stopping") or demo_id != self._demo_id:
                return
            self._pending.put(("event", demo_id, record, None))

    def submit_post_event(self, demo_id: str, record: dict[str, object]) -> None:
        """Queue reset outcome behind the sealed snapshot for the final copy."""
        with self._lock:
            self._check_error()
            if self._phase != "sealing" or self._demo_id != demo_id:
                raise RuntimeError("Reset event arrived outside demo finalization")
            self._pending.put(("event", demo_id, record, None))

    def submit_step(self, demo_id: str, record: dict[str, object]) -> None:
        with self._lock:
            self._check_error()
            if self._phase not in ("recording", "stopping") or demo_id != self._demo_id:
                raise RuntimeError("Executed step arrived after demo seal")
            self._pending.put(("step", demo_id, record, None))

    def submit_query(self, demo_id: str, chunk_id: int, payload: dict[str, object], timestamps: dict[str, float]) -> None:
        with self._lock:
            self._check_error()
            if self._phase not in ("recording", "stopping") or demo_id != self._demo_id:
                return
            self._pending.put(("query", demo_id, (chunk_id, payload.copy(), timestamps.copy()), None))

    def submit_publication(self, record: dict[str, object]) -> None:
        with self._lock:
            if self._capture_publications:
                self._pending.put(("publication", None, record, None))

    def stop_publications(self) -> None:
        with self._lock:
            self._capture_publications = False

    def seal_demo(self, demo_id: str, end_time_s: float) -> Checkpoint:
        with self._lock:
            self._check_error()
            if self._phase not in ("recording", "stopping") or self._demo_id != demo_id:
                raise RuntimeError("Demo seal did not match active recording")
            self._phase = "sealing"
            ack = threading.Event()
            resume = threading.Event()
            self._pending.put(("seal", demo_id, (end_time_s, self._dropped_frames, resume), ack))
        return self._start_checkpoint(ack, resume, demo_id, final=False)

    def finalize_demo(
        self, demo_id: str, score: int | None, reset_ok: bool | None,
        reset_publications_done: bool = True,
    ) -> Checkpoint:
        if reset_publications_done:
            self.stop_publications()
        with self._lock:
            self._check_error()
            if self._phase != "sealing" or self._demo_id != demo_id:
                raise RuntimeError("Cannot finalize a demo that is not sealed")
            self._phase = "finalizing"
            ack = threading.Event()
            resume = threading.Event()
            self._pending.put(("finalize", demo_id, (score, reset_ok, resume), ack))
        return self._start_checkpoint(ack, resume, demo_id, final=True)

    def _start_checkpoint(
        self, ack: threading.Event, resume: threading.Event, demo_id: str, *, final: bool,
    ) -> Checkpoint:
        previous = self._checkpoint
        if previous is not None and not previous.done.is_set():
            raise RuntimeError("Overlapping rollout checkpoints")
        checkpoint = Checkpoint()
        self._checkpoint = checkpoint

        def run() -> None:
            try:
                while not ack.wait(0.1):
                    self._check_error()
                self._check_error()
                shutil.copyfile(self.working, self.temporary)
                _validate(self.temporary, demo_id)
                with self.temporary.open("r+b") as stream:
                    os.fsync(stream.fileno())
                os.replace(self.temporary, self.output)
                if os.name == "posix":
                    directory_fd = os.open(self.root, os.O_RDONLY)
                    try:
                        os.fsync(directory_fd)
                    finally:
                        os.close(directory_fd)
                if final:
                    with self._lock:
                        self._phase = "idle"
                        self._demo_id = None
            except Exception as exc:  # noqa: BLE001 - preserve last valid checkpoint
                checkpoint.error = exc
            finally:
                resume.set()
                checkpoint.done.set()

        checkpoint.thread = threading.Thread(target=run, name="rollout-checkpoint", daemon=False)
        checkpoint.thread.start()
        return checkpoint

    def close(self, completed: bool = False) -> None:
        checkpoint_error: Exception | None = None
        if self._checkpoint is not None:
            try:
                self._checkpoint.wait()
            except Exception as exc:  # noqa: BLE001 - still join the writer
                checkpoint_error = exc
        if self._worker.is_alive():
            self._pending.put(("close", None, None, None))
            self._worker.join()
        self._check_error()
        if checkpoint_error is not None:
            raise checkpoint_error
        if completed:
            self.working.unlink(missing_ok=True)
            self.temporary.unlink(missing_ok=True)

    def _run(self, metadata: dict[str, object], requested_demos: int) -> None:
        file: h5py.File | None = None
        try:
            file = h5py.File(self.working, "r+" if self._resuming else "w")
            if self._resuming:
                if file.attrs.get("schema_kind") != "fastwam_live_rollout_v1":
                    raise RuntimeError("Cannot resume rollout with the wrong schema")
                if int(file.attrs["requested_demos"]) != requested_demos:
                    raise RuntimeError("Resume requested-demo count does not match checkpoint")
                existing_scores = [
                    int(demo.attrs["accuracy_score"])
                    for demo in file["data"].values()
                    if "accuracy_score" in demo.attrs
                ]
                file.attrs["scored_demos"] = len(existing_scores)
                file.attrs["success_count"] = sum(score == 100 for score in existing_scores)
                file.attrs["success_rate"] = file.attrs["success_count"] / requested_demos
                file.attrs["score_total"] = sum(existing_scores)
                file.attrs["score_max"] = requested_demos * 100
            else:
                file.attrs["schema_kind"] = "fastwam_live_rollout_v1"
                file.attrs["run_metadata_json"] = _json(metadata)
                file.attrs["requested_demos"] = requested_demos
                file.attrs["scored_demos"] = 0
                file.attrs["success_count"] = 0
                file.attrs["success_rate"] = 0.0
                file.attrs["score_total"] = 0
                file.attrs["score_max"] = requested_demos * 100
                file.create_group("data")
                _array(file, "command_publications_json", (), TEXT_DTYPE)
            self._ready.set()
            while True:
                kind, demo_id, payload, ack = self._pending.get()
                try:
                    if kind == "close":
                        return
                    assert file is not None
                    if kind == "start":
                        task, hz, start = payload
                        demo = file["data"].create_group(demo_id)
                        demo.attrs["task_description"] = task
                        demo.attrs["control_hz"] = hz
                        demo.attrs["recording_start_time"] = start
                        demo.attrs["status"] = "recording"
                        demo.attrs["success"] = False
                        for name in ("joint_states", "absolute_actions", "actions"):
                            _array(demo, name, (14,), np.float32)
                        _array(demo, "frame_timestamps_s", (), np.float64)
                        _array(demo, "sample_timestamps_s", (len(SAMPLE_NAMES),), np.float64)
                        _array(demo, "has_command", (), np.bool_)
                        obs = demo.create_group("obs")
                        for name in CAMERAS.values():
                            _array(obs, name, (), JPEG_DTYPE)
                        rollout = demo.create_group("rollout")
                        _array(rollout, "events_json", (), TEXT_DTYPE)
                        rollout.create_group("queries")
                        steps = rollout.create_group("steps")
                        for name in ("model_action", "commanded_target", "measured_before", "measured_after"):
                            _array(steps, name, (14,), np.float32)
                        for name in ("chunk_id", "action_step"):
                            _array(steps, name, (), np.int32)
                        _array(steps, "record_json", (), TEXT_DTYPE)
                    elif kind == "frame":
                        stamp, state, images, sample_stamps, target, has_target = payload
                        demo = file["data"][demo_id]
                        for name, value in (
                            ("joint_states", state), ("absolute_actions", target),
                            ("actions", target - state), ("frame_timestamps_s", stamp),
                            ("sample_timestamps_s", sample_stamps), ("has_command", has_target),
                        ):
                            _append(demo[name], value)
                        for source, destination in CAMERAS.items():
                            _append(demo["obs"][destination], _jpeg(images[source]))
                    elif kind == "event":
                        _append(file["data"][demo_id]["rollout/events_json"], _json(payload))
                    elif kind == "step":
                        steps = file["data"][demo_id]["rollout/steps"]
                        for name, value in (
                            ("model_action", payload["model_action"]),
                            ("commanded_target", payload["commanded_target"]),
                            ("measured_before", payload["measured_before"]),
                            ("measured_after", payload["measured_after"]),
                            ("chunk_id", payload["chunk_id"]),
                            ("action_step", payload["action_step"]),
                            ("record_json", _json(payload)),
                        ):
                            _append(steps[name], value)
                    elif kind == "query":
                        chunk_id, query, stamps = payload
                        group = file["data"][demo_id]["rollout/queries"].create_group(f"chunk_{chunk_id:04d}")
                        group.attrs["chunk_id"] = chunk_id
                        group.attrs["sample_timestamps_json"] = _json(stamps)
                        group.create_dataset("state", data=np.asarray(query["state"], dtype=np.float32))
                        for source, field in (("head", "head_image"), ("left_wrist", "left_wrist_image"), ("right_wrist", "right_wrist_image")):
                            group.create_dataset(source + "_jpeg", data=np.frombuffer(base64.b64decode(query[field], validate=True), dtype=np.uint8))
                    elif kind == "publication":
                        _append(file["command_publications_json"], _json(payload))
                    elif kind == "seal":
                        end_time, dropped, resume = payload
                        demo = file["data"][demo_id]
                        demo.attrs["recording_end_time"] = end_time
                        demo.attrs["num_samples"] = len(demo["joint_states"])
                        demo.attrs["dropped_frames"] = dropped
                        demo.attrs["status"] = "stopped_unscored"
                        file.flush()
                        file.close()
                        file = None
                        assert ack is not None
                        ack.set()
                        resume.wait()
                        file = h5py.File(self.working, "a")
                    elif kind == "finalize":
                        score, reset_ok, resume = payload
                        demo = file["data"][demo_id]
                        if score is not None:
                            demo.attrs["accuracy_score"] = score
                            demo.attrs["success"] = score == 100
                            file.attrs["scored_demos"] = int(file.attrs["scored_demos"]) + 1
                            file.attrs["success_count"] = int(file.attrs["success_count"]) + int(score == 100)
                            file.attrs["score_total"] = int(file.attrs["score_total"]) + score
                        demo.attrs["reset_ok"] = bool(reset_ok) if reset_ok is not None else False
                        demo.attrs["status"] = "scored" if score is not None else "interrupted"
                        file.attrs["success_rate"] = int(file.attrs["success_count"]) / int(file.attrs["requested_demos"])
                        file.flush()
                        file.close()
                        file = None
                        assert ack is not None
                        ack.set()
                        resume.wait()
                        file = h5py.File(self.working, "a")
                    else:
                        raise RuntimeError(f"Unknown HDF5 record kind: {kind}")
                finally:
                    if kind == "frame":
                        self._frame_slots.release()
                    self._pending.task_done()
        except Exception as exc:  # noqa: BLE001 - surface writer failure
            self._error = exc
            self._ready.set()
        finally:
            if file is not None:
                file.close()
