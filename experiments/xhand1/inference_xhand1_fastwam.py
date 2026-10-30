#!/usr/bin/env python3
"""Run FastWAM on the xArm7 + XHand1 from the robot workstation.

serve_fastwam_xhand1.py (GPU host) does all FastWAM work; this client owns the
cameras, measured state, interpolation, safety gates, and both robot SDKs.

The hardware layer (xhand1_hardware.py) is vendored from vpp_real's
scripts/deploy_policy.py: Open-Teach camera streams and ROI crops,
SafeRobotExecutor (workspace / IK / per-step / XHand limits), XHand startup
reset, HDF5 rollout recording, and terminal scoring. FastWAM therefore runs
behind the same guards as the VPP robot-client. The loop below is
deploy_policy._run_hardware_loop with these differences:
  * the model cameras come from the server's /metadata (the checkpoint's
    shape_meta, e.g. ego + wrist); exo is subscribed only if the model uses it
    or for --record-h5;
  * actions are absolute targets by default; with --action-mode relative the
    server's delta dims are per-step increments accumulated from the state;
  * the model horizon (32 steps) is longer than a VPP chunk, so only the first
    --execute-steps targets are executed before re-planning.

Usage (shadow mode by default; Open-Teach provides the XHand driver and cameras):
  python experiments/xhand1/inference_xhand1_fastwam.py \
      --openteach-root /path/to/Open-Teach \
      --server http://127.0.0.1:8000 --prompt "stack the cups"
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
from scipy.spatial.transform import Rotation

import xhand1_hardware as hw

STATE_DIM = ACTION_DIM = 18
# FastWAM camera key (training shape_meta) -> hardware stream name.
CAMERA_STREAMS = {"exo": "exo", "ego": "static", "wrist": "wrist"}


def _read_json(request: Request, timeout_s: float) -> dict[str, object]:
    try:
        with urlopen(request, timeout=timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise RuntimeError(f"Policy server HTTP {exc.code}: {exc.read().decode(errors='replace')}") from exc
    except URLError as exc:
        raise RuntimeError(f"Could not reach policy server: {exc.reason}") from exc


def get_json(endpoint: str, timeout_s: float) -> dict[str, object]:
    return _read_json(Request(endpoint, method="GET"), timeout_s)


def post_json(endpoint: str, payload: dict[str, object], timeout_s: float) -> dict[str, object]:
    request = Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return _read_json(request, timeout_s)


class FastWAMClient:
    """Synchronous HTTP client for serve_fastwam_xhand1.py."""

    def __init__(self, server: str, prompt: str, timeout_s: float, action_mode: str) -> None:
        self.server = server.rstrip("/")
        self.prompt = prompt
        self.timeout_s = timeout_s
        self.action_mode = action_mode
        self.relative_dims = None
        self.image_keys = None
        self.run_id = None
        self.demo_id = "demo_1"
        self.chunk_id = 0
        self.policy_hz = None
        self.substeps = None
        self.action_horizon = None

    def check_contract(self, execution_hz: float, execute_steps: int) -> dict[str, object]:
        """Refuse to start unless the checkpoint matches what this client executes."""
        meta = get_json(self.server + "/metadata", self.timeout_s)
        problems = []
        if meta.get("state_dim") != STATE_DIM or meta.get("action_dim") != ACTION_DIM:
            problems.append(f"state/action dims {meta.get('state_dim')}/{meta.get('action_dim')} != 18/18")
        if meta.get("action_semantics") != self.action_mode:
            problems.append(f"action_semantics={meta.get('action_semantics')!r}, client --action-mode={self.action_mode!r}")
        relative_dims = [bool(x) for x in meta.get("relative_dims", [])]
        if len(relative_dims) != ACTION_DIM:
            problems.append(f"relative_dims has {len(relative_dims)} entries, expected {ACTION_DIM}")
        elif len(set(relative_dims[3:6])) != 1:
            problems.append("rotation dims 3:6 must be all relative or all absolute")
        image_keys = list(meta.get("image_keys", []))
        if not image_keys or not set(image_keys) <= set(CAMERA_STREAMS):
            problems.append(f"image_keys={image_keys}, expected a subset of {sorted(CAMERA_STREAMS)}")
        if int(meta.get("action_horizon", 0)) < execute_steps:
            problems.append(f"action_horizon={meta.get('action_horizon')} < --execute-steps={execute_steps}")
        policy_hz = float(meta.get("policy_hz", 0.0))
        substeps = execution_hz / policy_hz if policy_hz > 0 else 0.0
        if substeps < 1 or abs(substeps - round(substeps)) > 1e-6:
            problems.append(f"policy_hz={policy_hz:g} does not divide the {execution_hz:g} Hz executor")
        if problems:
            raise RuntimeError("FastWAM server contract mismatch: " + "; ".join(problems))
        self.policy_hz = policy_hz
        self.substeps = int(round(substeps))
        self.action_horizon = int(meta["action_horizon"])
        self.relative_dims = np.asarray(relative_dims, dtype=bool)
        self.image_keys = image_keys
        return meta

    def start_run(self, execute: bool) -> str:
        result = post_json(
            self.server + "/runs/start",
            {
                "prompt": self.prompt,
                "num_demos": 1,
                "execute": execute,
                "control_hz": self.policy_hz * self.substeps,
            },
            self.timeout_s,
        )
        self.run_id = str(result["run_id"])
        return self.run_id

    def predict_targets(self, packets: dict[str, dict], state: np.ndarray) -> np.ndarray:
        """Send the post-ROI JPEGs exactly as recorded; returns (horizon, 18) absolute targets."""
        self.chunk_id += 1
        payload = {
            "prompt": self.prompt,
            "state": np.asarray(state, dtype=float).tolist(),
            "images": {key: base64.b64encode(packets[key]["jpeg"]).decode("ascii") for key in self.image_keys},
            "run_id": self.run_id,
            "demo_id": self.demo_id,
            "chunk_id": self.chunk_id,
        }
        started = time.perf_counter()
        result = post_json(self.server + "/infer", payload, self.timeout_s)
        if result.get("action_semantics") != self.action_mode:
            raise RuntimeError(
                "Client/server action-mode mismatch: "
                f"client={self.action_mode!r}, server={result.get('action_semantics')!r}"
            )
        actions = np.asarray(result.get("actions"), dtype=float)
        if actions.ndim != 2 or actions.shape[1] != ACTION_DIM or not np.isfinite(actions).all():
            raise RuntimeError(f"Policy returned invalid actions: {actions.shape}")
        server_ms = result.get("server_timing", {}).get("infer_ms", float("nan"))
        print(
            f"[client] chunk {self.chunk_id}: {actions.shape[0]} targets, "
            f"round trip {(time.perf_counter() - started) * 1000.0:.0f} ms (server {server_ms:.0f} ms)"
        )
        return actions


class OpenTeachModelCameras(hw.OpenTeachCameraPair):
    """The vendored camera pair (+ optional exo) read by FastWAM camera key."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if "exo" in self.streams:
            self.last_read_source_ns["exo"] = None

    def read_model_packets(self, keys: list[str]) -> dict[str, dict]:
        """Fresh post-ROI frames for the given model cameras, keyed by FastWAM camera key."""
        packets = {}
        for key in keys:
            name = CAMERA_STREAMS[key]
            if name not in self.streams:
                raise hw.SafetyViolation(f"model camera {key} ({name}) was not opened")
            packet = self.streams[name].wait_for_new(self.last_read_source_ns[name], self.timeout_ms)
            self._validate_age(name, packet)
            self.last_read_source_ns[name] = packet["host_monotonic_ns"]
            packets[key] = self.streams[name].prepare(packet)
        return packets


def to_absolute_targets(state: np.ndarray, actions: np.ndarray, relative_dims: np.ndarray) -> np.ndarray:
    """Accumulate per-step increments on the relative dims, starting from the measured state.

    FastWAM's delta_action_dim_mask pads delta dims with 0 (= hold), i.e. each
    delta is one step's increment, as the r1lite client adds them to live
    feedback. Rotation increments compose on the right (tool frame), as
    vpp_real's VPPPolicy does for this robot. All-absolute dims pass through.
    """
    actions = np.asarray(actions, dtype=float)
    if not relative_dims.any():
        return actions.copy()
    targets = np.empty_like(actions)
    previous = np.asarray(state, dtype=float)
    for index, action in enumerate(actions):
        target = np.where(relative_dims, previous + action, action)
        if relative_dims[3:6].all():
            target[3:6] = (Rotation.from_rotvec(previous[3:6]) * Rotation.from_rotvec(action[3:6])).as_rotvec()
        targets[index] = target
        previous = target
    return targets


def validate_targets(targets: np.ndarray) -> np.ndarray:
    """deploy_policy._validate_policy_targets without its fixed VPP chunk shape."""
    targets = np.asarray(targets, dtype=float).copy()
    if targets.ndim != 2 or targets.shape[1] != hw.EE_DIM + hw.HAND_DIM:
        raise hw.SafetyViolation(f"policy targets: expected (N, 18), got {targets.shape}")
    if not np.all(np.isfinite(targets)):
        bad = np.argwhere(~np.isfinite(targets)).tolist()
        raise hw.SafetyViolation(f"policy targets contain non-finite values at {bad}")
    for index, target in enumerate(targets):
        pose_mm = target[:hw.EE_DIM].copy()
        pose_mm[:3] *= 1000.0
        if not hw._inside_workspace(pose_mm, hw.WORKSPACE_MM):
            raise hw.SafetyViolation(
                f"policy target {index} xArm pose outside workspace: {pose_mm[:3].tolist()}mm")
        target[hw.EE_DIM:] = hw._clip_xhand_command(target[hw.EE_DIM:], f"policy target {index}")
    return targets


def annotate_recording(recorder, args: argparse.Namespace, client: FastWAMClient) -> None:
    """Overwrite the VPP chunk metadata InferenceRecorder writes at creation."""
    attrs = recorder.demo.attrs
    attrs["policy_family"] = "fastwam"
    attrs["action_semantics"] = client.action_mode
    attrs["fastwam_server"] = client.server
    attrs["fastwam_run_id"] = client.run_id
    attrs["model_cameras"] = list(client.image_keys)
    attrs["model_action_horizon"] = client.action_horizon
    attrs["policy_hz"] = client.policy_hz
    attrs["chunk_size"] = args.execute_steps
    attrs["commands_per_chunk"] = args.execute_steps * client.substeps
    attrs["camera_image_note"] = (
        "obs/cam0_rgb_jpeg (ego), obs/cam1_rgb_jpeg (wrist), and obs/cam2_rgb_jpeg (exo) store "
        f"post-ROI 480x480 JPEGs sampled on the fixed-rate 30Hz timeline; FastWAM inputs: {client.image_keys}"
    )
    recorder.file.flush()


def run_hardware_loop(args: argparse.Namespace, client: FastWAMClient, preopened_hand=None):
    """deploy_policy._run_hardware_loop with server-chosen model cameras and receding-horizon execution."""
    cameras = None
    executor = None
    pending_hand = preopened_hand
    recorder = None
    chunks = 0
    failure = None
    task_score = None
    discard_requested = False
    termination_reason = None
    command_hz = client.policy_hz * client.substeps
    try:
        if args.record_h5:
            recorder = hw.InferenceRecorder(args.record_h5, args)
            annotate_recording(recorder, args, client)
        cameras = OpenTeachModelCameras(
            args.camera_host, args.static_camera_port, args.wrist_camera_port,
            args.camera_timeout_ms, args.camera_max_age_ms, args.camera_rois,
            exo_port=(
                args.exo_camera_port
                if "exo" in client.image_keys or recorder is not None else None))
        # Camera serial/freshness passes before the xArm SDK is opened/configured.
        cameras.read_model_packets(client.image_keys)
        if recorder is not None:
            cameras.wait_for_recording_packets()
        hand_for_executor = pending_hand
        pending_hand = None
        executor = hw.SafeRobotExecutor(args, preopened_hand=hand_for_executor)
        with hw.TerminalTaskScoreReader() as score_reader:
            def terminal_stop_requested():
                nonlocal task_score, discard_requested
                decision = score_reader.read_decision()
                if decision == hw.TerminalTaskScoreReader.DISCARD_DECISION:
                    discard_requested = True
                elif decision is not None:
                    task_score = decision
                return discard_requested or task_score is not None

            while args.max_chunks == 0 or chunks < args.max_chunks:
                if recorder is not None:
                    recorder.raise_if_fixed_rate_failed()
                if terminal_stop_requested():
                    termination_reason = "discard" if discard_requested else "human_score"
                    break
                packets = cameras.read_model_packets(client.image_keys)
                state = hw._validate_model_state(executor.read_state())
                step_monotonic_ns = time.monotonic_ns()
                if recorder is not None:
                    if not recorder.is_running:
                        recorder.start_fixed_rate(
                            cameras,
                            state=state,
                            state_monotonic_ns=step_monotonic_ns,
                            chunk_index=chunks,
                            arm_joints=executor.last_arm_joint_state,
                            hand_state_sample=executor.last_hand_state_sample,
                        )
                    else:
                        recorder.update_latest_robot_sample(
                            state=state,
                            arm_joints=executor.last_arm_joint_state,
                            state_source_ns=step_monotonic_ns,
                            state_available_ns=step_monotonic_ns,
                            policy_chunk_index=chunks,
                            command_index=-1,
                            hand_state_sample=executor.last_hand_state_sample,
                        )
                attempted_commands = 0
                predicted = client.predict_targets(packets, state)
                targets = validate_targets(
                    to_absolute_targets(state, predicted[:args.execute_steps], client.relative_dims))
                stream = hw.interpolate_targets(state, targets, client.substeps)
                t_next = time.monotonic()
                terminal_stop_requested()
                for command_index, cmd in enumerate(stream):
                    if discard_requested or task_score is not None:
                        break
                    t_next += 1.0 / command_hz
                    time.sleep(max(0.0, t_next - time.monotonic()))
                    if terminal_stop_requested():
                        break
                    attempted_commands += 1
                    command_start = time.monotonic_ns()
                    result = executor.send(cmd)
                    command_end = time.monotonic_ns()
                    if recorder is not None:
                        recorder.update_latest_robot_sample(
                            state=result["measured_state"],
                            arm_joints=result["arm_joint_state"],
                            executor_target=result["effective_target"],
                            state_source_ns=command_start,
                            state_available_ns=command_end,
                            executor_target_ns=command_end,
                            policy_chunk_index=chunks,
                            command_index=command_index,
                            hand_state_sample=executor.last_hand_state_sample,
                        )
                        recorder.raise_if_fixed_rate_failed()
                    terminal_stop_requested()
                if attempted_commands == len(stream):
                    chunks += 1
                    print(f"[robot] completed chunk {chunks} ({args.execution})")
                if discard_requested or task_score is not None:
                    termination_reason = "discard" if discard_requested else "human_score"
                    break
            if discard_requested:
                print("[score] discard requested; ending rollout without saving it")
            elif task_score is not None:
                print(f"[score] task score {task_score}; ending rollout")
            elif termination_reason is None:
                termination_reason = "max_chunks"
    except BaseException as exc:
        failure = exc
        raise
    finally:
        record_stop_error = None
        if recorder is not None:
            try:
                recorder.stop_fixed_rate()
            except BaseException as exc:
                record_stop_error = exc
        if executor is not None:
            executor.close()
        elif pending_hand is not None:
            pending_hand.close()
        if cameras is not None:
            cameras.close()
        if recorder is not None:
            try:
                if discard_requested:
                    recorder.discard()
                else:
                    recorder.close(
                        failure if failure is not None else record_stop_error,
                        task_score,
                        termination_reason,
                    )
            except Exception as record_exc:
                if failure is None:
                    raise
                print(f"[recording] close failed: {record_exc}")
        if record_stop_error is not None and failure is None:
            raise record_stop_error
        print(f"[robot] stopped after {chunks} chunks")
    return task_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--server", required=True, help="FastWAM server base URL, e.g. http://127.0.0.1:8000")
    parser.add_argument("--prompt", required=True, help="Task text; must match a precomputed embedding")
    parser.add_argument("--execute-steps", type=int, default=8,
                        help="Targets executed per inference before re-planning (policy steps)")
    parser.add_argument("--request-timeout-s", type=float, default=30.0)
    parser.add_argument(
        "--action-mode",
        choices=("relative", "absolute"),
        default="absolute",
        help=(
            "How to interpret model outputs; must match the server's action_semantics. "
            "absolute executes them as targets; relative accumulates the server's delta dims onto the state."
        ),
    )

    # Hardware options: same names and defaults as deploy_policy.py, which the vendored layer reads from args.
    parser.add_argument("--execution", choices=["shadow", "controller-sim", "physical"], default="shadow")
    parser.add_argument("--print-safety-only", action="store_true")
    parser.add_argument("--max-chunks", type=int, default=0, help="0 = run until interrupted")
    parser.add_argument("--openteach-root", default=os.environ.get("OPENTEACH_ROOT"))
    parser.add_argument("--camera-rois", default=hw.CAMERA_ROIS_PATH)
    parser.add_argument("--robot-ip", default=os.environ.get("RIGHT_ARM_IP", "192.168.1.198"))
    parser.add_argument("--xhand-port", default="/dev/ttyUSB0")
    parser.add_argument("--xhand-baudrate", type=int, default=3_000_000)
    parser.add_argument("--xhand-device-id", type=int, default=0)
    parser.add_argument("--xhand-comm-type", choices=["RS485", "ETHERCAT"], default="RS485")
    parser.add_argument("--xhand-ifname", default="eth0")
    parser.add_argument("--camera-host", default="127.0.0.1")
    parser.add_argument("--static-camera-port", type=int, default=10005, help="ego camera")
    parser.add_argument("--wrist-camera-port", type=int, default=10006)
    parser.add_argument("--exo-camera-port", type=int, default=10007)
    parser.add_argument("--camera-timeout-ms", type=int, default=2000)
    parser.add_argument("--camera-max-age-ms", type=int, default=500)
    parser.add_argument("--record-h5", help="workstation HDF5 path; appends one demo_N per invocation")
    parser.add_argument("--record-task", help="task description stored in --record-h5; defaults to --prompt")
    parser.add_argument("--max-arm-step-mm", type=float, default=8.0)
    parser.add_argument("--max-arm-step-rad", type=float, default=0.05)
    parser.add_argument("--max-ik-joint-step-rad", type=float, default=0.35)
    parser.add_argument("--xarm-speed-mm-s", type=float, default=200.0)
    parser.add_argument("--xarm-accel-mm-s2", type=float, default=2000.0)
    args = parser.parse_args()

    if not args.print_safety_only and not args.openteach_root:
        parser.error("--openteach-root (or OPENTEACH_ROOT) is required")
    if args.max_chunks < 0:
        parser.error("--max-chunks must be >= 0")
    if args.execute_steps < 1:
        parser.error("--execute-steps must be >= 1")
    positive = (args.request_timeout_s, args.camera_timeout_ms, args.camera_max_age_ms,
                args.max_arm_step_mm, args.max_arm_step_rad, args.max_ik_joint_step_rad,
                args.xarm_speed_mm_s, args.xarm_accel_mm_s2)
    if not all(np.isfinite(value) and value > 0 for value in positive):
        parser.error("timeout/camera/control safety parameters must be finite and positive")
    try:
        args.task_text = hw._validate_task_text(args.prompt)
        if args.record_task is not None:
            args.record_task = hw._validate_task_text(args.record_task)
    except ValueError as exc:
        parser.error(str(exc))
    # Fields the vendored safety banner and recorder read.
    args.mode = "robot-client"
    args.inference_endpoint = args.server
    args.inference_timeout_ms = int(args.request_timeout_s * 1000)
    return args


def main() -> None:
    args = parse_args()
    hw.print_safety_constraints(args)
    print(f"FastWAM execution            first {args.execute_steps} {args.action_mode} actions per inference, then re-plan")
    if args.print_safety_only:
        print("[safety] no network, camera, or hardware opened")
        return

    client = FastWAMClient(args.server, args.task_text, args.request_timeout_s, args.action_mode)
    # The server loads its model before binding, so this proves it is ready
    # before the workstation opens either hardware SDK.
    client.check_contract(hw.EXECUTION_HZ, args.execute_steps)
    print(
        f"[client] FastWAM server ready at {client.server}: cameras {client.image_keys}, horizon {client.action_horizon} @ "
        f"{client.policy_hz:g} Hz, executing {args.execute_steps} "
        f"({args.execute_steps / client.policy_hz:.2f} s) per chunk at {client.policy_hz * client.substeps:g} Hz"
    )
    client.start_run(execute=args.execution == "physical")
    preopened_hand = None
    try:
        preopened_hand = hw._reset_xhand_before_confirmation(args)
        if not hw._confirm_physical_execution(args):
            return
        hand_for_loop = preopened_hand
        preopened_hand = None
        run_hardware_loop(args, client, preopened_hand=hand_for_loop)
    finally:
        if preopened_hand is not None:
            preopened_hand.close()


if __name__ == "__main__":
    main()
