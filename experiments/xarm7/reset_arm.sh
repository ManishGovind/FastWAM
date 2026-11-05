#!/usr/bin/env bash
# Move the xArm7 to the rest pose and open the gripper.
# Dry run by default: connects read-only and prints what would happen.
# Pass --execute to actually clear errors, move, and open the gripper.
#
# Target (default: the generic rest joints below):
#   --task NAME [--demo N]   first-frame joints of data/NAME demo N (1-based, default 1);
#                            without --demo, the mean start joints of all demos of NAME
#   --joints a,b,c,d,e,f,g   explicit joint angles in degrees
#   --speed DEG_PER_S        joint speed, default 20 (allowed 1-50)
#
#   ./reset_arm.sh                                   # dry run, default rest joints
#   ./reset_arm.sh --task stack_cubes --demo 1       # dry run, match stack_cubes demo 1
#   ./reset_arm.sh --task stack_cubes --demo 1 --execute   # move the arm
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXECUTE=0
TASK=""
DEMO=""
JOINTS=""
SPEED="20"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --execute) EXECUTE=1; shift ;;
    --task) TASK="${2:?--task needs a name}"; shift 2 ;;
    --demo) DEMO="${2:?--demo needs a number}"; shift 2 ;;
    --joints) JOINTS="${2:?--joints needs 7 comma-separated degrees}"; shift 2 ;;
    --speed) SPEED="${2:?--speed needs deg/s}"; shift 2 ;;
    -h|--help) sed -n '2,16p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
if [[ -n "$JOINTS" && ( -n "$TASK" || -n "$DEMO" ) ]]; then
  echo "use either --joints or --task/--demo, not both" >&2; exit 2
fi
if [[ -n "$DEMO" && -z "$TASK" ]]; then echo "--demo requires --task" >&2; exit 2; fi

CONDA_ROOT="${CONDA_ROOT:-/home/skapse/workspace/miniconda3}"
PYTHON_BIN="${CONDA_ROOT}/envs/fastwam_xarm7/bin/python"
ROBOT_IP="${ROBOT_IP:-192.168.1.198}"

exec "${PYTHON_BIN}" - "$ROBOT_IP" "$EXECUTE" "$SCRIPT_DIR" "$TASK" "$DEMO" "$JOINTS" "$SPEED" <<'PY'
import glob
import os
import sys
import time

import numpy as np

from xarm.wrapper import XArmAPI

GRIPPER_OPEN_POS = 850
TCP_OFFSET = [0, 0, 0, 0, 0, 0]
# Same as HOME_POSE_MM_DEG in xarm_data_collection.py (x, y, z mm; roll, pitch, yaw deg).
REST_POSE = (300, 0, 300, 180, 0, 0)
# Joint-space rest (deg, joint_1..7): mean of EXPERIMENT_STARTING_JOINT_POSITIONS in
# xarm_fastwam_inference.py. Cartesian set_position() can fail with C19 (IK) from
# arbitrary poses and picks a history-dependent elbow solution; a joint move does neither.
REST_JOINTS_DEG = [1.345, -28.24, -2.263, 23.495, -0.512, 51.732, 0.61]


def require_ok(name: str, result):
    code = result[0] if isinstance(result, tuple) else result
    if code != 0:
        raise RuntimeError(f"{name} failed with code={code}")
    return result


robot_ip = sys.argv[1]
execute = sys.argv[2] == "1"
script_dir, task, demo_arg, joints_arg = sys.argv[3:7]
speed = float(sys.argv[7])
if not 1 <= speed <= 50:
    raise SystemExit(f"--speed must be between 1 and 50 deg/s, got {speed}")


def resolve_target():
    """Returns (joint list in deg, description)."""
    if joints_arg:
        joints = [float(v) for v in joints_arg.split(",")]
        if len(joints) != 7:
            raise SystemExit(f"--joints needs 7 values, got {len(joints)}")
        return joints, "explicit --joints"
    if not task:
        return list(REST_JOINTS_DEG), "default rest joints"
    import h5py

    files = sorted(glob.glob(os.path.join(script_dir, "data", task, "*", "*.hdf5")))
    if not files:
        raise SystemExit(f"no HDF5 sessions under {script_dir}/data/{task}")
    starts = []
    for path in files:
        with h5py.File(path, "r") as f:
            for name in sorted(f["data"], key=lambda n: int(n.split("_")[1])):
                starts.append(np.asarray(f["data"][name]["joint_states"][0], dtype=float)[:7])
    if demo_arg:
        n = int(demo_arg)
        if not 1 <= n <= len(starts):
            raise SystemExit(f"--demo must be in [1, {len(starts)}] for {task}")
        return starts[n - 1].round(3).tolist(), f"{task} demo {n} first-frame joints"
    return np.mean(starts, axis=0).round(3).tolist(), f"{task} mean start joints ({len(starts)} demos)"


target_joints, target_desc = resolve_target()
mode = "EXECUTE" if execute else "DRY RUN (no motion; pass --execute to move)"
print(f"[reset] {mode}")
print(f"[reset] connecting to xArm at {robot_ip}...")
arm = XArmAPI(robot_ip)
time.sleep(0.5)

try:
    # Read-only state, shown in both modes.
    code, err_warn = arm.get_err_warn_code()
    require_ok("get_err_warn_code", code)
    error_code = err_warn[0] if err_warn and len(err_warn) > 0 else 0
    warn_code = err_warn[1] if err_warn and len(err_warn) > 1 else 0
    pose = require_ok("get_position", arm.get_position(is_radian=False))[1]
    gripper = arm.get_gripper_position()[1]
    print(f"[reset] current pose (mm/deg): {[round(v, 1) for v in pose]}")
    print(f"[reset] current gripper position: {gripper}")
    current_joints = np.asarray(arm.get_joint_states(is_radian=False)[1][0][:7], dtype=float)
    current_joints = current_joints.tolist()
    delta = (np.asarray(target_joints) - np.asarray(current_joints)).tolist()
    print(f"[reset] target: {target_desc}")
    print(f"[reset] target joints (deg):  {[round(v, 3) for v in target_joints]}")
    print(f"[reset] current joints (deg): {[round(v, 3) for v in current_joints]}")
    print(f"[reset] delta (deg):          {[round(v, 3) for v in delta]}  max|delta|={max(abs(d) for d in delta):.2f}  speed={speed:g} deg/s")
    if max(abs(d) for d in delta) > 45:
        print("[reset] WARNING: large joint move (>45 deg). The joint-space path is not collision-checked; "
              "make sure the workspace is clear and keep the e-stop in reach.")
    print(f"[reset] error code={error_code} warn code={warn_code}")

    if not execute:
        if warn_code:
            print(f"[reset] would clear warning {warn_code}")
        if error_code:
            print(f"[reset] would clear controller error {error_code}")
        print(f"[reset] would move to {target_joints} at {speed:g} deg/s")
        print(f"[reset] would open gripper to {GRIPPER_OPEN_POS}")
        print("[reset] dry run only; nothing was changed. Re-run with --execute to move.")
        sys.exit(0)

    arm.set_tcp_offset(TCP_OFFSET)
    require_ok("motion_enable", arm.motion_enable(True))
    require_ok("set_mode", arm.set_mode(0))
    require_ok("set_state", arm.set_state(0))

    if warn_code:
        print(f"[reset] clearing warning {warn_code}...")
        require_ok("clean_warn", arm.clean_warn())

    if error_code:
        print(f"[reset] clearing controller error {error_code}...")
        require_ok("clean_error", arm.clean_error())
        require_ok("motion_enable", arm.motion_enable(enable=True, servo_id=8))
        require_ok("set_state", arm.set_state(0))

    print(f"[reset] moving to {target_joints}...")
    require_ok(
        "set_servo_angle",
        arm.set_servo_angle(angle=target_joints, is_radian=False, speed=speed, wait=True),
    )

    print(f"[reset] opening gripper to {GRIPPER_OPEN_POS}...")
    require_ok("set_gripper_mode", arm.set_gripper_mode(0))
    require_ok("set_gripper_enable", arm.set_gripper_enable(True))
    require_ok("set_gripper_speed", arm.set_gripper_speed(5000))
    require_ok("set_gripper_position", arm.set_gripper_position(GRIPPER_OPEN_POS, wait=True))

    print("[reset] arm is at the rest pose.")
finally:
    arm.disconnect()
PY
