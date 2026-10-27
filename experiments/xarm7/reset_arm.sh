#!/usr/bin/env bash
# Move the xArm7 to the rest pose and open the gripper.
# Dry run by default: connects read-only and prints what would happen.
# Pass --execute to actually clear errors, move, and open the gripper.
#
#   ./reset_arm.sh              # dry run
#   ./reset_arm.sh --execute    # move the arm
set -euo pipefail

EXECUTE=0
for arg in "$@"; do
  case "$arg" in
    --execute) EXECUTE=1 ;;
    -h|--help) sed -n '2,7p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown argument: $arg (only --execute is supported)" >&2; exit 2 ;;
  esac
done

CONDA_ROOT="${CONDA_ROOT:-/home/skapse/workspace/miniconda3}"
PYTHON_BIN="${CONDA_ROOT}/envs/fastwam_xarm7/bin/python"
ROBOT_IP="${ROBOT_IP:-192.168.1.198}"

exec "${PYTHON_BIN}" - "$ROBOT_IP" "$EXECUTE" <<'PY'
import sys
import time

from xarm.wrapper import XArmAPI

GRIPPER_OPEN_POS = 850
TCP_OFFSET = [0, 0, 0, 0, 0, 0]
# Same as HOME_POSE_MM_DEG in xarm_data_collection.py (x, y, z mm; roll, pitch, yaw deg).
REST_POSE = (300, 0, 300, 180, 0, 0)


def require_ok(name: str, result):
    code = result[0] if isinstance(result, tuple) else result
    if code != 0:
        raise RuntimeError(f"{name} failed with code={code}")
    return result


robot_ip = sys.argv[1]
execute = sys.argv[2] == "1"
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
    print(f"[reset] error code={error_code} warn code={warn_code}")

    if not execute:
        if warn_code:
            print(f"[reset] would clear warning {warn_code}")
        if error_code:
            print(f"[reset] would clear controller error {error_code}")
        print(f"[reset] would move to rest pose {REST_POSE}")
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

    print(f"[reset] moving to rest pose {REST_POSE}...")
    require_ok("set_position", arm.set_position(*REST_POSE, wait=True))

    print(f"[reset] opening gripper to {GRIPPER_OPEN_POS}...")
    require_ok("set_gripper_mode", arm.set_gripper_mode(0))
    require_ok("set_gripper_enable", arm.set_gripper_enable(True))
    require_ok("set_gripper_speed", arm.set_gripper_speed(5000))
    require_ok("set_gripper_position", arm.set_gripper_position(GRIPPER_OPEN_POS, wait=True))

    print("[reset] arm is at the rest pose.")
finally:
    arm.disconnect()
PY
