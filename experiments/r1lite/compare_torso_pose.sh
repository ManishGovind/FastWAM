#!/usr/bin/env bash
# Read-only comparison of live R1 Lite torso feedback to the saved start pose.
# This script never creates a publisher and never commands the robot.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
POSITION_FILE="${R1LITE_INITIAL_POSITION_FILE:-${SCRIPT_DIR}/initial_robot_position.json}"
TOLERANCE_RAD="${1:-0.02}"

[[ -r "${POSITION_FILE}" ]] || { echo "Saved pose is missing: ${POSITION_FILE}" >&2; exit 1; }

set +u
export FASTRTPS_DEFAULT_PROFILES_FILE="${FASTRTPS_DEFAULT_PROFILES_FILE:-/opt/galaxea/find_server/super_client_configuration_file.xml}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-61}"
source /opt/ros/humble/setup.bash
source "${HOME}/galaxea/install/setup.bash"
set -u

exec /usr/bin/python3 - "${POSITION_FILE}" "${TOLERANCE_RAD}" <<'PY'
import json
import math
import sys
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

position_file, tolerance = sys.argv[1], float(sys.argv[2])
with open(position_file, encoding="utf-8") as handle:
    target = [float(value) for value in json.load(handle)["position"][14:18]]
if len(target) != 4 or not all(math.isfinite(value) for value in target):
    raise RuntimeError(f"Expected four finite saved torso values in {position_file}")

class TorsoFeedback(Node):
    def __init__(self):
        super().__init__("r1lite_torso_pose_compare")
        self.position = None
        self.create_subscription(JointState, "/hdas/feedback_torso", self.callback, 10)

    def callback(self, message):
        if len(message.position) >= 4:
            self.position = [float(value) for value in message.position[:4]]

rclpy.init()
node = TorsoFeedback()
try:
    deadline = time.monotonic() + 5.0
    while node.position is None and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
    if node.position is None:
        raise RuntimeError("Timed out waiting for /hdas/feedback_torso")
    error = [saved - live for saved, live in zip(target, node.position)]
    max_error = max(abs(value) for value in error)
    print("Live torso:  ", [round(value, 5) for value in node.position])
    print("Saved torso: ", [round(value, 5) for value in target])
    print("Saved - live:", [round(value, 5) for value in error])
    print(f"Max error: {max_error:.5f} rad; tolerance: {tolerance:.5f} rad")
    print("MATCH" if max_error <= tolerance else "NOT MATCHED")
finally:
    node.destroy_node()
    rclpy.shutdown()
PY
