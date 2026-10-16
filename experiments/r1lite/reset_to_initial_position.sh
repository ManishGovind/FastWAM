#!/usr/bin/env bash
# Move R1 Lite to the position captured in initial_robot_position.json.
#
# Default behavior is dry-run. Add --execute only after checking the reported
# current and target positions with the workcell clear. This script publishes
# targets; the corresponding arm, gripper, and torso controllers must be active.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export R1LITE_INITIAL_POSITION_FILE="${R1LITE_INITIAL_POSITION_FILE:-${SCRIPT_DIR}/initial_robot_position.json}"

set +u  # ROS setup scripts reference variables that may not be set yet.
export FASTRTPS_DEFAULT_PROFILES_FILE=/opt/galaxea/find_server/super_client_configuration_file.xml
export ROS_DOMAIN_ID=61
source /opt/ros/humble/setup.bash
source ~/galaxea/install/setup.bash
set -u

exec /usr/bin/python3 - "$@" <<'PY'
import argparse
import json
import math
import os
import signal
import time

import rclpy
from hdas_msg.msg import MotorControl
from rclpy.node import Node
from sensor_msgs.msg import JointState


INITIAL_POSITION_FILE = os.environ["R1LITE_INITIAL_POSITION_FILE"]
with open(INITIAL_POSITION_FILE, "r", encoding="utf-8") as position_file:
    TARGET = json.load(position_file)["position"]
if len(TARGET) != 18 or not all(isinstance(value, (int, float)) and math.isfinite(value) for value in TARGET):
    raise RuntimeError(f"Expected 18 finite positions in {INITIAL_POSITION_FILE}")
TARGET = [float(value) for value in TARGET]

ARM_FEEDBACK = {
    "left": "/hdas/feedback_arm_left",
    "right": "/hdas/feedback_arm_right",
}
GRIPPER_FEEDBACK = {
    "left": "/hdas/feedback_gripper_left",
    "right": "/hdas/feedback_gripper_right",
}
TORSO_FEEDBACK = "/hdas/feedback_torso"
ARM_TARGET = {
    "left": "/motion_target/target_joint_state_arm_left",
    "right": "/motion_target/target_joint_state_arm_right",
}
GRIPPER_TARGET = {
    "left": "/motion_target/target_position_gripper_left",
    "right": "/motion_target/target_position_gripper_right",
}
TORSO_CONTROL = "/motion_control/control_torso"

# Galaxea's torso interface accepts MotorControl directly; unlike the arms,
# there is no running JointState tracker on a /motion_target torso topic.
# These gains match the SDK's R1 torso position-control example.
TORSO_KP = [140.0, 200.0, 120.0, 20.0]
TORSO_KD = [10.0, 50.0, 5.0, 1.0]


class ResetNode(Node):
    def __init__(self):
        super().__init__("r1lite_initial_position_reset")
        self.arm = {"left": None, "right": None}
        self.arm_names = {"left": None, "right": None}
        self.gripper = {"left": None, "right": None}
        self.torso = None
        self.torso_names = None
        self.last_update = {
            "left_arm": 0.0,
            "right_arm": 0.0,
            "left_gripper": 0.0,
            "right_gripper": 0.0,
            "torso": 0.0,
        }

        for side, topic in ARM_FEEDBACK.items():
            self.create_subscription(JointState, topic, self._arm_callback(side), 10)
        for side, topic in GRIPPER_FEEDBACK.items():
            self.create_subscription(JointState, topic, self._gripper_callback(side), 10)
        self.create_subscription(JointState, TORSO_FEEDBACK, self._torso_callback, 10)

        self.arm_pub = {
            side: self.create_publisher(JointState, topic, 10)
            for side, topic in ARM_TARGET.items()
        }
        self.gripper_pub = {
            side: self.create_publisher(JointState, topic, 10)
            for side, topic in GRIPPER_TARGET.items()
        }
        self.torso_pub = self.create_publisher(MotorControl, TORSO_CONTROL, 10)

    def _arm_callback(self, side):
        def callback(msg):
            if len(msg.position) >= 6:
                self.arm[side] = [float(value) for value in msg.position[:6]]
                self.arm_names[side] = list(msg.name[:6]) if len(msg.name) >= 6 else None
                self.last_update[f"{side}_arm"] = time.monotonic()
        return callback

    def _gripper_callback(self, side):
        def callback(msg):
            if msg.position:
                self.gripper[side] = float(msg.position[0])
                self.last_update[f"{side}_gripper"] = time.monotonic()
        return callback

    def _torso_callback(self, msg):
        if len(msg.position) >= 4:
            self.torso = [float(value) for value in msg.position[:4]]
            self.torso_names = list(msg.name[:4]) if len(msg.name) >= 4 else None
            self.last_update["torso"] = time.monotonic()

    def ready(self):
        return (
            all(value is not None for value in self.arm.values())
            and all(value is not None for value in self.gripper.values())
            and self.torso is not None
        )

    def feedback_is_fresh(self):
        now = time.monotonic()
        return all(now - stamp <= 0.5 for stamp in self.last_update.values())

    def current_state(self):
        return (
            self.arm["left"]
            + self.arm["right"]
            + [self.gripper["left"], self.gripper["right"]]
            + self.torso
        )

    def publish(self, state):
        stamp = self.get_clock().now().to_msg()
        for index, side in enumerate(("left", "right")):
            base = index * 6
            arm_msg = JointState()
            arm_msg.header.stamp = stamp
            if self.arm_names[side]:
                arm_msg.name = self.arm_names[side]
            arm_msg.position = state[base:base + 6]

            gripper_msg = JointState()
            gripper_msg.header.stamp = stamp
            gripper_msg.position = [state[12 + index]]

            self.arm_pub[side].publish(arm_msg)
            self.gripper_pub[side].publish(gripper_msg)

        torso_msg = MotorControl()
        torso_msg.header.stamp = stamp
        torso_msg.name = "torso"
        torso_msg.p_des = state[14:18]
        torso_msg.v_des = [0.0, 0.0, 0.0, 0.0]
        torso_msg.kp = TORSO_KP
        torso_msg.kd = TORSO_KD
        torso_msg.t_ff = [0.0, 0.0, 0.0, 0.0]
        self.torso_pub.publish(torso_msg)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="publish the reset trajectory")
    parser.add_argument("--duration", type=float, default=10.0, help="trajectory duration in seconds")
    parser.add_argument(
        "--hold-final",
        action="store_true",
        help="keep publishing the final target until interrupted (for controller handoff)",
    )
    parser.add_argument(
        "--wait-for-takeover",
        action="store_true",
        help="wait for SIGUSR1 before taking control (for gap-free handoff)",
    )
    parser.add_argument(
        "--start-arm-gripper",
        type=float,
        nargs=14,
        metavar="VALUE",
        help="exact 14-D arm/gripper hold command to use at takeover",
    )
    args = parser.parse_args()
    if args.duration <= 0:
        parser.error("--duration must be positive")
    if args.wait_for_takeover and not args.execute:
        parser.error("--wait-for-takeover requires --execute")
    if args.start_arm_gripper is not None and not all(
        math.isfinite(value) for value in args.start_arm_gripper
    ):
        parser.error("--start-arm-gripper values must be finite")

    takeover_requested = False

    def request_takeover(_signal_number, _frame):
        nonlocal takeover_requested
        takeover_requested = True

    if args.wait_for_takeover:
        signal.signal(signal.SIGUSR1, request_takeover)

    rclpy.init()
    node = ResetNode()
    try:
        deadline = time.monotonic() + 5.0
        while not node.ready() and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
        if not node.ready():
            raise RuntimeError("Timed out waiting for arm, gripper, and torso feedback")
        if not node.feedback_is_fresh():
            raise RuntimeError("Robot feedback is stale; refusing reset")

        start = node.current_state()
        arm_delta = max(abs(target - current) for target, current in zip(TARGET[:12], start[:12]))
        torso_delta = max(abs(target - current) for target, current in zip(TARGET[14:], start[14:]))
        print("Current state:", [round(value, 5) for value in start])
        print("Initial-position target:", [round(value, 5) for value in TARGET])
        print("Initial-position source:", INITIAL_POSITION_FILE)
        print(f"Maximum arm travel: {arm_delta:.3f} rad")
        print(f"Maximum torso travel: {torso_delta:.3f} rad")

        if not args.execute:
            print("DRY RUN: no targets published. Re-run with --execute to move over the requested trajectory.")
            return

        if args.wait_for_takeover:
            print("RESET_READY", flush=True)
            while not takeover_requested:
                rclpy.spin_once(node, timeout_sec=0.05)
                if not node.feedback_is_fresh():
                    raise RuntimeError("Robot feedback became stale while waiting for takeover")

            # Use the exact arm/gripper command held by inference, rather than
            # independently sampled feedback that may differ by one ROS cycle.
            # Torso is not controlled by inference, so use its live feedback.
            takeover_feedback = node.current_state()
            if args.start_arm_gripper is not None:
                start = list(args.start_arm_gripper) + takeover_feedback[14:18]
            else:
                start = takeover_feedback
            node.publish(start)
            print("RESET_TOOK_OVER", flush=True)
            print("Reset takeover pose:", [round(value, 5) for value in start])

        rate_hz = 20.0
        steps = max(1, math.ceil(args.duration * rate_hz))
        print(f"Executing {args.duration:.1f}s reset trajectory at {rate_hz:.0f} Hz.")
        for step in range(1, steps + 1):
            rclpy.spin_once(node, timeout_sec=0.0)
            if not node.feedback_is_fresh():
                raise RuntimeError("Robot feedback became stale; stopping reset")
            # Quintic minimum-jerk blend: zero velocity and acceleration at
            # both ends, avoiding the step in commanded velocity produced by
            # plain linear interpolation during reset takeover.
            phase = step / steps
            alpha = phase * phase * phase * (10.0 + phase * (-15.0 + 6.0 * phase))
            state = [current + alpha * (target - current) for current, target in zip(start, TARGET)]
            node.publish(state)
            time.sleep(1.0 / rate_hz)
        print("Reset trajectory complete.")
        if args.hold_final:
            print("RESET_HOLDING_FINAL", flush=True)
            try:
                while rclpy.ok():
                    rclpy.spin_once(node, timeout_sec=0.0)
                    if not node.feedback_is_fresh():
                        raise RuntimeError("Robot feedback became stale while holding final pose")
                    node.publish(TARGET)
                    time.sleep(1.0 / rate_hz)
            except KeyboardInterrupt:
                print("Final-pose hold handed off.", flush=True)
    finally:
        node.destroy_node()
        # SIGINT may already have shut down the default rclpy context during
        # the gap-free reset-to-teleop handoff.
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
PY
