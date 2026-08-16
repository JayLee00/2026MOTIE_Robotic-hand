#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Move the KISTAR hand fingers GRADUALLY to an absolute 16-joint target pose,
then exit (leaving the target/servo as-is so a following node can continue).

Same receiver contract + rad->encoder-count encoding as
hand_joint_target_publisher.py (cube_rotate.cpp scaling with per-joint
corrections), so a --target given in radians lands the SAME physical pose as
the HDF5 trajectory frames. The move is a soft ramp from the CURRENT measured
pose (/hand/{side}/joint_states, counts) to the encoded target over
--ramp-secs, so the fingers slew smoothly instead of snapping.

Receiver contract (robot PC):
  /hand/{side}/q_target   Float32MultiArray, EXACTLY 16 RAW ENCODER COUNTS,
                          BEST_EFFORT depth 1.
  /hand/{side}/cmd_servo  Bool, RELIABLE.
  /hand/{side}/cmd_mode   Int32 (1=position, 2=circular), RELIABLE.
  /hand/{side}/joint_states  measured counts (16), BEST_EFFORT.

Startup: cmd_mode=1 (skip with --no-mode) -> cmd_servo=True (skip with
--no-servo-on) -> soft ramp measured->target -> hold --hold-secs -> exit.
Does NOT servo-off on exit.

Requires the ROS2-matched interpreter (Humble -> /usr/bin/python3, 3.10):

    source /opt/ros/humble/setup.bash
    /usr/bin/python3 in-hand/hand_goto_target.py --side right
"""

import argparse
import signal
import sys
import time

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import (
    Bool,
    Float32MultiArray,
    Int32,
    MultiArrayDimension,
    MultiArrayLayout,
)

NUM_JOINTS = 16

# cube_rotate.cpp encoding constants (identical to hand_joint_target_publisher.py).
SCALE_FACTOR = 4096.0
SCALE_FACTOR2 = 2.1
PI = 3.14159265358979323846
EXTRA_GAIN = 1.12
INT16_MIN, INT16_MAX = -32768, 32767

# Per-joint multipliers reproduced from cube_rotate.cpp (source order).
JOINT_CORRECTION = np.array(
    [
        1.0, 1.36, 0.9, 1.0,
        0.9, 1.0, 0.8, 0.8,
        1.12 * 0.75, 0.95, 0.92, 0.78,
        0.75, 1.0, 1.0, 1.0,
    ],
    dtype=np.float64,
)

# Default absolute target (radians), same convention as the HDF5 frames.
DEFAULT_TARGET = [
    1.575, -1.571, 0.162, 0.692, -0.298, 1.123, 0.732, 0.168,
    -0.183, 1.129, 0.321, 0.295, 0.526, 1.379, 0.662, 0.009,
]


def encode_ticks(rad16, apply_corrections=True, gain=1.0, invert=False):
    """radians[16] -> float32 encoder counts (cube_rotate encoding + clamp)."""
    data = np.asarray(rad16, dtype=np.float64)
    if apply_corrections:
        data = data * JOINT_CORRECTION
    data = data * gain
    ticks_f = data * SCALE_FACTOR * SCALE_FACTOR2 / PI * EXTRA_GAIN
    if invert:
        ticks_f = -ticks_f
    clamped = np.clip(ticks_f, INT16_MIN, INT16_MAX)
    return clamped.astype(np.int16).astype(np.float32)


class HandGotoTarget(Node):
    def __init__(self, args):
        super().__init__("hand_goto_target")

        if len(args.target) != NUM_JOINTS:
            raise ValueError(
                f"--target needs exactly {NUM_JOINTS} values, got {len(args.target)}"
            )
        self._target_ticks = encode_ticks(
            args.target, args.apply_corrections, args.gain, args.invert
        )
        self._rate_hz = float(args.rate_hz)
        self._hold_frames = max(1, int(round(args.hold_secs * self._rate_hz)))
        self.finished = False

        self._cmd_topic = f"/hand/{args.side}/q_target"
        servo_topic = f"/hand/{args.side}/cmd_servo"
        mode_topic = f"/hand/{args.side}/cmd_mode"
        state_topic = f"/hand/{args.side}/joint_states"

        q_qos = QoSProfile(
            depth=1, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST
        )
        reliable_qos = QoSProfile(
            depth=1, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST
        )

        self._pub = self.create_publisher(Float32MultiArray, self._cmd_topic, q_qos)
        self._servo_pub = self.create_publisher(Bool, servo_topic, reliable_qos)
        self._mode_pub = self.create_publisher(Int32, mode_topic, reliable_qos)

        self._measured_q = None
        self._state_sub = self.create_subscription(
            JointState, state_topic, self._on_state, q_qos
        )

        self._wait_for_subscriber(args.discovery_timeout)
        if args.mode is not None:
            self._send_mode(args.mode, repeat=3)
            self.get_logger().info(f"Sent cmd_mode = {args.mode} to '{mode_topic}'.")
            time.sleep(0.1)
        if args.servo_on:
            self._send_servo(True, repeat=5)
            self.get_logger().info(f"Sent servo-on (True) to '{servo_topic}'.")
            time.sleep(0.3)

        # Soft ramp: measured pose (counts) -> encoded target over --ramp-secs.
        self._ramp = self._build_ramp(state_topic, args.ramp_secs, args.rate_hz)
        self._ramp_frame = 0
        self._hold_count = 0
        self.get_logger().info(
            f"Target (counts): "
            f"{np.array2string(self._target_ticks.astype(int), max_line_width=200)}"
        )

        self._timer = self.create_timer(1.0 / self._rate_hz, self._on_timer)

    def _on_state(self, msg):
        if len(msg.position) >= NUM_JOINTS:
            self._measured_q = np.asarray(msg.position[:NUM_JOINTS], dtype=np.float32)

    def _build_ramp(self, state_topic, ramp_secs, rate_hz):
        """Interpolate measured pose (counts) -> target. Returns [M, 16] or None."""
        if ramp_secs <= 0:
            return None
        deadline = time.monotonic() + 3.0
        while self._measured_q is None and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
        if self._measured_q is None:
            self.get_logger().warn(
                f"No measured state on '{state_topic}'; going to target WITHOUT the "
                f"soft ramp (may cause a jump)."
            )
            return None
        start = self._measured_q.astype(np.float32)
        goal = self._target_ticks
        steps = max(2, int(round(ramp_secs * rate_hz)))
        alphas = np.linspace(0.0, 1.0, steps, dtype=np.float32)[:, None]
        ramp = start[None, :] * (1.0 - alphas) + goal[None, :] * alphas
        self.get_logger().info(
            f"Soft ramp: measured pose -> target over {ramp_secs:.1f}s "
            f"({steps} steps, max |delta| = {float(np.max(np.abs(goal - start))):.0f} counts)."
        )
        return np.ascontiguousarray(ramp.astype(np.float32))

    def _wait_for_subscriber(self, timeout_s):
        if timeout_s <= 0:
            return
        waited = 0.0
        while waited < timeout_s:
            if self._pub.get_subscription_count() > 0:
                self.get_logger().info(f"Subscriber detected on '{self._cmd_topic}'.")
                return
            time.sleep(0.1)
            waited += 0.1
        self.get_logger().warn(
            f"No subscriber on '{self._cmd_topic}' after {timeout_s:.1f}s; "
            f"publishing anyway (is hand_target_receiver running?)."
        )

    def _send_servo(self, value, repeat=1):
        msg = Bool()
        msg.data = bool(value)
        for _ in range(repeat):
            self._servo_pub.publish(msg)
            time.sleep(0.05)

    def _send_mode(self, value, repeat=1):
        msg = Int32()
        msg.data = int(value)
        for _ in range(repeat):
            self._mode_pub.publish(msg)
            time.sleep(0.05)

    def _make_msg(self, row):
        msg = Float32MultiArray()
        dim = MultiArrayDimension()
        dim.label = "joints"
        dim.size = NUM_JOINTS
        dim.stride = NUM_JOINTS
        msg.layout = MultiArrayLayout(dim=[dim], data_offset=0)
        msg.data = [float(v) for v in row]
        return msg

    def _on_timer(self):
        # Phase 1: soft ramp to the target.
        if self._ramp is not None and self._ramp_frame < self._ramp.shape[0]:
            self._pub.publish(self._make_msg(self._ramp[self._ramp_frame]))
            self._ramp_frame += 1
            if self._ramp_frame == self._ramp.shape[0]:
                self.get_logger().info("Reached target; holding.")
            return
        # Phase 2: hold the target for --hold-secs, then finish.
        self._pub.publish(self._make_msg(self._target_ticks))
        self._hold_count += 1
        if self._hold_count >= self._hold_frames:
            self.get_logger().info("Done; target left in place, servo untouched.")
            self._timer.cancel()
            self.finished = True


def parse_args(argv):
    p = argparse.ArgumentParser(
        description="Gradually move the KISTAR hand to an absolute 16-joint target "
        "(radians, cube_rotate encoding) via /hand/{side}/q_target, then exit."
    )
    p.add_argument("--side", choices=("left", "right"), default="right",
                   help="Which hand to command (default: right).")
    p.add_argument("--target", type=float, nargs=NUM_JOINTS, default=list(DEFAULT_TARGET),
                   help="16 absolute joint targets in radians (real/topic order).")
    p.add_argument("--ramp-secs", type=float, default=3.0,
                   help="Soft-ramp duration from the measured pose to the target "
                   "(default: 3.0; 0 = jump straight to target).")
    p.add_argument("--hold-secs", type=float, default=0.5,
                   help="Keep publishing the target this long after arriving (default: 0.5).")
    p.add_argument("--rate-hz", type=float, default=50.0,
                   help="Publish rate in Hz (default: 50).")
    p.add_argument("--gain", type=float, default=1.0,
                   help="Optional global scalar applied (in radians) before encoding.")
    p.add_argument("--mode", type=int, choices=(1, 2), default=1,
                   help="cmd_mode sent at startup (1=position; default: 1).")
    p.add_argument("--no-mode", dest="mode", action="store_const", const=None,
                   help="Do not send cmd_mode at startup.")
    p.add_argument("--no-corrections", dest="apply_corrections", action="store_false",
                   help="Skip cube_rotate's per-joint scaling (encode raw radians->counts).")
    p.add_argument("--invert", dest="invert", action="store_true",
                   help="Negate all commands before sending (default: off).")
    p.add_argument("--no-servo-on", dest="servo_on", action="store_false",
                   help="Do not auto-send cmd_servo = True at startup.")
    p.add_argument("--discovery-timeout", type=float, default=5.0,
                   help="Seconds to wait for the receiver to subscribe (default: 5).")
    p.set_defaults(apply_corrections=True, servo_on=True, invert=False)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv if argv is not None else sys.argv[1:])
    rclpy.init()
    signal.signal(signal.SIGINT, signal.default_int_handler)

    node = None
    try:
        node = HandGotoTarget(args)
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        if node is not None:
            node.get_logger().info("Interrupted by user (Ctrl+C).")
    except rclpy.executors.ExternalShutdownException:
        pass  # SIGTERM
    except Exception as exc:
        sys.stderr.write(f"Startup/run failed: {exc}\n")
        if node is None:
            if rclpy.ok():
                rclpy.shutdown()
            return 1
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
