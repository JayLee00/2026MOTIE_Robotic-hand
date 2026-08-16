#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Manual squeeze controller (ROS2) for the robot-PC KISTAR hand receiver.

Captures the CURRENT measured finger pose from /hand/{side}/joint_states as
the baseline, then lets you tighten/loosen the grip interactively: each '+'
adds a small positive offset (counts) to the flexion joints so the fingers
close slightly harder on the object; '-' releases.

Interface (same receiver contract as hand_joint_target_publisher_2.py,
verified against hand_target_receiver_node.cpp):
  /hand/{side}/q_target      Float32MultiArray, EXACTLY 16 values, RAW ENCODER
                             COUNTS (1 tick = pi/8192 rad), BEST_EFFORT depth 1.
  /hand/{side}/cmd_servo     Bool, RELIABLE, applied immediately.
  /hand/{side}/cmd_mode      Int32 (1=position, 2=circular), RELIABLE.
  /hand/{side}/joint_states  sensor_msgs/JointState, measured counts (16),
                             BEST_EFFORT. REQUIRED here: without a measured
                             baseline the node refuses to start.

Joint layout in real (topic) order, derived from data_loader.py S2R_INDICES:
  [ 0] idx MCP   [ 1] idx PIP   [ 2] idx DIP   [ 3] idx abduction
  [ 4] mid MCP   [ 5] mid PIP   [ 6] mid DIP   [ 7] mid abduction
  [ 8] rng MCP   [ 9] rng PIP   [10] rng DIP   [11] rng abduction
  [12] thm opposition  [13] thm flex2  [14] thm flex3  [15] thm abduction
The default squeeze mask moves PIP/DIP of index/middle/ring plus the thumb
distal flex joints (13/14) in the POSITIVE direction; finger MCP (0/4/8),
finger abduction (3/7/11), thumb opposition (12) and the thumb base joint
(15, drives thumb_link_1) stay untouched at their baseline. Override with
--joints.

The node streams the current target at --rate-hz continuously; goal changes
are followed with a speed limit (--speed counts/s) so each squeeze step lands
smoothly instead of snapping. The offset from baseline is clamped to
[--min-offset, --max-offset].

Commands (stdin):
  +  /  =          tighten by one step
  -                loosen by one step
  set <counts>     jump to an absolute offset from the baseline
  step <counts>    change the per-press step size
  reset            go back to the baseline pose (offset 0)
  rebase           re-capture the CURRENT measured pose as the new baseline
  show             print baseline / goal / measured
  servo on|off     publish cmd_servo
  quit / exit      stop (target stays where it is; servo untouched)

Requires the ROS2-matched interpreter (Humble -> /usr/bin/python3, 3.10):

    source /opt/ros/humble/setup.bash
    /usr/bin/python3 in-hand/hand_manual_squeeze.py --side right
"""

import argparse
import signal
import sys
import threading
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
INT16_MIN, INT16_MAX = -32768, 32767

# Real-order flexion mask (1 = joint participates in the squeeze).
# Indices per the layout table in the module docstring. For index/middle/ring
# only PIP (j2) and DIP (j3) squeeze — MCP (j1) and abduction (j0) hold their
# baseline. For the thumb only the two distal flex joints (13, 14) squeeze —
# opposition (12) and the base joint (15, moves thumb_link_1) hold their
# baseline.
DEFAULT_SQUEEZE_JOINTS = (1, 2, 5, 6, 9, 10, 13, 14)


class HandManualSqueeze(Node):
    def __init__(self, args):
        super().__init__("hand_manual_squeeze")

        self._step = float(args.step)
        self._min_offset = float(args.min_offset)
        self._max_offset = float(args.max_offset)
        self._speed = float(args.speed)  # counts/s
        self._rate_hz = float(args.rate_hz)

        mask = np.zeros(NUM_JOINTS, dtype=np.float32)
        for j in args.joints:
            if not 0 <= j < NUM_JOINTS:
                raise ValueError(f"--joints index {j} out of range 0..{NUM_JOINTS - 1}")
            mask[j] = 1.0
        self._mask = mask

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

        # Baseline = current measured pose. Refuse to run blind.
        baseline = self._capture_measured(state_topic, timeout_s=args.state_timeout)
        if baseline is None:
            raise RuntimeError(
                f"no measured pose on '{state_topic}' within {args.state_timeout:.1f}s "
                f"— cannot squeeze safely without a baseline (is the robot RT loop up?)"
            )

        # All state below is shared with the stdin thread; guard with a lock.
        self._lock = threading.Lock()
        self._baseline = baseline.astype(np.float32)
        self._offset_goal = 0.0    # commanded offset from baseline (counts)
        self._target = self._baseline.copy()  # what we publish right now

        self.get_logger().info(
            f"Baseline captured from '{state_topic}': "
            f"{np.array2string(self._baseline.astype(int), max_line_width=200)}"
        )
        self.get_logger().info(
            f"Squeeze joints: {[i for i in range(NUM_JOINTS) if mask[i]]} | "
            f"step={self._step:.0f} counts, offset range "
            f"[{self._min_offset:.0f}, {self._max_offset:.0f}], "
            f"speed={self._speed:.0f} counts/s"
        )

        self._timer = self.create_timer(1.0 / self._rate_hz, self._on_timer)

    # ------------------------------------------------------------------ ROS
    def _on_state(self, msg):
        if len(msg.position) >= NUM_JOINTS:
            self._measured_q = np.asarray(msg.position[:NUM_JOINTS], dtype=np.float32)

    def _capture_measured(self, state_topic, timeout_s):
        deadline = time.monotonic() + timeout_s
        while self._measured_q is None and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
        return None if self._measured_q is None else self._measured_q.copy()

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
        """Move the published target toward baseline + offset_goal * mask,
        limited to --speed counts per second, and publish it every tick."""
        with self._lock:
            goal = self._baseline + self._offset_goal * self._mask
            max_delta = self._speed / self._rate_hz
            err = goal - self._target
            self._target = self._target + np.clip(err, -max_delta, max_delta)
            np.clip(self._target, INT16_MIN, INT16_MAX, out=self._target)
            row = self._target.copy()
        self._pub.publish(self._make_msg(row))

    # ------------------------------------------------------ stdin commands
    def tighten(self, delta):
        with self._lock:
            self._offset_goal = float(
                np.clip(self._offset_goal + delta, self._min_offset, self._max_offset)
            )
            return self._offset_goal

    def set_offset(self, value):
        with self._lock:
            self._offset_goal = float(np.clip(value, self._min_offset, self._max_offset))
            return self._offset_goal

    def set_step(self, value):
        self._step = abs(float(value))
        return self._step

    @property
    def step(self):
        return self._step

    def rebase(self):
        if self._measured_q is None:
            return None
        with self._lock:
            self._baseline = self._measured_q.copy()
            self._offset_goal = 0.0
            return self._baseline.copy()

    def snapshot(self):
        with self._lock:
            return (
                self._baseline.copy(),
                self._offset_goal,
                self._target.copy(),
                None if self._measured_q is None else self._measured_q.copy(),
            )


def input_loop(node, stop_event):
    """Blocking stdin loop; runs in a daemon thread."""
    help_text = (
        "\ncommands: +|= tighten | - loosen | set <counts> | step <counts> | "
        "reset | rebase | show | servo on|off | quit\n"
    )
    print(help_text)
    while not stop_event.is_set():
        try:
            line = input(f"[squeeze step={node.step:.0f}] >> ").strip()
        except EOFError:
            stop_event.set()
            return
        if not line:
            continue
        tokens = line.split()
        cmd = tokens[0].lower()
        try:
            if cmd in ("quit", "exit", "q"):
                stop_event.set()
                return
            elif cmd in ("+", "="):
                off = node.tighten(+node.step)
                print(f"offset -> {off:+.0f} counts")
            elif cmd == "-":
                off = node.tighten(-node.step)
                print(f"offset -> {off:+.0f} counts")
            elif cmd == "set" and len(tokens) == 2:
                off = node.set_offset(float(tokens[1]))
                print(f"offset -> {off:+.0f} counts")
            elif cmd == "step" and len(tokens) == 2:
                print(f"step -> {node.set_step(float(tokens[1])):.0f} counts")
            elif cmd == "reset":
                node.set_offset(0.0)
                print("offset -> +0 (back to baseline)")
            elif cmd == "rebase":
                base = node.rebase()
                if base is None:
                    print("no measured pose available; rebase skipped")
                else:
                    print(f"baseline <- measured: {base.astype(int).tolist()}")
            elif cmd == "show":
                base, off, target, meas = node.snapshot()
                print(f"baseline: {base.astype(int).tolist()}")
                print(f"offset  : {off:+.0f} counts")
                print(f"target  : {target.astype(int).tolist()}")
                print(f"measured: {'n/a' if meas is None else meas.astype(int).tolist()}")
            elif cmd == "servo" and len(tokens) == 2 and tokens[1] in ("on", "off"):
                node._send_servo(tokens[1] == "on", repeat=3)
                print(f"cmd_servo -> {tokens[1]}")
            else:
                print(f"unknown command: {line!r}{help_text}")
        except ValueError as e:
            print(f"parse error: {e}")


def parse_args(argv):
    p = argparse.ArgumentParser(
        description="Interactively squeeze/release the KISTAR hand around its "
        "current pose via /hand/{side}/q_target (float32 encoder counts)."
    )
    p.add_argument("--side", choices=("left", "right"), default="right",
                   help="Which hand to command (default: right).")
    p.add_argument("--step", type=float, default=250.0,
                   help="Counts added per '+' press on each squeeze joint "
                   "(default: 250 ~ 5.5 deg; 1 count = pi/8192 rad).")
    p.add_argument("--max-offset", type=float, default=2500.0,
                   help="Upper clamp for the squeeze offset in counts (default: 1500).")
    p.add_argument("--min-offset", type=float, default=-1500.0,
                   help="Lower clamp for the squeeze offset in counts (default: -1500; "
                   "a release safety clamp so '-' cannot hyper-extend far past the "
                   "baseline — widen it here if you need more release travel).")
    p.add_argument("--speed", type=float, default=800.0,
                   help="Target slew rate in counts/s toward the goal (default: 800).")
    p.add_argument("--rate-hz", type=float, default=50.0,
                   help="Publish rate in Hz (default: 50).")
    p.add_argument("--joints", type=int, nargs="+", default=list(DEFAULT_SQUEEZE_JOINTS),
                   help="Real-order joint indices that take part in the squeeze "
                   f"(default: {list(DEFAULT_SQUEEZE_JOINTS)} = flexion joints, "
                   "no abduction / thumb opposition).")
    p.add_argument("--mode", type=int, choices=(1, 2), default=1,
                   help="cmd_mode sent at startup (1=position; default: 1).")
    p.add_argument("--no-mode", dest="mode", action="store_const", const=None,
                   help="Do not send cmd_mode at startup.")
    p.add_argument("--no-servo-on", dest="servo_on", action="store_false",
                   help="Do not auto-send cmd_servo = True at startup.")
    p.add_argument("--discovery-timeout", type=float, default=5.0,
                   help="Seconds to wait for the receiver to subscribe (default: 5).")
    p.add_argument("--state-timeout", type=float, default=5.0,
                   help="Seconds to wait for a measured pose before giving up (default: 5).")
    p.set_defaults(servo_on=True)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv if argv is not None else sys.argv[1:])
    rclpy.init()
    # rclpy.init() installs its own SIGINT handler, which was observed not to stop
    # the node mid-stream. Reinstall Python's default handler so Ctrl+C always
    # raises KeyboardInterrupt promptly.
    signal.signal(signal.SIGINT, signal.default_int_handler)

    node = None
    stop_event = threading.Event()
    try:
        node = HandManualSqueeze(args)
        t = threading.Thread(target=input_loop, args=(node, stop_event), daemon=True)
        t.start()
        while rclpy.ok() and not stop_event.is_set():
            rclpy.spin_once(node, timeout_sec=0.1)
        node.get_logger().info("Exiting; target left at its last value, servo untouched.")
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
        stop_event.set()
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
