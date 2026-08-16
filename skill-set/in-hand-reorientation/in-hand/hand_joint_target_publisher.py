#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real-time hand finger action publisher (ROS2) for the robot-PC KISTAR hand
receiver (hand_target_receiver).

Reads the SAME HDF5 trajectory used by cube_rotate.cpp
(dataset /data/demo_1/sim_joint_real, shape [N, 16] float32 radians),
reproduces cube_rotate.cpp's per-joint corrections AND its int16 motor-tick
encoding   ticks = int16( rad * 4096 * 2.1 / pi * 1.12 ),
then streams each frame to the receiver command topic
/hand/{side}/q_target as std_msgs/Float32MultiArray at a fixed rate
(default 50 Hz == cube_rotate's 20 ms FRAME_INTERVAL_MS).

SIGN CONVENTION: commands are sent with the raw cube_rotate tick sign
(default). The hand must be in cmd_mode = 1 (position) for targets to be
interpreted correctly — this node sends it at startup. Negated commands
(--invert) were tried on 2026-07-07 and fall outside the joint range.

Receiver contract (robot PC, verified against hand_target_receiver_node.cpp):
  /hand/{side}/q_target   Float32MultiArray, EXACTLY 16 values, RAW ENCODER
                          COUNTS (1 tick = pi/8192 rad), BEST_EFFORT depth 1.
                          Dropped while the robot RT loop is not running
                          (measured q all-zero) or while control ownership is
                          required but not held (/sequence/request_control).
  /hand/{side}/cmd_servo  Bool, RELIABLE, applied immediately (not gated).
  /hand/{side}/cmd_mode   Int32 (1=position, 2=circular), RELIABLE, immediate.

Startup sequence: cmd_mode = 1 (position; change with --mode, skip with
--no-mode), then cmd_servo = True (disable with --no-servo-on), then a soft
ramp from the CURRENT measured pose (/hand/{side}/joint_states) to the first
trajectory frame over --ramp-secs (default 2 s) to avoid a snap, then the
trajectory itself. It does NOT servo-off on exit (use --servo-off-on-exit).

Requires the ROS2-matched interpreter (Humble -> /usr/bin/python3, 3.10):

    source /opt/ros/humble/setup.bash
    /usr/bin/python3 in-hand/hand_joint_target_publisher.py \
        --side right --file in-hand/data/test_int.hdf5
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

try:
    import h5py
except ImportError:  # pragma: no cover
    sys.stderr.write(
        "ERROR: h5py not found for this interpreter.\n"
        "Install it for the ROS2 python, e.g.: /usr/bin/python3 -m pip install --user h5py\n"
    )
    raise

NUM_JOINTS = 16

# cube_rotate.cpp encoding constants.
SCALE_FACTOR = 4096.0    # SCALE_FACTOR
SCALE_FACTOR2 = 2.1      # SCALE3FACTOR2
PI = 3.14159265358979323846
EXTRA_GAIN = 1.13        # the trailing "* 1.12" in cube_rotate's int16 cast
INT16_MIN, INT16_MAX = -32768, 32767

# Per-joint multipliers reproduced from cube_rotate.cpp (applied in source order;
# joint 8 is scaled twice: 1.12 in the 'middle' block then 0.75 in the 'index' block).
#   thumb : j1*=1.36, j2*=0.9
#   index : j4*=0.9, j6*=0.8, j7*=0.8, j8*=0.75
#   middle: j8*=1.12, j9*=0.95, j10*=0.92, j11*=0.78
#   ring  : j12*=0.75, j13..15*=1.0
JOINT_CORRECTION = np.array(
    [
        1.0,          # 0
        1.2,         # 1  thumb
        0.9,          # 2  thumb
        1.0,          # 3
        0.9,          # 4  index
        1.0,          # 5
        0.8,          # 6  index
        0.8,          # 7  index
        1.12 * 0.75,  # 8  middle*index (= 0.84)
        0.95,         # 9  middle
        0.92,         # 10 middle
        # 0.78,         # 11 middle
        0.8,         # 11 middle
        # 0.75,         # 12 ring
        0.8,         # 12 ring
        1.0,          # 13 ring
        1.0,          # 14 ring
        1.0,          # 15 ring
    ],
    dtype=np.float64,
)


class HandJointTargetPublisher(Node):
    def __init__(self, args):
        super().__init__("hand_joint_target_publisher")

        self._loop = args.loop
        self._servo_off_on_exit = args.servo_off_on_exit
        self._frame = 0
        self.finished = False  # set True when the trajectory ends (main loop exits)

        ticks, n_clamped = self._load_trajectory(
            args.file, args.dataset, args.apply_corrections, args.gain, args.invert
        )
        total_loaded = ticks.shape[0]

        # Optional start offset: begin playback (and the soft-ramp target) at
        # --init-frame instead of frame 0. Frames before it are dropped, so the
        # ramp targets this frame and streaming starts here.
        init_frame = max(0, int(args.init_frame))
        if init_frame >= total_loaded:
            raise ValueError(
                f"--init-frame {init_frame} is out of range "
                f"(trajectory has {total_loaded} frames)."
            )
        if init_frame > 0:
            ticks = np.ascontiguousarray(ticks[init_frame:])
        self._init_frame = init_frame

        # Play at most --max-frames frames (default 2000), counting from init_frame.
        capped_by_max = (
            args.max_frames is not None
            and args.max_frames > 0
            and ticks.shape[0] > args.max_frames
        )
        if capped_by_max:
            ticks = np.ascontiguousarray(ticks[: args.max_frames])
        self._ticks = ticks
        self.get_logger().info(
            f"Loaded {total_loaded} frames x {ticks.shape[1]} joints from "
            f"'{args.file}':{args.dataset} "
            f"(start=frame {init_frame}, "
            f"corrections={'on' if args.apply_corrections else 'off'}, "
            f"gain={args.gain}, invert={'on' if args.invert else 'off'})"
        )
        if capped_by_max:
            self.get_logger().info(
                f"Capped to {ticks.shape[0]} frames "
                f"(--max-frames={args.max_frames}, from frame {init_frame})."
            )
        if n_clamped:
            self.get_logger().warn(
                f"{n_clamped} tick value(s) exceeded int16 range and were clamped "
                f"to [{INT16_MIN}, {INT16_MAX}]."
            )

        self._cmd_topic = f"/hand/{args.side}/q_target"
        servo_topic = f"/hand/{args.side}/cmd_servo"
        mode_topic = f"/hand/{args.side}/cmd_mode"
        state_topic = f"/hand/{args.side}/joint_states"

        # q_target: match the receiver (BEST_EFFORT depth 1) unless --reliable.
        q_reliability = (
            ReliabilityPolicy.RELIABLE if args.reliable else ReliabilityPolicy.BEST_EFFORT
        )
        q_qos = QoSProfile(depth=1, reliability=q_reliability, history=HistoryPolicy.KEEP_LAST)
        # cmd_servo / cmd_mode are RELIABLE on the receiver.
        reliable_qos = QoSProfile(
            depth=1, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST
        )
        state_qos = QoSProfile(
            depth=1, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST
        )

        self._pub = self.create_publisher(Float32MultiArray, self._cmd_topic, q_qos)
        self._servo_pub = self.create_publisher(Bool, servo_topic, reliable_qos)
        self._mode_pub = self.create_publisher(Int32, mode_topic, reliable_qos)

        self._measured_q = None
        self._state_sub = self.create_subscription(
            JointState, state_topic, self._on_state, state_qos
        )

        # Wait for the receiver to subscribe, then set mode and enable the hand.
        self._wait_for_subscriber(args.discovery_timeout)
        if args.mode is not None:
            self._send_mode(args.mode, repeat=3)
            self.get_logger().info(f"Sent cmd_mode = {args.mode} to '{mode_topic}'.")
            time.sleep(0.1)
        if args.servo_on:
            self._send_servo(True, repeat=5)
            self.get_logger().info(f"Sent servo-on (True) to '{servo_topic}'.")
            time.sleep(0.3)

        # Soft ramp: measured current pose -> first trajectory frame.
        self._ramp = self._build_ramp(state_topic, args.ramp_secs, args.rate_hz)
        self._ramp_frame = 0

        period = 1.0 / args.rate_hz
        self._timer = self.create_timer(period, self._on_timer)
        self.get_logger().info(
            f"Publishing float32 encoder counts -> '{self._cmd_topic}' at {args.rate_hz:.1f} Hz "
            f"(reliability={q_reliability.name}, {'looping' if self._loop else 'one-shot'})"
        )

    def _load_trajectory(self, path, dataset, apply_corrections, gain, invert):
        with h5py.File(path, "r") as f:
            if dataset not in f:
                raise KeyError(
                    f"dataset '{dataset}' not in {path}. "
                    f"Available leaves: {self._list_datasets(f)}"
                )
            data = np.asarray(f[dataset][:], dtype=np.float64)

        if data.ndim != 2 or data.shape[1] != NUM_JOINTS:
            raise ValueError(f"expected [N, {NUM_JOINTS}] dataset, got shape {data.shape}")

        if apply_corrections:
            data = data * JOINT_CORRECTION  # broadcast over rows
        data = data * gain

        # cube_rotate.cpp: (raw * SCALE_FACTOR * SCALE_FACTOR2) / PI * 1.12  -> int16
        # (kept identical so the resulting encoder counts match the proven motion).
        ticks_f = data * SCALE_FACTOR * SCALE_FACTOR2 / PI * EXTRA_GAIN
        if invert:
            ticks_f = -ticks_f  # SHM/EtherCAT direction is opposite to cube_rotate ticks
        clamped = np.clip(ticks_f, INT16_MIN, INT16_MAX)
        n_clamped = int(np.count_nonzero(clamped != ticks_f))
        ticks = clamped.astype(np.int16).astype(np.float32)
        return np.ascontiguousarray(ticks), n_clamped

    @staticmethod
    def _list_datasets(f):
        names = []
        f.visititems(lambda n, o: names.append(n) if isinstance(o, h5py.Dataset) else None)
        return names

    def _on_state(self, msg):
        if len(msg.position) >= NUM_JOINTS:
            self._measured_q = np.asarray(msg.position[:NUM_JOINTS], dtype=np.float32)

    def _build_ramp(self, state_topic, ramp_secs, rate_hz):
        """Interpolate from the measured pose to frame 0. Returns [M, 16] or None."""
        if ramp_secs <= 0:
            return None
        deadline = time.monotonic() + 3.0
        while self._measured_q is None and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
        if self._measured_q is None:
            self.get_logger().warn(
                f"No measured state on '{state_topic}'; starting WITHOUT the soft "
                f"ramp (first frame may cause a jump)."
            )
            return None
        start = self._measured_q.astype(np.float32)
        goal = self._ticks[0]
        steps = max(2, int(round(ramp_secs * rate_hz)))
        alphas = np.linspace(0.0, 1.0, steps, dtype=np.float32)[:, None]
        ramp = start[None, :] * (1.0 - alphas) + goal[None, :] * alphas
        self.get_logger().info(
            f"Soft ramp: measured pose -> frame 0 over {ramp_secs:.1f}s "
            f"({steps} steps, max |delta| = {float(np.max(np.abs(goal - start))):.0f} counts)."
        )
        return np.ascontiguousarray(ramp.astype(np.float32))

    def _wait_for_subscriber(self, timeout_s):
        if timeout_s <= 0:
            return
        deadline = timeout_s
        waited = 0.0
        step = 0.1
        while waited < deadline:
            if self._pub.get_subscription_count() > 0:
                self.get_logger().info(
                    f"Subscriber detected on '{self._cmd_topic}'."
                )
                return
            time.sleep(step)
            waited += step
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
        # Phase 1: soft ramp to the first trajectory frame.
        if self._ramp is not None and self._ramp_frame < self._ramp.shape[0]:
            self._pub.publish(self._make_msg(self._ramp[self._ramp_frame]))
            self._ramp_frame += 1
            if self._ramp_frame == self._ramp.shape[0]:
                self.get_logger().info("Soft ramp done; streaming trajectory.")
            return

        # Phase 2: the trajectory itself.
        n = self._ticks.shape[0]
        if self._frame >= n:
            if self._loop:
                self._frame = 0
            else:
                self.get_logger().info("Trajectory finished; shutting down.")
                self._timer.cancel()
                self.finished = True
                return

        row = self._ticks[self._frame]
        self._pub.publish(self._make_msg(row))

        if self._frame % 50 == 0:
            self.get_logger().info(
                f"frame {self._frame + 1}/{n} "
                f"(abs {self._init_frame + self._frame + 1})"
            )
        self._frame += 1

    def shutdown_hand(self):
        if self._servo_off_on_exit:
            try:
                self._send_servo(False, repeat=3)
                self.get_logger().info("Sent servo-off (False) on exit.")
            except Exception:
                pass


def parse_args(argv):
    p = argparse.ArgumentParser(
        description="Stream cube_rotate HDF5 hand trajectory to the robot-PC "
        "receiver topic /hand/{side}/q_target as float32 encoder counts."
    )
    p.add_argument("--side", choices=("left", "right"), default="right",
                   help="Which hand to command (default: right).")
    p.add_argument("--file", default="in-hand/data/test_int.hdf5",
                   help="HDF5 trajectory file (default: cube_rotate's test_int.hdf5).")
    p.add_argument("--dataset", default="/data/demo_1/sim_joint_real",
                   help="Dataset path inside the HDF5 file (default: /data/demo_1/sim_joint_real).")
    p.add_argument("--init-frame", type=int, default=300,
                   help="Start playback (and the soft-ramp target) at this frame "
                   "index instead of 0 (e.g. 300). Earlier frames are skipped; "
                   "--max-frames then counts from here (default: 0).")
    p.add_argument("--rate-hz", type=float, default=60.0,
                   help="Publish rate in Hz (default: 50 Hz == cube_rotate's 20 ms frame interval).")
    p.add_argument("--gain", type=float, default=1.0,
                   help="Optional global scalar applied (in radians) before encoding (default: 1.0).")
    p.add_argument("--mode", type=int, choices=(1, 2), default=1,
                   help="cmd_mode sent at startup (1=position, 2=circular; default: 1).")
    p.add_argument("--no-mode", dest="mode", action="store_const", const=None,
                   help="Do not send cmd_mode at startup.")
    p.add_argument("--ramp-secs", type=float, default=2.0,
                   help="Soft-ramp duration from the measured pose to frame 0 "
                   "(default: 2.0; 0 disables the ramp).")
    p.add_argument("--discovery-timeout", type=float, default=5.0,
                   help="Seconds to wait for the receiver to subscribe before streaming (default: 5).")
    p.add_argument("--no-corrections", dest="apply_corrections", action="store_false",
                   help="Skip cube_rotate's per-joint scaling (still encodes raw radians to counts).")
    p.add_argument("--invert", dest="invert", action="store_true",
                   help="Negate all commands before sending (default: off — raw "
                   "cube_rotate tick sign; inverted commands were measured to fall "
                   "outside the joint range on 2026-07-07).")
    p.add_argument("--no-servo-on", dest="servo_on", action="store_false",
                   help="Do not auto-send cmd_servo = True at startup.")
    p.add_argument("--servo-off-on-exit", action="store_true",
                   help="Send cmd_servo = False when the node exits (default: leave servo as-is).")
    p.add_argument("--reliable", action="store_true",
                   help="Use RELIABLE reliability for q_target instead of BEST_EFFORT "
                   "(receiver is BEST_EFFORT; default matches it).")
    p.add_argument("--loop", action="store_true",
                   help="Loop the trajectory instead of stopping at the last frame.")
    p.add_argument("--max-frames", type=int, default=500,
                   help="Play at most this many trajectory frames (default: 2000; "
                   "<=0 disables the cap).")
    p.set_defaults(apply_corrections=True, servo_on=True, invert=False)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv if argv is not None else sys.argv[1:])
    rclpy.init()
    # rclpy.init() installs its own SIGINT handler, which was observed not to stop
    # the node mid-stream. Reinstall Python's default handler so Ctrl+C always
    # raises KeyboardInterrupt promptly (during startup sleeps and while spinning).
    signal.signal(signal.SIGINT, signal.default_int_handler)

    node = None
    try:
        node = HandJointTargetPublisher(args)
        # Manual spin loop: exits on trajectory end (node.finished) or Ctrl+C.
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
            node.shutdown_hand()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
