#!/usr/bin/env python3
"""
KISTAR Hand GUI Bridge

Subscribes to the joint_state_publisher(_gui) output (URDF joint names,
radians) and streams the 16 hand joints to the robot-PC hand receiver as
raw encoder counts, so moving a slider in the GUI moves the real fingers.

Receiver contract (robot PC hand_target_receiver, verified 2026-07-07):
- /hand/{side}/q_target expects EXACTLY 16 float32 values in RAW ENCODER
  COUNTS (1 tick = pi/8192 rad), BEST_EFFORT depth 1. Targets are dropped
  while the robot RT loop is down (measured q all-zero) or while
  require_control is enabled without ownership (/sequence/request_control).
- /hand/{side}/cmd_servo (Bool) / cmd_mode (Int32, 1=position) are RELIABLE
  and applied immediately. The hand must be in mode 1 for position targets.

Real (topic) joint order — verified PHYSICALLY on hardware 2026-07-07
(moving GUI sliders and watching which finger moves):
  [ 0.. 3] thumb_joint_0..3
  [ 4.. 7] index_joint_0..3
  [ 8..11] middle_joint_0..3
  [12..15] ring_joint_0..3
CAUTION: /joint_states_r labels the hand block index-first — those NAME
labels do NOT match the actual SHM actuation order. Trust the hardware.

URDF names are {joint_prefix}{finger}_joint_{0..3}. Conversion is
  counts = rad * 8192 / pi   (negated when the 'invert' parameter is true).

The published target is slew-rate limited ('speed' counts/s) and initialized
from the measured hand pose (/hand/{side}/joint_states) when available, so
enabling the bridge does not snap the fingers.

Author: Chanyoung Ahn
Date: 2026
"""

import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float32MultiArray, Int32

NUM_JOINTS = 16
COUNTS_PER_RAD = 8192.0 / math.pi

# real-order slot -> URDF joint name suffix ({finger}_joint_{n}).
# Thumb-first, physically verified on hardware (2026-07-07). NOTE that the
# /joint_states_r NAME labels (index-first) do NOT match this actuation order.
REAL_ORDER_URDF_SUFFIXES = [
    "thumb_joint_0", "thumb_joint_1", "thumb_joint_2", "thumb_joint_3",
    "index_joint_0", "index_joint_1", "index_joint_2", "index_joint_3",
    "middle_joint_0", "middle_joint_1", "middle_joint_2", "middle_joint_3",
    "ring_joint_0", "ring_joint_1", "ring_joint_2", "ring_joint_3",
]


class HandGuiBridge(Node):
    def __init__(self):
        super().__init__("hand_gui_bridge")

        self.declare_parameter("side", "right")
        self.declare_parameter("joint_prefix", "")
        self.declare_parameter("rate_hz", 50.0)
        self.declare_parameter("speed", 800.0)  # counts/s slew limit
        self.declare_parameter("counts_per_rad", COUNTS_PER_RAD)
        self.declare_parameter("invert", False)
        self.declare_parameter("send_mode", 1)  # <=0: do not touch cmd_mode
        self.declare_parameter("servo_on", True)

        side = self.get_parameter("side").value
        if side not in ("left", "right"):
            raise ValueError(f"side must be 'left' or 'right', got: {side!r}")
        prefix = self.get_parameter("joint_prefix").value
        self._rate_hz = float(self.get_parameter("rate_hz").value)
        self._speed = float(self.get_parameter("speed").value)
        scale = float(self.get_parameter("counts_per_rad").value)
        self._scale = -scale if self.get_parameter("invert").value else scale

        self._urdf_names = [prefix + s for s in REAL_ORDER_URDF_SUFFIXES]
        self._goal = None    # [16] counts, from the GUI
        self._target = None  # [16] counts, slew-limited, what we publish
        # GUI-init handshake: the measured pose is offered to the GUI (rad) on
        # 'hand_gui_init_joint_states' (jsp_gui source_list) so the sliders
        # start AT the real pose instead of 0. Until the GUI echoes that pose
        # back (or the sync deadline passes), GUI goals are IGNORED — the hand
        # does not move at startup.
        self._gui_synced = False
        self._sync_deadline = None  # set on the first GUI message

        best_effort = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        reliable = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self._q_pub = self.create_publisher(
            Float32MultiArray, f"/hand/{side}/q_target", best_effort
        )
        self._servo_pub = self.create_publisher(Bool, f"/hand/{side}/cmd_servo", reliable)
        self._mode_pub = self.create_publisher(Int32, f"/hand/{side}/cmd_mode", reliable)

        # GUI joint states: relative topic so it resolves inside the launch namespace.
        self.create_subscription(JointState, "joint_states", self._on_gui, 10)
        # Measured hand pose (absolute robot topic): used once to seed the slew start.
        self.create_subscription(
            JointState, f"/hand/{side}/joint_states", self._on_measured, best_effort
        )
        # Init pose for the GUI sliders (relative; jsp_gui subscribes via source_list).
        self._init_pub = self.create_publisher(
            JointState, "hand_gui_init_joint_states", 10
        )

        send_mode = int(self.get_parameter("send_mode").value)
        if send_mode > 0:
            msg = Int32()
            msg.data = send_mode
            for _ in range(3):
                self._mode_pub.publish(msg)
                time.sleep(0.05)
            self.get_logger().info(f"Sent cmd_mode = {send_mode}.")
        if bool(self.get_parameter("servo_on").value):
            msg = Bool()
            msg.data = True
            for _ in range(5):
                self._servo_pub.publish(msg)
                time.sleep(0.05)
            self.get_logger().info("Sent cmd_servo = True.")

        # Wait briefly for the measured pose BEFORE the timer can publish:
        # remote DDS discovery takes ~1-2 s, and starting the slew from the GUI
        # goal instead of the real pose would snap a flexed hand.
        deadline = time.monotonic() + 3.0
        while self._target is None and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
        if self._target is None:
            self.get_logger().warn(
                "No measured hand pose within 3 s; the slew will start from the "
                "first GUI goal (fingers may move to the slider pose immediately)."
            )

        self.create_timer(1.0 / self._rate_hz, self._on_timer)
        self.get_logger().info(
            f"Bridging GUI joint_states -> /hand/{side}/q_target "
            f"(prefix='{prefix}', scale={self._scale:.1f} counts/rad, "
            f"speed={self._speed:.0f} counts/s). Waiting for the first GUI message..."
        )

    def _on_measured(self, msg):
        # Seed the slew start from the real pose exactly once, before the first
        # GUI goal is applied.
        if self._target is None and len(msg.position) >= NUM_JOINTS:
            self._target = [float(v) for v in msg.position[:NUM_JOINTS]]
            self.get_logger().info("Slew start seeded from measured hand pose.")

    def _on_gui(self, msg):
        idx = {name: i for i, name in enumerate(msg.name)}
        missing = [n for n in self._urdf_names if n not in idx]
        if missing:
            self.get_logger().warn(
                f"GUI joint_states missing {len(missing)} hand joints "
                f"(e.g. {missing[0]}) — ignoring message.",
                throttle_duration_sec=5.0,
            )
            return
        goal = [float(msg.position[idx[n]]) * self._scale for n in self._urdf_names]

        if not self._gui_synced:
            if self._sync_deadline is None:
                self._sync_deadline = time.monotonic() + 5.0
            if self._target is not None and all(
                abs(g - t) <= 0.05 * abs(self._scale)  # 0.05 rad in counts
                for g, t in zip(goal, self._target)
            ):
                self._gui_synced = True
                self.get_logger().info(
                    "GUI sliders synced to the measured pose; following the GUI now."
                )
            elif time.monotonic() > self._sync_deadline:
                self._gui_synced = True
                self.get_logger().warn(
                    "GUI never echoed the measured init pose (is jsp_gui's "
                    "source_list set?); following GUI goals anyway — the hand "
                    "will move to the current slider pose."
                )
            else:
                return  # ignore GUI goals until synced — hand stays put

        self._goal = goal

    def _on_timer(self):
        # Until the GUI has picked up the measured init pose, keep offering it
        # (~5 Hz) so the sliders start at the real pose instead of 0.
        if not self._gui_synced and self._target is not None:
            self._init_ticks = getattr(self, "_init_ticks", 0) + 1
            if self._init_ticks % 10 == 1:
                init = JointState()
                init.header.stamp = self.get_clock().now().to_msg()
                init.name = list(self._urdf_names)
                init.position = [float(t) / self._scale for t in self._target]
                self._init_pub.publish(init)
        if self._goal is None:
            return  # no GUI input yet — publish nothing, hand stays put
        if self._target is None:
            # No measured pose available: start from the first goal (no motion
            # until a slider moves away from it).
            self._target = list(self._goal)
            self.get_logger().warn(
                "No measured hand pose received; slew starts from the first GUI goal."
            )
        max_step = self._speed / self._rate_hz
        self._target = [
            t + max(-max_step, min(max_step, g - t))
            for t, g in zip(self._target, self._goal)
        ]
        out = Float32MultiArray()
        out.data = [float(v) for v in self._target]
        self._q_pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = HandGuiBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Hand GUI bridge shutting down...")
    except rclpy.executors.ExternalShutdownException:
        pass  # SIGTERM (e.g. launch teardown)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
