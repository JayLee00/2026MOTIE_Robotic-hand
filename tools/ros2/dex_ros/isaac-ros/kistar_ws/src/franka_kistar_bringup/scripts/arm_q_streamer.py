#!/usr/bin/env python3
"""
Arm Joint-Position (q_target) Streamer

Interactive CUI that streams a 7-element joint-position target to a single
FR3 arm's robot-PC receiver, and reports live tracking error against
/franka/{side}/joint_states feedback.

Receiver contract (robot PC, do not deviate without checking with the
robot-side maintainer):
- /franka/{side}/q_target expects EXACTLY 7 float64 values, RADIANS, in
  fr3_joint1..7 order. Wrong array size is silently dropped.
- The receiver clamps any per-message delta to min(0.2 rad, 3.0 rad/s * dt),
  so a target must be STREAMED repeatedly (this node does that at rate_hz),
  not sent once.
- The receiver silently drops targets while the robot's RT loop is not
  running (its measured q reads all-zero), or while require_control is
  enabled and this sender does not hold control ownership
  (see /sequence/request_control).
- NEVER stream q_target and ee_target to the SAME arm at the same time —
  both are latched/streamed independently on the robot PC and will fight
  each other for the same joints.

Author: Chanyoung Ahn
Date: 2026
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
import threading
import time

FR3_JOINT_NAMES = [f'fr3_joint{i}' for i in range(1, 8)]


class ArmQStreamer(Node):
    """
    Interactive q_target streamer for a single FR3 arm.
    """

    def __init__(self):
        super().__init__('arm_q_streamer')

        # Parameters
        self.declare_parameter('side', 'right')
        self.declare_parameter('rate_hz', 15.0)
        self.declare_parameter('tolerance_rad', 0.02)
        self.declare_parameter('timeout_sec', 20.0)
        # Empty string means "derive from side" (see below).
        self.declare_parameter('q_target_topic', '')
        self.declare_parameter('joint_states_topic', '')

        self.side = self.get_parameter('side').value
        if self.side not in ('left', 'right'):
            raise ValueError(f"side must be 'left' or 'right', got: {self.side!r}")
        self.rate_hz = float(self.get_parameter('rate_hz').value)
        self.tolerance_rad = float(self.get_parameter('tolerance_rad').value)
        self.timeout_sec = float(self.get_parameter('timeout_sec').value)

        q_target_override = self.get_parameter('q_target_topic').value
        joint_states_override = self.get_parameter('joint_states_topic').value
        self.q_target_topic = q_target_override or f'/franka/{self.side}/q_target'
        self.joint_states_topic = (
            joint_states_override or f'/franka/{self.side}/joint_states'
        )

        # Streaming state (read/written from the timer callback; only the
        # input thread starts a new stream, so no lock is needed for this
        # single-writer/single-reader pattern).
        self._target = None
        self._latest_q = None
        self._got_feedback_since_start = False
        self._warned_blind = False
        self._active = False
        self._stream_start_time = 0.0

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self._pub = self.create_publisher(Float64MultiArray, self.q_target_topic, qos)
        self._sub = self.create_subscription(
            JointState, self.joint_states_topic, self._on_joint_state, qos
        )

        # Single persistent timer at rate_hz; it is a no-op whenever no
        # stream is active, so creating/destroying timers from the input
        # thread is avoided entirely.
        self._timer = self.create_timer(1.0 / self.rate_hz, self._on_timer)

        # Start input thread
        self.input_thread = threading.Thread(target=self._input_loop, daemon=True)
        self.input_thread.start()

        self.get_logger().info('=' * 70)
        self.get_logger().info('Arm Q Streamer Started')
        self.get_logger().info(f'  Side: {self.side}')
        self.get_logger().info(f'  q_target topic: {self.q_target_topic}')
        self.get_logger().info(f'  joint_states topic: {self.joint_states_topic}')
        self.get_logger().info(f'  Rate: {self.rate_hz} Hz')
        self.get_logger().info(f'  Tolerance: {self.tolerance_rad} rad')
        self.get_logger().info(f'  Timeout: {self.timeout_sec} s')
        self.get_logger().info('-' * 70)
        self.get_logger().info(
            'Enter targets in RADIANS in fr3_joint1..7 order (exactly 7 values).'
        )
        self.get_logger().info(
            'Receiver drops the target when array size != 7.'
        )
        self.get_logger().info(
            'Receiver also drops targets when the robot RT loop is not running '
            '(measured q reads all-zero), or while this sender lacks control '
            'ownership (see /sequence/request_control).'
        )
        self.get_logger().info(
            'Receiver clamps each message to at most 0.2 rad/msg (rate-limited '
            'by 3.0 rad/s), so targets are streamed continuously at rate_hz, not '
            'sent once.'
        )
        self.get_logger().warn(
            'never stream q_target and ee_target to the SAME arm at the same time.'
        )
        self.get_logger().info('=' * 70)

    def _on_joint_state(self, msg: JointState):
        positions = self._extract_positions(msg)
        if positions is not None:
            self._latest_q = positions
            self._got_feedback_since_start = True

    @staticmethod
    def _extract_positions(msg: JointState):
        """Extract [q1..q7] in fr3_joint1..7 order from a JointState message."""
        if msg.name:
            by_name = dict(zip(msg.name, msg.position))
            try:
                return [by_name[n] for n in FR3_JOINT_NAMES]
            except KeyError:
                return None
        if len(msg.position) >= 7:
            return list(msg.position[:7])
        return None

    def _input_loop(self):
        """Input thread - blocking CUI input."""
        while rclpy.ok():
            print('\n' + '=' * 70)
            print(f'Arm Q Streamer [{self.side}] - enter target (RADIANS, 7 values):')
            print('  Format: q1 q2 q3 q4 q5 q6 q7')
            print("  (or 'q'/'quit'/'exit' to quit)")
            print('=' * 70)

            user_input = input('>> ').strip()

            if user_input.lower() in ('q', 'quit', 'exit'):
                self.get_logger().info('Shutting down...')
                rclpy.shutdown()
                break

            if not user_input:
                continue

            try:
                values = [float(x) for x in user_input.split()]
            except ValueError as e:
                print(f'Error parsing input: {e}')
                continue

            if len(values) != 7:
                print(f'Error: Expected 7 values, got {len(values)}')
                continue

            self._start_stream(values)

    def _start_stream(self, values):
        self._target = values
        self._stream_start_time = time.monotonic()
        self._got_feedback_since_start = False
        self._warned_blind = False
        self._active = True
        print(f'[STREAM] Starting stream to {self.q_target_topic} at {self.rate_hz} Hz')
        print(f'[STREAM] Target: {[round(v, 4) for v in values]}')

    def _on_timer(self):
        if not self._active or self._target is None:
            return

        elapsed = time.monotonic() - self._stream_start_time

        msg = Float64MultiArray()
        msg.data = list(self._target)
        self._pub.publish(msg)

        if (
            not self._got_feedback_since_start
            and not self._warned_blind
            and elapsed >= 1.0
        ):
            self.get_logger().warn(
                f'WARNING: no feedback on {self.joint_states_topic} — streaming '
                f'blind for {self.timeout_sec}s'
            )
            self._warned_blind = True

        if self._latest_q is not None:
            max_err = max(
                abs(a - b) for a, b in zip(self._latest_q, self._target)
            )
            print(f'[STREAM] t={elapsed:5.1f}s  max|q_meas - q_target| = {max_err:.4f} rad')

            if max_err < self.tolerance_rad:
                print('[STREAM] reached')
                self._active = False
                self._target = None
                return
        else:
            print(f'[STREAM] t={elapsed:5.1f}s  (no feedback yet)')

        if elapsed >= self.timeout_sec:
            if self._got_feedback_since_start:
                print(
                    f'[STREAM] TIMEOUT: target not reached within {self.timeout_sec}s. '
                    'Check that the robot RT loop is running (measured q not all-zero) '
                    'and that this sender holds control ownership '
                    '(/sequence/request_control).'
                )
            else:
                print(
                    f'[STREAM] Blind streaming timeout ({self.timeout_sec}s) elapsed '
                    f'with no feedback ever received on {self.joint_states_topic}.'
                )
            self._active = False
            self._target = None


def main(args=None):
    rclpy.init(args=args)
    node = ArmQStreamer()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Arm Q Streamer shutting down...')
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
