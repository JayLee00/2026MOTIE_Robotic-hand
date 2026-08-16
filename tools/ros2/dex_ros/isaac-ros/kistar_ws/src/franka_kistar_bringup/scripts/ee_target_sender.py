#!/usr/bin/env python3
"""
End-Effector World-Frame Target Sender

Interactive CUI that publishes a one-shot (repeated a few times to survive
BEST_EFFORT loss) PoseStamped target for a single FR3 arm's robot-PC
Cartesian receiver.

Receiver contract (robot PC, do not deviate without checking with the
robot-side maintainer):
- /franka/{side}/ee_target_world is BEST_EFFORT depth 1. The receiver
  LATCHES the last target it saw and re-applies it at 100 Hz forever, EVEN
  AFTER THIS SENDER EXITS, until a new target overwrites it. There is no
  "stop" message — sending a new pose is the only way to change behaviour.
- The receiver ignores header.frame_id and header.stamp entirely; the pose
  is interpreted directly in the robot's world frame as configured on the
  robot PC.
- CALIBRATION WARNING: T_base_world and T_flange_ee default to IDENTITY on
  the robot PC. Until they are calibrated, a "world" pose sent here does
  NOT correspond to the true robot-base/world or flange/end-effector
  transforms — positions will be wrong relative to the physical world.
- The receiver silently drops targets while the robot RT loop is not
  running (measured q reads all-zero), or while require_control is enabled
  and this sender does not hold control ownership
  (see /sequence/request_control).
- NEVER stream ee_target and q_target to the SAME arm at the same time —
  both act on the same joints independently on the robot PC and will fight
  each other.

Author: Chanyoung Ahn
Date: 2026
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from geometry_msgs.msg import PoseStamped, Point, Quaternion
import threading
import math
import time


class EeTargetSender(Node):
    """
    Interactive end-effector world-frame target sender for a single FR3 arm.
    """

    def __init__(self):
        super().__init__('ee_target_sender')

        # Parameters
        self.declare_parameter('side', 'right')
        self.declare_parameter('frame_id', 'world')
        self.declare_parameter('repeat', 5)
        self.declare_parameter('repeat_interval', 0.05)
        # Empty string means "derive from side" (see below).
        self.declare_parameter('ee_target_topic', '')

        self.side = self.get_parameter('side').value
        if self.side not in ('left', 'right'):
            raise ValueError(f"side must be 'left' or 'right', got: {self.side!r}")
        self.frame_id = self.get_parameter('frame_id').value
        self.repeat = int(self.get_parameter('repeat').value)
        self.repeat_interval = float(self.get_parameter('repeat_interval').value)

        topic_override = self.get_parameter('ee_target_topic').value
        self.ee_target_topic = topic_override or f'/franka/{self.side}/ee_target_world'

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self._pub = self.create_publisher(PoseStamped, self.ee_target_topic, qos)

        # Start input thread
        self.input_thread = threading.Thread(target=self._input_loop, daemon=True)
        self.input_thread.start()

        self.get_logger().info('=' * 70)
        self.get_logger().info('EE Target Sender Started')
        self.get_logger().info(f'  Side: {self.side}')
        self.get_logger().info(f'  Topic: {self.ee_target_topic}')
        self.get_logger().info(f'  frame_id (header only): {self.frame_id}')
        self.get_logger().info(f'  Repeat: {self.repeat}x @ {self.repeat_interval}s')
        self.get_logger().info('-' * 70)
        self.get_logger().warn(
            'CALIBRATION WARNING: T_base_world and T_flange_ee default to identity '
            'on the robot PC. Until calibrated, world-frame targets will NOT land '
            'at the intended physical position.'
        )
        self.get_logger().info(
            'The receiver latches the last target and re-applies it at 100 Hz '
            'forever — this persists even after this sender exits, until a new '
            'target is sent.'
        )
        self.get_logger().info(
            'Receiver drops targets while the robot RT loop is not running '
            '(measured q reads all-zero), or while this sender lacks control '
            'ownership (see /sequence/request_control).'
        )
        self.get_logger().warn(
            'never stream ee_target and q_target to the SAME arm at the same time.'
        )
        self.get_logger().info('=' * 70)

    def _input_loop(self):
        """Input thread - blocking CUI input."""
        while rclpy.ok():
            print('\n' + '=' * 70)
            print(f'EE Target Sender [{self.side}] - enter target pose (world frame):')
            print('  Format: x y z qx qy qz qw')
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

            x, y, z, qx, qy, qz, qw = values

            norm = math.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
            if norm < 0.01:
                print('Error: Invalid quaternion (near-zero norm)')
                continue
            qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm

            self._send_target(x, y, z, qx, qy, qz, qw)

    def _send_target(self, x, y, z, qx, qy, qz, qw):
        print(f'[SEND] Publishing to {self.ee_target_topic} x{self.repeat} '
              f'@ {self.repeat_interval}s intervals')
        print(f'[SEND] Position: ({x:.4f}, {y:.4f}, {z:.4f})')
        print(f'[SEND] Orientation: ({qx:.4f}, {qy:.4f}, {qz:.4f}, {qw:.4f})')

        for i in range(self.repeat):
            pose = PoseStamped()
            pose.header.frame_id = self.frame_id
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.pose.position = Point(x=x, y=y, z=z)
            pose.pose.orientation = Quaternion(x=qx, y=qy, z=qz, w=qw)
            self._pub.publish(pose)
            if i < self.repeat - 1:
                time.sleep(self.repeat_interval)

        print('[SEND] Done. Reminder: this target now persists on the robot PC '
              'at 100 Hz even after this sender exits.')


def main(args=None):
    rclpy.init(args=args)
    node = EeTargetSender()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('EE Target Sender shutting down...')
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
