#!/usr/bin/env python3
"""
KISTAR Hand Command Sender

Interactive CUI that publishes joint targets and mode/servo commands to a
single KISTAR hand's robot-PC receiver.

Receiver contract (robot PC, do not deviate without checking with the
robot-side maintainer):
- /hand/{side}/q_target expects EXACTLY 16 float32 values, in RAW ENCODER
  COUNTS (1 tick = pi/8192 rad), NOT radians. BEST_EFFORT depth 1. The
  receiver silently drops targets while the robot RT loop is not running
  (measured q reads all-zero), or while require_control is enabled and this
  sender does not hold control ownership (see /sequence/request_control).
- /hand/{side}/cmd_servo (Bool) and /hand/{side}/cmd_mode (Int32,
  1=position, 2=circular) are RELIABLE depth 1 and are applied IMMEDIATELY —
  they are NOT control-gated, unlike q_target.

Author: Chanyoung Ahn
Date: 2026
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from std_msgs.msg import Float32MultiArray, Bool, Int32
import threading


class HandTargetSender(Node):
    """
    Interactive command sender for a single KISTAR hand.
    """

    def __init__(self):
        super().__init__('hand_target_sender')

        # Parameters
        self.declare_parameter('side', 'right')

        self.side = self.get_parameter('side').value
        if self.side not in ('left', 'right'):
            raise ValueError(f"side must be 'left' or 'right', got: {self.side!r}")

        self.q_target_topic = f'/hand/{self.side}/q_target'
        self.cmd_servo_topic = f'/hand/{self.side}/cmd_servo'
        self.cmd_mode_topic = f'/hand/{self.side}/cmd_mode'

        best_effort_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        reliable_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self._q_pub = self.create_publisher(
            Float32MultiArray, self.q_target_topic, best_effort_qos
        )
        self._servo_pub = self.create_publisher(
            Bool, self.cmd_servo_topic, reliable_qos
        )
        self._mode_pub = self.create_publisher(
            Int32, self.cmd_mode_topic, reliable_qos
        )

        # Start input thread
        self.input_thread = threading.Thread(target=self._input_loop, daemon=True)
        self.input_thread.start()

        self.get_logger().info('=' * 70)
        self.get_logger().info('Hand Target Sender Started')
        self.get_logger().info(f'  Side: {self.side}')
        self.get_logger().info(f'  q_target topic: {self.q_target_topic}')
        self.get_logger().info(f'  cmd_servo topic: {self.cmd_servo_topic}')
        self.get_logger().info(f'  cmd_mode topic: {self.cmd_mode_topic}')
        self.get_logger().info('-' * 70)
        self.get_logger().warn(
            'q_target values are RAW ENCODER COUNTS (1 tick = pi/8192 rad), '
            'NOT radians. Exactly 16 values are required, in the same order '
            'every time.'
        )
        self.get_logger().info(
            'q_target is dropped while the robot RT loop is not running '
            '(measured q reads all-zero), or while this sender lacks control '
            'ownership (see /sequence/request_control).'
        )
        self.get_logger().info(
            'servo and mode commands are applied immediately and are NOT '
            'control-gated (unlike q_target).'
        )
        self.get_logger().info('=' * 70)

    def _input_loop(self):
        """Input thread - blocking CUI input."""
        while rclpy.ok():
            print('\n' + '=' * 70)
            print(f'Hand Target Sender [{self.side}] - commands:')
            print('  q <v1> ... <v16>   publish q_target (raw encoder counts)')
            print('  servo on|off       publish cmd_servo')
            print('  mode 1|2           publish cmd_mode (1=position, 2=circular)')
            print("  quit               (or 'q' alone is ambiguous - use 'quit'/'exit')")
            print('=' * 70)

            user_input = input('>> ').strip()

            if user_input.lower() in ('quit', 'exit'):
                self.get_logger().info('Shutting down...')
                rclpy.shutdown()
                break

            if not user_input:
                continue

            tokens = user_input.split()
            cmd = tokens[0].lower()

            try:
                if cmd == 'q':
                    self._handle_q(tokens[1:])
                elif cmd == 'servo':
                    self._handle_servo(tokens[1:])
                elif cmd == 'mode':
                    self._handle_mode(tokens[1:])
                else:
                    print(f"Error: Unknown command '{cmd}'")
            except ValueError as e:
                print(f'Error parsing input: {e}')

    def _handle_q(self, args):
        try:
            values = [float(x) for x in args]
        except ValueError as e:
            print(f'Error parsing values: {e}')
            return

        if len(values) != 16:
            print(f'Error: Exactly 16 values required, got {len(values)}. Not published.')
            return

        msg = Float32MultiArray()
        msg.data = [float(v) for v in values]
        self._q_pub.publish(msg)
        print(f'[SEND] q_target ({self.q_target_topic}): {[round(v, 1) for v in values]}')

    def _handle_servo(self, args):
        if len(args) != 1 or args[0].lower() not in ('on', 'off'):
            print("Error: Usage: 'servo on' or 'servo off'")
            return
        state = args[0].lower() == 'on'
        msg = Bool()
        msg.data = state
        self._servo_pub.publish(msg)
        print(f'[SEND] cmd_servo ({self.cmd_servo_topic}): {state}')

    def _handle_mode(self, args):
        if len(args) != 1 or args[0] not in ('1', '2'):
            print("Error: Usage: 'mode 1' (position) or 'mode 2' (circular)")
            return
        mode = int(args[0])
        msg = Int32()
        msg.data = mode
        self._mode_pub.publish(msg)
        print(f'[SEND] cmd_mode ({self.cmd_mode_topic}): {mode}')


def main(args=None):
    rclpy.init(args=args)
    node = HandTargetSender()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Hand Target Sender shutting down...')
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
