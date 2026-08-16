#!/usr/bin/env python3
"""
제어 PC 팔 상태 → /joint_states  (QoS 브리지)

MoveIt move_group / robot_state_publisher 는 현재 로봇 자세를 알기 위해
/joint_states (JointState) 가 필요하다. 제어 PC는 상태를 best_effort 로
발행하지만 move_group 의 CurrentStateMonitor 는 기본 reliable 로 구독하므로
QoS 가 안 맞아 상태를 못 받는다. 이 노드가 그 사이를 잇는다.

기본 소스 = /franka/right/joint_states (팔 7관절, name=fr3_joint1..7 → URDF와 일치).
  ⚠️ /joint_states_r 는 이름이 fr3_r_joint1.. (접두사) 라 URDF와 안 맞으므로 쓰지 않는다.
--source 로 다른 토픽 지정 가능. 메시지 name/position 을 그대로 통과(변환 없음).
"""
import argparse

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState


class JointStateBridge(Node):
    def __init__(self, source_topic: str):
        super().__init__('franka_joint_state_relay')
        sub_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        # move_group CurrentStateMonitor 기본 구독 = RELIABLE
        self._pub = self.create_publisher(JointState, '/joint_states', 10)
        self.create_subscription(JointState, source_topic, self._cb, sub_qos)
        self.get_logger().info(
            f'{source_topic} (best_effort) -> /joint_states (reliable) bridge started')

    def _cb(self, msg: JointState):
        msg.header.stamp = self.get_clock().now().to_msg()
        self._pub.publish(msg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', default='/franka/right/joint_states',
                    help='제어 PC 팔 상태 토픽 (기본: /franka/right/joint_states, fr3_joint1..7)')
    args, _ = ap.parse_known_args()

    rclpy.init()
    node = JointStateBridge(args.source)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
