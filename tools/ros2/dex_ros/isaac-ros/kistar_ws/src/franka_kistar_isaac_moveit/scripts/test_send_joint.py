#!/usr/bin/env python3
import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


class TestSendJoint(Node):
    def __init__(self):
        super().__init__("test_send_joint")

        # Isaac에서 joint_states 받아오기
        self.js_sub = self.create_subscription(
            JointState,
            "isaac_joint_states",  # Isaac 쪽 PublishJointState topicName
            self.joint_state_callback,
            10,
        )

        # Isaac으로 joint_commands 보내기
        self.cmd_pub = self.create_publisher(
            JointState,
            "isaac_joint_commands",  # Isaac 쪽 SubscribeJointState topicName
            10,
        )

        # joint name / 초기 각도 저장용
        self.joint_names = None
        self.q0 = None
        self.t = 0.0

        # 100 Hz 정도로 명령 보내기
        self.timer = self.create_timer(0.01, self.timer_callback)

        self.get_logger().info("TestSendJoint node started.")

    def joint_state_callback(self, msg: JointState):
        # 처음 한 번만 joint name / 초기 각도 저장
        if self.joint_names is None:
            if not msg.name or not msg.position:
                self.get_logger().warn(
                    "Received empty JointState. Waiting for valid data..."
                )
                return

            self.joint_names = list(msg.name)
            self.q0 = list(msg.position)
            self.get_logger().info(
                f"Got joint names from Isaac (N={len(self.joint_names)}). "
                f"First few: {self.joint_names[:5]}"
            )

    def timer_callback(self):
        # 아직 joint_states를 못 받았으면 아무 것도 하지 않음
        if self.joint_names is None or self.q0 is None:
            return

        self.t += 0.01

        # 기본은 초기 자세 q0 유지하면서, 첫 번째 조인트만 살짝 sin 파로 흔듦
        q_cmd = list(self.q0)
        if len(q_cmd) > 0:
            q_cmd[0] = self.q0[0] + 0.2 * math.sin(0.5 * self.t)

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position = q_cmd

        self.cmd_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = TestSendJoint()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
