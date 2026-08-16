#!/usr/bin/env python3
import time

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, GoalResponse, CancelResponse

from sensor_msgs.msg import JointState
from control_msgs.action import FollowJointTrajectory


class IsaacMoveItBridge(Node):
    """MoveIt <-> IsaacSim 브릿지 노드.

    - /isaac_joint_states  → /joint_states 로 재전송 (타임스탬프 벽시계로 교체)
    - /fr3_arm_controller/follow_joint_trajectory 액션 → /isaac_joint_commands 로 포워딩
    """

    def __init__(self):
        super().__init__("isaac_moveit_bridge")

        # 토픽 이름 파라미터
        self.declare_parameter("isaac_joint_states_topic", "/isaac_joint_states")
        self.declare_parameter("isaac_joint_commands_topic", "/isaac_joint_commands")
        self.declare_parameter("joint_states_topic", "/joint_states")

        isaac_js_topic = (
            self.get_parameter("isaac_joint_states_topic")
            .get_parameter_value()
            .string_value
        )
        joint_states_topic = (
            self.get_parameter("joint_states_topic").get_parameter_value().string_value
        )
        isaac_cmd_topic = (
            self.get_parameter("isaac_joint_commands_topic")
            .get_parameter_value()
            .string_value
        )

        # Isaac → MoveIt: joint_states 브릿지
        self.js_pub = self.create_publisher(JointState, joint_states_topic, 10)
        self.js_sub = self.create_subscription(
            JointState, isaac_js_topic, self._js_cb, 10
        )

        # MoveIt → Isaac: FollowJointTrajectory 액션 서버 + 커맨드 퍼블리셔
        self.cmd_pub = self.create_publisher(JointState, isaac_cmd_topic, 10)
        self.action_server = ActionServer(
            self,
            FollowJointTrajectory,
            "/fr3_arm_controller/follow_joint_trajectory",
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
        )

        self.get_logger().info("IsaacMoveItBridge started.")

    # ===== joint_states 브릿지 =====
    def _js_cb(self, msg: JointState):
        # IsaacSim 시뮬레이션 타임 말고, MoveIt이 쓰는 벽시계로 덮어쓰기
        msg.header.stamp = self.get_clock().now().to_msg()
        self.js_pub.publish(msg)
        # 디버그용으로 한 번만 보고 싶으면 주석 풀기
        # self.get_logger().info(f"[JS] names={list(msg.name)} pos={list(msg.position)}")

    # ===== FollowJointTrajectory 액션 서버 =====
    def goal_callback(self, goal_request):
        traj = goal_request.trajectory
        self.get_logger().info(
            f"[Action] Goal received: {len(traj.points)} points, "
            f"joints={list(traj.joint_names)}"
        )
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        self.get_logger().info("[Action] Cancel requested")
        return CancelResponse.ACCEPT

    def execute_callback(self, goal_handle):
        traj = goal_handle.request.trajectory
        n_points = len(traj.points)
        self.get_logger().info(f"[Action] Executing trajectory ({n_points} points)")

        last_t = 0.0
        for i, pt in enumerate(traj.points):
            if goal_handle.is_cancel_requested:
                self.get_logger().info("[Action] Goal canceled")
                goal_handle.canceled()
                return FollowJointTrajectory.Result()

            cmd = JointState()
            cmd.header.stamp = self.get_clock().now().to_msg()
            cmd.name = list(traj.joint_names)
            cmd.position = list(pt.positions)
            # velocity/effort는 Isaac에서 안 쓰면 생략 가능

            self.cmd_pub.publish(cmd)
            self.get_logger().info(f"[Action] Sent point {i+1}/{n_points}")

            # time_from_start 기준으로 다음 포인트까지 sleep
            t = pt.time_from_start.sec + pt.time_from_start.nanosec * 1e-9
            dt = max(t - last_t, 0.02)  # 최소 20ms
            time.sleep(dt)
            last_t = t

        self.get_logger().info("[Action] Trajectory done")

        result = FollowJointTrajectory.Result()
        result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
        goal_handle.succeed()
        return result


def main(args=None):
    rclpy.init(args=args)
    node = IsaacMoveItBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
