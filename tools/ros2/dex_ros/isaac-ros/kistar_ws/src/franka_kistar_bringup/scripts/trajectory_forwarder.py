#!/usr/bin/env python3
"""
Trajectory Forwarder Node

MoveIt의 FollowJointTrajectory Action을 받아서
JointTrajectory Topic으로 publish하는 노드

PC1 (Planning Computer)에서 실행:
- MoveIt → FollowJointTrajectory Action → 이 노드
- 이 노드 → /trajectory_commands Topic → PC2 (Robot Computer)

Author: Chanyoung Ahn
Date: 2025
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory
from std_msgs.msg import Header


class TrajectoryForwarder(Node):
    """
    FollowJointTrajectory Action을 받아서 JointTrajectory Topic으로 forward
    """

    def __init__(self):
        super().__init__('trajectory_forwarder')

        # Parameters
        self.declare_parameter('action_name', '/fr3_arm_controller/follow_joint_trajectory')
        self.declare_parameter('topic_name', '/trajectory_commands')
        self.declare_parameter('publish_rate', 10.0)  # Hz

        action_name = self.get_parameter('action_name').get_parameter_value().string_value
        topic_name = self.get_parameter('topic_name').get_parameter_value().string_value

        # Action Server (MoveIt이 여기로 연결)
        self.action_server = ActionServer(
            self,
            FollowJointTrajectory,
            action_name,
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback
        )

        # Topic Publisher (PC2로 전송)
        self.traj_pub = self.create_publisher(
            JointTrajectory,
            topic_name,
            10
        )

        # Status tracking
        self.current_goal_handle = None
        self.trajectory_count = 0

        self.get_logger().info('=' * 70)
        self.get_logger().info('Trajectory Forwarder Node Started')
        self.get_logger().info(f'  Action Server: {action_name}')
        self.get_logger().info(f'  Topic Publisher: {topic_name}')
        self.get_logger().info('  Waiting for MoveIt trajectories...')
        self.get_logger().info('=' * 70)

    def goal_callback(self, goal_request):
        """MoveIt으로부터 trajectory goal 수신"""
        trajectory = goal_request.trajectory
        n_points = len(trajectory.points)

        self.get_logger().info('')
        self.get_logger().info('=' * 70)
        self.get_logger().info('[GOAL RECEIVED]')
        self.get_logger().info(f'  Joints: {trajectory.joint_names}')
        self.get_logger().info(f'  Waypoints: {n_points}')
        if n_points > 0:
            duration = (
                trajectory.points[-1].time_from_start.sec +
                trajectory.points[-1].time_from_start.nanosec * 1e-9
            )
            self.get_logger().info(f'  Duration: {duration:.3f}s')
        self.get_logger().info('=' * 70)

        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        """Cancel 요청"""
        self.get_logger().warn('[CANCEL] Trajectory execution canceled')
        return CancelResponse.ACCEPT

    def execute_callback(self, goal_handle):
        """
        MoveIt으로부터 받은 trajectory를 Topic으로 publish
        """
        trajectory = goal_handle.request.trajectory
        self.current_goal_handle = goal_handle
        self.trajectory_count += 1

        self.get_logger().info('')
        self.get_logger().info('[FORWARDING] Publishing trajectory to network...')

        # Trajectory를 Topic으로 publish (PC2로 전송)
        self.traj_pub.publish(trajectory)

        self.get_logger().info(f'[SUCCESS] Trajectory #{self.trajectory_count} published')
        self.get_logger().info(f'  Topic: {self.traj_pub.topic_name}')
        self.get_logger().info(f'  Waypoints: {len(trajectory.points)}')
        self.get_logger().info('')

        # Action은 즉시 성공으로 반환
        # (실제 실행은 PC2에서 이루어지므로)
        goal_handle.succeed()

        result = FollowJointTrajectory.Result()
        result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
        result.error_string = f'Trajectory forwarded to remote execution (trajectory #{self.trajectory_count})'

        return result


def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryForwarder()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Trajectory Forwarder shutting down...')
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
