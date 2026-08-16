#!/usr/bin/env python3
"""
Trajectory Subscriber Node

PC1에서 전송한 JointTrajectory Topic을 받아서
로컬 JointTrajectoryController에게 Action으로 전달하는 노드

PC2 (Robot Computer)에서 실행:
- PC1 → /trajectory_commands Topic → 이 노드
- 이 노드 → FollowJointTrajectory Action → JointTrajectoryController → Robot

Author: Chanyoung Ahn
Date: 2025
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory
import time


class TrajectorySubscriber(Node):
    """
    JointTrajectory Topic을 받아서 로컬 Controller에게 Action으로 전달
    """

    def __init__(self):
        super().__init__('trajectory_subscriber')

        # Callback group for concurrent execution
        self.cb_group = ReentrantCallbackGroup()

        # Parameters
        self.declare_parameter('topic_name', '/trajectory_commands')
        self.declare_parameter('action_name', '/fr3_arm_controller/follow_joint_trajectory')
        self.declare_parameter('timeout_sec', 60.0)

        topic_name = self.get_parameter('topic_name').get_parameter_value().string_value
        action_name = self.get_parameter('action_name').get_parameter_value().string_value
        self.timeout_sec = self.get_parameter('timeout_sec').get_parameter_value().double_value

        # Topic Subscriber (PC1에서 수신)
        self.traj_sub = self.create_subscription(
            JointTrajectory,
            topic_name,
            self.trajectory_callback,
            10,
            callback_group=self.cb_group
        )

        # Action Client (로컬 Controller에게 전송)
        self.action_client = ActionClient(
            self,
            FollowJointTrajectory,
            action_name,
            callback_group=self.cb_group
        )

        # Status tracking
        self.trajectory_count = 0
        self.is_executing = False

        self.get_logger().info('=' * 70)
        self.get_logger().info('Trajectory Subscriber Node Started')
        self.get_logger().info(f'  Topic Subscriber: {topic_name}')
        self.get_logger().info(f'  Action Client: {action_name}')
        self.get_logger().info('  Waiting for controller...')

        # Wait for controller action server
        if not self.action_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error('=' * 70)
            self.get_logger().error('[ERROR] Controller action server not available!')
            self.get_logger().error(f'  Expected: {action_name}')
            self.get_logger().error('  Please check if JointTrajectoryController is running.')
            self.get_logger().error('=' * 70)
        else:
            self.get_logger().info('[READY] Controller connected successfully')
            self.get_logger().info('  Waiting for trajectories from PC1...')
            self.get_logger().info('=' * 70)

    def trajectory_callback(self, trajectory: JointTrajectory):
        """
        PC1에서 전송한 trajectory 수신 → 로컬 controller로 전달
        """
        if self.is_executing:
            self.get_logger().warn('[BUSY] Already executing trajectory, skipping...')
            return

        self.trajectory_count += 1
        n_points = len(trajectory.points)

        self.get_logger().info('')
        self.get_logger().info('=' * 70)
        self.get_logger().info(f'[RECEIVED] Trajectory #{self.trajectory_count} from PC1')
        self.get_logger().info(f'  Joints: {trajectory.joint_names}')
        self.get_logger().info(f'  Waypoints: {n_points}')
        if n_points > 0:
            duration = (
                trajectory.points[-1].time_from_start.sec +
                trajectory.points[-1].time_from_start.nanosec * 1e-9
            )
            self.get_logger().info(f'  Duration: {duration:.3f}s')
        self.get_logger().info('=' * 70)

        # Action Goal 생성
        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory = trajectory

        self.get_logger().info('[SENDING] Forwarding to local controller...')

        # Action 전송 (비동기)
        self.is_executing = True
        send_goal_future = self.action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )
        send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        """Action Goal이 accept되었는지 확인"""
        goal_handle = future.result()

        if not goal_handle.accepted:
            self.get_logger().error('[REJECTED] Goal was rejected by controller')
            self.is_executing = False
            return

        self.get_logger().info('[ACCEPTED] Controller started execution')

        # Result 대기
        get_result_future = goal_handle.get_result_async()
        get_result_future.add_done_callback(self.get_result_callback)

    def feedback_callback(self, feedback_msg):
        """실행 중 feedback 수신 (선택적)"""
        # feedback = feedback_msg.feedback
        # self.get_logger().info(f'[FEEDBACK] Executing...', throttle_duration_sec=1.0)
        pass

    def get_result_callback(self, future):
        """Action 완료 result 수신"""
        result = future.result().result
        self.is_executing = False

        self.get_logger().info('')
        self.get_logger().info('=' * 70)

        if result.error_code == FollowJointTrajectory.Result.SUCCESSFUL:
            self.get_logger().info(f'[SUCCESS] Trajectory #{self.trajectory_count} executed successfully')
        else:
            self.get_logger().error(f'[FAILED] Trajectory #{self.trajectory_count} execution failed')
            self.get_logger().error(f'  Error code: {result.error_code}')
            self.get_logger().error(f'  Error message: {result.error_string}')

        self.get_logger().info('  Ready for next trajectory...')
        self.get_logger().info('=' * 70)
        self.get_logger().info('')


def main(args=None):
    rclpy.init(args=args)
    node = TrajectorySubscriber()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Trajectory Subscriber shutting down...')
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
