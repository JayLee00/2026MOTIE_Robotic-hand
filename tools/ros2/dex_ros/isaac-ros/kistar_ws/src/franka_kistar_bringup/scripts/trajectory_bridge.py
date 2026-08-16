#!/usr/bin/env python3
"""Trajectory bridge: remap joint names and forward to robot's action server.

MoveIt sends trajectories with URDF joint names (e.g. left_fr3_joint1).
Robot PC expects its own naming (e.g. fr3_l_joint1).
This bridge remaps and forwards, then returns the robot's result to MoveIt.

Safety: subscribes to /joint_states_relay and rejects FollowJointTrajectory goals when
the relay stream is stale beyond max_age_sec (default 0.5s). Setting max_age_sec=0.0
disables the guard entirely (operator escape). The freshness-check subscription runs
on a dedicated MutuallyExclusiveCallbackGroup so it is not starved by goal-callback
queueing under load.
"""
import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, ActionClient, GoalResponse, CancelResponse
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from control_msgs.action import FollowJointTrajectory
from sensor_msgs.msg import JointState


class TrajectoryBridge(Node):
    def __init__(self):
        super().__init__("trajectory_bridge")

        self.declare_parameter("moveit_action", "/left_arm_controller/follow_joint_trajectory")
        self.declare_parameter("robot_action", "/fr3_l_arm_controller/follow_joint_trajectory")
        # Reverse remap: URDF names → robot names (comma-separated "src:dst")
        self.declare_parameter("joint_name_remap", "left_fr3_:fr3_l_,left_:l_")
        # Staleness guard (safety): max age of /joint_states_relay before goals are rejected.
        # 0.0 disables the guard entirely (operator escape).
        self.declare_parameter("max_age_sec", 0.5)
        self.declare_parameter("relay_topic", "/joint_states_relay")

        moveit_action = self.get_parameter("moveit_action").get_parameter_value().string_value
        robot_action = self.get_parameter("robot_action").get_parameter_value().string_value
        remap_str = self.get_parameter("joint_name_remap").get_parameter_value().string_value
        self._max_age_sec = float(self.get_parameter("max_age_sec").get_parameter_value().double_value)
        relay_topic = self.get_parameter("relay_topic").get_parameter_value().string_value

        self._remap_rules = []
        if remap_str:
            for rule in remap_str.strip().split(","):
                src, dst = rule.strip().split(":")
                self._remap_rules.append((src, dst))
            self._remap_rules.sort(key=lambda r: -len(r[0]))

        # Freshness state — written by relay callback, read by goal_callback.
        self._last_relay_stamp_ns = None

        # Dedicated callback group so the relay subscription is not starved by
        # goal_callback execution under load.
        self._relay_cb_group = MutuallyExclusiveCallbackGroup()
        self._relay_sub = self.create_subscription(
            JointState,
            relay_topic,
            self._on_relay,
            10,
            callback_group=self._relay_cb_group,
        )

        # ActionServer: MoveIt connects here
        self._server = ActionServer(
            self, FollowJointTrajectory, moveit_action,
            execute_callback=self._execute,
            goal_callback=self._goal_callback,
            cancel_callback=lambda _: CancelResponse.ACCEPT,
        )

        # ActionClient: forward to robot
        self._client = ActionClient(self, FollowJointTrajectory, robot_action)

        self.get_logger().info(f"Bridge: {moveit_action} -> {robot_action}")
        self.get_logger().info(f"Remap: {self._remap_rules}")
        if self._max_age_sec > 0.0:
            self.get_logger().info(
                f"Staleness guard ENABLED: relay_topic='{relay_topic}' max_age_sec={self._max_age_sec:.3f}s"
            )
        else:
            self.get_logger().warn(
                "Staleness guard DISABLED (max_age_sec=0.0). Operator-mode override is active; "
                "goals will be accepted regardless of /joint_states_relay freshness."
            )

    # ----- Staleness guard helpers -----

    def _on_relay(self, msg):
        # Use the wall-clock receive time so a stalled upstream stamp does not look fresh.
        self._last_relay_stamp_ns = self.get_clock().now().nanoseconds

    def _relay_age_sec(self):
        if self._last_relay_stamp_ns is None:
            return None
        now_ns = self.get_clock().now().nanoseconds
        return max(0.0, (now_ns - self._last_relay_stamp_ns) * 1e-9)

    def _goal_callback(self, _goal_request):
        max_age = self._max_age_sec
        # Operator escape: max_age_sec == 0.0 disables the guard entirely.
        if max_age <= 0.0:
            return GoalResponse.ACCEPT
        age = self._relay_age_sec()
        if age is None:
            self.get_logger().warn(
                f"REJECT goal: /joint_states_relay has not been seen yet "
                f"(age=None, configured max_age_sec={max_age:.3f}s). "
                f"To bypass, set max_age_sec:=0.0."
            )
            return GoalResponse.REJECT
        if age > max_age:
            self.get_logger().warn(
                f"REJECT goal: /joint_states_relay is stale "
                f"(measured age={age:.3f}s > configured max_age_sec={max_age:.3f}s). "
                f"To bypass, set max_age_sec:=0.0."
            )
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    # ----- Trajectory forwarding -----

    def _remap_name(self, name):
        for src, dst in self._remap_rules:
            if name.startswith(src):
                return dst + name[len(src):]
        return name

    def _remap_trajectory(self, trajectory):
        """Remap joint names in the trajectory."""
        trajectory.joint_names = [self._remap_name(n) for n in trajectory.joint_names]
        return trajectory

    async def _execute(self, goal_handle):
        trajectory = goal_handle.request.trajectory
        original_names = list(trajectory.joint_names)
        remapped = self._remap_trajectory(trajectory)

        self.get_logger().info("=" * 60)
        self.get_logger().info(f"Waypoints: {len(trajectory.points)}")
        self.get_logger().info(f"Joints (original): {original_names}")
        self.get_logger().info(f"Joints (remapped): {list(remapped.joint_names)}")
        if trajectory.points:
            p0 = trajectory.points[0]
            pN = trajectory.points[-1]
            self.get_logger().info(f"Start positions: {[round(v,4) for v in p0.positions]}")
            self.get_logger().info(f"End positions:   {[round(v,4) for v in pN.positions]}")
            dur = pN.time_from_start.sec + pN.time_from_start.nanosec * 1e-9
            self.get_logger().info(f"Duration: {dur:.3f}s")
        self.get_logger().info("=" * 60)

        # Wait for robot action server
        if not self._client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Robot action server not available!")
            goal_handle.abort()
            result = FollowJointTrajectory.Result()
            result.error_code = FollowJointTrajectory.Result.INVALID_GOAL
            result.error_string = "Robot action server not available"
            return result

        # Forward to robot
        robot_goal = FollowJointTrajectory.Goal()
        robot_goal.trajectory = remapped

        send_future = self._client.send_goal_async(robot_goal)
        robot_goal_handle = await send_future

        if not robot_goal_handle.accepted:
            self.get_logger().error("Robot rejected trajectory!")
            goal_handle.abort()
            result = FollowJointTrajectory.Result()
            result.error_code = FollowJointTrajectory.Result.INVALID_GOAL
            result.error_string = "Robot rejected trajectory"
            return result

        self.get_logger().info("Robot accepted, waiting for execution...")
        result_future = robot_goal_handle.get_result_async()
        robot_result = await result_future

        # Pass robot's result back to MoveIt
        result = robot_result.result
        if result.error_code == FollowJointTrajectory.Result.SUCCESSFUL:
            self.get_logger().info("Execution successful")
            goal_handle.succeed()
        else:
            self.get_logger().warn(f"Execution failed: {result.error_string}")
            goal_handle.abort()

        return result


def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryBridge()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
