#!/usr/bin/env python3
"""
Pose Commander Node

Interactive end-effector pose control with MoveIt planning

Features:
- CUI input for target pose (Quaternion: x y z qx qy qz qw)
- MoveIt planning via MoveGroup action
- RViz trajectory visualization (DisplayTrajectory)
- User confirmation (CUI)
- Trajectory execution via trajectory_bridge (/left|right_arm_controller/
  follow_joint_trajectory), which remaps joint names and forwards to the robot
  PC's /fr3_{l,r}_arm_controller action server; the robot's REAL terminal
  status is awaited and reported here.

Requires dual_fr3_kistar_planning_pc_v2.launch.py (move_group + bridges) to be
running. WARNING: never stream /franka/{side}/q_target or ee_target_world to
the SAME arm while executing a MoveIt trajectory — the robot-side receivers
overwrite each other's targets.

Author: Chanyoung Ahn
Date: 2025
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    DisplayTrajectory,
    Constraints,
    PositionConstraint,
    OrientationConstraint,
    BoundingVolume,
    RobotState,
)
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from shape_msgs.msg import SolidPrimitive
import threading
import math
import time


class PoseCommander(Node):
    """
    Interactive pose commander for MoveIt planning
    """

    def __init__(self):
        super().__init__('pose_commander')

        # Parameters
        self.declare_parameter('gui', True)
        self.declare_parameter('planning_group', 'fr3_arm')
        self.declare_parameter('end_effector_link', 'fr3_hand_tcp')
        self.declare_parameter('planning_time', 5.0)
        self.declare_parameter('reference_frame', 'world')
        # FollowJointTrajectory action server name. Override per planning_group:
        #   left_arm  -> /left_arm_controller/follow_joint_trajectory
        #   right_arm -> /right_arm_controller/follow_joint_trajectory
        self.declare_parameter(
            'traj_action', '/fr3_arm_controller/follow_joint_trajectory'
        )
        # 부드러운(느린) 실행용 스케일 — 0.5 → 0.1 로 낮춤 (가속 낮추면 더 부드럽다).
        self.declare_parameter('max_velocity_scaling', 0.1)
        self.declare_parameter('max_acceleration_scaling', 0.1)
        # OMPL 은 랜덤 플래너라 같은 목표도 어떤 시도는 사후 충돌검증에서
        # INVALID_MOTION_PLAN(-2) 로 버려진다 → 성공할 때까지 재시도할 횟수.
        self.declare_parameter('planning_retries', 8)

        self.gui = self.get_parameter('gui').value
        self.planning_group = self.get_parameter('planning_group').value
        self.ee_link = self.get_parameter('end_effector_link').value
        self.planning_time = self.get_parameter('planning_time').value
        self.ref_frame = self.get_parameter('reference_frame').value
        self.traj_action = self.get_parameter('traj_action').value
        self.vel_scaling = float(self.get_parameter('max_velocity_scaling').value)
        self.acc_scaling = float(self.get_parameter('max_acceleration_scaling').value)
        self.planning_retries = int(self.get_parameter('planning_retries').value)

        # Callback group for threading
        self.cb_group = ReentrantCallbackGroup()

        # Clients and Publishers
        self._setup_clients()

        # Start input thread
        self.input_thread = threading.Thread(
            target=self._input_loop,
            daemon=True
        )
        self.input_thread.start()

        self.get_logger().info('=' * 70)
        self.get_logger().info('Pose Commander Started')
        self.get_logger().info(f'  Mode: {"GUI" if self.gui else "CUI"}')
        self.get_logger().info(f'  Planning group: {self.planning_group}')
        self.get_logger().info(f'  End-effector: {self.ee_link}')
        self.get_logger().info(f'  Reference frame: {self.ref_frame}')
        self.get_logger().info(f'  Trajectory action: {self.traj_action}')
        self.get_logger().info(f'  Planning timeout: {self.planning_time}s')
        self.get_logger().info(
            f'  Vel/Acc scaling: {self.vel_scaling}/{self.acc_scaling}  '
            f'(부드러운 실행)')
        self.get_logger().info(f'  Planning retries: {self.planning_retries}')
        self.get_logger().info('=' * 70)

    def _setup_clients(self):
        """Setup action clients and publishers"""

        # MoveGroup action client (planning)
        self.move_group_client = ActionClient(
            self,
            MoveGroup,
            '/move_action',
            callback_group=self.cb_group
        )

        # DisplayTrajectory publisher (RViz visualization)
        self.display_pub = self.create_publisher(
            DisplayTrajectory,
            '/display_planned_path',
            10
        )

        # FollowJointTrajectory action client (trajectory_forwarder).
        # Name is parameterised so dual-arm groups can target the correct controller.
        self.traj_client = ActionClient(
            self,
            FollowJointTrajectory,
            self.traj_action,
            callback_group=self.cb_group
        )

        self.get_logger().info('Waiting for MoveGroup action server...')
        if not self.move_group_client.wait_for_server(timeout_sec=15.0):
            raise RuntimeError(
                'MoveGroup action server (/move_action) not available within 15s — '
                'is dual_fr3_kistar_planning_pc_v2.launch.py running?'
            )
        if not self.traj_client.wait_for_server(timeout_sec=15.0):
            raise RuntimeError(
                f'Trajectory action server ({self.traj_action}) not available within '
                '15s — is the trajectory_bridge up (part of the v2 launch)?'
            )
        self.get_logger().info('Action servers connected')
        self.get_logger().warn(
            'never stream /franka/{side}/q_target or ee_target_world to the SAME '
            'arm while executing a trajectory from here.'
        )

    def _input_loop(self):
        """Input thread - blocking CUI input"""
        while rclpy.ok():
            print("\n" + "=" * 70)
            print("Enter target pose (Quaternion):")
            print("  Format: x y z qx qy qz qw")
            print("  Example: 0.5 0.0 0.4 0 0.707 0 0.707")
            print("  (or 'quit' to exit)")
            print("=" * 70)

            user_input = input(">> ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                self.get_logger().info('Shutting down...')
                rclpy.shutdown()
                break

            # Parse input
            try:
                values = [float(x) for x in user_input.split()]
                if len(values) != 7:
                    print(f"Error: Expected 7 values, got {len(values)}")
                    continue

                x, y, z, qx, qy, qz, qw = values

                # Normalize quaternion
                norm = math.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
                if norm < 0.01:
                    print("Error: Invalid quaternion (near-zero norm)")
                    continue

                qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm

                # Create pose
                target_pose = self._create_pose(x, y, z, qx, qy, qz, qw)

                # Plan and execute
                self._plan_and_execute(target_pose)

            except ValueError as e:
                print(f"Error parsing input: {e}")
            except Exception as e:
                self.get_logger().error(f"Error: {e}")
                import traceback
                traceback.print_exc()

    def _create_pose(self, x, y, z, qx, qy, qz, qw):
        """Create PoseStamped message"""
        pose = PoseStamped()
        pose.header.frame_id = self.ref_frame
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position = Point(x=x, y=y, z=z)
        pose.pose.orientation = Quaternion(x=qx, y=qy, z=qz, w=qw)
        return pose

    def _plan_and_execute(self, target_pose: PoseStamped):
        """Plan trajectory to target pose and execute if confirmed"""

        print(f"\n[PLANNING] Target pose:")
        print(f"  Position: ({target_pose.pose.position.x:.3f}, "
              f"{target_pose.pose.position.y:.3f}, "
              f"{target_pose.pose.position.z:.3f})")
        print(f"  Orientation: ({target_pose.pose.orientation.x:.3f}, "
              f"{target_pose.pose.orientation.y:.3f}, "
              f"{target_pose.pose.orientation.z:.3f}, "
              f"{target_pose.pose.orientation.w:.3f})")

        # Build the (deterministic) planning goal once, then retry planning:
        # OMPL is a randomized planner, so the SAME goal can return a valid path
        # on one attempt and INVALID_MOTION_PLAN(-2) on the next (a randomly-found
        # path clips the static collision scene and fails post-plan validation).
        # Retry until SUCCESS instead of giving up on the first -2/-1.
        goal = self._build_plan_goal(target_pose)

        result = None
        for attempt in range(1, self.planning_retries + 1):
            print(f"[PLANNING] attempt {attempt}/{self.planning_retries} — "
                  "requesting trajectory from MoveGroup...")
            r = self._request_plan(goal)
            if r is not None and r.error_code.val == 1:  # SUCCESS = 1
                result = r
                break
            code = r.error_code.val if r is not None else 'timeout/None'
            print(f"[PLANNING] attempt {attempt} 실패 (error_code={code}) — 재시도")
            if r is not None:
                self._print_error_code(r.error_code.val)
            time.sleep(0.2)

        if result is None:
            print(f"[ERROR] Planning failed after {self.planning_retries} retries "
                  "— 실행 중단")
            return

        print(f"[SUCCESS] Planning succeeded!")
        trajectory = result.planned_trajectory
        n_points = len(trajectory.joint_trajectory.points)
        print(f"  Waypoints: {n_points}")

        if n_points > 0:
            duration = (
                trajectory.joint_trajectory.points[-1].time_from_start.sec +
                trajectory.joint_trajectory.points[-1].time_from_start.nanosec * 1e-9
            )
            print(f"  Duration: {duration:.3f}s")

        # Visualize in RViz
        if self.gui:
            self._publish_display_trajectory(result)
            print("[RVIZ] Trajectory published to /display_planned_path")

        # User confirmation
        if not self._confirm_execution():
            print("[CANCELED] Execution canceled by user")
            return

        # Execute trajectory
        self._execute_trajectory(trajectory.joint_trajectory)

    def _build_plan_goal(self, target_pose: PoseStamped) -> MoveGroup.Goal:
        """pose 목표에 대한 plan_only MoveGroup goal 생성 (재시도마다 재사용)."""
        goal = MoveGroup.Goal()
        goal.request.group_name = self.planning_group
        # 첫 시도 성공률을 높이기 위해 내부 시도 횟수도 늘림 (5 → 10).
        goal.request.num_planning_attempts = 10
        goal.request.allowed_planning_time = self.planning_time
        goal.request.max_velocity_scaling_factor = self.vel_scaling
        goal.request.max_acceleration_scaling_factor = self.acc_scaling

        goal_constraint = Constraints()

        pos_constraint = PositionConstraint()
        pos_constraint.header = target_pose.header
        pos_constraint.link_name = self.ee_link
        pos_constraint.constraint_region = BoundingVolume()
        sphere = SolidPrimitive()
        sphere.type = SolidPrimitive.SPHERE
        sphere.dimensions = [0.0001]  # Tight tolerance
        pos_constraint.constraint_region.primitives = [sphere]
        pos_constraint.constraint_region.primitive_poses = [target_pose.pose]
        pos_constraint.weight = 1.0

        ori_constraint = OrientationConstraint()
        ori_constraint.header = target_pose.header
        ori_constraint.link_name = self.ee_link
        ori_constraint.orientation = target_pose.pose.orientation
        ori_constraint.absolute_x_axis_tolerance = 0.01
        ori_constraint.absolute_y_axis_tolerance = 0.01
        ori_constraint.absolute_z_axis_tolerance = 0.01
        ori_constraint.weight = 1.0

        goal_constraint.position_constraints = [pos_constraint]
        goal_constraint.orientation_constraints = [ori_constraint]
        goal.request.goal_constraints = [goal_constraint]
        goal.planning_options.plan_only = True
        return goal

    def _request_plan(self, goal: MoveGroup.Goal):
        """MoveGroup 에 plan 요청 1회 → result (또는 timeout/거부 시 None)."""
        future = self.move_group_client.send_goal_async(goal)
        timeout = self.planning_time + 2.0
        start_time = time.time()
        while not future.done() and (time.time() - start_time) < timeout:
            time.sleep(0.01)
        if not future.done():
            print("[ERROR] Planning request timed out")
            return None
        goal_handle = future.result()
        if not goal_handle.accepted:
            print("[ERROR] Planning goal rejected")
            return None
        result_future = goal_handle.get_result_async()
        start_time = time.time()
        while not result_future.done() and (time.time() - start_time) < timeout:
            time.sleep(0.01)
        if not result_future.done():
            print("[ERROR] Planning result timed out")
            return None
        return result_future.result().result

    def _publish_display_trajectory(self, planning_result):
        """Publish trajectory to RViz for visualization"""
        display_traj = DisplayTrajectory()
        display_traj.model_id = self.planning_group
        display_traj.trajectory_start = planning_result.trajectory_start
        display_traj.trajectory.append(planning_result.planned_trajectory)

        # Give RViz time to process
        time.sleep(0.1)
        self.display_pub.publish(display_traj)
        time.sleep(0.5)  # Allow RViz to render before asking for confirmation

    def _confirm_execution(self):
        """Ask user confirmation for execution"""
        while True:
            response = input("\nExecute trajectory? (y/n): ").strip().lower()
            if response == 'y':
                return True
            elif response == 'n':
                return False
            else:
                print("Please enter 'y' or 'n'")

    def _execute_trajectory(self, joint_trajectory):
        """Send trajectory to the bridge and wait for the robot's real result."""
        print(f"[EXECUTING] Sending trajectory to {self.traj_action} ...")

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = joint_trajectory

        future = self.traj_client.send_goal_async(goal)

        # Wait for goal acceptance
        timeout = 5.0
        start_time = time.time()
        while not future.done() and (time.time() - start_time) < timeout:
            time.sleep(0.01)

        if not future.done():
            print("[ERROR] Trajectory execution request timed out")
            return

        goal_handle = future.result()
        if not goal_handle.accepted:
            print("[ERROR] Trajectory execution rejected by the bridge.")
            print("        Common causes: /joint_states_relay stale/absent "
                  "(robot not streaming, or fake mode), see the bridge log.")
            return

        # The bridge forwards to the robot and returns its REAL terminal status.
        # The robot prepends a current-pose ramp (1-10s) and runs safety-limited
        # (speed_factor), so execution takes much longer than the planned
        # duration — wait generously instead of guessing tightly.
        plan_duration = 0.0
        if joint_trajectory.points:
            last = joint_trajectory.points[-1].time_from_start
            plan_duration = last.sec + last.nanosec * 1e-9
        result_timeout = max(30.0, 3.0 * plan_duration + 20.0)
        print(f"[EXECUTING] Robot accepted. Waiting for completion "
              f"(up to {result_timeout:.0f}s; ramp + speed_factor make this slow)...")

        result_future = goal_handle.get_result_async()
        start_time = time.time()
        while not result_future.done() and (time.time() - start_time) < result_timeout:
            time.sleep(0.05)

        if not result_future.done():
            print(f"[WARN] No result within {result_timeout:.0f}s — the robot may "
                  "still be executing. Check the robot-side log (q_pos vs q_tar).")
            return

        result = result_future.result().result
        if result.error_code == FollowJointTrajectory.Result.SUCCESSFUL:
            print(f"[SUCCESS] Robot reported completion: '{result.error_string}'")
        else:
            print(f"[ERROR] Execution failed (code {result.error_code}): "
                  f"'{result.error_string}'")

    def _print_error_code(self, code):
        """Print MoveIt error code description"""
        error_codes = {
            -1: "PLANNING_FAILED",
            -2: "INVALID_MOTION_PLAN",
            -3: "MOTION_PLAN_INVALIDATED_BY_ENVIRONMENT_CHANGE",
            -4: "CONTROL_FAILED",
            -5: "UNABLE_TO_AQUIRE_SENSOR_DATA",
            -6: "TIMED_OUT",
            -7: "PREEMPTED",
            -10: "START_STATE_IN_COLLISION",
            -11: "START_STATE_VIOLATES_PATH_CONSTRAINTS",
            -12: "GOAL_IN_COLLISION",
            -13: "GOAL_VIOLATES_PATH_CONSTRAINTS",
            -14: "GOAL_CONSTRAINTS_VIOLATED",
            -15: "INVALID_GROUP_NAME",
            -16: "INVALID_GOAL_CONSTRAINTS",
            -17: "INVALID_ROBOT_STATE",
            -18: "INVALID_LINK_NAME",
            -19: "INVALID_OBJECT_NAME",
            -20: "FRAME_TRANSFORM_FAILURE",
            -21: "COLLISION_CHECKING_UNAVAILABLE",
            -22: "ROBOT_STATE_STALE",
            -23: "SENSOR_INFO_STALE",
            -31: "NO_IK_SOLUTION",
        }
        desc = error_codes.get(code, f"UNKNOWN_ERROR ({code})")
        print(f"  Error: {desc}")


def main(args=None):
    rclpy.init(args=args)
    node = PoseCommander()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Pose Commander shutting down...')
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
