#!/usr/bin/env python3
"""
GraspExecutor + frame-transform helpers (extracted from robot_executor.py).

Import this module instead of robot_executor when you only need the class:
    from utils.grasp import GraspExecutor, world_to_base
"""

import math
import os
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np
import rclpy
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    BoundingVolume,
    Constraints,
    DisplayTrajectory,
    JointConstraint,
    OrientationConstraint,
    PositionConstraint,
)
from moveit_msgs.srv import GetCartesianPath, GetPositionIK, GetPositionFK
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import Point, PoseStamped, Quaternion
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import Bool, Float32MultiArray, Float64, Float64MultiArray, Int16MultiArray
from sensor_msgs.msg import JointState   # 새 제어 PC 상태 토픽 (/…/right/joint_states)
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration as RosDuration

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))

from utils.step import (
    step_approach, step_approach_descend, step_descend, step_lift,
    step_init_hand, step_close_hand, step_release_hand, step_go_home,
)
from utils.hand import HAND_INIT_ENC, HAND_RELEASE_ENC, HAND_STEPS, HAND_PERIOD
from utils.arm import (
    PLANNING_GROUP, EE_LINK, REF_FRAME, PLANNING_TIME,
    HOME_JOINT_NAMES, HOME_JOINT_VALUES,
)

TRAJ_TOPIC = '/franka/target_trajectory'
# pose_commander.py 와 동일한 FollowJointTrajectory 액션 서버.
# launch_moveit.py 의 trajectory_bridge_right 가 이 액션을 제공하고
# joint 이름을 remap(right_fr3_→fr3_r_)하여 로봇 /fr3_r_arm_controller 로 전달한다.
ARM_TRAJ_ACTION = '/right_arm_controller/follow_joint_trajectory'
DEG_TO_RAW = 8192.0 / 180.0


# ── Frame transform helpers ───────────────────────────────────────────────────

def _quat_to_rotmat(xyzw):
    x, y, z, w = xyzw
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y-z*w),   2*(x*z+y*w)],
        [  2*(x*y+z*w), 1-2*(x*x+z*z),   2*(y*z-x*w)],
        [  2*(x*z-y*w),   2*(y*z+x*w), 1-2*(x*x+y*y)],
    ], dtype=np.float64)


def _rotmat_to_quat(R):
    tr = R[0,0] + R[1,1] + R[2,2]
    if tr > 0:
        s = 0.5 / math.sqrt(tr + 1.0)
        return [float((R[2,1]-R[1,2])*s), float((R[0,2]-R[2,0])*s),
                float((R[1,0]-R[0,1])*s), float(0.25/s)]
    if R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = 2.0 * math.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
        return [float(0.25*s), float((R[0,1]+R[1,0])/s),
                float((R[0,2]+R[2,0])/s), float((R[2,1]-R[1,2])/s)]
    if R[1,1] > R[2,2]:
        s = 2.0 * math.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
        return [float((R[0,1]+R[1,0])/s), float(0.25*s),
                float((R[1,2]+R[2,1])/s), float((R[0,2]-R[2,0])/s)]
    s = 2.0 * math.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
    return [float((R[0,2]+R[2,0])/s), float((R[1,2]+R[2,1])/s),
            float(0.25*s), float((R[1,0]-R[0,1])/s)]


def world_to_base(T_world_base_list, xyz_world, quat_world_xyzw):
    """Transform EE pose from world frame to base frame.

    T_world_base: 4x4 list (world<-base, i.e. p_world = T @ p_base).
    Returns (xyz_base, quat_base_xyzw).
    """
    T_wb = np.array(T_world_base_list, dtype=np.float64)
    R_wb = T_wb[:3, :3]
    t_wb = T_wb[:3, 3]
    R_bw = R_wb.T
    t_bw = -R_bw @ t_wb
    p_b  = R_bw @ np.array(xyz_world, dtype=np.float64) + t_bw
    R_we = _quat_to_rotmat(quat_world_xyzw)
    R_be = R_bw @ R_we
    q_b  = _rotmat_to_quat(R_be)
    return p_b.tolist(), q_b


def base_to_world(T_world_base_list, xyz_base, quat_base_xyzw):
    """Transform EE pose from base frame to world frame.

    T_world_base: 4x4 list (p_world = T @ p_base).
    Returns (xyz_world, quat_world_xyzw).
    """
    T_wb = np.array(T_world_base_list, dtype=np.float64)
    R_wb = T_wb[:3, :3]
    t_wb = T_wb[:3, 3]
    p_w  = R_wb @ np.array(xyz_base, dtype=np.float64) + t_wb
    R_be = _quat_to_rotmat(quat_base_xyzw)
    R_we = R_wb @ R_be
    q_w  = _rotmat_to_quat(R_we)
    return p_w.tolist(), q_w


# ── Node ─────────────────────────────────────────────────────────────────────

class GraspExecutor(Node):

    def __init__(self, summary: dict, execute_mode: str, speed_factor: float,
                 approach_offset: float, summary_json_path: str = '',
                 disable_collision: bool = False):
        super().__init__('grasp_executor')
        self._summary           = summary
        self._mode              = execute_mode
        self._speed             = speed_factor
        self._approach_offset   = summary.get('approach_offset', approach_offset)
        self._summary_json_path = summary_json_path
        self._disable_collision = disable_collision
        self._hand_deg          = None
        self._last_hand_enc     = None
        self._success           = False
        self._current_joints    = None
        self._approach_traj     = None
        self._cb_group          = ReentrantCallbackGroup()
        if disable_collision:
            self.get_logger().warning(
                '[COLLISION] avoid_collisions=False 모드 — collision 검사 비활성화')
        self._setup_ros()
        self._thread = threading.Thread(target=self._run_guarded, daemon=True)
        self._thread.start()

    # ── ROS setup ─────────────────────────────────────────────────────────────

    def _setup_ros(self):
        self._mg_client   = ActionClient(self, MoveGroup, '/move_action',
                                         callback_group=self._cb_group)
        self._cart_client = self.create_client(GetCartesianPath,
                                               '/compute_cartesian_path',
                                               callback_group=self._cb_group)
        self._ik_client   = self.create_client(GetPositionIK, '/compute_ik',
                                               callback_group=self._cb_group)
        self._display_pub = self.create_publisher(
            DisplayTrajectory, '/display_planned_path', 10)
        # ── 새 제어 PC 토픽 (Dual_Arm_Hand_Ctrl) ──────────────────────────────
        # self._hand_pub = self.create_publisher(
        #     Int16MultiArray, '/hand/target_joint', 10)                    # 옛 토픽
        self._hand_pub = self.create_publisher(
            Float32MultiArray, '/hand/right/q_target', 10)                  # 손 목표 (Float32[16] count)
        self._hand_servo_pub = self.create_publisher(
            Bool, '/hand/right/cmd_servo', 10)                             # 손 서보 on/off
        _be_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        # self.create_subscription(Float32MultiArray, '/hand/joint_position',
        #                          self._hand_cb, _be_qos)                  # 옛 상태
        self.create_subscription(JointState, '/hand/right/joint_states',
                                 self._hand_cb, _be_qos)                    # 손 상태 (JointState)
        # self.create_subscription(Float64MultiArray, '/franka/joint_position',
        #                          self._franka_joint_cb, _be_qos)         # 옛 상태
        self.create_subscription(JointState, '/franka/right/joint_states',
                                 self._franka_joint_cb, _be_qos)           # 팔 상태 (JointState)
        # self._franka_target_pub = self.create_publisher(
        #     Float64MultiArray, '/franka/target_joint', 10)                # 옛 토픽
        self._franka_target_pub = self.create_publisher(
            Float64MultiArray, '/franka/right/q_target', 10)               # 팔 목표 (Float64[7] rad)
        self._franka_speed_pub  = self.create_publisher(
            Float64, '/franka/target_speed_factor', 10)
        self._traj_smooth_pub = self.create_publisher(
            JointTrajectory, TRAJ_TOPIC, 1)
        # pose_commander.py 방식 실행용 FollowJointTrajectory 액션 클라이언트.
        # (trajectory_bridge_right 가 제공 — 없으면 q_target 스트리밍으로 fallback)
        self._traj_client = ActionClient(
            self, FollowJointTrajectory, ARM_TRAJ_ACTION,
            callback_group=self._cb_group)

        self.get_logger().info('Waiting for MoveGroup action server...')
        if not self._mg_client.wait_for_server(timeout_sec=15.0):
            self.get_logger().error(
                'MoveGroup 서버를 찾을 수 없습니다 (15 s timeout).\n'
                '  → 컨테이너 내에서 MoveIt이 실행 중인지 확인하세요:\n'
                '    start_robot_gwb.sh 를 먼저 실행한 뒤 다시 시도하세요.')
            rclpy.shutdown(); sys.exit(1)
        self.get_logger().info('Waiting for /compute_cartesian_path service...')
        if not self._cart_client.wait_for_service(timeout_sec=15.0):
            self.get_logger().error(
                '/compute_cartesian_path 서비스를 찾을 수 없습니다 (15 s timeout).')
            rclpy.shutdown(); sys.exit(1)
        self.get_logger().info('Waiting for /compute_ik service...')
        if not self._ik_client.wait_for_service(timeout_sec=15.0):
            self.get_logger().error('/compute_ik 서비스를 찾을 수 없습니다 (15 s timeout).')
            rclpy.shutdown(); sys.exit(1)
        self._fk_client = self.create_client(GetPositionFK, '/compute_fk',
                                             callback_group=self._cb_group)
        self.get_logger().info('Waiting for /compute_fk service...')
        if not self._fk_client.wait_for_service(timeout_sec=15.0):
            self.get_logger().error('/compute_fk 서비스를 찾을 수 없습니다 (15 s timeout).')
            rclpy.shutdown(); sys.exit(1)
        # FollowJointTrajectory 액션 서버 (부드러운 실행). 없으면 q_target fallback.
        self.get_logger().info(f'Waiting for {ARM_TRAJ_ACTION} action server...')
        if self._traj_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().info('  Trajectory action server ready — 부드러운 실행 사용')
        else:
            self.get_logger().warning(
                f'  {ARM_TRAJ_ACTION} 없음 → q_target 스트리밍 fallback (뚝뚝 끊길 수 있음).\n'
                '    launch_moveit.py 의 trajectory_bridge_right 가 떠 있는지 확인하세요.')
            self._traj_client = None
        self.get_logger().info(f'Servers ready  (mode={self._mode})')

    def _hand_cb(self, msg: JointState):
        # if len(msg.data) == 16: self._hand_deg = list(msg.data)   # 옛 Float32MultiArray
        if len(msg.position) == 16:
            self._hand_deg = list(msg.position)

    def _franka_joint_cb(self, msg: JointState):
        # if len(msg.data) == 7: self._current_joints = list(msg.data)   # 옛 Float64MultiArray
        if len(msg.position) == 7:
            self._current_joints = list(msg.position)

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _make_pose(self, x, y, z, qx, qy, qz, qw) -> PoseStamped:
        p = PoseStamped()
        p.header.frame_id  = REF_FRAME
        p.header.stamp     = self.get_clock().now().to_msg()
        p.pose.position    = Point(x=float(x), y=float(y), z=float(z))
        p.pose.orientation = Quaternion(x=float(qx), y=float(qy),
                                        z=float(qz), w=float(qw))
        return p

    def _wait(self, future, timeout: float) -> bool:
        t0 = time.time()
        while not future.done():
            if time.time() - t0 > timeout:
                return False
            time.sleep(0.01)
        return True

    def _confirm(self, prompt: str) -> bool:
        if os.environ.get('GRASP_AUTO_YES') == '1':
            print(prompt + 'y (auto)')
            return True
        while True:
            try:
                resp = input(prompt).strip().lower()
            except EOFError:
                return False
            if resp == 'y':
                return True
            if resp == 'n':
                return False
            print("  'y' 또는 'n' 을 입력하세요.")

    def _display_trajectory(self, start_state, trajectory, label: str):
        disp = DisplayTrajectory()
        # model_id 는 로봇 모델명(URDF robot name)용 — 그룹명(PLANNING_GROUP)이 아니다.
        # 비워두면 RViz 가 모델 일치 검사를 건너뛰어 어떤 모델(dual/single)에서도 경고 없이 표시됨.
        disp.model_id = ''
        disp.trajectory_start = start_state
        disp.trajectory.append(trajectory)
        time.sleep(0.1)
        self._display_pub.publish(disp)
        time.sleep(0.3)
        self.get_logger().info(f'[{label}] RViz에 경로 표시됨 → 확인 후 y/n 입력')

    # ── Planning ──────────────────────────────────────────────────────────────

    _JOINT_VEL_LIMITS = [2.62, 2.62, 2.62, 2.62, 5.26, 4.18, 5.26]
    _JOINT_ACC_LIMITS = [2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0]   # dual_v2_joint_limits.yaml

    def _add_timestamps(self, jt: JointTrajectory,
                        resample_dt: float = 0.01) -> JointTrajectory:
        """waypoint 궤적 → TOTG-lite 재타이밍 + 10ms 리샘플.

        기존(구간 등속 + 중앙차분 속도)은 가속도 프로파일이 없어 waypoint 마디마다
        속도가 계단식으로 바뀌어 실행이 덜컹였다. 여기서는:
          ① 속도/가속 한계(×speed_factor) 기반 사다리꼴 프로파일
             (전진/후진 패스, 방향전환 waypoint 자동 감속)
          ② cubic spline 으로 100Hz 리샘플 → positions+velocities 연속
        """
        import numpy as np
        from scipy.interpolate import CubicSpline

        pts_in = list(jt.points)
        names  = list(jt.joint_names) if jt.joint_names else list(HOME_JOINT_NAMES)
        if len(pts_in) < 2 or len(pts_in[0].positions) < 7:
            return jt
        speed = max(0.001, min(1.0, self._speed))

        # ── waypoint 추출 + 연속 중복 제거 ──
        Q = np.array([list(p.positions)[:7] for p in pts_in], dtype=float)
        keep = [0]
        for i in range(1, len(Q)):
            if np.max(np.abs(Q[i] - Q[keep[-1]])) > 1e-6:
                keep.append(i)
        Q = Q[keep]
        if len(Q) < 2:
            return jt

        vlim = np.array(self._JOINT_VEL_LIMITS) * speed
        alim = np.array(self._JOINT_ACC_LIMITS) * speed

        # ── 경로 파라미터 s: 구간별 "등속 최단시간" 누적 (순항 경로속도 = 1) ──
        dQ = np.diff(Q, axis=0)
        ds = np.maximum(np.max(np.abs(dQ) / vlim, axis=1), 1e-4)
        # 구간별 경로가속 한계: |q̈_j| = |dq_j/ds|·a_path ≤ alim_j
        a_seg = np.min(alim / np.maximum(np.abs(dQ) / ds[:, None], 1e-9), axis=1)

        # ── 노드 속도 상한: 경로가 꺾이는 waypoint 에서 감속 ──
        v_cap = np.ones(len(Q))
        v_cap[0] = v_cap[-1] = 0.0
        seg_norm = np.linalg.norm(dQ, axis=1)
        u = dQ / np.maximum(seg_norm[:, None], 1e-12)
        for i in range(1, len(Q) - 1):
            cos_t = float(np.dot(u[i - 1], u[i]))
            if cos_t < 0.995:                    # ~5° 이상 방향 전환
                v_cap[i] = max(0.05, ((1.0 + cos_t) * 0.5) ** 2)

        # ── 전진/후진 패스 (사다리꼴 프로파일, 가속 한계 준수) ──
        v = v_cap.copy()
        for i in range(len(Q) - 1):
            v[i + 1] = min(v[i + 1], math.sqrt(v[i] ** 2 + 2.0 * a_seg[i] * ds[i]))
        for i in range(len(Q) - 2, -1, -1):
            v[i] = min(v[i], math.sqrt(v[i + 1] ** 2 + 2.0 * a_seg[i] * ds[i]))
        dt_seg = 2.0 * ds / np.maximum(v[:-1] + v[1:], 1e-6)
        t_node = np.concatenate([[0.0], np.cumsum(dt_seg)])
        if t_node[-1] < 0.5:                     # 너무 짧은 이동도 최소 0.5s
            t_node = t_node * (0.5 / t_node[-1])

        # ── spline + 리샘플. 한계 초과 시 시간 늘려 재시도 ──
        for _ in range(3):
            spline = CubicSpline(t_node, Q, axis=0, bc_type='clamped')
            total  = float(t_node[-1])
            n_out  = max(2, int(math.ceil(total / resample_dt)) + 1)
            ts     = np.linspace(0.0, total, n_out)
            qs     = spline(ts)
            vs     = spline(ts, 1)
            accs   = spline(ts, 2)
            r = max(float(np.max(np.abs(vs) / vlim)),
                    math.sqrt(max(float(np.max(np.abs(accs) / alim)), 1e-9)))
            if r <= 1.02:
                break
            t_node = t_node * min(r, 2.0)

        result = JointTrajectory()
        result.joint_names = names
        last = len(ts) - 1
        for k in range(len(ts)):
            p = JointTrajectoryPoint()
            p.positions  = [float(x) for x in qs[k]]
            p.velocities = ([0.0] * qs.shape[1] if k in (0, last)
                            else [float(x) for x in vs[k]])
            sec  = int(ts[k])
            nsec = int(round((ts[k] - sec) * 1e9))
            p.time_from_start = RosDuration(sec=sec, nanosec=nsec)
            result.points.append(p)
        self.get_logger().info(
            f'[TIMESTAMP] {len(pts_in)} wp → {len(result.points)} pts '
            f'@{1.0 / resample_dt:.0f}Hz  total={ts[-1]:.2f}s')
        return result

    def _plan_cartesian(self, goal: PoseStamped, label: str,
                        jnames=None, jvals=None, max_step: float = 0.005):
        req = GetCartesianPath.Request()
        req.header           = goal.header
        req.group_name       = PLANNING_GROUP
        req.link_name        = EE_LINK
        req.waypoints        = [goal.pose]
        req.max_step         = max_step
        req.jump_threshold   = 0.0
        req.avoid_collisions = not self._disable_collision
        if jnames is not None:
            req.start_state.is_diff = False
            req.start_state.joint_state.name     = list(jnames)
            req.start_state.joint_state.position = list(jvals)
        else:
            req.start_state.is_diff = True
        self.get_logger().info(
            f'[{label}] Cartesian → '
            f'({goal.pose.position.x:.3f}, {goal.pose.position.y:.3f}, '
            f'{goal.pose.position.z:.3f})')
        future = self._cart_client.call_async(req)
        if not self._wait(future, 12.0):
            self.get_logger().error(f'[{label}] Cartesian timeout')
            return None
        res = future.result()
        if res.fraction < 0.9:
            self.get_logger().warning(
                f'[{label}] Cartesian {res.fraction:.0%} < 90% — 경로 계획 불완전')
            return None
        n = len(res.solution.joint_trajectory.points)
        self.get_logger().info(f'[{label}] Cartesian OK {res.fraction:.0%}  {n} pts')
        return res

    def _plan_ompl(self, goal: PoseStamped, label: str):
        g = MoveGroup.Goal()
        g.request.group_name                      = PLANNING_GROUP
        g.request.num_planning_attempts           = 20
        g.request.allowed_planning_time           = PLANNING_TIME
        g.request.max_velocity_scaling_factor     = 0.5
        g.request.max_acceleration_scaling_factor = 0.5
        ws = g.request.workspace_parameters
        ws.header.frame_id = REF_FRAME
        ws.min_corner.x = ws.min_corner.y = ws.min_corner.z = -5.0
        ws.max_corner.x = ws.max_corner.y = ws.max_corner.z =  5.0
        g.request.start_state.is_diff = True
        gc  = Constraints()
        pc  = PositionConstraint()
        pc.header            = goal.header
        pc.link_name         = EE_LINK
        pc.constraint_region = BoundingVolume()
        sph = SolidPrimitive()
        sph.type             = SolidPrimitive.SPHERE
        sph.dimensions       = [0.005]
        pc.constraint_region.primitives      = [sph]
        pc.constraint_region.primitive_poses = [goal.pose]
        pc.weight            = 1.0
        oc  = OrientationConstraint()
        oc.header            = goal.header
        oc.link_name         = EE_LINK
        oc.orientation       = goal.pose.orientation
        oc.absolute_x_axis_tolerance = 0.05
        oc.absolute_y_axis_tolerance = 0.05
        oc.absolute_z_axis_tolerance = 0.05
        oc.weight            = 1.0
        gc.position_constraints    = [pc]
        gc.orientation_constraints = [oc]
        g.request.goal_constraints = [gc]
        g.planning_options.plan_only = True
        self.get_logger().info(
            f'[{label}] OMPL → '
            f'({goal.pose.position.x:.3f}, {goal.pose.position.y:.3f}, '
            f'{goal.pose.position.z:.3f})')
        future  = self._mg_client.send_goal_async(g)
        timeout = PLANNING_TIME + 4.0
        if not self._wait(future, timeout):
            self.get_logger().error(f'[{label}] OMPL goal timeout'); return None
        gh = future.result()
        if not gh.accepted:
            self.get_logger().error(f'[{label}] OMPL goal rejected'); return None
        rf = gh.get_result_async()
        if not self._wait(rf, timeout):
            self.get_logger().error(f'[{label}] OMPL result timeout'); return None
        res = rf.result().result
        if res.error_code.val != 1:
            self.get_logger().error(f'[{label}] OMPL failed  code={res.error_code.val}')
            return None
        n = len(res.planned_trajectory.joint_trajectory.points)
        self.get_logger().info(f'[{label}] OMPL OK  {n} pts')
        return res

    def _plan_step(self, goal: PoseStamped, label: str,
                   jnames=None, jvals=None, confirm: bool = True):
        """Cartesian -> seeded IK+OMPL -> seeded IK+PtoP 순서로 시도."""
        seed         = list(jvals) if jvals is not None else self._current_joints
        jstart_names = HOME_JOINT_NAMES if seed is not None else None
        jstart_vals  = seed

        for max_step in [0.01, 0.02, 0.05, 0.1]:
            print(f'\n[{label}] Cartesian path (max_step={max_step:.3f})...')
            cart = self._plan_cartesian(goal, label,
                                        jnames=jstart_names, jvals=jstart_vals,
                                        max_step=max_step)
            if cart is not None:
                jt    = self._add_timestamps(cart.solution.joint_trajectory)
                frac  = cart.fraction
                partial_final = list(jt.points[-1].positions)
                completion_jt = None
                if frac < 1.0 and seed is not None:
                    self.get_logger().info(
                        f'[{label}] fraction={frac:.0%} → IK 보정 시도...')
                    j_comp = self._compute_ik_seeded(goal, partial_final, label + '_COMP')
                    if j_comp is not None:
                        completion_jt = self._make_joint_traj(j_comp, start=partial_final)
                        self.get_logger().info(f'[{label}] 보정 trajectory 생성 완료')
                self._display_trajectory(cart.start_state, cart.solution, label)
                if confirm and not self._confirm(f'  [{label}] 실행하시겠습니까? (y/n): '):
                    print(f'  [{label}] 취소됨.')
                    return None
                if completion_jt is not None:
                    if getattr(self, '_traj_client', None) is not None:
                        self._exec_traj_action(jt)
                    else:
                        self._exec_traj_smooth(jt)
                        dur = (jt.points[-1].time_from_start.sec +
                               jt.points[-1].time_from_start.nanosec * 1e-9)
                        self._wait_for_traj(dur)
                    return completion_jt
                return jt
            print(f'  fraction 부족 — 재시도 (max_step 완화)...')

        self.get_logger().warning(f'[{label}] Cartesian 전부 실패 → OMPL 시도...')
        if seed is not None:
            joints_target = self._compute_ik_seeded(goal, seed, label)
            if joints_target is not None:
                res = self._plan_joints(HOME_JOINT_NAMES, joints_target, label)
                if res is not None:
                    self._display_trajectory(
                        res.trajectory_start, res.planned_trajectory, label)
                    if confirm and not self._confirm(
                            f'  [{label}(OMPL)] 실행하시겠습니까? (y/n): '):
                        print(f'  [{label}] 취소됨.')
                        return None
                    return res.planned_trajectory.joint_trajectory
                self.get_logger().warning(f'[{label}] OMPL 실패 → direct PtoP...')
                delta = [abs(j - s) for j, s in zip(joints_target, seed)]
                print(f'  IK OK  max_Δ={math.degrees(max(delta)):.1f}°')
                if confirm and not self._confirm(
                        f'  [{label}(PtoP)] 실행하시겠습니까? (y/n): '):
                    print(f'  [{label}] 취소됨.')
                    return None
                return self._make_joint_traj(joints_target, start=seed)

        self.get_logger().error(f'[{label}] 경로 계획 실패. 중단.')
        return None

    def _plan_joints(self, joint_names, joint_values, label: str):
        g = MoveGroup.Goal()
        g.request.group_name                      = PLANNING_GROUP
        g.request.num_planning_attempts           = 10
        g.request.allowed_planning_time           = PLANNING_TIME
        g.request.max_velocity_scaling_factor     = 0.3
        g.request.max_acceleration_scaling_factor = 0.3
        if self._current_joints is not None:
            g.request.start_state.is_diff = False
            g.request.start_state.joint_state.name     = list(HOME_JOINT_NAMES)
            g.request.start_state.joint_state.position = [float(v) for v in self._current_joints]
        else:
            g.request.start_state.is_diff = True
        gc = Constraints()
        for name, val in zip(joint_names, joint_values):
            jc               = JointConstraint()
            jc.joint_name    = name
            jc.position      = float(val)
            jc.tolerance_above = 0.01
            jc.tolerance_below = 0.01
            jc.weight        = 1.0
            gc.joint_constraints.append(jc)
        g.request.goal_constraints = [gc]
        g.planning_options.plan_only = True
        self.get_logger().info(f'[{label}] Joint-space planning...')
        future  = self._mg_client.send_goal_async(g)
        timeout = PLANNING_TIME + 4.0
        if not self._wait(future, timeout):
            self.get_logger().error(f'[{label}] goal timeout'); return None
        gh = future.result()
        if not gh.accepted:
            self.get_logger().error(f'[{label}] goal rejected'); return None
        rf = gh.get_result_async()
        if not self._wait(rf, timeout):
            self.get_logger().error(f'[{label}] result timeout'); return None
        res = rf.result().result
        if res.error_code.val != 1:
            self.get_logger().error(f'[{label}] failed  code={res.error_code.val}')
            return None
        n = len(res.planned_trajectory.joint_trajectory.points)
        self.get_logger().info(f'[{label}] OK  {n} pts')
        return res

    def _compute_ik_seeded(self, goal: PoseStamped, seed_joints: list,
                           label: str) -> list | None:
        req = GetPositionIK.Request()
        req.ik_request.group_name       = PLANNING_GROUP
        req.ik_request.ik_link_name     = EE_LINK
        req.ik_request.pose_stamped     = goal
        req.ik_request.avoid_collisions = not self._disable_collision
        req.ik_request.timeout.sec      = 1
        req.ik_request.timeout.nanosec  = 0
        req.ik_request.robot_state.joint_state.name     = HOME_JOINT_NAMES
        req.ik_request.robot_state.joint_state.position = [float(v) for v in seed_joints]
        req.ik_request.robot_state.is_diff              = False
        future = self._ik_client.call_async(req)
        if not self._wait(future, 5.0):
            self.get_logger().error(f'[{label}] IK timeout'); return None
        res = future.result()
        if res.error_code.val != 1:
            self.get_logger().warning(f'[{label}] IK failed  code={res.error_code.val}')
            # avoid_collisions=False 로 재시도해서 원인 진단 + joint 값 확인
            req2 = GetPositionIK.Request()
            req2.ik_request.group_name       = PLANNING_GROUP
            req2.ik_request.ik_link_name     = EE_LINK
            req2.ik_request.pose_stamped     = goal
            req2.ik_request.avoid_collisions = False
            req2.ik_request.timeout.sec      = 2
            req2.ik_request.robot_state.joint_state.name     = HOME_JOINT_NAMES
            req2.ik_request.robot_state.joint_state.position = [float(v) for v in seed_joints]
            req2.ik_request.robot_state.is_diff              = False
            f2 = self._ik_client.call_async(req2)
            if self._wait(f2, 5.0) and f2.result().error_code.val == 1:
                r2  = f2.result()
                js2 = r2.solution.joint_state
                jm2 = dict(zip(js2.name, js2.position))
                jv2 = [jm2[n] for n in HOME_JOINT_NAMES]
                self.get_logger().warning(
                    f'[{label}] IK (no-collision) 성공 → collision scene이 차단 중')
                print(f'  [DEBUG][{label}] IK target joints (avoid_collisions=False):')
                for name, val in zip(HOME_JOINT_NAMES, jv2):
                    print(f'    {name}: {val:.6f}  ({math.degrees(val):.2f}°)')
                print(f'  [DEBUG][{label}] list: {[round(v, 6) for v in jv2]}')
            else:
                self.get_logger().warning(
                    f'[{label}] IK (no-collision) 도 실패 → pose 자체가 도달 불가')
                print(f'  [DEBUG][{label}] target pose (base frame):')
                p = goal.pose.position
                o = goal.pose.orientation
                print(f'    xyz  = ({p.x:.4f}, {p.y:.4f}, {p.z:.4f})')
                print(f'    quat = ({o.x:.4f}, {o.y:.4f}, {o.z:.4f}, {o.w:.4f})')
            return None
        js        = res.solution.joint_state
        joint_map = dict(zip(js.name, js.position))
        joints    = [joint_map[n] for n in HOME_JOINT_NAMES]
        delta     = [abs(j - s) for j, s in zip(joints, seed_joints)]
        self.get_logger().info(
            f'[{label}] IK OK  max_Δ={max(delta):.3f} rad  '
            f'joints={[round(v,4) for v in joints]}')
        return joints

    def _compute_fk(self, joint_values: list,
                    link_name: str = EE_LINK) -> list | None:
        req = GetPositionFK.Request()
        req.header.frame_id   = REF_FRAME
        req.fk_link_names     = [link_name]
        req.robot_state.joint_state.name     = list(HOME_JOINT_NAMES)
        req.robot_state.joint_state.position = [float(v) for v in joint_values]
        future = self._fk_client.call_async(req)
        if not self._wait(future, 5.0):
            self.get_logger().error('[FK] timeout'); return None
        res = future.result()
        if res.error_code.val != 1:
            self.get_logger().error(f'[FK] failed code={res.error_code.val}')
            return None
        p = res.pose_stamped[0].pose
        return [p.position.x, p.position.y, p.position.z,
                p.orientation.x, p.orientation.y,
                p.orientation.z, p.orientation.w]

    def _plan_step_ik(self, goal: PoseStamped, label: str,
                      seed_joints: list = None) -> list | None:
        seed = seed_joints if seed_joints is not None else self._current_joints
        if seed is not None:
            print(f'\n[{label}] Seeded IK (seed from current joints)...')
            joints = self._compute_ik_seeded(goal, seed, label)
            if joints is not None:
                delta = [abs(j - s) for j, s in zip(joints, seed)]
                print(f'  max_Δjoint = {max(delta):.3f} rad  '
                      f'({math.degrees(max(delta)):.1f}°)')
                if not self._confirm(f'  [{label}] 실행하시겠습니까? (y/n): '):
                    print(f'  [{label}] 취소됨.')
                    return None
                return joints
            self.get_logger().warning(f'[{label}] Seeded IK 실패 → Cartesian/OMPL fallback')
        jnames_seed = HOME_JOINT_NAMES if seed is not None else None
        jt = self._plan_step(goal, label, jnames=jnames_seed, jvals=seed)
        if jt is None:
            return None
        return list(jt.points[-1].positions)

    def _exec_joints(self, joint_values: list) -> None:
        spd      = Float64()
        spd.data = max(0.001, min(1.0, float(self._speed)))
        self._franka_speed_pub.publish(spd)
        time.sleep(0.05)
        tgt      = Float64MultiArray()
        tgt.data = [float(v) for v in joint_values]
        self._franka_target_pub.publish(tgt)
        self.get_logger().info(f'Joint cmd sent: {[round(v,4) for v in tgt.data]}')

    def _wait_for_traj(self, dur: float, margin: float = 0.3) -> None:
        wait = max(0.1, dur + margin)
        self.get_logger().info(f'[WAIT] {wait:.2f}s (dur={dur:.2f}s)')
        time.sleep(wait)

    def _wait_for_motion(self, target_joints: list,
                         timeout: float = 30.0, tol: float = 0.05) -> bool:
        time.sleep(0.1)
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self._current_joints is not None:
                err = max(abs(c - t)
                          for c, t in zip(self._current_joints, target_joints))
                if err < tol:
                    self.get_logger().info(
                        f'[WAIT] 도달 (err={err:.4f} rad, t={time.time()-t0:.1f}s)')
                    return True
            time.sleep(0.05)
        self.get_logger().warning(f'[WAIT] {timeout:.0f}s 초과')
        return False

    def _hold_hand_position(self, duration: float = 2.0) -> None:
        if not rclpy.ok():
            return
        enc = self._last_hand_enc
        if enc is None:
            return
        # msg = Int16MultiArray(); msg.data = [int(v) for v in enc]   # 옛 (Int16)
        msg = Float32MultiArray()
        msg.data = [float(v) for v in enc]
        rate     = 30
        interval = 1.0 / rate
        count    = int(duration * rate)
        self.get_logger().info(f'[HOLD] 손 현재 위치 유지 ({duration:.1f}s)...')
        for _ in range(count):
            if not rclpy.ok():
                break
            self._hand_pub.publish(msg)
            time.sleep(interval)
        self.get_logger().info('[HOLD] 완료')

    def _make_joint_traj(self, target: list, start: list = None) -> JointTrajectory:
        if start is None:
            start = self._current_joints or list(HOME_JOINT_VALUES)
        max_delta = max(abs(t - s) for t, s in zip(target, start))
        duration  = max(3.0, max_delta / max(0.01, self._speed * 1.5))
        jt = JointTrajectory()
        jt.joint_names = list(HOME_JOINT_NAMES)
        pt0 = JointTrajectoryPoint()
        pt0.positions       = [float(v) for v in start]
        pt0.time_from_start = RosDuration(sec=0, nanosec=0)
        pt1 = JointTrajectoryPoint()
        pt1.positions       = [float(v) for v in target]
        sec  = int(duration)
        nsec = int((duration - sec) * 1e9)
        pt1.time_from_start = RosDuration(sec=sec, nanosec=nsec)
        jt.points = [pt0, pt1]
        self.get_logger().info(
            f'[TRAJ_DIRECT] max_Δ={max_delta:.3f} rad  dur={duration:.1f}s')
        # 2점 직행도 TOTG-lite 리샘플 → S-커브 프로파일로 부드럽게
        return self._add_timestamps(jt)

    def _do_release_hand(self):
        start  = self._last_hand_enc if self._last_hand_enc else list(HAND_RELEASE_ENC)
        target = list(HAND_RELEASE_ENC)
        self.get_logger().info(f'[RELEASE] {HAND_STEPS}steps × {HAND_PERIOD}s')
        for i in range(1, HAND_STEPS + 1):
            t      = i / HAND_STEPS
            interp = [max(-32768, min(32767, int(round(s + t * (g - s)))))
                      for s, g in zip(start, target)]
            # msg = Int16MultiArray(); msg.data = interp   # 옛 (Int16)
            msg      = Float32MultiArray()
            msg.data = [float(v) for v in interp]
            self._hand_pub.publish(msg)
            if i < HAND_STEPS:
                time.sleep(HAND_PERIOD)
        self._last_hand_enc = list(HAND_RELEASE_ENC)
        self.get_logger().info('[RELEASE] 완료')

    # ── Arm execution ─────────────────────────────────────────────────────────

    def _exec_arm(self, jt: JointTrajectory):
        """execute_mode 에 맞는 팔 실행 방식 선택.

        - direct_franka_topic : /franka/right/q_target 로 waypoint 스트리밍 (제어 PC가 받는 입구)
        - trajectory_forwarder: /franka/target_trajectory 로 JointTrajectory 발행 (별도 forwarder 필요)

        이 제어 PC(Dual_Arm_Hand_Ctrl)는 q_target 만 받으므로 direct_franka_topic 사용.
        """
        if self._mode == 'trajectory_forwarder':
            return self._exec_traj_smooth(jt)
        return self._exec_waypoints(jt)

    def _exec_traj_smooth(self, jt: JointTrajectory) -> list:
        if not jt.points:
            self.get_logger().error('[TRAJ] Empty trajectory')
            return []
        msg = JointTrajectory()
        msg.joint_names = list(jt.joint_names) if jt.joint_names else list(HOME_JOINT_NAMES)
        msg.points = jt.points
        spd = Float64()
        spd.data = max(0.001, min(1.0, float(self._speed)))
        self._franka_speed_pub.publish(spd)
        time.sleep(0.02)
        self._traj_smooth_pub.publish(msg)
        dur   = jt.points[-1].time_from_start.sec + \
                jt.points[-1].time_from_start.nanosec * 1e-9
        final = list(jt.points[-1].positions)
        self.get_logger().info(
            f'[TRAJ] sent {len(jt.points)} waypoints, duration={dur:.2f}s, '
            f'final={[round(v,4) for v in final]}')
        return final

    def _exec_traj_action(self, jt) -> bool:
        """전체 JointTrajectory 를 FollowJointTrajectory 액션으로 전송하고
        로봇의 실제 완료 결과를 대기한다 (pose_commander.py 방식과 동일).

        컨트롤러가 waypoint 사이를 스플라인 보간하여 한 번에 부드럽게 실행하므로
        q_target setpoint 스트리밍의 '뚝뚝 끊김' 이 없다. 액션 result 가 완료 신호라
        타이머/폴링 기반 추정이 필요없다.
        """
        if not jt.points:
            self.get_logger().error('[TRAJ] Empty trajectory'); return False

        dur = (jt.points[-1].time_from_start.sec +
               jt.points[-1].time_from_start.nanosec * 1e-9)
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = jt
        self.get_logger().info(
            f'[TRAJ] {len(jt.points)} pts, dur={dur:.2f}s → {ARM_TRAJ_ACTION}')

        send_future = self._traj_client.send_goal_async(goal)
        if not self._wait(send_future, 10.0):
            self.get_logger().error('[TRAJ] goal 전송 timeout'); return False
        gh = send_future.result()
        if not gh.accepted:
            self.get_logger().error(
                '[TRAJ] 로봇이 goal 거부 — /joint_states_relay stale 이거나 '
                'bridge/컨트롤러 확인 필요'); return False

        # 로봇은 현재자세 램프 + speed_factor 안전제한으로 계획 duration 보다
        # 오래 걸리므로 넉넉히 대기 (pose_commander.py 와 동일 산식).
        result_timeout = max(30.0, 3.0 * dur + 20.0)
        self.get_logger().info(
            f'[TRAJ] 수락됨. 완료 대기 (최대 {result_timeout:.0f}s)...')
        rf = gh.get_result_async()
        if not self._wait(rf, result_timeout):
            self.get_logger().warning(
                f'[TRAJ] {result_timeout:.0f}s 내 결과 없음 — 로봇이 아직 실행 중일 수 있음')
            return False
        res = rf.result().result
        if res.error_code == FollowJointTrajectory.Result.SUCCESSFUL:
            self.get_logger().info(f"[TRAJ] 실행 완료: '{res.error_string}'")
            return True
        self.get_logger().warning(
            f'[TRAJ] 실행 실패 (code={res.error_code}): {res.error_string}')
        return False

    def _exec_waypoints(self, jt):
        if not jt.points:
            self.get_logger().error('Empty trajectory'); return
        spd      = Float64()
        spd.data = max(0.001, min(1.0, float(self._speed)))
        self._franka_speed_pub.publish(spd)
        time.sleep(0.02)
        t_start = time.time()
        for i, pt in enumerate(jt.points):
            if len(pt.positions) != 7:
                self.get_logger().error(
                    f'Waypoint[{i}] positions len={len(pt.positions)}, expected 7')
                return
            t_target = pt.time_from_start.sec + pt.time_from_start.nanosec * 1e-9
            t_now    = time.time() - t_start
            sleep_t  = t_target - t_now
            if sleep_t > 0:
                time.sleep(sleep_t)
            tgt      = Float64MultiArray()
            tgt.data = [float(v) for v in pt.positions]
            self._franka_target_pub.publish(tgt)
        self.get_logger().info(
            f'Waypoints sent: {len(jt.points)} pts, '
            f'final={[round(v,4) for v in jt.points[-1].positions]}')

    def _exec_direct(self, jt):
        if not jt.points:
            self.get_logger().error('Empty trajectory'); return
        final = jt.points[-1]
        if len(final.positions) != 7:
            self.get_logger().error(
                f'Expected 7 joint positions, got {len(final.positions)}'); return
        spd      = Float64()
        spd.data = max(0.001, min(1.0, float(self._speed)))
        self._franka_speed_pub.publish(spd)
        time.sleep(0.05)
        tgt      = Float64MultiArray()
        tgt.data = list(final.positions)
        self._franka_target_pub.publish(tgt)
        self.get_logger().info(f'Direct arm cmd sent: {[round(v,4) for v in tgt.data]}')

    # ── Hand execution ────────────────────────────────────────────────────────

    def _exec_hand(self, target_enc: list):
        start  = self._last_hand_enc if self._last_hand_enc else list(HAND_INIT_ENC)
        target = [int(v) for v in target_enc]
        total  = HAND_STEPS * HAND_PERIOD
        self.get_logger().info(
            f'[HAND] {HAND_STEPS}steps × {HAND_PERIOD}s = {total:.1f}s')
        self.get_logger().info(f'  target={target}')
        for i in range(1, HAND_STEPS + 1):
            t      = i / HAND_STEPS
            interp = [max(-32768, min(32767, int(round(s + t * (g - s)))))
                      for s, g in zip(start, target)]
            # msg = Int16MultiArray(); msg.data = interp   # 옛 (Int16)
            msg      = Float32MultiArray()
            msg.data = [float(v) for v in interp]
            self._hand_pub.publish(msg)
            self.get_logger().info(f'  [{i:2d}/{HAND_STEPS}] t={t:.1f}  raw={interp}')
            if i < HAND_STEPS:
                time.sleep(HAND_PERIOD)
        self._last_hand_enc = target
        self.get_logger().info('[HAND] 파지 완료')

    # ── Main execution ────────────────────────────────────────────────────────

    def _run_guarded(self):
        try:
            self._execute()
        except Exception as e:
            self.get_logger().error(f'Unhandled exception: {e}')
            traceback.print_exc()
        finally:
            try:
                self._finalize()
            except Exception as e:
                self.get_logger().error(f'[FINALIZE] 예외 발생: {e}')
            rclpy.shutdown()

    def _execute(self):
        grasp  = self._summary['grasps'][0]
        bp     = grasp['base_pose']
        xyz_w  = bp['xyz']
        quat_w = bp['quat_xyzw']
        enc    = grasp['joint_angles_enc']
        T_wb   = self._summary.get('T_world_base')

        if T_wb is not None:
            xyz_b, quat_b = world_to_base(T_wb, xyz_w, quat_w)
            xyz_w_app     = [xyz_w[0], xyz_w[1], xyz_w[2] + self._approach_offset]
            xyz_b_app, quat_b_app = world_to_base(T_wb, xyz_w_app, quat_w)
        else:
            self.get_logger().warning('T_world_base not in summary — using xyz as-is in base frame')
            xyz_b, quat_b = xyz_w, quat_w
            xyz_b_app     = [xyz_w[0], xyz_w[1], xyz_w[2] + self._approach_offset]
            quat_b_app    = quat_w

        print('\n' + '=' * 60)
        print('  Grasp Executor')
        print(f'  EE target (world): xyz={[round(v,3) for v in xyz_w]}')
        print(f'  EE target (base) : xyz={[round(v,3) for v in xyz_b]}')
        print(f'  quat (base)      : {[round(v,3) for v in quat_b]}')
        print(f'  Hand enc (HW)    : {enc}')
        print(f'  Approach offset  : {self._approach_offset:.2f} m')
        print('\n  Grasp Executor  (STEP 1~4)')
        print(f'  Grasp (base): {[round(v,3) for v in xyz_b]}')
        print(f'  Approach z  : {xyz_b_app[2]:.3f} m')
        print('=' * 60)

        time.sleep(1.0)
        # 손 서보 켜기 (새 제어 PC — q_target 초기값 세팅 후 servo on, 튐 방지)  #추가
        _init_msg = Float32MultiArray()
        _init_msg.data = [float(v) for v in HAND_INIT_ENC]
        self._hand_pub.publish(_init_msg)                 # 현재 목표 먼저
        time.sleep(0.1)
        self._hand_servo_pub.publish(Bool(data=True))     # 서보 on
        time.sleep(0.2)
        step_init_hand(self)
        step_go_home(self, confirm=True)
        target   = self._make_pose(*xyz_b,     *quat_b)
        approach = self._make_pose(*xyz_b_app, *quat_b_app)

        # APPROACH+TARGET 통합 (접근점 무정지). 계획 실패 시 기존 2-step 폴백.
        result = step_approach_descend(self, approach, target, confirm=True)
        if result is False:
            return                                   # 사용자 취소
        if result is not None:
            j2, self._approach_traj, descend_traj = result
        else:
            self.get_logger().warning(
                '[APPROACH+TARGET] 통합 Cartesian 실패 → 기존 2단계(접근점 정지) 폴백')
            result = step_approach(self, approach, confirm=True)
            if result is None: return
            j1, self._approach_traj = result

            result = step_descend(self, target, seed=j1, confirm=True)
            if result is None: return
            j2, descend_traj = result

        print(f'\n  STEP 3/4  HAND 파지  enc={enc}')
        if not step_close_hand(self, enc, confirm=True): return

        j4 = step_lift(self, approach, seed=j2,
                       descend_traj=descend_traj, confirm=True)
        if j4 is None: return

        self._success = True
        print('\n  [OK] 파지 완료')

    def _finalize(self):
        print('\n' + '─' * 60)
        print('  [FINALIZE]')
        print('─' * 60)
        if self._success:
            print('  파지 완료 → 릴리즈 및 홈 복귀')
        else:
            print('  ⚠  실행이 중단되었습니다.')
        step_release_hand(self, confirm=True)
        step_go_home(self, confirm=True)
