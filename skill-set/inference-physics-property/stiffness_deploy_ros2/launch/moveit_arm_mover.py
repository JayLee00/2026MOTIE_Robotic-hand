#!/usr/bin/env python3
"""moveit_arm_mover.py — dex_ros MoveIt 로 '충돌회피 끝점 이동'을 plan-only 로 얻어
기존 팔 경로(q_target)로 재생하는 독립 모듈 (Option B).

설계 (모듈식 · 완전 디커플):
  · collect_ros2 / deploy 를 전혀 import 하지 않는다. 팔 명령은 'arm_sink' 인자로만 받는다.
    arm_sink = write_partial(*, Arm_j_tar=((7,)))  를 가진 아무 객체 (= Ros2ShmBridge).
  · MoveIt(/move_action)에 plan_only=True 로 충돌회피된 관절 경로만 요청 → 그 경로를
    arm_sink.write_partial(Arm_j_tar=...) 로 스트리밍 재생.
  ⇒ 팔 제어 경로가 q_target '하나'로 유지 → 제어권 전쟁/미확인 arm_controller 서버 의존 없음.
     충돌회피 가치는 '플랜'이 이미 충돌검사됐으므로 보존.

사용 (collect_ros2 등에서):
    from moveit_arm_mover import MoveItArmMover
    mover = MoveItArmMover(group="right_arm", ee_link="right_fr3_link8", frame="world")
    executor.add_node(mover)                     # ★ 반드시 스피닝 executor 에 추가(브리지와 동일)
    ok = mover.move_to_pose(bridge, position=(x,y,z), orientation=(qx,qy,qz,qw))

전제 / 게이트:
  · 플래닝 PC 에서 move_group 기동 (도메인 9, joint_state_mode:=direct — 정확한 시작상태 필요).
      ros2 launch franka_kistar_bringup dual_fr3_kistar_moveit.launch.py joint_state_mode:=direct ...
      확인: ros2 action list | grep move_action
  · 메시지 패키지 필요(라이브러리 아님 — 클라이언트는 msg 만):
      sudo apt install ros-humble-moveit-msgs ros-humble-shape-msgs
  · ★ EE 는 FR3 '플랜지'(right_fr3_link8) — KISTAR 손끝이 아니다. 목표 포즈는 플랜지 기준.
  · arm_sink(q_target) 외의 팔 쓰기(ee_target 등)를 동시에 하지 말 것(로봇 PC 에서 충돌).
"""
from __future__ import annotations

import time

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (Constraints, PositionConstraint, OrientationConstraint,
                             JointConstraint, BoundingVolume, MoveItErrorCodes)
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose, Point, Quaternion


class MoveItArmMover(Node):
    """MoveIt /move_action 로 끝점 포즈 plan-only → q_target 재생 (Option B)."""

    def __init__(self, *, group: str = "right_arm", ee_link: str = "right_fr3_link8",
                 frame: str = "world", move_action: str = "/move_action",
                 pos_tol: float = 1e-3, ori_tol: float = 0.01, joint_tol: float = 0.01,
                 plan_time: float = 5.0, plan_attempts: int = 5,
                 vel_scale: float = 0.3, acc_scale: float = 0.3,
                 joint_names=None, node_name: str = "moveit_arm_mover"):
        super().__init__(node_name)
        self.group = group
        self.ee_link = ee_link
        self.frame = frame
        self.pos_tol = float(pos_tol)
        self.ori_tol = float(ori_tol)
        self.joint_tol = float(joint_tol)
        # 관절각 목표(JointConstraint)용 관절 이름. 기본 = FR3 오른팔 joint1..7 (dex_ros).
        self.joint_names = list(joint_names) if joint_names else [f"right_fr3_joint{i}" for i in range(1, 8)]
        self.plan_time = float(plan_time)
        self.plan_attempts = int(plan_attempts)
        self.vel_scale = float(vel_scale)
        self.acc_scale = float(acc_scale)
        self._ac = ActionClient(self, MoveGroup, move_action)
        self._move_action = move_action

    def wait_ready(self, timeout: float = 5.0) -> bool:
        """move_group(/move_action) 서버가 떠 있는지 확인. (외부 executor 스핀 전제)"""
        return self._ac.wait_for_server(timeout_sec=timeout)

    # ── 내부: 외부 executor 가 스핀 중이라는 전제하에 future 폴링 대기 ──────────
    def _wait(self, future, timeout: float):
        end = time.monotonic() + timeout
        while not future.done():
            if time.monotonic() > end:
                return None
            time.sleep(0.01)
        return future.result()

    def _build_goal(self, position, orientation) -> MoveGroup.Goal:
        goal = MoveGroup.Goal()
        req = goal.request
        req.group_name = self.group
        req.num_planning_attempts = self.plan_attempts
        req.allowed_planning_time = self.plan_time
        req.max_velocity_scaling_factor = self.vel_scale
        req.max_acceleration_scaling_factor = self.acc_scale
        # (start_state 를 비워두면 move_group 이 현재 /joint_states 를 시작상태로 사용)

        c = Constraints()

        # 위치 제약: ee_link 가 target 점 주변 반경 pos_tol 구(sphere) 안.
        pc = PositionConstraint()
        pc.header.frame_id = self.frame
        pc.link_name = self.ee_link
        sphere = SolidPrimitive()
        sphere.type = SolidPrimitive.SPHERE
        sphere.dimensions = [self.pos_tol]
        region = BoundingVolume()
        region.primitives = [sphere]
        region_pose = Pose()
        region_pose.position = Point(x=float(position[0]), y=float(position[1]),
                                     z=float(position[2]))
        region_pose.orientation.w = 1.0
        region.primitive_poses = [region_pose]
        pc.constraint_region = region
        pc.weight = 1.0
        c.position_constraints = [pc]

        # 방향 제약 (orientation=None 이면 위치만 — 자유 방향 플랜).
        if orientation is not None:
            oc = OrientationConstraint()
            oc.header.frame_id = self.frame
            oc.link_name = self.ee_link
            oc.orientation = Quaternion(x=float(orientation[0]), y=float(orientation[1]),
                                        z=float(orientation[2]), w=float(orientation[3]))
            oc.absolute_x_axis_tolerance = self.ori_tol
            oc.absolute_y_axis_tolerance = self.ori_tol
            oc.absolute_z_axis_tolerance = self.ori_tol
            oc.weight = 1.0
            c.orientation_constraints = [oc]

        req.goal_constraints = [c]
        goal.planning_options.plan_only = True   # ★ 실행은 안 함 — 경로만 받는다
        return goal

    def _build_joint_goal(self, joints) -> MoveGroup.Goal:
        """관절각 목표(JointConstraint). joints = list(self.joint_names 순서) 또는 dict{name:val}."""
        goal = MoveGroup.Goal()
        req = goal.request
        req.group_name = self.group
        req.num_planning_attempts = self.plan_attempts
        req.allowed_planning_time = self.plan_time
        req.max_velocity_scaling_factor = self.vel_scale
        req.max_acceleration_scaling_factor = self.acc_scale
        pairs = list(joints.items()) if isinstance(joints, dict) else list(zip(self.joint_names, joints))
        c = Constraints()
        for nm, val in pairs:
            jc = JointConstraint()
            jc.joint_name = nm
            jc.position = float(val)
            jc.tolerance_above = self.joint_tol
            jc.tolerance_below = self.joint_tol
            jc.weight = 1.0
            c.joint_constraints.append(jc)
        req.goal_constraints = [c]
        goal.planning_options.plan_only = True   # ★ 실행은 안 함 — 경로만
        return goal

    # ── 내부: goal 전송 → 충돌회피된 관절 경로(JointTrajectory) 반환, 실패 시 None ─────
    def _send(self, goal, timeout: float, desc: str):
        if not self._ac.wait_for_server(timeout_sec=3.0):
            self.get_logger().error(f"{self._move_action} 서버 없음 — move_group 미기동?")
            return None
        gh = self._wait(self._ac.send_goal_async(goal), timeout)
        if gh is None or not gh.accepted:
            self.get_logger().error(f"MoveGroup {desc} 거부 또는 타임아웃")
            return None
        res = self._wait(gh.get_result_async(), timeout)
        if res is None:
            self.get_logger().error("플랜 결과 타임아웃")
            return None
        result = res.result
        if result.error_code.val != MoveItErrorCodes.SUCCESS:
            self.get_logger().error(f"플랜 실패({desc}) error_code={result.error_code.val}")
            return None
        traj = result.planned_trajectory.joint_trajectory
        self.get_logger().info(f"플랜 성공({desc}): {len(traj.points)} waypoints")
        return traj

    # ── 플랜만(plan_only): Cartesian pose / joint 각도 ─────────────────────────
    def plan_to_pose(self, position, orientation=None, *, timeout: float = 20.0):
        return self._send(self._build_goal(position, orientation), timeout, "pose")

    def plan_to_joints(self, joints, *, timeout: float = 20.0):
        return self._send(self._build_joint_goal(joints), timeout, "joints")

    # ── 2) 재생: JointTrajectory 를 arm_sink(q_target) 로 rate_hz 스트리밍 ──────
    def replay_trajectory(self, arm_sink, traj, *, rate_hz: float = 100.0,
                          start_gap_tol: float = 0.15, dry_run: bool = False,
                          lookahead: float = 0.15, settle_tol: float = 0.02,
                          settle_timeout: float = 3.0, stall_timeout: float = 5.0) -> bool:
        """경로를 q_target 으로 스트리밍 재생하고 **도달까지 기다린다**.

        ★ 수신 노드(arm_q_target_receiver)는 q_target 을 절대위치가 아니라
          **실측 기준 델타 클램프**로 처리한다: q_tar = q_meas + clamp(target - q_meas),
          delta_cap = min(max_joint_delta(0.2), max_joint_vel(3.0)·dt).
          100Hz 로 보내면 dt≈0.01 → 한 메시지가 실측보다 0.03 rad 앞까지만 목표를 옮긴다.
          따라서 (a) 팔은 계획된 시간표보다 항상 뒤처지고, (b) 스트림을 멈추면 그 순간
          위치+0.03 에서 목표가 얼어붙어 **가다가 멈춘다**.
          → 시간 기반으로 흘려보내면 안 되고, **실측 진행에 맞춰 waypoint 를 전진**시키고
            마지막엔 **도달할 때까지 목표를 계속 재발행**해야 한다.

        lookahead      : 현재 waypoint 와 실측의 허용 간격[rad]. 이보다 벌어지면 기다린다.
        settle_tol     : 최종 도달 판정 허용오차[rad].
        settle_timeout : 최종 수렴 대기 상한[s].
        stall_timeout  : 진전이 없을 때 중단하는 시간[s].
        """
        pts = traj.points
        if not pts:
            self.get_logger().warn("빈 trajectory — 재생 생략")
            return False

        samples = _resample(traj, rate_hz)          # (nsamples, njoints), 시간 선형보간

        # 안전: 실제 팔 위치와 경로 첫 점의 간격 체크 (플랜 시작상태가 틀리면 큰 점프 = 위험).
        if hasattr(arm_sink, "read"):
            try:
                cur = np.asarray(arm_sink.read().Arm_j_pos[0], dtype=float)
                gap = float(np.max(np.abs(cur - samples[0]))) if cur.size == samples.shape[1] else 0.0
                if gap > start_gap_tol:
                    self.get_logger().error(
                        f"경로 첫 점이 현재 팔 자세와 {gap:.3f} rad 벌어짐(>tol {start_gap_tol}) "
                        "— 플랜 시작상태 불일치 의심, 재생 중단(joint_state_mode:=direct 확인).")
                    return False
            except Exception as e:  # read 인터페이스가 달라도 재생은 진행(경고만)
                self.get_logger().warn(f"시작 간격 체크 생략: {e}")

        if dry_run:
            self.get_logger().info(f"[dry_run] {len(samples)} setpoints 스트리밍 생략")
            return True

        dt = 1.0 / rate_hz
        can_read = hasattr(arm_sink, "read")

        def _meas():
            return np.asarray(arm_sink.read().Arm_j_pos[0], dtype=float)

        def _pub(q):
            arm_sink.write_partial(Arm_j_tar=(tuple(float(x) for x in q),))

        if not can_read:
            # 실측을 못 읽으면 진행 게이팅/도달 대기가 불가 → 옛 동작(시간 기반)으로 폴백.
            self.get_logger().warn(
                "arm_sink.read() 없음 → 시간기반 재생(도달 보장 없음: 델타 클램프로 중간에 멈출 수 있음)")
            for q in samples:
                _pub(q)
                time.sleep(dt)
            return True

        # ── 1) 경로 추종: 실측이 현재 waypoint 를 따라올 때까지 기다리며 전진 ──
        #    (시간 기반으로 흘려보내면 팔이 뒤처진 채 경로가 끝나 버린다 — docstring 참고)
        for q in samples:
            t_wp = time.monotonic()
            best = float("inf")
            while True:
                _pub(q)                                  # 계속 재발행 = carrot 유지
                time.sleep(dt)
                err = float(np.max(np.abs(_meas() - q)))
                if err <= lookahead:
                    break
                if err < best - 1e-4:                    # 진전 있음 → stall 타이머 리셋
                    best, t_wp = err, time.monotonic()
                elif time.monotonic() - t_wp > stall_timeout:
                    self.get_logger().error(
                        f"경로 추종 정지(stall): {stall_timeout}s 동안 진전 없음, 잔차 {err:.3f} rad. "
                        "충돌/관절限/제어권(require_control) 확인.")
                    return False

        # ── 2) 최종 수렴: 목표를 계속 재발행하며 도달 대기 ──
        #    스트림을 멈추면 q_tar 가 'q_meas + 작은 δ' 로 얼어붙어 그 자리에서 멈춘다.
        goal = samples[-1]
        t0 = time.monotonic()
        err = float("inf")
        while time.monotonic() - t0 < settle_timeout:
            _pub(goal)
            time.sleep(dt)
            err = float(np.max(np.abs(_meas() - goal)))
            if err <= settle_tol:
                self.get_logger().info(f"도달 완료: 최대 오차 {err:.4f} rad "
                                       f"({time.monotonic() - t0:.1f}s)")
                return True
        self.get_logger().warn(
            f"도달 미완(settle_timeout {settle_timeout}s): 최대 오차 {err:.4f} rad "
            f"(tol {settle_tol}). 임피던스 정상오차면 settle_tol 을 키울 것.")
        return False

    # ── 편의: plan + replay ───────────────────────────────────────────────
    def move_to_pose(self, arm_sink, position, orientation=None, *,
                     rate_hz: float = 100.0, start_gap_tol: float = 0.15,
                     dry_run: bool = False) -> bool:
        traj = self.plan_to_pose(position, orientation)
        if traj is None:
            return False
        return self.replay_trajectory(arm_sink, traj, rate_hz=rate_hz,
                                      start_gap_tol=start_gap_tol, dry_run=dry_run)

    def move_to_joints(self, arm_sink, joints, *,
                       rate_hz: float = 100.0, start_gap_tol: float = 0.15,
                       dry_run: bool = False) -> bool:
        traj = self.plan_to_joints(joints)
        if traj is None:
            return False
        return self.replay_trajectory(arm_sink, traj, rate_hz=rate_hz,
                                      start_gap_tol=start_gap_tol, dry_run=dry_run)


def _resample(traj, rate_hz: float) -> np.ndarray:
    """JointTrajectory 를 time_from_start 기준 고정 rate 로 선형보간 재샘플 → (n, njoints)."""
    pts = traj.points
    times = np.array([p.time_from_start.sec + p.time_from_start.nanosec * 1e-9 for p in pts])
    Q = np.array([list(p.positions) for p in pts], dtype=float)   # (npts, njoints)
    T = float(times[-1]) if len(times) else 0.0
    if len(pts) == 1 or T <= 0:
        return Q[-1:].copy()
    ts = np.arange(0.0, T + 1e-9, 1.0 / rate_hz)
    out = np.empty((len(ts), Q.shape[1]), dtype=float)
    for j in range(Q.shape[1]):
        out[:, j] = np.interp(ts, times, Q[:, j])
    return out


def main():
    """스탠드얼론 스모크 테스트: 포즈로 plan-only 만 수행하고 결과 출력(실 이동 없음).
       사용: python3 moveit_arm_mover.py X Y Z [QX QY QZ QW]"""
    import sys
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if len(args) < 3:
        print("usage: moveit_arm_mover.py X Y Z [QX QY QZ QW]  (plan-only 테스트)")
        return
    pos = tuple(float(a) for a in args[:3])
    ori = tuple(float(a) for a in args[3:7]) if len(args) >= 7 else None

    rclpy.init()
    mover = MoveItArmMover()
    from rclpy.executors import SingleThreadedExecutor
    import threading
    ex = SingleThreadedExecutor()
    ex.add_node(mover)
    threading.Thread(target=ex.spin, daemon=True).start()
    try:
        traj = mover.plan_to_pose(pos, ori)
        if traj is None:
            print("[smoke] 플랜 실패")
        else:
            samp = _resample(traj, 100.0)
            print(f"[smoke] 플랜 OK: waypoints={len(traj.points)}, "
                  f"재샘플(100Hz)={len(samp)} setpoints, 관절={list(traj.joint_names)}")
    finally:
        ex.shutdown()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
