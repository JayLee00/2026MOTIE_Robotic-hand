#!/usr/bin/env python3
"""goto_q.py — 오른팔을 관절각 7개 목표로 MoveIt 충돌회피 이동시키는 단독 도구.

    python3 tools/goto_q.py -0.2866 1.4185 0.2677 -1.9216 0.7769 1.2157 2.0401
    python3 tools/goto_q.py <j1..j7> --plan-only     # 플랜(충돌검사)만, 이동 없음
    python3 tools/goto_q.py <j1..j7> --yes           # 확인 프롬프트 생략 (파이프라인용)

동작: MoveGroup(/move_action) 에 JointConstraint 목표를 plan_only 로 요청해
충돌회피된 관절 경로를 받고, 승인되면 /right_arm_controller/follow_joint_trajectory
로 실행한다 — pose_commander.py 와 같은 실행 경로(제어권 게이트 동일), 목표만
Cartesian 대신 관절각이다. 플랜 자체가 planning scene(정적 박스 + 자가충돌)
검사를 통과한 것이므로 "걸리는 것 없는지"는 플랜 성공이 보증한다.

임피던스 제어 안전 (2026-08-16):
  · 실행 체인은 [이 도구] → trajectory_bridge(관절명 리맵만) → 제어 PC
    trajectory_receiver → 임피던스 제어기다. MoveIt 플랜의 웨이포인트 간격은
    ~0.1s 수준이라, 수신기가 보간을 안 하면 임피던스 팔이 스텝마다 튈 수 있다.
    → 전송 전에 궤적을 --resample-hz(기본 100Hz)로 선형 재샘플해 스텝을
    항상 미세하게 만든다 (수신기 보간 여부와 무관하게 안전).
  · 속도/가속 스케일 기본 0.1 — 기존 검증된 pose_commander.py 와 동일 프로필.

전제: MoveIt 트윈 기동(move_group 1개, joint_state_mode:=direct — 실기 시작상태 필요),
      source tools/env/setup_env.sh, /usr/bin/python3.
종료 코드: 0=성공(플랜/실행), 1=실패.
"""
from __future__ import annotations

import argparse
import sys
import time

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

import numpy as np

from control_msgs.action import FollowJointTrajectory
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint, MoveItErrorCodes
from trajectory_msgs.msg import JointTrajectoryPoint

from builtin_interfaces.msg import Duration


def resample_traj(traj, hz: float):
    """JointTrajectory 를 고정 rate 로 선형 재샘플 (positions + velocities).

    MoveIt 시간 매개변수화는 ~0.1s 간격의 웨이포인트를 준다. 임피던스 제어기로
    가는 경로에서 스텝을 미세하게 유지하기 위해 hz 간격으로 촘촘히 채운다.
    (stiffness 모듈 moveit_arm_mover.py 의 _resample 과 같은 접근)
    """
    pts = traj.points
    if len(pts) < 2 or hz <= 0:
        return traj
    t = np.array([p.time_from_start.sec + p.time_from_start.nanosec * 1e-9 for p in pts])
    Q = np.array([list(p.positions) for p in pts], dtype=float)
    has_vel = all(len(p.velocities) == Q.shape[1] for p in pts)
    V = (np.array([list(p.velocities) for p in pts], dtype=float) if has_vel else None)
    T = float(t[-1])
    ts = np.arange(0.0, T, 1.0 / hz)
    ts = np.append(ts, T)                      # 마지막 점(목표)은 정확히 포함
    new_pts = []
    for tk in ts:
        p = JointTrajectoryPoint()
        p.positions = [float(np.interp(tk, t, Q[:, j])) for j in range(Q.shape[1])]
        if V is not None:
            p.velocities = [float(np.interp(tk, t, V[:, j])) for j in range(Q.shape[1])]
        p.time_from_start = Duration(sec=int(tk), nanosec=int((tk - int(tk)) * 1e9))
        new_pts.append(p)
    step = np.max(np.abs(np.diff(np.array([q.positions for q in new_pts]), axis=0))) \
        if len(new_pts) > 1 else 0.0
    traj.points = new_pts
    print(f"[goto_q] 재샘플 {hz:.0f}Hz — {len(pts)} → {len(new_pts)} points, "
          f"최대 스텝 {step:.4f} rad (임피던스 안전)", flush=True)
    return traj


class GotoQ(Node):
    def __init__(self, a):
        super().__init__("goto_q")
        self.a = a
        self.joint_names = [f"{a.prefix}_joint{i}" for i in range(1, 8)]
        self._move = ActionClient(self, MoveGroup, "/move_action")
        self._exec = ActionClient(self, FollowJointTrajectory, a.traj_action)

    # ── plan ────────────────────────────────────────────────────────────
    def _joint_goal(self, joints):
        goal = MoveGroup.Goal()
        req = goal.request
        req.group_name = self.a.group
        req.num_planning_attempts = 5
        req.allowed_planning_time = self.a.plan_time
        req.max_velocity_scaling_factor = self.a.vel_scale
        req.max_acceleration_scaling_factor = self.a.acc_scale
        c = Constraints()
        for nm, val in zip(self.joint_names, joints):
            jc = JointConstraint()
            jc.joint_name = nm
            jc.position = float(val)
            jc.tolerance_above = jc.tolerance_below = 0.01
            jc.weight = 1.0
            c.joint_constraints.append(jc)
        req.goal_constraints = [c]
        goal.planning_options.plan_only = True
        return goal

    def _wait(self, future, timeout):
        deadline = time.monotonic() + timeout
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if future.done():
                return future.result()
        return None

    def plan(self, joints):
        """plan_only 요청. OMPL 은 랜덤 플래너라 같은 목표도 실패할 수 있어 재시도한다
        (pose_commander.py 와 동일한 이유). 성공 시 JointTrajectory 반환."""
        if not self._move.wait_for_server(timeout_sec=10.0):
            print("[goto_q] /move_action 서버 없음 — 트윈(move_group) 기동 필요", file=sys.stderr)
            return None
        goal = self._joint_goal(joints)
        for attempt in range(1, self.a.retries + 1):
            print(f"[goto_q] plan attempt {attempt}/{self.a.retries} ...", flush=True)
            gh = self._wait(self._move.send_goal_async(goal), 15.0)
            if gh is None or not gh.accepted:
                print("[goto_q] MoveGroup goal 거부/타임아웃", file=sys.stderr)
                continue
            res = self._wait(gh.get_result_async(), self.a.plan_time + 20.0)
            if res is None:
                print("[goto_q] 플랜 결과 타임아웃", file=sys.stderr)
                continue
            code = res.result.error_code.val
            if code == MoveItErrorCodes.SUCCESS:
                traj = res.result.planned_trajectory.joint_trajectory
                dur = (traj.points[-1].time_from_start.sec
                       + traj.points[-1].time_from_start.nanosec * 1e-9) if traj.points else 0.0
                print(f"[goto_q] 플랜 성공 — waypoints={len(traj.points)} duration={dur:.2f}s "
                      f"(충돌검사 통과)", flush=True)
                return traj
            print(f"[goto_q] 플랜 실패 error_code={code} — 재시도", file=sys.stderr)
            time.sleep(0.2)
        return None

    # ── execute ─────────────────────────────────────────────────────────
    def execute(self, traj):
        if not self._exec.wait_for_server(timeout_sec=10.0):
            print(f"[goto_q] {self.a.traj_action} 서버 없음 — 제어 PC 브리지 확인",
                  file=sys.stderr)
            return False
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj
        dur = (traj.points[-1].time_from_start.sec
               + traj.points[-1].time_from_start.nanosec * 1e-9) if traj.points else 0.0
        gh = self._wait(self._exec.send_goal_async(goal), 15.0)
        if gh is None or not gh.accepted:
            print("[goto_q] 실행 goal 거부/타임아웃 (제어권 owner 확인)", file=sys.stderr)
            return False
        res = self._wait(gh.get_result_async(), dur + 30.0)
        if res is None:
            print("[goto_q] 실행 결과 타임아웃", file=sys.stderr)
            return False
        ok = res.result.error_code == 0
        print(f"[goto_q] 실행 {'완료' if ok else f'실패 (error_code={res.result.error_code})'}",
              flush=True)
        return ok


def main():
    p = argparse.ArgumentParser(
        description="오른팔 관절각 7개 목표로 MoveIt 충돌회피 이동 (plan → confirm → execute)")
    p.add_argument("joints", nargs=7, type=float, metavar="J",
                   help="관절각 7개 [rad] (right_fr3_joint1..7 순)")
    p.add_argument("--group", default="right_arm")
    p.add_argument("--prefix", default="right_fr3", help="관절 이름 접두 (기본 right_fr3)")
    p.add_argument("--traj-action", default="/right_arm_controller/follow_joint_trajectory")
    p.add_argument("--vel-scale", type=float, default=0.1,
                   help="속도 스케일 (기본 0.1 = 기존 pose_commander 검증 프로필)")
    p.add_argument("--acc-scale", type=float, default=0.1)
    p.add_argument("--plan-time", type=float, default=5.0)
    p.add_argument("--retries", type=int, default=5)
    p.add_argument("--resample-hz", type=float, default=100.0,
                   help="전송 전 궤적 재샘플 주파수 [Hz], 0=재샘플 안 함")
    p.add_argument("--plan-only", action="store_true", help="플랜(충돌검사)만 하고 이동하지 않음")
    p.add_argument("--yes", action="store_true", help="실행 확인 프롬프트 생략")
    a = p.parse_args()

    rclpy.init()
    node = GotoQ(a)
    try:
        traj = node.plan(a.joints)
        if traj is None:
            print("[goto_q] 플랜 실패 — 이동하지 않음", file=sys.stderr)
            return 1
        if a.plan_only:
            print("[goto_q] --plan-only — 여기서 종료 (이동 없음)")
            return 0
        if not a.yes:
            ans = input("[goto_q] 실행할까요? [y/N] ").strip().lower()
            if ans not in ("y", "yes"):
                print("[goto_q] 취소")
                return 1
        traj = resample_traj(traj, a.resample_hz)
        return 0 if node.execute(traj) else 1
    except KeyboardInterrupt:
        return 130
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
