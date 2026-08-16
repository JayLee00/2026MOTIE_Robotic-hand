#!/usr/bin/env python3
"""test_moveit_mover.py — MoveIt 충돌회피 끝점/관절 이동(moveit_arm_mover) 테스트 도구.

기본은 **plan-only**(충돌회피 경로만 계산, 팔 안 움직임 — 안전). `--execute` 를 주면
그 경로를 q_target 으로 재생해 **실제로 이동**한다(확인 프롬프트 후).

사용 (source env.sh 후; 팔 이동엔 플래닝 PC move_group 필요):
  # 현재 팔 관절각 출력(ARM_POSES/캡처용)
  python3 stiffness_deploy_ros2/launch/test_moveit_mover.py --print-current
  # 관절 goal 플랜(이동 없음)
  python3 stiffness_deploy_ros2/launch/test_moveit_mover.py --joints 0.71 0.68 -0.24 -2.05 -0.58 2.10 -1.37
  # Cartesian goal 플랜(플랜지 right_fr3_link8, world)
  python3 stiffness_deploy_ros2/launch/test_moveit_mover.py --pose 0.45 -0.10 0.35
  # 실제 이동(팔 움직임!)
  python3 stiffness_deploy_ros2/launch/test_moveit_mover.py --joints ... --execute

전제: 제어 PC 상태 토픽(/franka/right/joint_states) + 플래닝 PC move_group(joint_state_mode:=direct)
  이 같은 ROS_DOMAIN_ID(9)로 실행 중. 상태 토픽 없으면 현재자세/실행 불가(플랜만 시도).
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time

import numpy as np

import rclpy
from rclpy.executors import SingleThreadedExecutor

_LAUNCH_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_LAUNCH_DIR, ".."))
sys.path.insert(0, _LAUNCH_DIR)

import deploy_ros2 as DR                          # noqa: E402  (Ros2ShmBridge = 팔 상태/이동 경로)
from moveit_arm_mover import MoveItArmMover        # noqa: E402

Ros2ShmBridge = DR.Ros2ShmBridge


def _wait_arm(bridge, timeout: float = 3.0) -> bool:
    """팔 상태(_arm_pos)만 대기 — 손 없이도 팔 테스트 가능.
    (bridge.attach 는 팔+손 상태를 둘 다 요구하므로 arm-only 테스트엔 쓰지 않는다.)"""
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        with bridge._lock:
            if bridge._arm_pos is not None:
                return True
        time.sleep(0.05)
    with bridge._lock:
        return bridge._arm_pos is not None


def _report(plan) -> None:
    pts = plan.points
    T = pts[-1].time_from_start.sec + pts[-1].time_from_start.nanosec * 1e-9
    print(f"[test] 플랜 성공: waypoints={len(pts)}, duration={T:.2f}s")
    print(f"       관절 = {list(plan.joint_names)}")
    print(f"       시작 = {[round(x, 3) for x in pts[0].positions]}")
    print(f"       도착 = {[round(x, 3) for x in pts[-1].positions]}")


def main():
    ap = argparse.ArgumentParser(description="MoveIt 충돌회피 팔 이동 테스트 (기본 plan-only)")
    ap.add_argument("--joints", nargs=7, type=float, metavar="J", help="관절 goal (7개)")
    ap.add_argument("--pose", nargs="+", type=float, metavar="V",
                    help="Cartesian goal: x y z [qx qy qz qw]")
    ap.add_argument("--print-current", action="store_true", help="현재 팔 관절각만 출력(캡처용)")
    ap.add_argument("--execute", action="store_true", help="실제 이동(팔 움직임!). 기본은 plan-only")
    ap.add_argument("--rate", type=float, default=100.0, help="실행 재생 rate(Hz)")
    ap.add_argument("--group", default="right_arm")
    ap.add_argument("--ee-link", default="right_fr3_link8")
    ap.add_argument("--frame", default="world")
    args = ap.parse_args()

    rclpy.init()
    bridge = Ros2ShmBridge()
    mover = MoveItArmMover(group=args.group, ee_link=args.ee_link, frame=args.frame)
    ex = SingleThreadedExecutor()
    ex.add_node(bridge)
    ex.add_node(mover)
    threading.Thread(target=ex.spin, daemon=True).start()
    try:
        # 팔 상태만 대기(손 없이도 arm-only 테스트 가능). bridge.attach 는 손까지 요구하므로 안 씀.
        attached = _wait_arm(bridge, 3.0)
        if not attached:
            print("[test] ⚠ 팔 상태 토픽(/franka/right/joint_states) 미수신 — 현재자세/실행 불가"
                  "(플랜은 move_group 자체 상태로 시도).")

        # 현재 자세 출력 (캡처용)
        if args.print_current:
            if not attached:
                raise SystemExit("현재자세 읽기 실패(상태 토픽 미수신).")
            cur = [float(x) for x in bridge.read().Arm_j_pos[0]]
            print(f"[test] 현재 팔 관절각(7) = [{', '.join(f'{x:.5f}' for x in cur)}]")
            print(f'       ARM_POSES 형식  = {{"joints": [{", ".join(f"{x:.5f}" for x in cur)}]}}')
            return

        if not mover.wait_ready(3.0):
            print("[test] ⚠ /move_action(move_group) 미수신 — 플랜 실패 예상. 플래닝 PC 확인.")

        # 목표 → 플랜 (plan-only)
        if args.joints:
            plan = mover.plan_to_joints(args.joints)
        elif args.pose:
            pos = args.pose[:3]
            ori = args.pose[3:7] if len(args.pose) >= 7 else None
            plan = mover.plan_to_pose(pos, ori)
        else:
            raise SystemExit("사용: --joints j0..j6 | --pose x y z [qx qy qz qw] | --print-current")

        if plan is None:
            raise SystemExit("[test] 플랜 실패 — move_group/자세/충돌 확인.")
        _report(plan)

        if attached:
            cur = np.asarray(bridge.read().Arm_j_pos[0], float)
            gap = float(np.max(np.abs(cur - np.asarray(plan.points[0].positions, float))))
            print(f"[test] 현재 vs 플랜 시작 최대차 = {gap:.3f} rad "
                  f"{'(큼 — 시작상태 불일치 주의)' if gap > 0.15 else ''}")

        # 실행 (확인 후)
        if args.execute:
            ans = input("\n[test] 실제로 팔을 이동합니다(q_target 재생). 계속? [y/N] ").strip().lower()
            if ans != "y":
                print("[test] 취소.")
                return
            ok = mover.replay_trajectory(bridge, plan, rate_hz=args.rate)
            print(f"[test] 이동 {'완료' if ok else '실패/중단(시작 간격/빈 경로)'}")
        else:
            print("[test] plan-only — 실제 이동 없음. 이동하려면 --execute.")
    finally:
        bridge.detach()
        ex.shutdown()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    _code = 0
    try:
        main()
    except SystemExit as e:
        if e.code and not isinstance(e.code, int):
            print(e.code, file=sys.stderr)
        _code = e.code if isinstance(e.code, int) else (1 if e.code else 0)
    except KeyboardInterrupt:
        _code = 130
    sys.stdout.flush(); sys.stderr.flush()   # P2#4: os._exit 전 flush (파이프/리다이렉트 시 마지막 출력 유실 방지)
    os._exit(_code)
