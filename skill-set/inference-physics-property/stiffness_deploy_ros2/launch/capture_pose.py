#!/usr/bin/env python3
"""capture_pose.py — 팔을 원하는 자세로 잡고 '관절각'을 캡처해 ARM_POSES 형식으로 출력.

방법 B(관절각) 캡처 도구. 로봇을 원하는 자세로 옮긴 뒤(RViz MotionPlanning 드래그+Execute,
또는 free-drive/핸드가이딩), 이 도구에서 포즈 이름을 입력하면 그 순간의 팔 관절각(7)을 저장한다.
빈 줄로 종료하면 collect_ros2.py 의 ARM_POSES 에 바로 붙일 형식으로 출력한다.

왜 관절각인가: 고정 teach 자세는 관절각이 IK 모호성 없이 정확 재현된다(palm-up/down 처럼
손목 방향만 다른 자세에 특히 유리). 캡처 순서는 /franka/right/joint_states 의 name 순서이며,
MoveItArmMover.joint_names(기본 right_fr3_joint1..7)와 같아야 한다(아래 출력에서 확인).

실행:
  source env.sh
  python3 stiffness_deploy_ros2/launch/capture_pose.py
  # 반복: 로봇 자세 잡기 → 이름 입력(safe/grip/palm_up/palm_down) → Enter → 저장
  # 빈 줄 Enter 로 종료 → ARM_POSES 블록 출력 (복사해서 collect_ros2.py 에 붙여넣기)

전제: 제어 PC 스택이 /franka/right/joint_states 를 발행 중(ROS_DOMAIN_ID=9).
"""
from __future__ import annotations

import os
import sys
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import JointState

ARM_DOF = 7
EXPECTED_NAMES = [f"right_fr3_joint{i}" for i in range(1, 8)]   # MoveItArmMover 기본 순서


class JointReader(Node):
    def __init__(self, topic: str):
        super().__init__("capture_pose")
        self._lock = threading.Lock()
        self._pos = None
        self._names = None
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(JointState, topic, self._on, qos)

    def _on(self, m):
        if len(m.position) >= ARM_DOF:
            with self._lock:
                self._pos = [float(m.position[j]) for j in range(ARM_DOF)]
                self._names = [str(m.name[j]) for j in range(ARM_DOF)] if len(m.name) >= ARM_DOF else None

    def read(self):
        with self._lock:
            return (list(self._pos) if self._pos else None,
                    list(self._names) if self._names else None)

    def wait(self, timeout: float = 5.0) -> bool:
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            if self.read()[0] is not None:
                return True
            time.sleep(0.05)
        return False


def main():
    topic = sys.argv[1] if len(sys.argv) > 1 else "/franka/right/joint_states"
    rclpy.init()
    node = JointReader(topic)
    ex = SingleThreadedExecutor()
    ex.add_node(node)
    threading.Thread(target=ex.spin, daemon=True).start()

    poses = {}
    try:
        if not node.wait(5.0):
            raise SystemExit(f"{topic} 미수신 — 제어 PC 스택/ROS_DOMAIN_ID(=9) 확인.")
        _, names = node.read()
        print(f"[capture] 토픽 = {topic}")
        print(f"[capture] 관절 이름 = {names}")
        if names and names != EXPECTED_NAMES:
            print(f"[capture] ⚠ 이름 순서가 MoveItArmMover 기본({EXPECTED_NAMES})과 다름 — "
                  "MoveItArmMover(joint_names=...) 로 맞추거나 dict 형식 사용 권장.")
        print("[capture] 로봇을 원하는 자세로 옮긴 뒤 포즈 이름 입력(빈 줄=종료).")
        print("          예: safe / grip / palm_up / palm_down\n")

        while True:
            name = input("포즈 이름 (Enter=종료) : ").strip()
            if not name:
                break
            pos, _ = node.read()
            if pos is None:
                print("  아직 관절 상태 미수신 — 잠시 후 재시도.")
                continue
            poses[name] = pos
            # 구분자 = 공백 하나(콤마·괄호 없음) → test_moveit_mover.py --joints 뒤에 그대로 복붙.
            # (아래 ARM_POSES 블록은 파이썬 dict 이므로 콤마를 유지한다)
            print(f"  ✔ '{name}' 캡처: {' '.join(f'{v:.5f}' for v in pos)}\n")

        if poses:
            print("\n" + "=" * 72)
            print("아래를 collect_ros2.py 의 ARM_POSES 에 붙여넣으세요 (해당 키만 교체):\n")
            print("ARM_POSES = {")
            for k, v in poses.items():
                pad = " " * max(1, 11 - len(k))
                print(f'    "{k}":{pad}{{"joints": [{", ".join(f"{x:.5f}" for x in v)}]}},')
            print("}")
            print("=" * 72)
        else:
            print("캡처된 포즈 없음.")
    finally:
        ex.shutdown()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        pass
    sys.stdout.flush(); sys.stderr.flush()   # P2#4: os._exit 는 파이썬 정리를 건너뜀 → 파이프/tee 시 마지막 출력(ARM_POSES) 유실 방지
    os._exit(0)
