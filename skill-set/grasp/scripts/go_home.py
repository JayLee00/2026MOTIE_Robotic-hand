#!/usr/bin/env python3
"""로봇 오른팔을 HOME(초기) 자세로 이동 — 명령 하나로.

arm.yaml 의 home.joint_values 를 /franka/right/q_target 로 스트리밍한다.
제어 PC 가 클램프(0.2 rad/msg, 3 rad/s)로 따라가므로 move_group 불필요.

용법 (호스트 py3.10, ROS 소싱 상태):
  /usr/bin/python3 scripts/go_home.py
  /usr/bin/python3 scripts/go_home.py --yes        # 확인 없이 바로
  /usr/bin/python3 scripts/go_home.py --speed 0.5  # 스트리밍 속도(수렴 스텝) 조절

전제: 제어 PC arm_q_target_receiver 실행 중 + (require_control 이면) 제어권 필요할 수 있음.
      HOME 이동은 충돌검사 없이 관절공간 직선 → 주변 안전 확인 후 실행.
"""
import argparse
import sys
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import Float64MultiArray, Float64, Float32MultiArray, Bool
from sensor_msgs.msg import JointState

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
from utils.arm import HOME_JOINT_VALUES
from utils.hand import HAND_INIT_ENC, HAND_STEPS, HAND_PERIOD

Q_TARGET_TOPIC   = '/franka/right/q_target'
STATE_TOPIC      = '/franka/right/joint_states'
SPEED_TOPIC      = '/franka/target_speed_factor'
HAND_TOPIC       = '/hand/right/q_target'      # Float32[16] count
HAND_SERVO_TOPIC = '/hand/right/cmd_servo'     # Bool servo on/off


class GoHome(Node):
    def __init__(self):
        super().__init__('go_home')
        self._pub   = self.create_publisher(Float64MultiArray, Q_TARGET_TOPIC, 10)
        self._spd   = self.create_publisher(Float64, SPEED_TOPIC, 10)
        self._hand      = self.create_publisher(Float32MultiArray, HAND_TOPIC, 10)
        self._hand_srv  = self.create_publisher(Bool, HAND_SERVO_TOPIC, 10)
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.cur = None
        self.create_subscription(JointState, STATE_TOPIC, self._cb, qos)

    def _cb(self, m: JointState):
        if len(m.position) >= 7:
            self.cur = list(m.position[:7])

    def home_hand(self):
        """손가락을 HAND_INIT_ENC(초기/대기 자세)로 이동. 서보 on 후 목표 스트리밍."""
        target = [float(v) for v in HAND_INIT_ENC]
        msg = Float32MultiArray(); msg.data = target
        # 1) 목표 먼저 세팅 → 서보 on (튐 방지, grasp.py 와 동일 순서)
        self._hand.publish(msg)
        time.sleep(0.1)
        self._hand_srv.publish(Bool(data=True))
        time.sleep(0.2)
        # 2) 목표 반복 발행 (HAND_STEPS × HAND_PERIOD 동안 → 제어 PC 가 이동)
        for _ in range(HAND_STEPS):
            self._hand.publish(msg)
            time.sleep(HAND_PERIOD)
        print(f"[go_home] 손가락 초기자세 완료 (HAND_INIT_ENC)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--yes', action='store_true', help='확인 없이 바로 실행')
    ap.add_argument('--speed', type=float, default=0.3, help='speed_factor (기본 0.3)')
    ap.add_argument('--timeout', type=float, default=12.0, help='최대 이동 시간(s)')
    ap.add_argument('--tol', type=float, default=0.02, help='도달 허용오차(rad)')
    ap.add_argument('--no_hand', action='store_true', help='손가락 초기화 생략 (팔만)')
    args = ap.parse_args()

    home = [float(v) for v in HOME_JOINT_VALUES]

    rclpy.init()
    node = GoHome()

    # 현재 자세 잠깐 수신
    t0 = time.time()
    while node.cur is None and time.time() - t0 < 3.0:
        rclpy.spin_once(node, timeout_sec=0.1)

    print(f"[go_home] HOME(rad) = {[round(v, 3) for v in home]}")
    if node.cur is not None:
        print(f"[go_home] 현재(rad) = {[round(v, 3) for v in node.cur]}")
        print(f"[go_home] 최대 Δ    = {max(abs(c-h) for c, h in zip(node.cur, home)):.3f} rad")
    else:
        print("[go_home] ⚠️ 현재 자세 미수신 (/franka/right/joint_states) — 제어 PC/도메인 확인")

    print("[go_home] HOME 으로 이동합니다 (충돌검사 없음, 주변 확인).")

    # speed_factor 설정
    spd = Float64(); spd.data = max(0.01, min(1.0, args.speed))
    node._spd.publish(spd)
    time.sleep(0.1)

    # HOME 까지 최소저크(quintic) 보간 스트리밍 → 부드러운 출발/정지.
    # (기존: 최종 목표를 한 번에 발행 → 제어 PC 클램프가 쫓아가며 출발이 튐)
    JOINT_VEL_LIMITS = [2.62, 2.62, 2.62, 2.62, 5.26, 4.18, 5.26]
    msg = Float64MultiArray(); msg.data = home
    if node.cur is not None:
        q0 = list(node.cur)
        # min-jerk 피크속도 = 1.875·Δ/T ≤ vlim×speed 가 되도록 T 결정
        T = max(1.5, 1.875 * max(
            abs(h - c) / (v * spd.data)
            for h, c, v in zip(home, q0, JOINT_VEL_LIMITS)))
        print(f"[go_home] 이동 중... (min-jerk {T:.1f}s, 50Hz 보간 스트리밍)")
        t0 = time.time()
        while rclpy.ok():
            tau = (time.time() - t0) / T
            if tau >= 1.0:
                break
            s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5   # quintic 0→1
            msg.data = [c + (h - c) * s for c, h in zip(q0, home)]
            node._pub.publish(msg)
            rclpy.spin_once(node, timeout_sec=0.0)
            time.sleep(0.02)
        msg.data = home
    else:
        print("[go_home] 이동 중... (현재자세 미수신 → 목표 직행 스트리밍)")

    # 수렴 확인 (최종 목표 발행 유지)
    t0 = time.time()
    reached = False
    while rclpy.ok() and time.time() - t0 < args.timeout:
        node._pub.publish(msg)
        rclpy.spin_once(node, timeout_sec=0.0)
        if node.cur is not None:
            err = max(abs(c - h) for c, h in zip(node.cur, home))
            if err < args.tol:
                reached = True
                print(f"[go_home] ✅ 도달 (err={err:.4f} rad)")
                break
        time.sleep(0.05)   # 20Hz 스트리밍

    if not reached:
        print("[go_home] ⚠️ timeout — 완전히 도달 못 함 (require_control 이면 제어권 필요할 수 있음)")
    # 마지막 타겟 유지 위해 몇 번 더 발행
    for _ in range(5):
        node._pub.publish(msg); time.sleep(0.05)

    # 손가락 초기자세
    if not args.no_hand:
        print("[go_home] 손가락 초기화 중...")
        node.home_hand()

    print("[go_home] 완료.")
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()
