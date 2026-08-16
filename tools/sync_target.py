#!/usr/bin/env python3
"""sync_target.py — 현재 측정 state 를 target 으로 동기화 (arm / hand / 동시).

    python3 tools/sync_target.py            # arm + hand 둘 다 (기본)
    python3 tools/sync_target.py --arm      # 팔만
    python3 tools/sync_target.py --hand     # 손만

왜 필요한가: 제어 PC 수신기는 마지막 target 을 래치한다. 이전 실행이 남긴 스테일
target 과 현재 실제 자세가 크게 다르면, 다음 제어 사이클/서보-온 순간 임피던스
제어기가 스테일 target 으로 한 번에 당겨 로봇이 튄다. 체인 시작 전에
"target = 지금 측정값" 으로 재앵커해 두면 그 점프가 원천 차단된다.

  · arm  : /joint_states_relay 에서 right_fr3_joint1..7 측정 → tools/goto_q.py 로
           그 관절각을 목표로 전송 (MoveIt 충돌검사 + 100Hz 재샘플 경로 재사용.
           현재=목표라 실제 이동은 사실상 0).
  · hand : /hand/right/joint_states(BEST_EFFORT, encoder counts) 측정 →
           /hand/right/cmd_mode=1(Position) 발행 후 /hand/right/q_target 에
           측정값을 N회 재발행 (BEST_EFFORT 유실 대비 — DEPLOY.md 의 시딩 ×2 관례).
           측정값을 그대로 명령하므로 손 움직임 없음. cmd_servo 는 건드리지 않는다.

종료 코드: 0=요청한 동기화 전부 성공, 1=실패.
전제: source tools/env/setup_env.sh, /usr/bin/python3.
arm 동기화는 move_group(트윈)이 필요하다 (goto_q 경유).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import rclpy
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray, Int32

HERE = Path(__file__).resolve().parent
GOTO_Q = str(HERE / "goto_q.py")

ARM_JOINTS = [f"right_fr3_joint{i}" for i in range(1, 8)]
BEST_EFFORT = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=1)


def read_state(node, topic, qos, pick, timeout):
    """JointState 1건 수신 → pick(msg) 결과 반환 (없으면 None)."""
    box = {}
    sub = node.create_subscription(JointState, topic, lambda m: box.update(m=m), qos)
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        rclpy.spin_once(node, timeout_sec=0.2)
        if "m" in box:
            out = pick(box["m"])
            if out is not None:
                node.destroy_subscription(sub)
                return out
            box.clear()          # 원하는 관절이 없는 샘플 — 다음 것
    node.destroy_subscription(sub)
    return None


def sync_arm(node, timeout):
    def pick(m):
        d = dict(zip(m.name, m.position))
        if all(j in d for j in ARM_JOINTS):
            return [d[j] for j in ARM_JOINTS]
        return None

    q = read_state(node, "/joint_states_relay", QoSProfile(depth=10), pick, timeout)
    if q is None:
        print("[sync] ✗ arm: /joint_states_relay 에서 right_fr3 관절 수신 실패", file=sys.stderr)
        return False
    print(f"[sync] arm 측정: {[round(v, 4) for v in q]}", flush=True)
    # goto_q 재사용: 현재=목표 → MoveIt 트리비얼 플랜 → 같은 실행 경로로 재앵커
    cmd = ["/usr/bin/python3", GOTO_Q] + [f"{v:.6f}" for v in q] + ["--yes"]
    rc = subprocess.call(cmd, timeout=120)
    ok = rc == 0
    print(f"[sync] arm {'✓ 동기화 완료' if ok else f'✗ 실패 (goto_q rc={rc})'}", flush=True)
    return ok


def sync_hand(node, repeat, timeout):
    def pick(m):
        return list(m.position[:16]) if len(m.position) >= 16 else None

    q = read_state(node, "/hand/right/joint_states", BEST_EFFORT, pick, timeout)
    if q is None:
        print("[sync] ✗ hand: /hand/right/joint_states 수신 실패", file=sys.stderr)
        return False
    print(f"[sync] hand 측정(counts): {[round(v) for v in q]}", flush=True)
    pub_mode = node.create_publisher(Int32, "/hand/right/cmd_mode", 10)
    pub_tgt = node.create_publisher(Float32MultiArray, "/hand/right/q_target", BEST_EFFORT)
    time.sleep(0.3)                                   # 구독 매칭 대기
    pub_mode.publish(Int32(data=1))                   # Position 모드 보장
    time.sleep(0.2)
    msg = Float32MultiArray(data=[float(v) for v in q])
    for _ in range(max(1, repeat)):                   # BEST_EFFORT 유실 대비 반복
        pub_tgt.publish(msg)
        time.sleep(0.05)
    print(f"[sync] hand ✓ 동기화 완료 (측정값 {repeat}회 재발행 — 움직임 없음)", flush=True)
    return True


def main():
    p = argparse.ArgumentParser(description="현재 state → target 동기화 (스테일 타겟 점프 방지)")
    p.add_argument("--arm", action="store_true", help="팔만 동기화")
    p.add_argument("--hand", action="store_true", help="손만 동기화")
    p.add_argument("--hand-repeat", type=int, default=5)
    p.add_argument("--timeout", type=float, default=10.0, help="상태 수신 대기 [s]")
    a = p.parse_args()
    do_arm = a.arm or not (a.arm or a.hand)     # 플래그 없으면 둘 다
    do_hand = a.hand or not (a.arm or a.hand)

    rclpy.init()
    node = rclpy.create_node("sync_target")
    try:
        ok = True
        if do_hand:
            ok = sync_hand(node, a.hand_repeat, a.timeout) and ok
        if do_arm:
            ok = sync_arm(node, a.timeout) and ok
        return 0 if ok else 1
    except KeyboardInterrupt:
        return 130
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
