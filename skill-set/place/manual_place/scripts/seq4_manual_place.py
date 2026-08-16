#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
seq 4 수동 place 체인 (신규 개발판) — 기존 seq 4(vision_pipeline place 서버)는 건드리지 않는다.

stiffness(seq 3) 종료 자세에서 이어받아:
  ① 진입 시 arm/hand state 저장 → hand 지정 관절 +delta 로 살짝 더 꽉 쥠 (mode 1 유지)
  ② 공통 경유점 FRANKA_PLACE_1→2→3 → FRANKA_PLACE_SAFE
  ③ FRANKA_PLACE_TOP[slot] → FRANKA_PLACE_DOWN[slot]
  ④ hand: servo-OFF 창 프로토콜로 mode 2(current) 전환(정착 2s) → 벌리는 타겟으로 천천히 릴리즈
  ⑤ FRANKA_PLACE_TOP[slot] → (hand mode 1 복귀) → FRANKA_PLACE_SAFE → done
체인이 pick(1)→inhand(2)→stiffness(3)→여기(4) 를 돌 때마다 slot 1→2→…→5 로 진행한다.

사용법 — 반드시 `rs` 로 환경 접속 후 /usr/bin/python3 (conda python 금지, rclpy 충돌):
  포인트 찍기(티칭):  /usr/bin/python3 skill-set/place/scripts/seq4_manual_place.py --capture wp1
  설정/토픽 검증:     ... --check
  ★ 4번만 1회 테스트: ... --goto-start --slot 1
  ★ place 전용 루프:  ... --loop 3           (매 바퀴 stiffness 자세 5s 보간 복귀 → 그립 →
                                              place slot 1→2→3 → SAFE 종료. --lap-pause 5 로
                                              바퀴마다 과일 장전 대기 가능)
  체인 모드(arbiter): ... --chain            (slot 은 상태파일로 자동 진행, --slot 로 강제 가능)

안전:
  - 모든 Franka 이동은 min-jerk 보간을 /franka/right/q_target 로 100Hz 스트리밍
    (임피던스 모드 타겟 점프 방지). 첫 이동 전 현재 측정자세를 타겟으로 재앵커.
  - hand 모드 전환은 2026-08-16 확립 프로토콜 준수: servo OFF 창 안에서만 모드 변경,
    명령당 50ms 틱 분리, 시드 타겟 2회 발행(BEST_EFFORT 유실 대비).
  - Ctrl-C 시 즉시 발행 중단 → 마지막 타겟이 래치되어 그 자리에서 홀드된다.
"""

# ══════════════════════════════════════════════════════════════════════════
# ★★ 사용자 설정 — 모든 좌표는 여기서 채운다 ★★
#
# 값 얻는 법: 로봇을 원하는 자세로 만든 뒤
#   /usr/bin/python3 skill-set/place/scripts/seq4_manual_place.py --capture 이름
# 출력된 줄을 그대로 복사해 붙여넣는다. (arm: rad 7개 / hand: encoder counts 16개)
# ══════════════════════════════════════════════════════════════════════════

# stiffness 종료(=seq4 시작) 팔 자세. --goto-start 를 주면 여기로 먼저 이동 후 시퀀스 시작.
# 기본값 = 제시 자세(goto_q, pick 후 이동 자세 = 손안조작/숙도측정 실행 위치).
STIFFNESS_END_ARM_Q = [-0.2866, 1.4185, 0.2677, -1.9216, 0.7769, 1.2157, 2.0401]

# place 전용 루프(--loop)에서 쓸 손 그립 타겟 — 매 바퀴 stiffness 자세 도착 후 이 타겟으로
# 쥐고, 이어서 HAND_EXTRA_GRIP_DELTA 만큼 더 꽉 쥔다.
Hand_target_for_only_place = [3529, -3284, 399, 4008, -700, 2062, 2632, 1489,
                              123, 2560, 2062, 1912, 901, 3233, 3352, 1733]

# ── 공통 Franka 타겟 4개 (매 체인 공통 경유) ── rad 7개씩. None 이면 --check 에서 잡아준다.
FRANKA_PLACE_1 = [-0.2886, 1.2059, 0.5555, -1.8884, 0.6670, 1.2765, 2.2614]
FRANKA_PLACE_2 = [-0.2428, 1.1350, 0.8361, -1.7110, -0.2692, 1.2881, 2.2516]
FRANKA_PLACE_3 = [-0.1324, 1.0662, 1.0202, -1.8228, -1.3190, 1.4277, 1.4881]
FRANKA_PLACE_SAFE = [0.2730, 1.4354, 0.7673, -1.2935, -1.3057, 1.5331, 0.6687]

# ── place 위치 5개: top(위) / down(내려놓는 위치) 쌍 ── 각각 rad 7개.
FRANKA_PLACE_TOP_1 = [0.1100, 1.2750, 0.5625, -1.4245, -1.3052, 2.0815, 0.6448]
FRANKA_PLACE_TOP_2 = [0.1453, 1.3683, 0.7682, -1.7102, -1.2967, 1.8500, 0.8195]
FRANKA_PLACE_TOP_3 = [0.2941, 1.4991, 0.6110, -1.1356, -1.2973, 1.8774, 0.5042]
FRANKA_PLACE_TOP_4 = [0.1100, 1.2750, 0.5625, -1.4245, -1.3052, 2.0815, 0.6448]
FRANKA_PLACE_TOP_5 = [0.1100, 1.2750, 0.5625, -1.4245, -1.3052, 2.0815, 0.6448]

FRANKA_PLACE_DOWN_1 = [0.2729, 1.4507, 0.4134, -1.1378, -1.2316, 1.9252, 0.5060]
FRANKA_PLACE_DOWN_2 = [0.2391, 1.5225, 0.6352, -1.4935, -1.3008, 1.8041, 0.7889]
FRANKA_PLACE_DOWN_3 = [0.3176, 1.6369, 0.5460, -0.9733, -1.2980, 1.8304, 0.8309]
FRANKA_PLACE_DOWN_4 = [0.2729, 1.4507, 0.4134, -1.1378, -1.2316, 1.9252, 0.5060]
FRANKA_PLACE_DOWN_5 = [0.2729, 1.4507, 0.4134, -1.1378, -1.2316, 1.9252, 0.5060]

FRANKA_PLACE_TOP = [FRANKA_PLACE_TOP_1, FRANKA_PLACE_TOP_2, FRANKA_PLACE_TOP_3,
                    FRANKA_PLACE_TOP_4, FRANKA_PLACE_TOP_5]
FRANKA_PLACE_DOWN = [FRANKA_PLACE_DOWN_1, FRANKA_PLACE_DOWN_2, FRANKA_PLACE_DOWN_3,
                     FRANKA_PLACE_DOWN_4, FRANKA_PLACE_DOWN_5]

# ── place 전용 루프(--loop) 설정 ──
RETURN_TO_START_S = 5.0        # 매 바퀴 시작: SAFE → stiffness 자세로 천천히 보간 이동(초)
HAND_PLACE_GRIP_RAMP_S = 3.0   # stiffness 자세에서 그립 타겟으로 쥐는 보간 시간(초)
LAP_END_PAUSE_S = 2.0          # 복귀(stiffness 자세) 후 다음 바퀴 시작 전 대기(초)

# ── hand 설정 ──
# 더 꽉 쥘 관절(1-index, 16관절 = thumb j0..3, index j0..3, middle j0..3, ring j0..3 순)
HAND_EXTRA_GRIP_JOINTS_1IDX = [2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15]
HAND_EXTRA_GRIP_DELTA = 100          # counts (+100 ≈ 2.2°). --grip-delta 로 실행 시 변경 가능
HAND_EXTRA_GRIP_RAMP_S = 1.0         # 더 꽉 쥐기 보간 시간

HAND_RELEASE_MODE = 2                # 2 = current 모드 (deploy.py: "hand_mode=2 (current)")
HAND_MODE2_SETTLE_S = 2.0            # mode 2 전환 후 정착 시간 ("2초 동안 hand mode 2 로 변경")
HAND_RELEASE_TARGET = [4096, -4096, 0, 0,
                       0, 1000, 1000, 1000,
                       0, 1000, 1000, 1000,
                       0, 1000, 1000, 1000]   # 벌리는 타겟 (counts)
HAND_RELEASE_RAMP_S = 2.0            # 천천히 놓는 보간 시간(초). --release-ramp 로 변경 가능
HAND_AFTER_RELEASE_BACK_TO_MODE1 = True  # 릴리즈 후 top 복귀 시 mode 1(Position) 재진입

# ── 속도/주기 ──
ARM_SPEED_SCALE = 0.10               # 검증된 pose_commander/goto_q 프로필과 동일 배율
ARM_STREAM_HZ = 100.0                # q_target 스트리밍 주기 (수신기 클램프: 0.2rad/msg, 3rad/s)
ARM_MIN_SEGMENT_S = 2.0              # 세그먼트 최소 시간 (아무리 가까워도 이보다 빨리 안 움직임)
ARM_BLEND_MIN_SEG_S = 1.2            # 연속 통과(스플라인) waypoint 간 최소 시간 — 멈추지 않으므로 짧게
ARM_JOINT_VEL_LIMITS = [2.62, 2.62, 2.62, 2.62, 5.26, 4.18, 5.26]  # FR3 rad/s
HAND_STREAM_HZ = 100.0               # hand 보간 주기
HAND_HOLD_HZ = 20.0                  # arm 이동 중 hand 타겟 재발행 주기 (place 서버와 동일)
HAND_SWITCH_TICK_S = 0.05            # 모드 전환 프로토콜 명령 간격 (HAND_SWITCH_TICK_MS=50)

# 체인 모드 slot 자동 진행 상태파일 (같은 디렉토리)
import os as _os
SLOT_STATE_FILE = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                ".seq4_manual_slot")

# ══════════════════════════════════════════════════════════════════════════
# 이하 구현 — 좌표 수정만 할 거면 아래는 볼 필요 없음
# ══════════════════════════════════════════════════════════════════════════
import argparse
import json
import sys
import time
from datetime import datetime

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Float64MultiArray, Float32MultiArray, Int32, Bool
from sensor_msgs.msg import JointState

ARM_DOF = 7
HAND_DOF = 16

TOPIC_ARM_TARGET = "/franka/right/q_target"
TOPIC_ARM_STATE = "/franka/right/joint_states"
TOPIC_HAND_TARGET = "/hand/right/q_target"
TOPIC_HAND_STATE = "/hand/right/joint_states"
TOPIC_HAND_MODE = "/hand/right/cmd_mode"
TOPIC_HAND_SERVO = "/hand/right/cmd_servo"


def _log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] {msg}", flush=True)


def _clamp_counts(v: float) -> int:
    return max(-32768, min(32767, int(round(v))))


class Seq4ManualPlace(Node):
    def __init__(self):
        super().__init__("seq4_manual_place")
        best = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                          history=HistoryPolicy.KEEP_LAST, depth=1)
        rel = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
        # q_target 은 반드시 BEST_EFFORT (RELIABLE 이면 publish() 가 수 초 블로킹)
        self.pub_arm = self.create_publisher(Float64MultiArray, TOPIC_ARM_TARGET, best)
        self.pub_hand = self.create_publisher(Float32MultiArray, TOPIC_HAND_TARGET, best)
        self.pub_mode = self.create_publisher(Int32, TOPIC_HAND_MODE, rel)
        self.pub_servo = self.create_publisher(Bool, TOPIC_HAND_SERVO, rel)
        self.arm_q = None      # 측정 팔 관절각 [7] rad
        self.hand_q = None     # 측정 손 관절 [16] counts
        self.create_subscription(JointState, TOPIC_ARM_STATE, self._on_arm, best)
        self.create_subscription(JointState, TOPIC_HAND_STATE, self._on_hand, best)
        self._arm_cmd = None   # 이 노드가 마지막으로 명령한 팔 타겟
        self._hand_cmd = None  # 이 노드가 마지막으로 명령한 손 타겟
        self._hand_mode = 1    # 이 노드 관점의 현재 hand 모드

    # ── 상태 수신 ──
    def _on_arm(self, m: JointState):
        if len(m.position) >= ARM_DOF:
            self.arm_q = [float(m.position[j]) for j in range(ARM_DOF)]

    def _on_hand(self, m: JointState):
        if len(m.position) >= HAND_DOF:
            self.hand_q = [int(round(m.position[j])) for j in range(HAND_DOF)]

    def sleep_spin(self, sec: float):
        end = time.monotonic() + sec
        while time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=min(0.01, max(0.0, end - time.monotonic())))

    def wait_states(self, timeout: float = 10.0) -> bool:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.arm_q is not None and self.hand_q is not None:
                return True
        return False

    # ── 발행 헬퍼 ──
    def send_arm(self, q):
        msg = Float64MultiArray()
        msg.data = [float(v) for v in q]
        self.pub_arm.publish(msg)

    def send_hand(self, q):
        msg = Float32MultiArray()
        msg.data = [float(_clamp_counts(v)) for v in q]
        self.pub_hand.publish(msg)
        self._hand_cmd = [_clamp_counts(v) for v in q]

    def send_mode(self, mode: int):
        self.pub_mode.publish(Int32(data=int(mode)))
        self._hand_mode = int(mode)

    def send_servo(self, on: bool):
        self.pub_servo.publish(Bool(data=bool(on)))

    # ── 팔: 현재 측정자세를 타겟으로 재앵커 (스테일 타겟 점프 방지, sync_target 패턴) ──
    def arm_anchor(self):
        assert self.arm_q is not None
        for _ in range(3):
            self.send_arm(self.arm_q)
            self.sleep_spin(0.05)
        self._arm_cmd = list(self.arm_q)
        _log(f"arm 재앵커 완료: {['%.4f' % v for v in self.arm_q]}")

    # ── 팔: min-jerk 보간 스트리밍 이동 ──
    def arm_move(self, target, label: str, speed_scale: float, fixed_T: float = None):
        assert len(target) == ARM_DOF
        start = self._arm_cmd if self._arm_cmd is not None else list(self.arm_q)
        dmax = max(abs(t - s) / (v * speed_scale)
                   for t, s, v in zip(target, start, ARM_JOINT_VEL_LIMITS))
        T = max(ARM_MIN_SEGMENT_S, 1.875 * dmax)   # min-jerk 평균/최대속도 비 = 1.875
        if fixed_T is not None:
            # 지정 시간 사용 — 단, 관절 절대 속도한계보다 빨라지지는 않게 하한을 둔다
            t_floor = 1.875 * max(abs(t - s) / v
                                  for t, s, v in zip(target, start, ARM_JOINT_VEL_LIMITS))
            T = max(fixed_T, t_floor)
        steps = max(1, int(T * ARM_STREAM_HZ))
        dt = 1.0 / ARM_STREAM_HZ
        hold_every = max(1, int(ARM_STREAM_HZ / HAND_HOLD_HZ))
        _log(f"arm 이동 [{label}] : {T:.2f}s / {steps} steps")
        for i in range(1, steps + 1):
            tau = i / steps
            s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5   # quintic min-jerk 0→1
            q = [a + (b - a) * s for a, b in zip(start, target)]
            self.send_arm(q)
            # 이동 중 hand 타겟 20Hz 재발행 (mode 1 일 때만 — 래치/수신기 재기동 대비)
            if self._hand_cmd is not None and self._hand_mode == 1 and i % hold_every == 0:
                self.send_hand(self._hand_cmd)
            self.sleep_spin(dt)
        for _ in range(5):                                # 마지막 타겟 재발행 (유실 대비)
            self.send_arm(target)
            self.sleep_spin(0.02)
        self._arm_cmd = list(target)
        _log(f"arm 이동 [{label}] 완료")

    # ── 팔: 여러 waypoint 를 멈추지 않고 부드럽게 연속 통과 ──
    # Catmull-Rom(Hermite) 스플라인: 각 waypoint 를 정확히 지나되 내부 waypoint 에서
    # 속도가 끊기지 않고(연속), 시작·끝에서만 속도 0. 100Hz 스트리밍.
    def arm_move_through(self, targets, label: str, speed_scale: float):
        assert all(len(t) == ARM_DOF for t in targets)
        knots = [self._arm_cmd if self._arm_cmd is not None else list(self.arm_q)]
        knots += [list(t) for t in targets]
        n = len(knots)
        # knot 시각: 세그먼트별 소요시간 누적 (개별 이동과 같은 속도 공식, 최소시간만 짧게)
        times = [0.0]
        for a, b in zip(knots[:-1], knots[1:]):
            dmax = max(abs(y - x) / (v * speed_scale)
                       for x, y, v in zip(a, b, ARM_JOINT_VEL_LIMITS))
            times.append(times[-1] + max(ARM_BLEND_MIN_SEG_S, 1.875 * dmax))
        total = times[-1]
        # 접선: 양끝 0(정지 출발/도착), 내부 waypoint 는 Catmull-Rom 평균 기울기
        tangents = [[0.0] * ARM_DOF for _ in range(n)]
        for i in range(1, n - 1):
            span = times[i + 1] - times[i - 1]
            tangents[i] = [(knots[i + 1][j] - knots[i - 1][j]) / span
                           for j in range(ARM_DOF)]
        steps = max(1, int(total * ARM_STREAM_HZ))
        dt = 1.0 / ARM_STREAM_HZ
        hold_every = max(1, int(ARM_STREAM_HZ / HAND_HOLD_HZ))
        _log(f"arm 연속 통과 [{label}] : {total:.2f}s / waypoint {len(targets)}개 (무정지)")
        seg = 0
        for i in range(1, steps + 1):
            t = min(total, i * dt)
            while seg < n - 2 and t > times[seg + 1]:
                seg += 1
            seg_T = times[seg + 1] - times[seg]
            s = (t - times[seg]) / seg_T
            h00 = 2 * s**3 - 3 * s**2 + 1
            h10 = s**3 - 2 * s**2 + s
            h01 = -2 * s**3 + 3 * s**2
            h11 = s**3 - s**2
            q = [h00 * knots[seg][j] + h10 * seg_T * tangents[seg][j]
                 + h01 * knots[seg + 1][j] + h11 * seg_T * tangents[seg + 1][j]
                 for j in range(ARM_DOF)]
            self.send_arm(q)
            if self._hand_cmd is not None and self._hand_mode == 1 and i % hold_every == 0:
                self.send_hand(self._hand_cmd)
            self.sleep_spin(dt)
        for _ in range(5):                                # 마지막 타겟 재발행 (유실 대비)
            self.send_arm(knots[-1])
            self.sleep_spin(0.02)
        self._arm_cmd = list(knots[-1])
        _log(f"arm 연속 통과 [{label}] 완료")

    # ── 손: 선형 보간 이동 ──
    def hand_ramp(self, target, duration_s: float, label: str):
        assert len(target) == HAND_DOF
        start = self._hand_cmd if self._hand_cmd is not None else list(self.hand_q)
        steps = max(1, int(duration_s * HAND_STREAM_HZ))
        dt = 1.0 / HAND_STREAM_HZ
        _log(f"hand 보간 [{label}] : {duration_s:.1f}s")
        for i in range(1, steps + 1):
            t = i / steps
            q = [a + (b - a) * t for a, b in zip(start, target)]
            self.send_hand(q)
            self.sleep_spin(dt)
        self.send_hand(target)   # 재발행 (BEST_EFFORT 유실 대비)
        _log(f"hand 보간 [{label}] 완료")

    # ── 손: 안전 재무장 (safe_hand_servo_on 패턴: mode1 → 측정자세 시드 ×2 → servo ON) ──
    def hand_safe_arm(self):
        assert self.hand_q is not None
        self.send_mode(1)
        self.sleep_spin(0.1)
        self.send_hand(self.hand_q)
        self.sleep_spin(HAND_SWITCH_TICK_S)
        self.send_hand(self.hand_q)          # 재발행 (BEST_EFFORT 유실 대비)
        self.sleep_spin(0.2)
        self.send_servo(True)
        self.sleep_spin(0.2)
        _log("hand 안전 재무장 완료 (mode 1, 측정자세 시드, servo ON)")

    # ── 손: 모드 전환 (2026-08-16 확립 프로토콜: servo OFF 창 + 50ms 틱 분리 + 시드 ×2) ──
    def hand_mode_switch(self, new_mode: int, seed_target, settle_s: float, label: str):
        _log(f"hand 모드 전환 [{label}] : mode {self._hand_mode} → {new_mode} "
             f"(servo OFF 창, 시드 ×2, 정착 {settle_s:.1f}s)")
        tick = HAND_SWITCH_TICK_S
        self.send_servo(False); self.sleep_spin(tick)   # ① servo OFF — 전환 창 열기
        self.send_mode(new_mode); self.sleep_spin(tick)  # ② 모드 변경 (off 창 안에서만)
        self.send_hand(seed_target); self.sleep_spin(tick)   # ③ 타겟 시딩
        self.send_hand(seed_target); self.sleep_spin(tick)   #    재발행 (best_effort 유실 대비)
        self.send_servo(True)                            # ④ servo ON — 시드 타겟으로 유지
        self.sleep_spin(max(tick, settle_s))
        _log(f"hand 모드 전환 [{label}] 완료")


# ──────────────────────────────────────────────────────────────────────────
def _validate_config(n_slots_needed=None):
    problems = []

    def chk_arm(name, v):
        if v is None:
            problems.append(f"{name} 이 비어 있음 (--capture 로 찍어서 채울 것)")
        elif len(v) != ARM_DOF:
            problems.append(f"{name} 은 rad 7개여야 함 (현재 {len(v)}개)")

    chk_arm("STIFFNESS_END_ARM_Q", STIFFNESS_END_ARM_Q)
    chk_arm("FRANKA_PLACE_1", FRANKA_PLACE_1)
    chk_arm("FRANKA_PLACE_2", FRANKA_PLACE_2)
    chk_arm("FRANKA_PLACE_3", FRANKA_PLACE_3)
    chk_arm("FRANKA_PLACE_SAFE", FRANKA_PLACE_SAFE)
    if len(FRANKA_PLACE_TOP) != len(FRANKA_PLACE_DOWN):
        problems.append(f"FRANKA_PLACE_TOP({len(FRANKA_PLACE_TOP)}) 와 "
                        f"FRANKA_PLACE_DOWN({len(FRANKA_PLACE_DOWN)}) 개수가 다름")
    slots = range(len(FRANKA_PLACE_TOP)) if n_slots_needed is None else [n_slots_needed - 1]
    for i in slots:
        chk_arm(f"FRANKA_PLACE_TOP[{i}] (slot {i+1})", FRANKA_PLACE_TOP[i])
        chk_arm(f"FRANKA_PLACE_DOWN[{i}] (slot {i+1})", FRANKA_PLACE_DOWN[i])
    if len(HAND_RELEASE_TARGET) != HAND_DOF:
        problems.append("HAND_RELEASE_TARGET 은 16개여야 함")
    if len(Hand_target_for_only_place) != HAND_DOF:
        problems.append("Hand_target_for_only_place 은 16개여야 함")
    bad = [j for j in HAND_EXTRA_GRIP_JOINTS_1IDX if not 1 <= j <= 16]
    if bad:
        problems.append(f"HAND_EXTRA_GRIP_JOINTS_1IDX 범위 오류: {bad}")
    return problems


def _read_slot_state() -> int:
    try:
        with open(SLOT_STATE_FILE) as f:
            return int(json.load(f)["next_slot"])
    except Exception:
        return 1


def _write_slot_state(next_slot: int):
    with open(SLOT_STATE_FILE, "w") as f:
        json.dump({"next_slot": next_slot, "stamp": datetime.now().isoformat()}, f)


def do_capture(node: Seq4ManualPlace, label: str):
    if not node.wait_states():
        _log("!! 상태 토픽 미수신 — rs 환경/DOMAIN 9/발행자 확인")
        sys.exit(1)
    print()
    print(f"# ── capture {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (label: {label}) ──")
    print(f"{label}_ARM_Q = [" + ", ".join(f"{v:.4f}" for v in node.arm_q) + "]")
    print(f"{label}_HAND_Q = [" + ", ".join(str(v) for v in node.hand_q) + "]")
    print("# 팔 좌표는 위 줄의 리스트를 FRANKA_PLACE_* 자리에 그대로 붙여넣으면 된다.")
    print()


def do_check(node: Seq4ManualPlace):
    problems = _validate_config()
    ok = node.wait_states(timeout=5.0)
    print()
    print("── seq4_manual_place --check ──")
    print(f"arm 상태 수신: {'OK ' + str(['%.4f' % v for v in node.arm_q]) if node.arm_q else 'X (미수신)'}")
    print(f"hand 상태 수신: {'OK ' + str(node.hand_q) if node.hand_q else 'X (미수신)'}")
    if not ok:
        problems.append("상태 토픽 미수신 — rs / DOMAIN 9 / Control PC 발행 확인")
    n = len(FRANKA_PLACE_TOP)
    print(f"place slot 수: {n}, 다음 체인 slot(상태파일): {_read_slot_state()}")
    if problems:
        print(f"\n채워야 할 것 {len(problems)}건:")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)
    print("\n설정 완비 — 실행 가능")


def _place_motion(node: Seq4ManualPlace, slot: int, speed: float, release_ramp: float,
                  mode1_release: bool = False):
    """공통 place 동작 (진입 시 손은 쥔 상태·mode 1 가정):
    경유점 1→2→3 → SAFE → TOP[slot] → DOWN[slot] → 릴리즈 → TOP → SAFE.
    mode1_release=True 면 모드 전환 없이 mode 1(Position)에서 오픈 타겟으로 천천히 벌린다
    (--loop 테스트용, place 서버의 Position 릴리즈와 동일 방식)."""
    # 공통 경유점 1→2→3→SAFE — waypoint 에서 멈추지 않고 부드럽게 연속 통과
    node.arm_move_through([FRANKA_PLACE_1, FRANKA_PLACE_2, FRANKA_PLACE_3, FRANKA_PLACE_SAFE],
                          "경유 1→2→3→SAFE", speed)

    # 해당 slot top → down
    node.arm_move(FRANKA_PLACE_TOP[slot - 1], f"FRANKA_PLACE_TOP[{slot}]", speed)
    node.arm_move(FRANKA_PLACE_DOWN[slot - 1], f"FRANKA_PLACE_DOWN[{slot}]", speed)

    if mode1_release:
        # 릴리즈(mode 1 고정): 모드 전환 없이 오픈 타겟으로 천천히
        node.hand_ramp(HAND_RELEASE_TARGET, release_ramp, "천천히 벌리기 (mode 1 유지)")
        node.sleep_spin(0.5)
        # DOWN 이후 TOP→SAFE 복귀 — 멈추지 않고 부드럽게 연속 통과
        node.arm_move_through([FRANKA_PLACE_TOP[slot - 1], FRANKA_PLACE_SAFE],
                              f"TOP[{slot}]→SAFE 복귀", speed)
        return
    else:
        # 릴리즈: mode 2(current) 전환(프로토콜+정착 2s) → 벌리는 타겟 천천히
        node.hand_mode_switch(HAND_RELEASE_MODE, node._hand_cmd,
                              HAND_MODE2_SETTLE_S, "릴리즈 준비(mode 2)")
        node.hand_ramp(HAND_RELEASE_TARGET, release_ramp, "천천히 벌리기")
        node.sleep_spin(0.5)
        node.arm_move(FRANKA_PLACE_TOP[slot - 1], f"FRANKA_PLACE_TOP[{slot}] 복귀", speed)
        if HAND_AFTER_RELEASE_BACK_TO_MODE1:
            # 벌린 자세 유지한 채 Position 복귀 (시드 = 현재 명령값, 재파지 아님)
            node.hand_mode_switch(1, node._hand_cmd, 0.3, "mode 1 복귀")

    node.arm_move(FRANKA_PLACE_SAFE, "FRANKA_PLACE_SAFE 복귀", speed)


def run_place_only_loop(node: Seq4ManualPlace, laps: int, args):
    """place 전용 루프 — pick/inhand/stiffness 없이 place 만 laps 바퀴 돈다.
    첫 진입: SAFE 정렬 → stiffness 자세로 RETURN_TO_START_S 초 이동.
    매 바퀴(모두 hand mode 1 고정, 무정지 연속 통과):
      그립(3s) → +delta(1s) → 1→2→3→SAFE 연속 → TOP → DOWN → 릴리즈(2s)
      → TOP→SAFE→3→2→1→stiffness 연속 복귀 → 대기 → 다음 바퀴."""
    if not node.wait_states():
        _log("!! 상태 토픽 미수신 — 중단")
        sys.exit(1)

    _log(f"═══ place 전용 루프 시작 — {laps}바퀴 (slot 1..{laps}) ═══")
    node.arm_anchor()
    node.hand_safe_arm()   # mode 1 + 측정자세 시드 + servo ON (스테일 타겟 구동 방지)

    # 첫 진입: 어디에 있든 SAFE 정렬 후 stiffness 자세로 천천히 이동
    node.arm_move(FRANKA_PLACE_SAFE, "시작 정렬: FRANKA_PLACE_SAFE", args.speed_scale)
    node.arm_move(STIFFNESS_END_ARM_Q,
                  f"stiffness 자세 이동 ({RETURN_TO_START_S:.0f}s 보간)",
                  args.speed_scale, fixed_T=RETURN_TO_START_S)

    speed = args.speed_scale
    for lap in range(1, laps + 1):
        slot = lap
        _log(f"───── 바퀴 {lap}/{laps} — slot {slot} ─────")

        # ① (선택) 과일 장전 대기
        if args.lap_pause > 0:
            _log(f"과일 장전 대기 {args.lap_pause:.0f}s ...")
            node.sleep_spin(args.lap_pause)

        # ② 그립 타겟으로 쥐기 → 더 꽉
        base = list(Hand_target_for_only_place)
        node.hand_ramp(base, HAND_PLACE_GRIP_RAMP_S, "place 전용 그립 타겟으로 쥐기")
        node.sleep_spin(0.3)
        tighten = list(base)
        for j1 in HAND_EXTRA_GRIP_JOINTS_1IDX:
            tighten[j1 - 1] += args.grip_delta
        node.hand_ramp(tighten, HAND_EXTRA_GRIP_RAMP_S, f"더 꽉 쥐기 (+{args.grip_delta} counts)")
        node.sleep_spin(0.3)

        # ③ 전진: 경유 1→2→3→SAFE 무정지 연속 → TOP → DOWN
        node.arm_move_through([FRANKA_PLACE_1, FRANKA_PLACE_2, FRANKA_PLACE_3,
                               FRANKA_PLACE_SAFE], "경유 1→2→3→SAFE", speed)
        node.arm_move(FRANKA_PLACE_TOP[slot - 1], f"FRANKA_PLACE_TOP[{slot}]", speed)
        node.arm_move(FRANKA_PLACE_DOWN[slot - 1], f"FRANKA_PLACE_DOWN[{slot}]", speed)

        # ④ 릴리즈 (mode 1 고정, 모드 전환 없음)
        node.hand_ramp(HAND_RELEASE_TARGET, args.release_ramp, "천천히 벌리기 (mode 1 유지)")
        node.sleep_spin(0.5)

        # ⑤ 복귀: TOP→SAFE→3→2→1→stiffness 무정지 연속 통과
        node.arm_move_through([FRANKA_PLACE_TOP[slot - 1], FRANKA_PLACE_SAFE,
                               FRANKA_PLACE_3, FRANKA_PLACE_2, FRANKA_PLACE_1,
                               STIFFNESS_END_ARM_Q],
                              f"복귀 TOP[{slot}]→SAFE→3→2→1→stiffness", speed)

        _log(f"───── 바퀴 {lap}/{laps} 완료 — stiffness 자세, {LAP_END_PAUSE_S:.0f}s 대기 ─────")
        node.sleep_spin(LAP_END_PAUSE_S)

    _log(f"═══ place 전용 루프 종료: {laps}바퀴 전부 완료, stiffness 자세에서 정지 ═══")


def run_sequence(node: Seq4ManualPlace, slot: int, args):
    """slot 은 1-base."""
    grip_delta = args.grip_delta
    release_ramp = args.release_ramp
    speed = args.speed_scale

    if not node.wait_states():
        _log("!! 상태 토픽 미수신 — 중단")
        sys.exit(1)

    n_slots = len(FRANKA_PLACE_TOP)
    _log(f"═══ seq4 manual place 시작 — slot {slot}/{n_slots} ═══")

    # 0) hand 모드 1 확인 발행(단계 진입 관례) + 팔 재앵커
    node.send_mode(1)
    node.sleep_spin(0.1)
    node.arm_anchor()

    # 0-1) (테스트용) stiffness 종료 자세로 먼저 이동
    if args.goto_start:
        node.arm_move(STIFFNESS_END_ARM_Q, "stiffness 종료 자세로 이동(--goto-start)", speed)

    # 1) 진입 상태 저장
    saved_arm = list(node.arm_q)
    saved_hand = list(node.hand_q)
    _log(f"진입 arm state 저장: {['%.4f' % v for v in saved_arm]}")
    _log(f"진입 hand state 저장: {saved_hand}")

    # 1-1) 더 꽉 쥐기: 지정 관절 +delta (mode 1 그대로)
    tighten = list(saved_hand)
    for j1 in HAND_EXTRA_GRIP_JOINTS_1IDX:
        tighten[j1 - 1] += grip_delta
    node._hand_cmd = list(saved_hand)      # 보간 시작점 = 저장한 hand state
    node.send_hand(saved_hand)             # 시드 (best_effort ×2)
    node.sleep_spin(0.05)
    node.send_hand(saved_hand)
    node.sleep_spin(0.05)
    node.hand_ramp(tighten, HAND_EXTRA_GRIP_RAMP_S, f"더 꽉 쥐기 (+{grip_delta} counts)")
    node.sleep_spin(0.3)

    _place_motion(node, slot, speed, release_ramp)

    _log(f"═══ slot {slot} place 완료 (done) ═══")

    # 체인 진행 상태 갱신
    nxt = slot + 1 if slot < n_slots else 1
    _write_slot_state(nxt)
    _log(f"다음 체인 slot = {nxt} (상태파일 {SLOT_STATE_FILE})")


def main():
    ap = argparse.ArgumentParser(description="seq4 수동 place 체인 (신규 개발판)")
    ap.add_argument("--capture", metavar="LABEL", help="현재 arm/hand 상태를 붙여넣기용으로 출력")
    ap.add_argument("--check", action="store_true", help="설정/토픽 검증만 (로봇 안 움직임)")
    ap.add_argument("--slot", type=int, help="place 위치 번호 (1..N). 생략 시 상태파일의 다음 slot")
    ap.add_argument("--loop", type=int, metavar="N",
                    help="place 전용 루프: N바퀴 (slot 1..N). 매 바퀴 stiffness 자세로 "
                         f"{RETURN_TO_START_S:.0f}s 보간 복귀 → 그립 → place → SAFE 종료")
    ap.add_argument("--lap-pause", type=float, default=0.0,
                    help="--loop 에서 바퀴마다 stiffness 자세 도착 후 과일 장전 대기 초 (기본 0)")
    ap.add_argument("--goto-start", action="store_true",
                    help="시작 전 STIFFNESS_END_ARM_Q 로 이동 (단독 테스트용)")
    ap.add_argument("--chain", action="store_true",
                    help="arbiter 규약(SequenceClient 4, seq3 DONE 대기)으로 실행. "
                         "※ 기존 place 서버(skill_server)와 동시 실행 금지 — client_id 4 충돌")
    ap.add_argument("--grip-delta", type=int, default=HAND_EXTRA_GRIP_DELTA,
                    help=f"더 꽉 쥐기 counts (기본 {HAND_EXTRA_GRIP_DELTA})")
    ap.add_argument("--release-ramp", type=float, default=HAND_RELEASE_RAMP_S,
                    help=f"벌리기 보간 시간 s (기본 {HAND_RELEASE_RAMP_S})")
    ap.add_argument("--speed-scale", type=float, default=ARM_SPEED_SCALE,
                    help=f"팔 속도 배율 (기본 {ARM_SPEED_SCALE})")
    args = ap.parse_args()

    rclpy.init()
    node = Seq4ManualPlace()
    try:
        if args.capture:
            do_capture(node, args.capture)
            return
        if args.check:
            do_check(node)
            return

        # 안전장치: 명시적 동작 인자 없이는 로봇을 절대 움직이지 않는다.
        # (터미널 줄바꿈으로 인자가 잘려 들어와 의도치 않게 실행되는 사고 방지)
        if args.loop is None and args.slot is None and not args.chain:
            print("동작 인자가 없어 아무것도 하지 않습니다. 다음 중 하나를 명시하세요:")
            print("  --loop N      place 전용 루프 N바퀴 (예: --loop 3)")
            print("  --slot N      단일 place 1회")
            print("  --chain       arbiter 체인 모드")
            print("  --check / --capture LABEL")
            return

        if args.loop:
            if not 1 <= args.loop <= len(FRANKA_PLACE_TOP):
                print(f"--loop 은 1..{len(FRANKA_PLACE_TOP)}")
                sys.exit(1)
            problems = _validate_config()
            if problems:
                print("설정 미비 — --check 로 확인:")
                for p in problems:
                    print(f"  - {p}")
                sys.exit(1)
            run_place_only_loop(node, args.loop, args)
            return

        slot = args.slot if args.slot is not None else _read_slot_state()
        problems = _validate_config(n_slots_needed=slot)
        if problems:
            print("설정 미비 — --check 로 확인:")
            for p in problems:
                print(f"  - {p}")
            sys.exit(1)
        if not 1 <= slot <= len(FRANKA_PLACE_TOP):
            print(f"--slot 은 1..{len(FRANKA_PLACE_TOP)}")
            sys.exit(1)

        if args.chain:
            # arbiter 규약: seq 3 DONE 대기 → 제어권 4번 획득 → 실행 → 반납(=DONE 신호)
            from sequence_client import SequenceClient  # 소스된 ws 필요
            from dual_arm_msgs.msg import SequenceState
            client = SequenceClient(SequenceState.SEQ_PLACE)
            _log("seq 3(stiffness) DONE 대기 중...")
            client.wait_for_previous_done(SequenceState.SEQ_STIFFNESS)
            with client:
                run_sequence(node, slot, args)
            client.shutdown()
        else:
            run_sequence(node, slot, args)
    except KeyboardInterrupt:
        _log("!! Ctrl-C — 발행 중단 (마지막 타겟 래치 상태로 홀드됨)")
    finally:
        try:
            node.destroy_node()
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
