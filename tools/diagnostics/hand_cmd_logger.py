#!/usr/bin/env python3
"""hand_cmd_logger.py — 핸드 명령 버스 블랙박스 (cmd_mode / cmd_servo / q_target / 측정).

2026-08-16 핸드 런어웨이 사고("내려놓으러 갈 때 mode 0 + 타겟 튐") 때 버스에 무엇이
흘렀는지 기록이 없어 만들었다. 러너가 체인 시작 시 자동 기동해 logs/run_*/hand_cmd.log
에 남긴다. 단독 실행도 가능:

    python3 tools/diagnostics/hand_cmd_logger.py

기록 규칙:
  · /hand/right/cmd_mode(Int32) · /hand/right/cmd_servo(Bool) — 모든 수신을 기록
  · /hand/right/q_target(Float32[16]) — 모든 수신을 기록 (16개 값 그대로)
  · /hand/right/mode(Int32MultiArray, 실제 모드 피드백) · joint_states(측정 counts)
    — 값이 바뀔 때 기록
  · ⚠ ALERT: cmd_mode==0(Voltage) 상태에서 |q_target| max > 600 수신
    = Position counts 가 duty 로 들어가는 폭주 패턴 — 사고 재현 시 결정적 증거
"""
from __future__ import annotations

import sys
import time

import rclpy
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float32MultiArray, Int32, Int32MultiArray

# 모든 구독을 BEST_EFFORT 로 — RELIABLE/BEST_EFFORT 발행자 모두와 호환된다
QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                 history=HistoryPolicy.KEEP_LAST, depth=50)

COUNTS_LIKE = 600.0            # |duty| 클램프(±500)보다 크면 counts 로 간주


def main():
    rclpy.init()
    n = rclpy.create_node("hand_cmd_logger")
    t0 = time.time()
    state = {"cmd_mode": None, "actual_mode": None, "servo": None, "last_meas": None}

    def log(tag, msg, alert=False):
        ts = time.strftime("%H:%M:%S") + f".{int((time.time() % 1) * 1000):03d}"
        flag = "⚠ ALERT " if alert else ""
        print(f"[{ts} +{time.time() - t0:8.2f}s] {flag}{tag:9s} {msg}", flush=True)

    def on_cmd_mode(m):
        prev = state["cmd_mode"]
        state["cmd_mode"] = int(m.data)
        log("cmd_mode", f"{prev} -> {m.data}"
            + ("  (0=Voltage: 이제 q_target 은 raw duty 로 해석됨)" if m.data == 0 else ""),
            alert=(m.data == 0))

    def on_servo(m):
        state["servo"] = bool(m.data)
        log("cmd_servo", f"{'ON' if m.data else 'OFF'}")

    def on_qtar(m):
        v = [round(float(x), 1) for x in m.data]
        mx = max(abs(x) for x in v) if v else 0.0
        bad = state["cmd_mode"] == 0 and mx > COUNTS_LIKE
        log("q_target", f"max|v|={mx:7.1f}  {v}",
            alert=bad)
        if bad:
            log("q_target", "→ Voltage 모드에서 counts 급 타겟! (폭주 패턴 — 발행자 추적 필요)",
                alert=True)

    def on_actual_mode(m):
        cur = list(m.data)
        if cur != state["actual_mode"]:
            log("mode(fb)", f"{state['actual_mode']} -> {cur}",
                alert=(0 in cur if cur else False))
            state["actual_mode"] = cur

    def on_js(m):
        cur = [round(float(x)) for x in m.position[:16]]
        prev = state["last_meas"]
        if prev is None or max(abs(a - b) for a, b in zip(cur, prev)) > 150:
            log("measured", f"{cur}")
            state["last_meas"] = cur

    n.create_subscription(Int32, "/hand/right/cmd_mode", on_cmd_mode, QOS)
    n.create_subscription(Bool, "/hand/right/cmd_servo", on_servo, QOS)
    n.create_subscription(Float32MultiArray, "/hand/right/q_target", on_qtar, QOS)
    n.create_subscription(Int32MultiArray, "/hand/right/mode", on_actual_mode, QOS)
    n.create_subscription(JointState, "/hand/right/joint_states", on_js, QOS)

    log("start", "핸드 명령 블랙박스 시작 — cmd_mode/cmd_servo/q_target 전건 기록")
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    except Exception:                       # SIGTERM 등 외부 셧다운 — 조용히 종료
        pass
    finally:
        n.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
