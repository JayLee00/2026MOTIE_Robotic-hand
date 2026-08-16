#!/usr/bin/env python3
"""stiffness_result_pub.py — 강성 추론 결과를 GUI로 연속 발행하는 ROS2 퍼블리셔.

deploy_task3_ros2 가 이 노드를 executor 에 추가하고,
  · 스퀴즈 시작 전  set_measuring()  → phase="measuring" (GUI: "측정 중...")
  · 추론 완료 후     set_result()     → phase="done"      (GUI: 막대+등급 표시)
를 호출한다.

타이머가 최신 상태를 PUBLISH_HZ 로 **계속(연속)** 발행하고, QoS 는
transient_local(latched) 이므로 나중에 켠 GUI(stiffness_gui.py)도 즉시 현재
상태를 받는다. 즉 "한 번만 보내는 게 아니라 상태를 지속적으로 알려주는" 방식.

payload: std_msgs/String, JSON (dual_arm_msgs 같은 커스텀 메시지 빌드 불필요).
  {phase, fruit, stiffness, cls, cname, norm_min, norm_max, boundaries, class_names}
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading

from rclpy.node import Node
from rclpy.qos import (QoSProfile, QoSReliabilityPolicy,
                       QoSDurabilityPolicy, QoSHistoryPolicy)
from std_msgs.msg import String

# GUI(stiffness_gui.py) 와 반드시 동일해야 하는 상수.
RESULT_TOPIC = "/stiffness/result"
PUBLISH_HZ = 5.0

# 결과 GUI 스크립트 (launch/ 기준 ../gui/stiffness_gui.py).
_GUI_SCRIPT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "gui", "stiffness_gui.py"))


def spawn_gui():
    """결과 GUI(stiffness_gui.py)를 별도 프로세스로 실행하고 Popen 반환(실패 시 None).
       현재 실행 중인 python(env.sh 의 시스템 python3)을 그대로 써서 ROS 환경을 상속한다.
       GUI 는 부가기능이라 실패해도 배포는 계속되고, 배포 종료 후에도 결과를 계속 보여주려
       프로세스를 자동으로 죽이지 않는다(창을 닫으면 정리). 이미 떠 있으면 --no-gui 로 실행."""
    if not os.path.exists(_GUI_SCRIPT):
        print(f"[stiffness_gui] 스크립트 없음: {_GUI_SCRIPT} — GUI 생략")
        return None
    try:
        proc = subprocess.Popen([sys.executable, _GUI_SCRIPT])
        print(f"[stiffness_gui] 결과 GUI 실행 (pid={proc.pid}).")
        return proc
    except Exception as e:  # noqa: BLE001  (GUI 실패가 배포를 막지 않도록)
        print(f"[stiffness_gui] GUI 실행 실패({e}) — GUI 없이 계속")
        return None


def latched_qos() -> QoSProfile:
    """latched(=늦게 접속해도 최신값 1개 수신) + reliable. 구독자도 동일해야 함."""
    return QoSProfile(
        depth=1,
        history=QoSHistoryPolicy.KEEP_LAST,
        reliability=QoSReliabilityPolicy.RELIABLE,
        durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    )


class StiffnessResultPublisher(Node):
    """최신 상태 dict 를 타이머로 계속 publish. deploy 스레드는 set_* 로 갱신만."""

    def __init__(self) -> None:
        super().__init__("stiffness_result_publisher")
        self._pub = self.create_publisher(String, RESULT_TOPIC, latched_qos())
        self._lock = threading.Lock()
        self._latest: dict = {"phase": "idle"}
        # 타이머(executor 스레드)로 연속 발행 + set_* 시 즉시 1회 발행.
        # 모든 publish 는 _lock 안에서만 → 두 스레드가 겹쳐도 직렬화(경합 없음).
        # 즉시 발행이 있어 set_result 직후 바로 종료해도 "done" 이 latched 로 나간다.
        self._timer = self.create_timer(1.0 / PUBLISH_HZ, self._tick)

    # ── deploy 스레드에서 호출하는 상태 갱신 API (갱신 + 즉시 발행) ──
    def set_idle(self) -> None:
        self._set({"phase": "idle"})

    def set_measuring(self, fruit, norm_min, norm_max, boundaries, class_names) -> None:
        self._set({
            "phase": "measuring",
            "fruit": fruit,
            "norm_min": float(norm_min), "norm_max": float(norm_max),
            "boundaries": [float(b) for b in boundaries],
            "class_names": list(class_names),
        })

    def set_result(self, fruit, stiffness, cls, cname,
                   norm_min, norm_max, boundaries, class_names) -> None:
        self._set({
            "phase": "done",
            "fruit": fruit,
            "stiffness": float(stiffness), "cls": int(cls), "cname": cname,
            "norm_min": float(norm_min), "norm_max": float(norm_max),
            "boundaries": [float(b) for b in boundaries],
            "class_names": list(class_names),
        })

    def set_error(self, message: str) -> None:
        self._set({"phase": "error", "message": str(message)})

    # ── 내부 ──
    def _set(self, payload: dict) -> None:
        with self._lock:
            self._latest = payload
            self._publish_locked()

    def _tick(self) -> None:
        with self._lock:
            self._publish_locked()

    def _publish_locked(self) -> None:
        """반드시 self._lock 을 잡은 상태에서 호출 (publish 직렬화)."""
        msg = String()
        msg.data = json.dumps(self._latest, ensure_ascii=False)
        self._pub.publish(msg)
