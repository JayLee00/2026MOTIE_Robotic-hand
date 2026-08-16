#!/usr/bin/env python3
"""property_result_pub.py — 3속성(mass·size·stif) 추론 결과를 property GUI 로 발행.

stiffness_result_pub.py 와 같은 구조(연속 발행 + latched QoS + set_* API)이되,
대상 GUI 가 gui/property_gui.py 이고 메시지 스키마가 /property/result 규약이다:
  {"phase": idle|measuring|done|error, "sample": ...,
   "stiffness": .., "stiffness_std": .., "stiffness_max": ..,
   "weight": .., "diameter": .., "message": ..}
(property_gui.py 모듈 docstring 의 예시 JSON 과 필드명을 맞춘다. 등급/과일명 없음.)
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

# gui/property_gui.py 의 RESULT_TOPIC 과 반드시 동일.
RESULT_TOPIC = "/property/result"
PUBLISH_HZ = 5.0

_GUI_SCRIPT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "gui", "property_gui.py"))


def spawn_property_gui():
    """property GUI 를 별도 프로세스로 실행 (stiffness_result_pub.spawn_gui 와 동일 규약:
       실패해도 배포는 계속, 배포 종료 후에도 창 유지)."""
    if not os.path.exists(_GUI_SCRIPT):
        print(f"[property_gui] 스크립트 없음: {_GUI_SCRIPT} — GUI 생략")
        return None
    try:
        proc = subprocess.Popen([sys.executable, _GUI_SCRIPT])
        print(f"[property_gui] 결과 GUI 실행 (pid={proc.pid}).")
        return proc
    except Exception as e:  # noqa: BLE001  (GUI 실패가 배포를 막지 않도록)
        print(f"[property_gui] GUI 실행 실패({e}) — GUI 없이 계속")
        return None


def latched_qos() -> QoSProfile:
    return QoSProfile(
        depth=1,
        history=QoSHistoryPolicy.KEEP_LAST,
        reliability=QoSReliabilityPolicy.RELIABLE,
        durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    )


class PropertyResultPublisher(Node):
    """최신 상태 dict 를 타이머로 계속 publish. deploy 스레드는 set_* 로 갱신만."""

    def __init__(self) -> None:
        super().__init__("property_result_publisher")
        self._pub = self.create_publisher(String, RESULT_TOPIC, latched_qos())
        self._lock = threading.Lock()
        self._latest: dict = {"phase": "idle"}
        self._n = 0                      # sample 자동 채번 (sample_1, sample_2 …)
        self._timer = self.create_timer(1.0 / PUBLISH_HZ, self._tick)

    # ── deploy 스레드에서 호출하는 상태 갱신 API ──
    def set_idle(self) -> None:
        self._set({"phase": "idle"})

    def set_measuring(self, sample: str | None = None) -> None:
        self._n += 1
        self._sample = sample or f"sample_{self._n}"
        self._set({"phase": "measuring", "sample": self._sample})

    def set_result(self, res: dict, stif_max: float | None = None) -> None:
        """res = EcoflexPropertyEngine.infer() 반환 dict {mass,size,stif,...}."""
        payload = {
            "phase": "done",
            "sample": getattr(self, "_sample", f"sample_{self._n or 1}"),
            "stiffness": float(res["stif"]),
            "weight": float(res["mass"]),
            "diameter": float(res["size"]),
        }
        if stif_max is not None:
            payload["stiffness_max"] = float(stif_max)
        if "anchor_stif" in res:       # 앵커 경로 보조 추정 → 편차 참고치로 전달
            payload["stiffness_std"] = abs(float(res["stif"]) - float(res["anchor_stif"]))
        self._set(payload)

    def set_error(self, message: str) -> None:
        self._set({"phase": "error", "message": str(message)})

    # ── 내부 (stiffness_result_pub 와 동일한 직렬화 규약) ──
    def _set(self, payload: dict) -> None:
        with self._lock:
            self._latest = payload
            self._publish_locked()

    def _tick(self) -> None:
        with self._lock:
            self._publish_locked()

    def _publish_locked(self) -> None:
        msg = String()
        msg.data = json.dumps(self._latest, ensure_ascii=False)
        self._pub.publish(msg)
