#!/usr/bin/env python3
"""deploy_ros2_exp_rawft.py — 힘 출처를 /raw(진짜 127점)로 바꾼 계측 실험 (H1 fix).

기존 코드(deploy.py / deploy_ros2.py / deploy_ros2_exp.py)를 전혀 수정하지 않고,
deploy_ros2_exp 가 쓰는 Ros2PaxiniBridge 만 '/raw 구독 버전'으로 monkeypatch 한다.

무엇이 달라지나:
  기존 브리지: /paxini/right/ft(4×3, 센서 FT블록) → (4,127,3) 의 point0 에만 적재
               → deploy 의 Σ127(점축 합)이 사실상 ft 로 붕괴 (힘-임계·추론 둘 다 ft).
  이 브리지 : /paxini/right/raw(4×127×3, 분포 원본) → 그대로 (4,127,3) 적재
               → Σ127 이 학습/수집(P2P)과 '동일한 진짜 127점 합' 이 됨.

기대(H1 이 맞다면):
  · [measure] thumb 최대 Fz 가 ~4N → 학습대(8~13N)로 상승, 스퀴즈 임계 도달.
  · 추론 절대강성이 기존 대비 변화 (입력 힘이 정상화되므로).
  · (부수효과) 힘-임계 컨트롤러가 정상 동작 → grip 7N 도 도달하며 물리적으로 제대로 쥠.
  H1 이 아니라면(ft ≈ Σ127) 변화가 없을 것 → H2(모션) 로 이동.

비교 실행 (동일 과일·회차):
  기존(ft, point0): source env.sh && python3 stiffness_deploy_ros2/launch/deploy_ros2_exp.py
  /raw(Σ127):       source env.sh && python3 stiffness_deploy_ros2/launch/deploy_ros2_exp_rawft.py

전제: shm_state_publisher 가 /paxini/right/raw 를 발행 중이어야 함
      (확인:  ros2 topic hz /paxini/right/raw  ·  ros2 topic echo /paxini/right/raw --once).
"""
from __future__ import annotations

import os
import sys
import threading
import time

import numpy as np

_LAUNCH_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_LAUNCH_DIR, ".."))
sys.path.insert(0, _LAUNCH_DIR)

from rclpy.node import Node                                        # noqa: E402
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy  # noqa: E402
from std_msgs.msg import Float32MultiArray                         # noqa: E402

import deploy_ros2 as DR          # noqa: E402
import deploy_ros2_exp as EXP     # noqa: E402  (계측 흐름 전체 재사용)

_POINTS = 127
_FINGERS = 4
_N = _FINGERS * _POINTS * 3       # 1524


class Ros2RawPaxiniBridge:
    """PaxiniShmReader.read() 흉내: /paxini/right/raw (4×127×3) 를 그대로 적재.
       기존 Ros2PaxiniBridge 와 동일 인터페이스 (node, topic) — topic 은 무시하고 /raw 구독."""

    def __init__(self, node: Node, topic: str = "/paxini/right/ft"):
        # topic 인자는 호환 위해 받되, 실제로는 /raw 를 구독한다.
        self._node = node
        self._lock = threading.Lock()
        self._tac = None                                       # (4,127,3)
        self._seq = 0
        raw_topic = "/paxini/right/raw"
        state_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                               history=HistoryPolicy.KEEP_LAST, depth=1)
        node.create_subscription(Float32MultiArray, raw_topic, self._on_raw, state_qos)
        print(f"[rawft] 힘 출처 = {raw_topic} (4×127×3, 진짜 Σ127) — point0 트릭 없음")

    def _on_raw(self, m):
        if len(m.data) >= _N:
            tac = np.asarray(m.data[:_N], np.float32).reshape(_FINGERS, _POINTS, 3)
            with self._lock:
                self._tac = tac
                self._seq += 1

    def attach(self, timeout_sec: float = 3.0) -> bool:
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout_sec:
            with self._lock:
                if self._tac is not None:
                    return True
            time.sleep(0.05)
        with self._lock:
            return self._tac is not None

    def read(self):
        with self._lock:
            tac = None if self._tac is None else self._tac.copy()
            seq = self._seq
        if tac is None:
            return (np.zeros((_FINGERS, _POINTS, 3), np.float32),
                    np.array(0, np.int64), np.array(0, np.int8), np.array(-1, np.int64))
        return (tac, np.array(time.monotonic_ns(), np.int64),
                np.array(1, np.int8), np.array(int(seq), np.int64))


def main() -> None:
    # deploy_ros2_exp 가 이름으로 참조하는 Ros2PaxiniBridge 를 /raw 버전으로 교체.
    EXP.Ros2PaxiniBridge = Ros2RawPaxiniBridge
    DR.Ros2PaxiniBridge = Ros2RawPaxiniBridge     # (안전용) 원 모듈도 함께 교체

    print("=" * 60)
    print("[rawft] 힘 출처 = /paxini/right/raw 의 진짜 Σ127 (기존 = /ft point0)")
    print("[rawft] 기존 deploy_ros2_exp.py 결과와 '[measure] thumb 최대 Fz' · 추론값을 비교하세요.")
    print("=" * 60)
    EXP.main()


if __name__ == "__main__":
    main()
