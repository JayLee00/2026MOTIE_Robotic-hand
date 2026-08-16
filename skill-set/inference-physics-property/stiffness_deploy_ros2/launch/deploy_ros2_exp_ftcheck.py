#!/usr/bin/env python3
"""deploy_ros2_exp_ftcheck.py — '힘 출처' 판별 진단 (H1 vs H2).

기존 코드를 전혀 수정하지 않는 read-only 진단 노드. 두 토픽을 동시에 구독해
손가락별로 다음을 나란히 출력한다:

    ft_Fz   = /paxini/right/ft  의 finger별 Fz            ← deploy 가 실제로 읽는 값
    Σ127_Fz = /paxini/right/raw 의 finger별 Σ(127점) Fz   ← 학습/수집(P2P)이 쓴 값

배경(코드로 확인됨):
  · deploy_ros2.Ros2PaxiniBridge 는 /ft(4×3)를 (4,127,3) 의 point0 에만 실어
    'Σ127' 이 사실상 ft 로 붕괴한다 → 힘-임계 컨트롤러도, 추론 입력도 모두 ft.
  · shm_state_publisher 는 /ft(=shm Paxini_ft, 센서 FT블록)와
    /raw(=shm Paxini_tac, 127점 분포)를 '다른 SHM 필드'에서 발행 → 구조적으로 ft ≠ Σ127.

판별:
  · 스퀴즈 중 thumb 의  ft_Fz ≪ Σ127_Fz  → H1 확정.
      → fix: 브리지를 /raw 구독으로 교체 (deploy_ros2_exp_rawft.py). 재학습·힘·자세 불필요.
  · ft_Fz ≈ Σ127_Fz  (둘 다 ~4N)         → H2. 실제 힘이 약함 → 모션/서보 점검.

사용:
  터미널 A: (로봇/센서 스택 구동 — shm_state_publisher, paxini writer 등)
  터미널 B: source env.sh && python3 stiffness_deploy_ros2/launch/deploy_ros2_exp_ftcheck.py
  손으로(또는 다른 배포 실행으로) thumb 를 스퀴즈시키며 두 값을 비교.
"""
from __future__ import annotations

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Float32MultiArray

_POINTS = 127
_FINGERS = 4
_FZ = 2  # axis index of Fz


class FtCheck(Node):
    def __init__(self, side: str = "right", period: float = 0.5):
        super().__init__("deploy_ros2_exp_ftcheck")
        self._ft = None       # (4,3)
        self._sum127 = None   # (4,3)  raw 를 127점 합산한 (4,3)
        self._ft_seen = self._raw_seen = 0
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(Float32MultiArray, f"/paxini/{side}/ft", self._on_ft, qos)
        self.create_subscription(Float32MultiArray, f"/paxini/{side}/raw", self._on_raw, qos)
        self.create_timer(period, self._report)
        print(f"[ftcheck] 구독: /paxini/{side}/ft  vs  /paxini/{side}/raw (Σ127)  — {period}s 주기")
        print("[ftcheck] thumb(finger0) 의 ft_Fz 와 Σ127_Fz 를 비교하세요.\n")

    def _on_ft(self, m):
        if len(m.data) >= _FINGERS * 3:
            self._ft = np.asarray(m.data[:_FINGERS * 3], np.float32).reshape(_FINGERS, 3)
            self._ft_seen += 1

    def _on_raw(self, m):
        n = _FINGERS * _POINTS * 3
        if len(m.data) >= n:
            raw = np.asarray(m.data[:n], np.float32).reshape(_FINGERS, _POINTS, 3)
            self._sum127 = np.nan_to_num(raw, nan=0.0).sum(axis=1)   # (4,3)
            self._raw_seen += 1

    def _report(self):
        if self._ft is None or self._sum127 is None:
            print(f"[ftcheck] 대기 — ft수신={self._ft_seen}  raw수신={self._raw_seen} "
                  f"({'ft 없음' if self._ft is None else 'raw 없음'})")
            return
        ft_fz = self._ft[:, _FZ]
        s127_fz = self._sum127[:, _FZ]
        print("  finger |    ft_Fz |  Σ127_Fz |  Σ127/ft")
        for i in range(_FINGERS):
            tag = " (thumb)" if i == 0 else ""
            ratio = (s127_fz[i] / ft_fz[i]) if abs(ft_fz[i]) > 1e-3 else float("nan")
            print(f"     {i}   | {ft_fz[i]:7.2f}N | {s127_fz[i]:7.2f}N | {ratio:6.2f}x{tag}")
        print("")


def main():
    rclpy.init()
    node = FtCheck()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
