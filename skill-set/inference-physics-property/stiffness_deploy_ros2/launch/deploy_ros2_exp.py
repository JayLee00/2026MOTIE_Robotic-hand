#!/usr/bin/env python3
"""deploy_ros2_exp.py — 실험/계측 전용 (기존 코드 보존).

deploy_ros2.py 의 브리지·헬퍼·시퀀스를 **그대로 재사용**하고, 추론 엔진만 얇게 감싸
(MeasureEngine) 스퀴즈 1회당 계측치를 출력한다. 기존 파일은 전혀 수정하지 않는다.

이 파일의 목적(Path A 개선 단계별):
  #3 (지금)   : 계측 — 스퀴즈당 add_sample 호출 수 · 유효 rate · downsample 스텝 · 도달 힘(Fz)
  #1 (다음)   : FACTOR 를 실제 rate 에 맞춰 시퀀스를 학습과 일치
  #2 (다음)   : 스퀴즈에 '힘 도달까지 curl' 추가
  #4 (다음)   : mN 캐시 / 불필요 구독 제거로 루프 rate↑

실행:
  source env.sh
  python3 stiffness_deploy_ros2/launch/deploy_ros2_exp.py   # 실행 후 과일 번호 입력
"""

from __future__ import annotations

import os
import sys
import time
import threading
from pathlib import Path

import numpy as np

import rclpy
from rclpy.executors import SingleThreadedExecutor

# deploy_ros2 재사용을 위한 sys.path 규약 (deploy_ros2.py 와 동일).
_LAUNCH_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_LAUNCH_DIR, ".."))
sys.path.insert(0, _LAUNCH_DIR)

# deploy_ros2 를 import 하면 path/shim 설정 + 브리지/헬퍼가 준비된다. (기존 파일 미수정)
from deploy_ros2 import (  # noqa: E402
    Ros2ShmBridge, Ros2PaxiniBridge, _grip, _squeeze_and_infer, _ask_next_action)
import deploy as D                          # noqa: E402
import real_deploy_inference_final as _RE   # noqa: E402  (FACTOR / MIN_LEN 참조)


class MeasureEngine:
    """실제 추론 엔진을 감싸 계측만 추가. deploy 는 이걸 '진짜 엔진'처럼 사용한다.
    (reset → add_sample×N → infer 흐름을 그대로 위임하면서 스퀴즈 통계를 집계)"""

    def __init__(self, engine):
        self._e = engine
        self._reset_stats()

    def __getattr__(self, name):
        # 정의 안 된 속성(fruit 등)은 실제 엔진으로 위임. (_e 없으면 AttributeError)
        return getattr(object.__getattribute__(self, "_e"), name)

    def _reset_stats(self) -> None:
        self._t = []                            # add_sample 호출 시각 (유효 rate 계산용)
        self._valid = 0                         # engine 이 실제 적재한(valid) 샘플 수
        self._calls = 0                         # add_sample 호출 총수
        self._peak = np.zeros(4, np.float32)    # finger별 최대 Fz (스퀴즈 구간)

    # ── 엔진 인터페이스 (deploy 가 호출) ───────────────────────────────
    def reset(self):
        self._reset_stats()                     # 이번 스퀴즈 통계 새로 시작
        return self._e.reset()

    def add_sample(self, shm, paxini):
        self._calls += 1
        self._t.append(time.monotonic())
        tac, _t, valid, _seq = paxini.read()    # 힘 추적(엔진과 중복 read 지만 캐시라 무해)
        if int(valid):
            fz = np.nan_to_num(tac)[:, :, 2].sum(axis=1)   # finger별 Fz 합 (4,)
            self._peak = np.maximum(self._peak, fz.astype(np.float32))
        ok = self._e.add_sample(shm, paxini)
        if ok:
            self._valid += 1
        return ok

    def infer(self):
        self._report()                          # 추론 직전에 이번 스퀴즈 계측 출력
        return self._e.infer()

    # ── 계측 리포트 ────────────────────────────────────────────────
    def _report(self) -> None:
        if self._calls == 0:
            print("[measure] add_sample 호출 없음 (스퀴즈 구간 미수집)")
            return
        dur = (self._t[-1] - self._t[0]) if len(self._t) >= 2 else 0.0
        rate = (len(self._t) - 1) / dur if dur > 0 else 0.0
        steps = self._valid // _RE.FACTOR
        print("\n" + "-" * 52)
        print("[measure] 스퀴즈 계측 (학습 대비 격차 확인용)")
        print(f"  add_sample 호출 = {self._calls},  valid(적재) = {self._valid}")
        print(f"  유효 rate       = {rate:6.1f} Hz   (수집시간 {dur:.2f}s)")
        print(f"  downsample 스텝 = {steps}   (FACTOR={_RE.FACTOR}, MIN_LEN={_RE.MIN_LEN})")
        print(f"  finger별 최대 Fz = {np.round(self._peak, 2).tolist()}")
        print(f"    thumb 최대 = {self._peak[0]:.2f} N   (스퀴즈 임계 {D.SQUEEZE_FORCE_THRESHOLD} N)")
        print("-" * 52 + "\n")


def main() -> None:
    args = D.parse_args()
    marker_path = Path(args.marker_file).resolve()
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    D._set_squeeze_flag(False)

    rclpy.init()
    bridge = Ros2ShmBridge()
    paxini = Ros2PaxiniBridge(bridge, "/paxini/right/ft")
    executor = SingleThreadedExecutor()
    executor.add_node(bridge)
    threading.Thread(target=executor.spin, daemon=True).start()

    try:
        if not bridge.attach():
            raise SystemExit(
                "상태 토픽 미수신 — Dual_Arm_Hand_Ctrl 스택/ROS_DOMAIN_ID 확인.")
        if not paxini.attach():
            print("[measure] 경고: /paxini/right/ft 미수신 — 힘=0 으로 진행")

        bridge.safe_hand_servo_on(mode=D.HAND_SAFE_MODE)

        # 과일 선택 → 모델/포즈/임계값/엔진 준비 (deploy_ros2 와 동일). 엔진만 MeasureEngine 로 감쌈.
        fruit = D.ask_fruit()
        model_path, pose_file, force_zero = D.resolve_fruit_config(fruit)
        D.set_pose_for_fruit(pose_file)
        D.set_thresholds_for_fruit(fruit)
        engine = MeasureEngine(D.StiffnessInferenceEngine(
            model_path=model_path, fruit=fruit, label_dir=D.LABEL_DIR, force_zero=force_zero))
        print(f"[measure] 준비 완료. 과일={fruit}, 모델={Path(model_path).name}")

        # 첫 실행: 안전 → 파지 → 스퀴즈 → (계측 출력) 추론
        demo_id = 0
        print(f"\n--- 데모 {demo_id} (계측) ---")
        D._write_marker(marker_path, "S", demo_id)
        grip_position = _grip(bridge, paxini)
        _squeeze_and_infer(bridge, paxini, engine, grip_position)
        D._write_marker(marker_path, "E", demo_id)

        # 추론 후 메뉴 (deploy_ros2 와 동일): 1=재스퀴즈, 2=안전복귀 후 재파지→스퀴즈, 3=종료
        while True:
            action = _ask_next_action()
            if action == "3":
                print("=================== 안전 위치 복귀 후 종료 ===================")
                D.move_hand_to(bridge, D.HAND_SAFE_POSITION, D.HAND_MOVE_DURATION)
                break
            demo_id += 1
            print(f"\n--- 데모 {demo_id} (계측) ---")
            D._write_marker(marker_path, "S", demo_id)
            if action == "2":
                grip_position = _grip(bridge, paxini)
            _squeeze_and_infer(bridge, paxini, engine, grip_position)
            D._write_marker(marker_path, "E", demo_id)
        print("\n계측 완료.")
    finally:
        D._set_squeeze_flag(False)
        bridge.detach()
        executor.shutdown()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
