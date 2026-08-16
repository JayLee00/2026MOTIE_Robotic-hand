#!/usr/bin/env python3
"""deploy_ros2_exp_forcecurl.py — deploy_ros2_exp 계측 + '스퀴즈 힘-도달 curl' 실험.

기존 코드(deploy.py / deploy_ros2.py / deploy_ros2_exp.py)를 **전혀 수정하지 않고**,
deploy.move_hand_to_squeeze 만 grip_curl=True 버전으로 monkeypatch 한다.
→ 스퀴즈 시 thumb 를 고정 위치까지만 닫던 것(기존=위치-정지) 대신, 접촉력이 임계에 도달할
   때까지 GRIP_CURL_MAX_DEG(기본 30°) 범위에서 2·3관절을 추가로 닫는다(=힘-도달 curl).
   (파지에 쓰던 grip_curl 로직을 스퀴즈에도 켜는 것 — 유일한 차이는 grip_curl=True 한 곳.)

목적(palm-up 유지): thumb 최대 Fz 가 실제로 오르는지 exp 계측으로 확인.
  · Fz 가 임계로 상승  → 코드로 힘 도달 가능(성공, B/C/D 로 더 밀어붙임).
  · ~5N 에서 정체       → 하드웨어/포즈 상한(코드로 불가) → Path B(재학습) 판단.

기존 코드와 비교 (동일 과일·회차로 각각 실행 후 [measure] 의 'thumb 최대 Fz' 만 비교):
  기존(위치-정지):  source env.sh && python3 stiffness_deploy_ros2/launch/deploy_ros2_exp.py
  힘-도달 curl:     source env.sh && python3 stiffness_deploy_ros2/launch/deploy_ros2_exp_forcecurl.py

curl 한계를 더 키워 재시도하려면 (Fz 가 정체할 때):
  GRIP_CURL_MAX_DEG 환경변수로 override — 예: FORCECURL_MAX_DEG=60 python3 ...forcecurl.py
"""
from __future__ import annotations

import os
import sys

_LAUNCH_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_LAUNCH_DIR, ".."))
sys.path.insert(0, _LAUNCH_DIR)

# ★ P2#5: deploy_ros2_exp 를 반드시 먼저 import 한다. 그 안의 `from deploy_ros2 import ...` 가
#   'launch.real_deploy_inference_old' → real_deploy_inference_final shim 을 설치하기 때문.
#   deploy 를 먼저 import 하면 구 엔진(real_deploy_inference_old)이 먼저 로드돼 setdefault shim 이
#   무효화된다(→ kiwi 미지원·USE_JKIN 경로·루프 58Hz 등 baseline 과 비교 불가).
import deploy_ros2_exp as EXP         # noqa: E402  (계측 흐름 전체 재사용 + shim 설치)
import deploy as D                    # noqa: E402


def _squeeze_forcecurl(shm, paxini, target_position, duration, threshold, return_position,
                       hold_sec=D.SQUEEZE_HOLD_SEC,
                       return_duration=D.HAND_SQUEEZE_RETURN_DURATION,
                       pre_wait_sec=D.SQUEEZE_PRE_WAIT_SEC, engine=None) -> None:
    """deploy.move_hand_to_squeeze 사본 + 스퀴즈 close 에 grip_curl=True (임계까지 추가 curl).
       비-엄지 finger 는 return_position(파지) 그대로 유지, thumb 만 추가 curl."""
    squeeze_target = list(return_position)
    for f in D.SQUEEZE_FORCE_FINGERS:
        for j in range(f * D.JOINTS_PER_FINGER, (f + 1) * D.JOINTS_PER_FINGER):
            squeeze_target[j] = target_position[j]

    print(f"  [squeeze/forcecurl] 스퀴즈 전 {pre_wait_sec}s 대기")
    D._hold_hand_position(shm, return_position, pre_wait_sec)

    D._set_squeeze_flag(True)
    if engine is not None:
        engine.reset()
    try:
        # ★ 기존과 유일한 차이: grip_curl=True → thumb 를 임계 힘까지 추가로 더 닫음.
        squeezed = D.move_hand_to_target_until_force(
            shm, paxini, squeeze_target, duration, threshold, mode=True,
            force_fingers=D.SQUEEZE_FORCE_FINGERS, grip_curl=True, engine=engine)
        print(f"  [squeeze/forcecurl] 힘-도달 curl 후 {hold_sec}s holding")
        D._hold_hand_position(shm, squeezed, hold_sec, engine=engine, paxini=paxini)
    finally:
        D._set_squeeze_flag(False)

    print("  [squeeze/forcecurl] thumb 만 파지 position 으로 복귀 (나머지 finger 는 파지 유지)")
    restore = list(squeezed)
    for f in D.SQUEEZE_FORCE_FINGERS:
        for j in range(f * D.JOINTS_PER_FINGER, (f + 1) * D.JOINTS_PER_FINGER):
            restore[j] = return_position[j]
    D.move_hand_to(shm, restore, return_duration)


def main() -> None:
    # (선택) curl 한계 override — Fz 가 정체하면 더 깊이 닫아보려고.
    max_deg = os.environ.get("FORCECURL_MAX_DEG")
    if max_deg:
        D.GRIP_CURL_MAX_DEG = float(max_deg)
        D._GRIP_CURL_MAX_COUNT = int(round(D.GRIP_CURL_MAX_DEG / 180.0 * 8192))

    D.move_hand_to_squeeze = _squeeze_forcecurl    # monkeypatch (원본 파일 미수정)

    print("=" * 60)
    print("[forcecurl] 스퀴즈 힘-도달 curl ON  "
          f"(force_fingers={D.SQUEEZE_FORCE_FINGERS}, GRIP_CURL_MAX_DEG={D.GRIP_CURL_MAX_DEG}°)")
    print("[forcecurl] 기존 deploy_ros2_exp.py 결과와 '[measure] thumb 최대 Fz' 를 비교하세요.")
    print("=" * 60)
    EXP.main()


if __name__ == "__main__":
    main()
