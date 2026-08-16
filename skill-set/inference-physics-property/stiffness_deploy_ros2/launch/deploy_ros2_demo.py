#!/usr/bin/env python3
"""deploy_ros2_demo.py — deploy_ros2.py 의 데모판: ecoflex2fruit 3속성 추론.

deploy_ros2.py(과일 강성 등급 데모) 대비 바뀐 것 4가지:
  1) 스퀴즈 모션 = collect_ros2_new.py 규약 — 파지 임계 6.0 N · 스퀴즈 임계
     6.0+5.0=11.0 N **고정**(과일별 임계 로드 안 함), 파지 close 2.5 s,
     파지 후 안정화 0.8 s(이 구간이 학습 ①pre_wait = 엔진 baseline), 스퀴즈
     호출·복귀 파라미터는 collect 와 동일(D.HAND_SQUEEZE_DURATION 등 그대로).
  2) 터미널 = 과일 선택이 아니라 **포즈 선택** (tomato/lemon/kiwi/plum/ecoflex.txt
     + pose1.txt~pose5.txt).
  3) GUI = gui/property_gui.py (크기·강성·무게 3속성 표시, /property/result).
  4) 모델 = deep_ws/src/ecoflex2fruit 챔피언(Champ_repair · 변형2+3+5 67ch) —
     mass·size·stif 동시 추론. 입력 전처리는 launch/ecoflex_engine.py 가
     학습 파이프라인을 복제한다. paxini 는 **/paxini/right/raw** 브리지 사용
     (변형3·5 는 127점 분포가 필요 — point0 트릭으로는 불가).

실행:  (자세한 절차·문제해결은 README.md §4-A)
  source env.sh
  python3 stiffness_deploy_ros2/launch/deploy_ros2_demo.py        # 포즈 번호 입력
  python3 stiffness_deploy_ros2/launch/deploy_ros2_demo.py --no-gui
전제: Ctrl 스택 + shm_state_publisher 가 /paxini/right/raw 를 발행 중이어야 함.
모델: models/ecoflex2fruit/Champ_repair_s42.pth 고정 (ECO_MODEL 무시 — 아래 main 참고).
"""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

import rclpy
from rclpy.executors import SingleThreadedExecutor

_LAUNCH_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_LAUNCH_DIR, ".."))  # project_root (core.* 해석)
sys.path.insert(0, _LAUNCH_DIR)                       # launch/

# ROS2 'launch' 패키지 선점 문제 회피 (deploy_ros2.py 와 동일 규약).
import importlib  # noqa: E402
sys.modules.setdefault(
    "launch.real_deploy_inference_old",
    importlib.import_module("real_deploy_inference_final"))

import deploy as D  # noqa: E402
# 브리지는 기존 것 재사용: 상태/명령 = deploy_ros2, paxini = raw(127점) 판.
from deploy_ros2 import Ros2ShmBridge  # noqa: E402
from deploy_ros2_exp_rawft import Ros2RawPaxiniBridge  # noqa: E402
from ecoflex_engine import (  # noqa: E402
    EcoflexPropertyEngine, load_labels, nearest_specimens, check_label_norm)
from property_result_pub import PropertyResultPublisher, spawn_property_gui  # noqa: E402

# ── ecoflex 개체 라벨 (실제값 대조) — 학습 라벨(oldstif)의 repo 내 사본 ──────
LABEL_DIR = os.path.join(_LAUNCH_DIR, "..", "labels", "object_labels_oldstif")
try:
    _LABELS, _LABEL_NORM = load_labels(LABEL_DIR)
except Exception as _e:  # noqa: BLE001  (라벨은 대조 표시용 — 없어도 추론은 계속)
    print(f"[deploy_ros2_demo] ⚠ 라벨 로드 실패({_e}) — 실제값 대조 없이 진행")
    _LABELS, _LABEL_NORM = {}, {}

# ── collect_ros2_new.py 와 동일한 고정 파라미터 ──────────────────────────────
GRIP_FORCE_THRESHOLD = 6.0    # 파지 접촉력 임계 [N] (고정)
SQUEEZE_DELTA_N = 5.0         # 스퀴즈 추가 압축량 [N] (고정) → 스퀴즈 임계 11.0
SQUEEZE_FORCE_THRESHOLD = GRIP_FORCE_THRESHOLD + SQUEEZE_DELTA_N
GRIP_CLOSE_DURATION = 2.5     # 파지 close 시간 [s] (deploy 기본 1.5 대신)
GRIP_SETTLE_SEC = 0.8         # 파지 후 안정화 [s] — 이 구간이 엔진 baseline(①pre_wait)

# ── 포즈 선택 (과일 선택 대체) ───────────────────────────────────────────────
POSES = [
    ("1", "tomato", "tomato.txt"), ("2", "lemon", "lemon.txt"),
    ("3", "kiwi", "kiwi.txt"), ("4", "plum", "plum.txt"),
    ("5", "ecoflex", "ecoflex.txt"),
    ("6", "pose1", "pose1.txt"), ("7", "pose2", "pose2.txt"),
    ("8", "pose3", "pose3.txt"), ("9", "pose4", "pose4.txt"),
    ("10", "pose5", "pose5.txt"),
]


def ask_pose() -> str:
    """포즈 파일 선택. 반환 = launch/ 의 포즈 txt 파일명."""
    menu = "  ".join(f"[{k}] {name}" for k, name, _ in POSES)
    while True:
        c = input(f"\n포즈 선택  {menu} : ").strip()
        for k, name, fn in POSES:
            if c == k or c.lower() == name:
                print(f"  → 포즈 = {name} ({fn})")
                return fn
        print(f"  1~{len(POSES)} 번호(또는 이름)로 입력하세요.")


def _grip(shm, paxini):
    """안전 위치 → 파지 (deploy_ros2._grip + collect 파라미터: close 2.5 s · 임계 6.0 N)."""
    shm.write_partial(servo_on=(D.HAND_SAFE_SERVO_ON,), hand_mode=(D.HAND_SAFE_MODE,))
    print("=================== 안전 위치 (손만) ===================")
    D.move_hand_to(shm, D.HAND_SAFE_POSITION, D.HAND_MOVE_DURATION)
    print("=================== 파지 위치 (손만) ===================")
    return D.move_hand_to_target_until_force(
        shm, paxini, D.HAND_GRIP_POINT, GRIP_CLOSE_DURATION, GRIP_FORCE_THRESHOLD,
        grip_curl=True)


def _print_properties(res, engine=None, labels=None, label_norm=None) -> None:
    print("\n" + "=" * 48)
    if res is not None:
        print("  [추론 결과] ecoflex2fruit 3속성")
        print(f"    무게(mass) = {res['mass']:.1f} g")
        print(f"    크기(size) = {res['size']:.1f} mm")
        print(f"    강성(stif) = {res['stif']:.3f}")
        if "anchor_stif" in res:
            print(f"    (앵커 경로 강성 = {res['anchor_stif']:.3f})")
        print(f"    사용 프레임 = {res['n_frames']}")
        # 라벨(oldstif) 최근접 개체 대조 — 실물테스트에서 실제값을 바로 읽는다.
        if labels:
            print(f"  [실제값 대조] 최근접 ecoflex 개체 (labels/object_labels_oldstif)")
            for rank, (oid, dist, lab) in enumerate(
                    nearest_specimens(res, labels, label_norm,
                                      stif_log=engine.stif_log, k=3), 1):
                print(f"    {rank}위 ecoflex_{oid:<2d} — "
                      f"mass {lab['mass']:6.1f} g (Δ{abs(res['mass']-lab['mass']):5.1f}) · "
                      f"size {lab['size']:5.2f} mm (Δ{abs(res['size']-lab['size']):4.2f}) · "
                      f"stif {lab['stif']:6.3f} (Δ{abs(res['stif']-lab['stif']):5.3f})"
                      f"   [정규화 거리 {dist:.3f}]")
    else:
        print("  [추론 결과] 샘플 부족 — 추론 불가")
    print("=" * 48 + "\n")


def _squeeze_and_infer(shm, paxini, engine, grip_position, result_pub):
    """baseline(0.8 s 안정화) → 스퀴즈(11.0 N 고정, collect 와 동일 호출) → 3속성 추론."""
    print(f"=============== 안정화 {GRIP_SETTLE_SEC}s (baseline 확보) ===============")
    result_pub.set_measuring()
    engine.capture_baseline(shm, paxini, sec=GRIP_SETTLE_SEC)
    print("=================== 스퀴즈 모션 ===================")
    D.move_hand_to_squeeze(
        shm, paxini, D.HAND_SAVE_POINT, D.HAND_SQUEEZE_DURATION,
        SQUEEZE_FORCE_THRESHOLD, grip_position,
        return_duration=D.HAND_SQUEEZE_RETURN_DURATION, engine=engine)
    res = engine.infer()
    _print_properties(res, engine, _LABELS, _LABEL_NORM)
    if res is not None:
        result_pub.set_result(res, stif_max=engine.norm_max)
    else:
        result_pub.set_error("샘플 부족 — 추론 불가")
    return res


def _ask_next_action() -> str:
    while True:
        c = input("\n다음 동작  [1] 다시 스퀴즈+추론  "
                  "[2] 안전위치 복귀 후 다시 스퀴즈  [3] 안전위치 복귀 후 종료 : ").strip()
        if c in ("1", "2", "3"):
            return c
        print("  1, 2, 3 중에서 입력하세요.")


def main() -> None:
    want_gui = "--no-gui" not in sys.argv
    if not want_gui:
        sys.argv = [a for a in sys.argv if a != "--no-gui"]
    if want_gui:
        spawn_property_gui()

    args = D.parse_args()
    marker_path = Path(args.marker_file).resolve()
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    D._set_squeeze_flag(False)

    rclpy.init()
    bridge = Ros2ShmBridge()
    # raw 브리지 하나가 힘 판정(진짜 Σ127)과 엔진 입력(127점 분포)을 모두 담당.
    paxini = Ros2RawPaxiniBridge(bridge)
    result_pub = PropertyResultPublisher()
    executor = SingleThreadedExecutor()
    executor.add_node(bridge)
    executor.add_node(result_pub)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    try:
        if not bridge.attach():
            raise SystemExit(
                "상태 토픽 미수신 — Dual_Arm_Hand_Ctrl 의 shm_state_publisher_node "
                "(및 C++ 컨트롤러)가 실행 중인지, ROS_DOMAIN_ID 가 맞는지 확인하세요.")
        if not paxini.attach():
            print("[deploy_ros2_demo] 경고: /paxini/right/raw 미수신 — 힘=0 으로 진행"
                  "(shm_state_publisher 의 raw 발행 확인).")

        bridge.safe_hand_servo_on(mode=D.HAND_SAFE_MODE)

        # 포즈 선택 → 포즈 로드. 임계값은 고정(collect 규약)이라 과일별 로드 없음.
        pose_file = ask_pose()
        D.set_pose_for_fruit(pose_file)
        print(f"[deploy_ros2_demo] 임계값 고정 — 파지 {GRIP_FORCE_THRESHOLD:.1f} N · "
              f"스퀴즈 {SQUEEZE_FORCE_THRESHOLD:.1f} N (collect_ros2_new 와 동일)")
        #   모델 고정 — models/ecoflex2fruit/Champ_repair_s42.pth (ECO_MODEL 무시)
        engine = EcoflexPropertyEngine(variant="gru")
        if _LABELS:
            print(f"[deploy_ros2_demo] 라벨 {len(_LABELS)}개체 로드 — {LABEL_DIR}")
            check_label_norm(engine, _LABEL_NORM)   # oldstif ↔ ckpt norm 정합 확인
        result_pub.set_idle()

        demo_id = 0
        print(f"\n--- 데모 {demo_id} ---")
        D._write_marker(marker_path, "S", demo_id)
        grip_position = _grip(bridge, paxini)
        _squeeze_and_infer(bridge, paxini, engine, grip_position, result_pub)
        D._write_marker(marker_path, "E", demo_id)

        while True:
            action = _ask_next_action()
            if action == "3":
                print("=================== 안전 위치 복귀 후 종료 ===================")
                D.move_hand_to(bridge, D.HAND_SAFE_POSITION, D.HAND_MOVE_DURATION)
                break
            demo_id += 1
            print(f"\n--- 데모 {demo_id} ---")
            D._write_marker(marker_path, "S", demo_id)
            if action == "2":
                grip_position = _grip(bridge, paxini)
            _squeeze_and_infer(bridge, paxini, engine, grip_position, result_pub)
            D._write_marker(marker_path, "E", demo_id)
        print("\n시퀀스 완료.")
    finally:
        D._set_squeeze_flag(False)
        bridge.detach()
        executor.shutdown()
        # spin 종료 순서 규약은 deploy_ros2.py 참고 (context 무효화 → join).
        if rclpy.ok():
            rclpy.shutdown()
        spin_thread.join(timeout=2.0)


if __name__ == "__main__":
    # 종료 규약(os._exit)은 deploy_ros2.py 와 동일 — 라이브 DDS 에서 teardown 멈춤 방지.
    _exit_code = 0
    try:
        main()
    except SystemExit as _e:
        if _e.code and not isinstance(_e.code, int):
            print(_e.code, file=sys.stderr)
        _exit_code = _e.code if isinstance(_e.code, int) else (1 if _e.code else 0)
    except KeyboardInterrupt:
        _exit_code = 130
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(_exit_code)
