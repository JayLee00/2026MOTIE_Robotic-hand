#!/usr/bin/env python3
"""deploy_task3_ros2_demo.py — deploy_task3_ros2.py 의 데모판: ecoflex2fruit 3속성 추론.

deploy_task3_ros2.py("이미 파지한 상태에서 스퀴즈만" + Pick→Inhand→Stiffness→Place
시퀀스 이어받기) 대비 바뀐 것 3가지:
  1) 스퀴즈 임계 = collect_ros2_new.py 규약 — 파지 6.0 N + delta 5.0 = **11.0 N 고정**
     (과일별 임계 로드 안 함). 파지 확인 hold(1 s)가 학습 ①pre_wait = 엔진 baseline.
     스퀴즈 호출 자체는 deploy_task3.run_one_sequence 와 동일(현재 파지 자세 기준
     save point · pre_wait 0 · thumb-only 복귀).
  2) GUI = gui/property_gui.py (크기·강성·무게 3속성, /property/result).
  3) 모델 = deep_ws/src/ecoflex2fruit 챔피언(Champ_repair · 변형2+3+5 67ch) —
     mass·size·stif 동시 추론. paxini 는 **/paxini/right/raw** 브리지
     (127점 분포 필요 — point0 트릭 불가). 과일 선택 프롬프트는 유지(포즈 파일용).

실행:  (자세한 절차·문제해결은 README.md §4-A)
  source env.sh
  source ~/Dual_Arm_Hand_Ctrl/ros2/install/setup.bash   # dual_arm_msgs + sequence_client
  python3 stiffness_deploy_ros2/launch/deploy_task3_ros2_demo.py
전제: deploy_task3_ros2.py 와 동일(파지 상태 시작 · sequence_arbiter · require_control)
      + shm_state_publisher 가 /paxini/right/raw 를 발행 중이어야 함.
모델: models/ecoflex2fruit/Champ_repair_s42.pth 고정 (ECO_MODEL 무시 — 아래 main 참고).
"""

from __future__ import annotations

import os
import sys
import threading
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

from deploy_ros2 import Ros2ShmBridge  # noqa: E402
from deploy_ros2_exp_rawft import Ros2RawPaxiniBridge  # noqa: E402
import deploy_task3 as D  # noqa: E402  (시퀀스 상수·헬퍼 재사용)
from core.shm_common import Hand_DOF  # noqa: E402

from dual_arm_msgs.msg import SequenceState  # noqa: E402
from sequence_client import SequenceClient   # noqa: E402

from ecoflex_engine import (  # noqa: E402
    EcoflexPropertyEngine, load_labels, nearest_specimens, check_label_norm)
from property_result_pub import PropertyResultPublisher, spawn_property_gui  # noqa: E402

# ── ecoflex 개체 라벨 (실제값 대조) — 학습 라벨(oldstif)의 repo 내 사본 ──────
LABEL_DIR = os.path.join(_LAUNCH_DIR, "..", "labels", "object_labels_oldstif")
try:
    _LABELS, _LABEL_NORM = load_labels(LABEL_DIR)
except Exception as _e:  # noqa: BLE001  (라벨은 대조 표시용 — 없어도 추론은 계속)
    print(f"[deploy_task3_ros2_demo] ⚠ 라벨 로드 실패({_e}) — 실제값 대조 없이 진행")
    _LABELS, _LABEL_NORM = {}, {}

# ── collect_ros2_new.py 와 동일한 고정 임계 ─────────────────────────────────
GRIP_FORCE_THRESHOLD = 6.0
SQUEEZE_DELTA_N = 5.0
SQUEEZE_FORCE_THRESHOLD = GRIP_FORCE_THRESHOLD + SQUEEZE_DELTA_N   # 11.0 N


def run_one_sequence_demo(shm, paxini, engine, result_pub):
    """deploy_task3.run_one_sequence 의 데모판 — 차이는 ① 파지 확인 hold 동안
    엔진 baseline 확보, ② 스퀴즈 임계 11.0 N 고정, ③ 추론 = 3속성 dict."""
    shm.write_partial(servo_on=(D.HAND_SAFE_SERVO_ON,), hand_mode=(D.HAND_SAFE_MODE,))

    print("=================== 1. 파지 상태 확인 ===================")
    msg = shm.read()
    grip_position = [int(msg.j_pos[0][j]) for j in range(Hand_DOF)]
    print(f"  현재 파지 자세 확인 — {D.GRASP_CONFIRM_SEC}s 유지(안정화)")
    hold = threading.Thread(target=D._hold_hand_position,
                            args=(shm, grip_position, D.GRASP_CONFIRM_SEC))
    hold.start()
    # hold(파지 유지) 와 병행으로 baseline 프레임 수집 = 학습 ①pre_wait 등가.
    engine.capture_baseline(shm, paxini, sec=max(0.2, D.GRASP_CONFIRM_SEC - 0.2))
    hold.join()

    print("=================== 2. 스퀴즈 모션 ===================")
    # save point: 현재 파지 자세에서 thumb_3 만 extra curl (deploy_task3 와 동일).
    save_point = list(grip_position)
    save_point[D._THUMB_3_IDX] += D._SQUEEZE_EXTRA_COUNT
    result_pub.set_measuring()
    D.move_hand_to_squeeze(
        shm, paxini, save_point, D.HAND_SQUEEZE_DURATION,
        SQUEEZE_FORCE_THRESHOLD, grip_position,
        return_duration=D.HAND_SQUEEZE_RETURN_DURATION,
        pre_wait_sec=0.0,   # 파지 확인 1s(=baseline)가 사전 대기를 대체
        engine=engine,
    )

    res = engine.infer()
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
        if _LABELS:
            print(f"  [실제값 대조] 최근접 ecoflex 개체 (labels/object_labels_oldstif)")
            for rank, (oid, dist, lab) in enumerate(
                    nearest_specimens(res, _LABELS, _LABEL_NORM,
                                      stif_log=engine.stif_log, k=3), 1):
                print(f"    {rank}위 ecoflex_{oid:<2d} — "
                      f"mass {lab['mass']:6.1f} g (Δ{abs(res['mass']-lab['mass']):5.1f}) · "
                      f"size {lab['size']:5.2f} mm (Δ{abs(res['size']-lab['size']):4.2f}) · "
                      f"stif {lab['stif']:6.3f} (Δ{abs(res['stif']-lab['stif']):5.3f})"
                      f"   [정규화 거리 {dist:.3f}]")
    else:
        print("  [추론 결과] 샘플 부족 — 추론 불가")
    print("=" * 48 + "\n")
    return res


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
            print("[deploy_task3_ros2_demo] 경고: /paxini/right/raw 미수신 — 힘=0 으로 진행"
                  "(shm_state_publisher 의 raw 발행 확인).")

        # 안전 서보-온: 현재(파지 중) 손 자세를 q_target 으로 먼저 발행 → servo_on.
        bridge.safe_hand_servo_on(mode=D.HAND_SAFE_MODE)

        # 과일 선택은 포즈 파일용으로만 유지 — 임계/모델은 과일과 무관하게 고정.
        fruit = D.ask_fruit()
        _model_path, pose_file, _force_zero = D.resolve_fruit_config(fruit)
        D.set_pose_for_fruit(pose_file)
        print(f"[deploy_task3_ros2_demo] 임계값 고정 — 스퀴즈 {SQUEEZE_FORCE_THRESHOLD:.1f} N "
              f"(collect_ros2_new 와 동일 · 과일별 임계 미사용)")
        #   모델 고정 — models/ecoflex2fruit/Champ_repair_s42.pth (ECO_MODEL 무시)
        engine = EcoflexPropertyEngine(variant="gru")
        if _LABELS:
            print(f"[deploy_task3_ros2_demo] 라벨 {len(_LABELS)}개체 로드 — {LABEL_DIR}")
            check_label_norm(engine, _LABEL_NORM)   # oldstif ↔ ckpt norm 정합 확인
        result_pub.set_idle()
        print("\n※ 이미 물체를 파지한 상태에서 시작합니다.")

        # ── 시퀀스 이어받기: Stiffness(3) — deploy_task3_ros2.py 와 동일 ──
        client = SequenceClient(SequenceState.SEQ_STIFFNESS)
        print(f"[sequence] 직전 Inhand(#{SequenceState.SEQ_INHAND}) DONE 대기...")
        client.wait_for_previous_done(SequenceState.SEQ_INHAND)
        print(f"[sequence] 제어권 획득 → Stiffness(#{SequenceState.SEQ_STIFFNESS}) 시작")

        D._write_marker(marker_path, "S", 0)
        with client:                       # 진입 = Start + 하트비트 자동
            res = run_one_sequence_demo(bridge, paxini, engine, result_pub)
        # with 정상 탈출 = End 자동(제어권 반납).
        if res is not None:
            result_pub.set_result(res, stif_max=engine.norm_max)
        else:
            result_pub.set_error("샘플 부족 — 추론 불가 (스퀴즈가 너무 짧거나 무효 프레임)")
        D._write_marker(marker_path, "E", 0)
        client.shutdown()
        print("\n스퀴즈 시퀀스 완료 — 제어권 반납, 다음 시퀀스(Place)로 이어받음.")
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
