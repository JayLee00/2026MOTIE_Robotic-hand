#!/usr/bin/env python3
"""deploy_task3_ros2.py — deploy_task3.py 를 Dual_Arm_Hand_Ctrl 의 ROS2 토픽으로 실행.

deploy_task3.py("이미 파지한 상태에서 스퀴즈만") 의 시퀀스를 그대로 재사용하고,
SHM 직접 접근만 deploy_ros2.py 의 ROS2 브리지(Ros2ShmBridge/Ros2PaxiniBridge)로
치환한다. (deploy_ros2.py 와 동일한 I/O 계층을 공유 — 브리지 코드 중복 없음)

흐름 (deploy_task3.run_one_sequence):
  1) 파지 상태 확인(현재 자세 hold, GRASP_CONFIRM_SEC)
  2) 기존 스퀴즈 모션
  3) 스퀴즈 직후 강성 추론 출력

시퀀스 이어받기(Pick→Inhand→Stiffness→Place, 나는 Stiffness=3):
  직전 Inhand(2) DONE 대기 → 제어권 획득(Start)+하트비트 → 위 스퀴즈 수행 →
  End(제어권 반납)로 완료를 알림 → 다음 Place(4) 가 이어받는다.
  (sequence_client 로 구현 — 별도 "완료 토픽" 불필요, End 가 완료 신호)

전제:
  - 실행 전에 물체를 이미 파지하고 있어야 한다(팔/손은 이동하지 않음, Hand 스퀴즈만).
  - Dual_Arm_Hand_Ctrl 의 C++ 컨트롤러 + shm_state_publisher_node
    + target receiver 노드 (+ paxini writer) + sequence_arbiter 가 함께 실행 중이어야 한다.
  - 운영(체인) 시 제어 PC 는 require_control:=true 로 launch 되어야 한다
    (제어권 없이 보낸 손 타겟은 무시됨).

실행:
  source /opt/ros/humble/setup.bash
  source /home/prime/Dual_Arm_Hand_Ctrl/ros2/install/setup.bash
  python3 .../launch/deploy_task3_ros2.py   # 실행 후 과일 번호 입력
"""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

import rclpy
from rclpy.executors import SingleThreadedExecutor

# deploy_task3 / deploy 재사용을 위한 sys.path 규약 (deploy_ros2.py 와 동일).
_LAUNCH_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_LAUNCH_DIR, ".."))  # project_root (core.* 해석)
sys.path.insert(0, _LAUNCH_DIR)                       # launch/ (deploy* 모듈)

# ROS2 의 'launch' 패키지가 sys.path 를 선점해 deploy.py 의
# `from launch.real_deploy_inference_old import ...` 가 실패하는 것을 막기 위해,
# 해당 모듈을 top-level 로 로드해 'launch.*' 이름으로 미리 등록한다. (deploy.py 미수정)
import importlib  # noqa: E402
sys.modules.setdefault(
    "launch.real_deploy_inference_old",
    importlib.import_module("real_deploy_inference_final"))

import real_deploy_inference_final as _RE  # noqa: E402  (SOTA 앙상블 설정 참조)
# ROS2 브리지는 deploy_ros2.py 것을 그대로 재사용 (동일한 토픽 I/O).
from deploy_ros2 import Ros2ShmBridge, Ros2PaxiniBridge  # noqa: E402
# 시퀀스 로직은 task3(스퀴즈 전용).
import deploy_task3 as D  # noqa: E402

# 시퀀스 이어받기(Pick→Inhand→Stiffness→Place) 클라이언트.
#   배정 번호(고정): 1=Pick, 2=Inhand, 3=Stiffness, 4=Place  (나는 Stiffness=3)
#   상수는 dual_arm_msgs/msg/SequenceState 의 SEQ_PICK…SEQ_PLACE 를 사용(하드코딩 숫자 대신).
#   ※ 제어 PC 저장소의 dual_arm_msgs + sequence_client 를 워크스페이스에 빌드/소스해야 import 가능.
#     (colcon build --packages-select dual_arm_msgs sequence_client && source install/setup.bash)
from dual_arm_msgs.msg import SequenceState  # noqa: E402
from sequence_client import SequenceClient   # noqa: E402

# 강성 추론 결과 → GUI 로 연속 발행하는 퍼블리셔 + GUI 자동 실행 (launch/ 에 위치).
from stiffness_result_pub import StiffnessResultPublisher, spawn_gui  # noqa: E402


def main() -> None:
    # --no-gui : GUI 자동 실행 끄기. D.parse_args(argparse) 가 모르는 인자라 미리 제거.
    want_gui = "--no-gui" not in sys.argv
    if not want_gui:
        sys.argv = [a for a in sys.argv if a != "--no-gui"]
    if want_gui:
        spawn_gui()

    args = D.parse_args()
    marker_path = Path(args.marker_file).resolve()
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    D._set_squeeze_flag(False)

    rclpy.init()
    bridge = Ros2ShmBridge()
    paxini = Ros2PaxiniBridge(bridge, "/paxini/right/ft")
    result_pub = StiffnessResultPublisher()   # 강성 결과 → GUI 연속 발행
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
            print("[deploy_task3_ros2] 경고: /paxini/right/ft 미수신 — 힘=0 으로 진행"
                  "(paxini writer 실행/토픽 확인).")

        # 안전 서보-온: 현재(파지 중) 손 자세를 q_target 으로 먼저 발행 → servo_on.
        # 이미 물체를 파지한 상태이므로, 서보 켤 때 손가락이 튀어 파지를 놓치지 않게 한다.
        bridge.safe_hand_servo_on(mode=D.HAND_SAFE_MODE)

        # 과일 선택 → 모델/포즈/엔진 준비 (deploy_task3 를 통해 deploy 로직 재사용).
        fruit = D.ask_fruit()
        model_path, pose_file, force_zero = D.resolve_fruit_config(fruit)
        D.set_pose_for_fruit(pose_file)
        D.set_thresholds_for_fruit(fruit)     # 과일별 스퀴즈 임계값 로드 (task3 는 squeeze 만 사용)
        # ★ SOTA 앙상블(Phase_SOTA.md §15): USE_SOTA_ENSEMBLE 면 5-seed 리스트로 교체
        #   (재파지·제어기 변경 없이 추론측만 바꿈). False 면 기존 단일모델 그대로.
        _model_arg = _RE.SOTA_ENSEMBLE_PATHS if _RE.USE_SOTA_ENSEMBLE else model_path
        engine = D.StiffnessInferenceEngine(
            model_path=_model_arg, fruit=fruit, label_dir=D.LABEL_DIR, force_zero=force_zero)
        _mdesc = (f"SOTA앙상블 {len(_RE.SOTA_ENSEMBLE_PATHS)}-seed"
                  if _RE.USE_SOTA_ENSEMBLE else Path(model_path).name)
        print(f"[deploy_task3_ros2] 추론엔진 준비 완료. 과일={fruit}, 모델={_mdesc}")
        print("\n※ 이미 물체를 파지한 상태에서 시작합니다.")

        # ── 시퀀스 이어받기: Stiffness(3) ──────────────────────────────────
        # 직전 Inhand(2) 가 DONE 될 때까지 대기(= 물체를 손에 쥐여준 상태) →
        # 제어권 획득(Start)+하트비트 → 스퀴즈 수행 → with 탈출 시 End(제어권 반납) →
        # 다음 Place(4) 가 이어받는다. (End 가 곧 "스퀴즈+추론 완료" 신호 역할)
        # ※ 직전이 실패(하트비트 타임아웃 회수)면 wait_for_previous_done 이 PreviousAborted 로
        #   중단됨 → finally 에서 정리 후 종료(정상 abort 경로).
        client = SequenceClient(SequenceState.SEQ_STIFFNESS)
        print(f"[sequence] 직전 Inhand(#{SequenceState.SEQ_INHAND}) DONE 대기...")
        client.wait_for_previous_done(SequenceState.SEQ_INHAND)
        print(f"[sequence] 제어권 획득 → Stiffness(#{SequenceState.SEQ_STIFFNESS}) 시작")

        D._write_marker(marker_path, "S", 0)
        # GUI: 스퀴즈 시작 → "측정 중..." (과일/범위/경계를 미리 넘겨 막대 스케일 준비).
        result_pub.set_measuring(engine.fruit, engine.norm_min, engine.norm_max,
                                 engine.boundaries, engine.class_names)
        with client:                       # 진입 = Start + 하트비트 자동
            result = D.run_one_sequence(bridge, paxini, engine=engine)
        # with 정상 탈출 = End 자동(제어권 반납). 동작 중 예외/크래시면 End 없이 3초 후 자동 회수.
        # GUI: 추론 결과 발행 (샘플 부족이면 error). latched 라 GUI 를 나중에 켜도 유지됨.
        if result is not None and result[0] is not None:
            stiffness, cls, cname = result
            result_pub.set_result(engine.fruit, stiffness, cls, cname,
                                  engine.norm_min, engine.norm_max,
                                  engine.boundaries, engine.class_names)
        else:
            result_pub.set_error("샘플 부족 — 추론 불가 (스퀴즈가 너무 짧거나 무효 프레임)")
        D._write_marker(marker_path, "E", 0)
        client.shutdown()
        print("\n스퀴즈 시퀀스 완료 — 제어권 반납, 다음 시퀀스(Place)로 이어받음.")
    finally:
        D._set_squeeze_flag(False)
        bridge.detach()
        executor.shutdown()
        # spin_thread 가 콜백 처리 중일 수 있으므로, 컨텍스트를 먼저 무효화해
        # spin() 이 ExternalShutdownException 으로 즉시 빠져나오게 한 뒤 join 한다.
        if rclpy.ok():
            rclpy.shutdown()
        spin_thread.join(timeout=2.0)


if __name__ == "__main__":
    # 라이브 DDS 네트워크(활성 노드 다수)에서는 rclpy.shutdown() 이후 남은 Node/
    # Publisher C 객체의 GC/인터프리터 종료 정리가 소켓 송신 대기로 오래 걸리거나
    # 멈출 수 있다. main() 안의 애플리케이션 레벨 정리(스레드 join 등)는 이미 끝났으니,
    # 이후의 네이티브 teardown 대기는 건너뛰고 바로 종료한다.
    _exit_code = 0
    try:
        main()
    except SystemExit as _e:
        if _e.code and not isinstance(_e.code, int):
            print(_e.code, file=sys.stderr)
        _exit_code = _e.code if isinstance(_e.code, int) else (1 if _e.code else 0)
    except KeyboardInterrupt:
        _exit_code = 130
    sys.stdout.flush(); sys.stderr.flush()   # P2#4: os._exit 전 flush (파이프/리다이렉트 시 마지막 출력 유실 방지)
    os._exit(_exit_code)
