"""deploy_task3.py — "이미 파지한 상태"에서 시작하는 스퀴즈 전용 시퀀스 (배포용)
=================================================================================
deploy.py 는 [안전 → (직접)파지 → 스퀴즈 → 원위치] 전체를 수행하지만,
deploy_task3.py 는 **이미 물체를 잡고 있는 시점**에서 시작해 스퀴즈만 수행한다.
모션·힘 판정·강성 추론 로직은 deploy.py 의 함수를 그대로 재사용한다.

흐름:
  1) 파지 상태 확인: 현재(파지) 손 자세를 GRASP_CONFIRM_SEC 초 유지(안정화).
  2) 스퀴즈 모션: 기존 스퀴즈(thumb curl → 접촉력 threshold → hold → 파지 복귀).
  3) 스퀴즈 직후 강성/등급 추론을 터미널에 출력.
  ※ (추후) 스퀴즈 + 추론 완료 시 "완료" 토픽을 전송 — 이 파일에는 아직 넣지 않는다.

deploy.py 와 달리:
  - Franka(팔) 는 움직이지 않는다(이미 파지 자세). Hand 스퀴즈만 수행.
  - 별도 파지(grip) 단계 없음 — 시작 시점의 손 위치를 그대로 파지 자세로 사용.
  - 데모 반복 없음(1회 실행). 여러 번 필요하면 프로그램을 다시 실행.

사용:  python deploy_task3.py   (실행 후 과일 번호 입력)
  ※ 실행 전에 물체를 이미 파지하고 있어야 한다.
"""

import os
import sys
from pathlib import Path

# deploy.py 와 동일한 sys.path 규약: '..'=project_root(core.* 해석), '.'=launch(deploy 등)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..")
sys.path.append(project_root)
sys.path.append(current_dir)

from core.shm_common import (ShmAccess, SHM_MSG_KEY, Hand_DOF)          # noqa: E402
from core.paxini_shm import PaxiniShmReader, PAXINI_SHM_KEY             # noqa: E402

# deploy.py 의 모션/힘/추론 로직·상수 재사용 (기존 기능 100% 유지)
import deploy as _D                                                    # noqa: E402
from deploy import (                                                   # noqa: E402
    # 과일/모델/추론
    ask_fruit, resolve_fruit_config, set_pose_for_fruit, set_thresholds_for_fruit,
    StiffnessInferenceEngine, LABEL_DIR,
    # 모션 헬퍼
    move_hand_to_squeeze, _hold_hand_position,
    # 스퀴즈 파라미터/상수
    _THUMB_3_IDX, _SQUEEZE_EXTRA_COUNT,
    HAND_SQUEEZE_DURATION, HAND_SQUEEZE_RETURN_DURATION,
    HAND_SAFE_SERVO_ON, HAND_SAFE_MODE,
    # 실행/마커
    parse_args, DEFAULT_MARKER_FILE, _set_squeeze_flag, _write_marker,
)
# SQUEEZE_FORCE_THRESHOLD 는 과일별로 바뀌므로 by-value import 하지 않고
# 호출 시점에 _D.SQUEEZE_FORCE_THRESHOLD(live) 로 참조한다 (set_thresholds_for_fruit 반영).

# 파지 상태 확인(안정화) 대기 시간 [초].
GRASP_CONFIRM_SEC: float = 1.0


def run_one_sequence(
    shm: ShmAccess,
    paxini: PaxiniShmReader,
    engine=None,
) -> None:
    """이미 물체를 파지한 상태에서 시작하는 스퀴즈 전용 시퀀스.

    1) 파지 상태 확인(현재 자세 hold, GRASP_CONFIRM_SEC)
    2) 기존 스퀴즈 모션(thumb curl → 접촉력 → hold → 파지 복귀)
    3) 스퀴즈 직후 강성 추론 출력
    """
    # Hand 를 position 모드로 두고(현재 자세 유지) 시작.
    shm.write_partial(servo_on=(HAND_SAFE_SERVO_ON,), hand_mode=(HAND_SAFE_MODE,))

    print("=================== 1. 파지 상태 확인 ===================")
    # 시작 시점의 현재(파지) 손 위치 = 스퀴즈 후 복귀할 grip_position.
    msg = shm.read()
    grip_position = [int(msg.j_pos[0][j]) for j in range(Hand_DOF)]
    print(f"  현재 파지 자세 확인 — {GRASP_CONFIRM_SEC}s 유지(안정화)")
    _hold_hand_position(shm, grip_position, GRASP_CONFIRM_SEC)

    print("=================== 2. 스퀴즈 모션 ===================")
    # 스퀴즈 save point: 현재 파지 자세에서 thumb_3 관절만 extra curl (deploy 와 동일 방식).
    #  (deploy 는 HAND_GRIP_POINT 기준으로 계산하지만, 여기서는 실제 파지 자세 기준으로 계산해
    #   grip_curl 등으로 파지 위치가 달라져도 정확한 상대 curl 이 되도록 한다.)
    save_point = list(grip_position)
    save_point[_THUMB_3_IDX] += _SQUEEZE_EXTRA_COUNT
    move_hand_to_squeeze(
        shm, paxini, save_point, HAND_SQUEEZE_DURATION,
        _D.SQUEEZE_FORCE_THRESHOLD, grip_position,   # 과일별 임계값(live) — set_thresholds_for_fruit 반영
        return_duration=HAND_SQUEEZE_RETURN_DURATION,
        pre_wait_sec=0.0,   # 스퀴즈 내부 사전 대기 제거 (파지 확인 1s 로 대체)
        engine=engine,
    )

    # 스퀴즈 직후 추론 (절대강성 + 등급 터미널 출력)
    result = None
    if engine is not None:
        stiffness, cls, cname = engine.infer()
        print("\n" + "=" * 48)
        if stiffness is not None:
            print(f"  [추론 결과] {engine.fruit}")
            print(f"    절대강성 = {stiffness:.3f}")
            print(f"    등급     = {cname}  (class {cls})")
            result = (stiffness, cls, cname)
        else:
            print("  [추론 결과] 샘플 부족 — 추론 불가 (스퀴즈가 너무 짧거나 무효 프레임)")
        print("=" * 48 + "\n")

    print("스퀴즈 완료.")
    # 호출자(deploy_task3_ros2)가 GUI 토픽으로 발행할 수 있도록 결과를 돌려준다.
    #   (stiffness, cls, cname) 또는 None(엔진 없음 / 샘플 부족으로 추론 불가).
    return result


def main() -> None:
    args = parse_args()
    marker_path = Path(args.marker_file).resolve()
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    _set_squeeze_flag(False)

    shm = ShmAccess(key=SHM_MSG_KEY)
    if not shm.attach():
        raise SystemExit("SHM attach 실패. C++ 프로세스(Franka+KISTAR)가 먼저 실행 중이어야 합니다.")

    paxini = PaxiniShmReader(PAXINI_SHM_KEY)
    if not paxini.attach():
        print(f"[deploy_task3] 경고: PaXini SHM {hex(PAXINI_SHM_KEY)} 없음 — "
              "접촉력 피드백 없이 position 이동만 수행합니다.")

    # 시작 시 과일을 묻고, 그 과일의 모델+포즈+채널처리를 적용.
    fruit = ask_fruit()
    model_path, pose_file, force_zero = resolve_fruit_config(fruit)
    set_pose_for_fruit(pose_file)
    set_thresholds_for_fruit(fruit)     # 과일별 스퀴즈 임계값 로드 (task3 는 squeeze 만 사용)
    engine = StiffnessInferenceEngine(model_path=model_path, fruit=fruit,
                                      label_dir=LABEL_DIR, force_zero=force_zero)
    print(f"[deploy_task3] 추론엔진 준비 완료. 과일={fruit}, 모델={Path(model_path).name}")
    print("\n※ 이미 물체를 파지한 상태에서 시작합니다.")

    try:
        _write_marker(marker_path, "S", 0)
        run_one_sequence(shm, paxini, engine=engine)
        _write_marker(marker_path, "E", 0)
        print("\n스퀴즈 시퀀스 완료.")
    finally:
        _set_squeeze_flag(False)
        shm.detach()


if __name__ == "__main__":
    main()
