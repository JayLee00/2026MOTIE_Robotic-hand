"""
deploy_motion_sequence.py  —  motion_sequence_A_self.py + 실시간 강성 추론 (배포용)
==================================================================================
원본 motion_sequence_A_self.py 와 동작은 동일하되, 각 데모의 스퀴즈 구간에서
추론 샘플을 모아 스퀴즈 직후 강성/등급을 터미널에 출력한다.

흐름:
  - 시작 시 1회: 터미널에서 과일 종류를 물어 추론엔진 생성.
  - 각 데모마다 시퀀스(안전→파지→스퀴즈) 실행. ※ Franka(팔)는 이동하지 않음(손만).
    · 스퀴즈+hold 구간(학습의 24_squeeze_on=1 과 동일)에서 engine.add_sample()
    · 스퀴즈 종료 직후 engine.infer() → 절대강성 + 등급 터미널 출력
  - 데이터수집용 squeeze 플래그파일(/tmp/gen3_squeeze_on.txt)은 그대로 두지만
    추론은 파일이 아니라 motion 이 직접 엔진 메서드를 호출한다.

모션 시퀀스 (손만 — Franka 이동 없음):
  1) Hand 안전 위치
  2) 파지: HAND_GRIP_POINT 방향으로 닫아 가벼운 접촉(GRIP_FORCE_THRESHOLD) 확보
  3) 스퀴즈: HAND_SAVE_POINT 방향으로 닫아 접촉력(SQUEEZE_FORCE_THRESHOLD)까지 → hold → 파지 복귀
  ※ '물체 원위치'(4단계)와 팔 이동은 제거됨 — 팔은 실행 전 자세에 고정.
     (FRANKA_*/move_franka_to 정의는 남아 있으나 시퀀스에서 호출하지 않음)

사용:  python deploy_motion_sequence.py   (실행 후 과일 번호 입력)
  ※ real_deploy_inference.py 의 MODEL_PATH 를 시연 모델로 먼저 설정할 것.
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

import time
import os
from typing import List

import numpy as np
import yaml

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..")
sys.path.append(project_root)
sys.path.append(current_dir)

from core.shm_common import (ShmAccess, SHM_MSG_KEY, Hand_DOF, Arm_DOF, Finger_Num)
from core.paxini_shm import PaxiniShmReader, PAXINI_SHM_KEY
from hand_pose_io import load_pose_counts

# ★ deploy: 실시간 강성 추론 엔진 (배포용 추가)
from launch.real_deploy_inference_old import (StiffnessInferenceEngine, ask_fruit,
                                   resolve_fruit_config, LABEL_DIR)

JOINTS_PER_FINGER = Hand_DOF // Finger_Num     # 16/4 = 4

FRANKA_SAFE_POSITION: List[float] = [0.7131, 0.6767, -0.2425, -2.0534, -0.5806, 2.0966, -1.3660] # 안전위치
HAND_SAFE_POSITION_FILE: str = str(Path(__file__).resolve().parent / "initial_pose.txt")  # 안전(초기) 포즈
HAND_SAFE_POSITION: List[int] = load_pose_counts(HAND_SAFE_POSITION_FILE, field="tar")

# HAND_GRIP_POINT 는 kiwi.txt 형식 포즈 파일의 tar(rad) 값을 count 로
_POSE_DIR: Path = Path(__file__).resolve().parent
# HAND_GRIP_POINT_FILE: str = str(_POSE_DIR / "tomaeto.txt")   # 파지 포즈
HAND_GRIP_POINT_FILE: str = str(_POSE_DIR / "kiwi.txt")   # 파지 포즈


FRANKA_GRIP_POINT: List[float] = [0.7112, 0.7969, -0.3489, -2.0718, -0.5825, 2.2636, -1.4434] # 파지위치
HAND_GRIP_POINT: List[int] = load_pose_counts(HAND_GRIP_POINT_FILE, field="tar")

# FRANKA_SAVE_POINT: List[float] = [0.7131, 0.6767, -0.2425, -2.0534, -0.5806, 2.0966, -1.3660] # 추론위치(손바닥 아래)
# FRANKA_SAVE_POINT: List[float] = [0.5084, 0.4859, -0.4427, -2.4877, 2.0655, 0.6453, -1.9251] # 추론위치(손바닥 위)

## 이 부분이 중요. 중심점으로 세팅 == 스퀴즈(중심점) 포즈: 파지 포즈(HAND_GRIP_POINT)에서 thumb_3 관절만 약 15도 더 닫아(curl) probing 용 스퀴즈 자세를 만든다.
SQUEEZE_EXTRA_DEG: float = 25.0
# count = rad · 8192/π 이므로 deg → count = deg/180 · 8192.
_SQUEEZE_EXTRA_COUNT: int = int(round(SQUEEZE_EXTRA_DEG / 180.0 * 8192))
# 관절 인덱스 = finger(thumb=0,index=1,middle=2,ring=3) · JOINTS_PER_FINGER + joint(0..3)
_THUMB_3_IDX: int = 0 * JOINTS_PER_FINGER + 3
HAND_SAVE_POINT: List[int] = list(HAND_GRIP_POINT)
HAND_SAVE_POINT[_THUMB_3_IDX] += _SQUEEZE_EXTRA_COUNT


def set_pose_for_fruit(pose_file_name):
    """과일 선택 후, 그 과일의 파지 포즈 txt 로 HAND_GRIP_POINT/HAND_SAVE_POINT 갱신.
       (포즈는 모듈 로드 때 1회 고정되므로, 과일별 분기를 위해 여기서 다시 로드한다.)"""
    global HAND_GRIP_POINT_FILE, HAND_GRIP_POINT, HAND_SAVE_POINT
    path = str(_POSE_DIR / pose_file_name)
    HAND_GRIP_POINT_FILE = path
    HAND_GRIP_POINT = load_pose_counts(path, field="tar")
    HAND_SAVE_POINT = list(HAND_GRIP_POINT)
    HAND_SAVE_POINT[_THUMB_3_IDX] += _SQUEEZE_EXTRA_COUNT
    print(f"[pose] 과일 포즈 적용: {pose_file_name} -> GRIP={HAND_GRIP_POINT}")
    return HAND_GRIP_POINT, HAND_SAVE_POINT

CONTACT_FORCE_THRESHOLD = 0.2

# ── 과일별 힘 임계값 (config: fruit_thresholds.yaml) ────────────────────────
# grip=파지 접촉력, squeeze=스퀴즈 정지력. 과일 선택 시 set_thresholds_for_fruit 로 갱신.
FRUIT_THRESHOLD_FILE: Path = _POSE_DIR / "fruit_thresholds.yaml"
_DEFAULT_GRIP_THRESHOLD: float = 7.0
_DEFAULT_SQUEEZE_THRESHOLD: float = 10.0


def _load_threshold_config(path: Path = FRUIT_THRESHOLD_FILE) -> dict:
    """fruit_thresholds.yaml → {'default': {...}, 'fruits': {...}}. 없거나 실패하면 {} (기본값 사용)."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        print(f"[threshold] config 없음({path}) — 기본값 "
              f"(grip={_DEFAULT_GRIP_THRESHOLD}, squeeze={_DEFAULT_SQUEEZE_THRESHOLD}) 사용")
        return {}
    except Exception as exc:  # noqa: BLE001  (config 오류로 전체가 죽지 않게 방어)
        print(f"[threshold] config 로드 실패({exc}) — 기본값 사용")
        return {}


_THRESHOLD_CONFIG: dict = _load_threshold_config()


def _threshold_for(fruit, key: str, default: float) -> float:
    """config 의 fruits[fruit][key] → 없으면 default[key] → 없으면 하드코딩 default."""
    fruits = _THRESHOLD_CONFIG.get("fruits") or {}
    cfg_default = _THRESHOLD_CONFIG.get("default") or {}
    val = (fruits.get(fruit) or {}).get(key, cfg_default.get(key, default))
    return float(val)


# 모듈 로드 시엔 config 의 default 값(없으면 하드코딩). 과일 선택 후 set_thresholds_for_fruit 로 덮어씀.
GRIP_FORCE_THRESHOLD = _threshold_for(None, "grip", _DEFAULT_GRIP_THRESHOLD)
SQUEEZE_FORCE_THRESHOLD = _threshold_for(None, "squeeze", _DEFAULT_SQUEEZE_THRESHOLD)


def set_thresholds_for_fruit(fruit):
    """과일 선택 후 그 과일의 파지/스퀴즈 임계값을 config 에서 읽어 모듈 전역에 반영.
       set_pose_for_fruit 과 동일 패턴 — deploy_ros2/task3 도 이 전역을 공유한다.
       (grip 은 직접 파지하는 deploy/deploy_ros2 만 사용, squeeze 는 전 파일 공통)"""
    global GRIP_FORCE_THRESHOLD, SQUEEZE_FORCE_THRESHOLD
    GRIP_FORCE_THRESHOLD = _threshold_for(fruit, "grip", _DEFAULT_GRIP_THRESHOLD)
    SQUEEZE_FORCE_THRESHOLD = _threshold_for(fruit, "squeeze", _DEFAULT_SQUEEZE_THRESHOLD)
    print(f"[threshold] {fruit}: 파지={GRIP_FORCE_THRESHOLD}N, 스퀴즈={SQUEEZE_FORCE_THRESHOLD}N")
    return GRIP_FORCE_THRESHOLD, SQUEEZE_FORCE_THRESHOLD


# 스퀴즈 정지(threshold) 판정을 적용할 손가락: thumb(0) 만.
# (finger 순서 = thumb,index,middle,ring) index(1)/middle(2)/ring(3) 은 힘과 무관하게 target 까지만 이동.
SQUEEZE_FORCE_FINGERS = (0,)

SQUEEZE_PRE_WAIT_SEC: float = 1.0  # 스퀴즈 시작 전 현재(파지) 자세로 대기하는 시간
SQUEEZE_HOLD_SEC: float = 1.0      # 스퀴즈 자세 유지(hold) 시간
HAND_PRESS_TIMEOUT_FACTOR: float = 2.0


FRANKA_SPEED_FACTOR: float = 0.3
FRANKA_POSITION_TOLERANCE_RAD: float = 0.002
HAND_MOVE_DURATION: float = 1.5
# 스퀴즈 구간 이동 시간(초). 값을 줄이면 더 빠르게 움직인다.
HAND_SQUEEZE_DURATION: float = 0.5         # 스퀴즈 '닫기'(HAND_SAVE_POINT 방향) 시간
HAND_SQUEEZE_RETURN_DURATION: float = 1.5  # 스퀴즈 후 파지 position '복귀' 시간

CONTROL_RATE_HZ: float = 100.0 # SHM 제어 주기 (Hz) — 이 주기로 목표치 갱신

# 파지(grip) 보강 curl: target 까지 갔는데도 finger 가 GRIP_FORCE_THRESHOLD 미달이면,
# 그 finger 의 2,3번 관절(joint offset 2,3)을 스퀴즈처럼 조금씩 더 닫아 파지력을 확보한다.
GRIP_CURL_JOINT_OFFSETS = (2, 3)             # finger 내 닫을 관절(offset)
GRIP_CURL_SPEED_DEG_PER_SEC: float = 20.0    # 추가 curl 속도(도/초)
GRIP_CURL_MAX_DEG: float = 30.0              # finger별 추가 curl 한계(도) — 무한정 닫지 않도록
# deg → count = deg/180 · 8192. 한 tick(=1/CONTROL_RATE) 당 증가 count.
_GRIP_CURL_STEP_COUNT: int = max(1, int(round(
    GRIP_CURL_SPEED_DEG_PER_SEC / 180.0 * 8192 / CONTROL_RATE_HZ)))
_GRIP_CURL_MAX_COUNT: int = int(round(GRIP_CURL_MAX_DEG / 180.0 * 8192))

NUM_DEMOS: int = 11 # 실험 데모 개수 (demo0 ~ demo10 = 11개). 각 데모마다 S(i), E(i) 마커 전달

# 마커 파일 경로: hdf5_logger 와 동일한 Gen3/logs 에, 같은 타임스탬프 형식으로 저장한다.
# (logger: logs/shm_YYYYMMDD_HHMMSS.h5  ↔  marker: logs/marker_YYYYMMDD_HHMMSS.txt)
# 나중에 h5 와 마커를 타임스탬프로 짝지어 함께 쓰기 위함.
_DEFAULT_LOG_DIR: Path = Path(__file__).resolve().parent.parent / "logs"
DEFAULT_MARKER_FILE: str = str(
    _DEFAULT_LOG_DIR / f"marker_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
)

# 스퀴즈 구간 플래그 파일: hdf5_logger 의 DEFAULT_SQUEEZE_FLAG_FILE 와 동일 경로.
# 스퀴즈+hold 구간에서 "1", 그 외 "0" 을 써서 로거의 24_squeeze_on 에 반영시킨다.
SQUEEZE_FLAG_FILE: Path = Path("/tmp/gen3_squeeze_on.txt")

# Hand 안전위치로 갈 때: servo_on=1, hand_mode=2 (current)
HAND_SAFE_SERVO_ON: int = 1
HAND_SAFE_MODE: int = 1  # position
# HAND_SAFE_MODE: int = 2  # current

# Hand 저장 포인트로 갈 때: servo_on=1, hand_mode=0 (volt), 0V 확인용 16개 0
HAND_SAVE_SERVO_ON: int = 1
HAND_SAVE_MODE: int = 0  # volt

# volt 모드에서 핸드 타겟(j_tar)에 넣을 값 — 0V 확인용
# HAND_VOLT_ZERO_TARGET: List[int] = [0, 0, -300, -300, 0, -300, -300, -300, 0, -300, -300, -300, 0, -300, -300, -300]
HAND_VOLT_ZERO_TARGET: List[int] = [200, 100, -300, 0,
                                    100, -300, 0, 0,
                                    100, -300, 0, 0,
                                    -100, -300, 0, 0]

# 시퀀스 한 사이클: S → (Franka안전→Hand안전→저장점→...) → Hand안전 → E → 대기 → 다음 데모에서 S 다시
PAUSE_BETWEEN_DEMOS_SEC: float = 0.5
# ======================================================

def _clamp_duration(d: float) -> float:
    return max(0.01, float(d))


def wait_franka_reached(shm: ShmAccess, target: List[float]) -> None:
    """Franka가 target에 도달할 때까지 무한 대기."""
    tol = max(1e-5, FRANKA_POSITION_TOLERANCE_RAD)
    dt = 1.0 / CONTROL_RATE_HZ
    while True:
        msg = shm.read()
        current = [float(msg.Arm_j_pos[0][j]) for j in range(Arm_DOF)]
        if all(abs(current[j] - target[j]) <= tol for j in range(Arm_DOF)):
            return
        time.sleep(dt)


def move_franka_to(shm: ShmAccess, target: List[float], duration_sec: float = 0.0) -> None:
    """Franka를 target 관절각으로 이동하고, 완전히 도달할 때까지 대기한 뒤 반환 (이후 Hand 동작 가능)."""
    shm.write_partial(
        Arm_j_tar=(tuple(target),),
        Arm_Speed_Factor=(FRANKA_SPEED_FACTOR,),
    )
    wait_franka_reached(shm, target)


def move_hand_to(shm: ShmAccess, target: List[int], duration_sec: float) -> None:
    """Hand를 현재 위치에서 target까지 duration_sec 동안 선형 보간."""
    duration_sec = _clamp_duration(duration_sec)
    shm.write_partial(servo_on=(HAND_SAFE_SERVO_ON,), hand_mode=(HAND_SAFE_MODE,))
    msg = shm.read()
    current = [int(msg.j_pos[0][j]) for j in range(Hand_DOF)]
    steps = max(1, int(duration_sec * CONTROL_RATE_HZ))
    dt = 1.0 / CONTROL_RATE_HZ

    for i in range(steps + 1):
        t = i / steps
        current = [int(round(current[j] + (target[j] - current[j]) * t)) for j in range(Hand_DOF)]
        shm.write_partial(
            j_tar=(tuple(current),),
            servo_on=(HAND_SAFE_SERVO_ON,),
            hand_mode=(HAND_SAFE_MODE,),
        )
        time.sleep(dt)


def move_fingers_to(shm: ShmAccess, target: List[int], fingers, duration_sec: float,
                    hold_position: List[int]) -> List[int]:
    """fingers 의 관절만 실측 현재값에서 target 으로 선형 보간해 움직이고, **나머지 관절은
    hold_position(직전 명령값)으로 고정 커밋**한다.

    물체를 쥔 손가락은 실측(j_pos)이 명령보다 얕다(물체에 막혀 목표까지 못 들어감).
    move_hand_to 처럼 16관절 전부를 실측 기준으로 재명령하면, 쥐고 있던 손가락의 서보
    목표가 실측 위치로 후퇴했다가 다시 조여져 '풀렸다 조이는' 움직임이 생긴다(스퀴즈 때
    엄지 외 손가락이 움직이던 원인). → 움직일 손가락 외의 명령값은 실측으로 재설정하지
    않는다. 반환: 마지막으로 명령한 16관절 position."""
    duration_sec = _clamp_duration(duration_sec)
    steps = max(1, int(duration_sec * CONTROL_RATE_HZ))
    dt = 1.0 / CONTROL_RATE_HZ
    shm.write_partial(servo_on=(HAND_SAFE_SERVO_ON,), hand_mode=(HAND_SAFE_MODE,))
    msg = shm.read()
    move_joints = [j for f in fingers
                   for j in range(f * JOINTS_PER_FINGER, (f + 1) * JOINTS_PER_FINGER)]
    start = {j: int(msg.j_pos[0][j]) for j in move_joints}
    current = [int(p) for p in hold_position]
    for i in range(steps + 1):
        t = i / steps
        for j in move_joints:
            current[j] = int(round(start[j] + (target[j] - start[j]) * t))
        shm.write_partial(j_tar=(tuple(current),), servo_on=(HAND_SAFE_SERVO_ON,),
                          hand_mode=(HAND_SAFE_MODE,))
        time.sleep(dt)
    return current


def calculate_contact_normal_force(paxini: PaxiniShmReader, mode: bool):
    """paxini.read(): (4,127,3), t_mono_ns, valid, seq,  return: (force(4,), direction(4,3))"""

    tactile_distribution, _t_mono_ns, valid, _seq = paxini.read()  # (4, 127, 3)
    if not int(valid):
        return np.zeros(4, dtype=np.float32), np.zeros((4, 3), dtype=np.float32)

    if not mode:
        # 곡률 고려: 각 tactile point 의 표면 법선 방향을 반영한 normal force.
        return np.zeros(4, dtype=np.float32), np.zeros((4, 3), dtype=np.float32)

    # 단순 합력: paxini (4, 127, 3) -> finger별로 127점의 z축(normal, force_z) 합.
    tactile_distribution = np.nan_to_num(tactile_distribution, nan=0.0)
    contact_normal_force = tactile_distribution[:, :, 2].sum(axis=1)  # (4,) finger별 force_z 합

    # 방향: finger별 (fx,fy,fz) 합벡터를 정규화 → 접촉 합력(=표면 normal) 방향(센서 프레임).
    force_vec = tactile_distribution.sum(axis=1)                      # (4,3) finger별 합력
    mag = np.linalg.norm(force_vec, axis=1, keepdims=True)            # (4,1)
    direction = np.divide(force_vec, mag, out=np.zeros_like(force_vec),
                          where=mag > 1e-9)                           # (4,3) 단위벡터, 무접촉=0

    force = contact_normal_force.astype(np.float32)
    direction = direction.astype(np.float32)
    return force, direction


def _hold_hand_position(shm: ShmAccess, position: List[int], duration_sec: float,
                        engine=None, paxini=None) -> None:
    """Hand 를 주어진 position 에 duration_sec 동안 그대로 유지(hold).
       engine+paxini 가 주어지면(스퀴즈 hold) 매 주기 추론 샘플도 적재."""
    duration_sec = _clamp_duration(duration_sec)
    steps = max(1, int(duration_sec * CONTROL_RATE_HZ))
    dt = 1.0 / CONTROL_RATE_HZ
    pos = tuple(int(p) for p in position)
    for _ in range(steps):
        shm.write_partial(j_tar=(pos,), servo_on=(HAND_SAFE_SERVO_ON,),
                          hand_mode=(HAND_SAFE_MODE,))
        if engine is not None and paxini is not None:   # ★ deploy: hold 구간 샘플 적재
            engine.add_sample(shm, paxini)
        time.sleep(dt)


def move_hand_to_target_until_force(
    shm: ShmAccess,
    paxini: PaxiniShmReader,
    target: List[int],
    duration_sec: float,
    threshold: float,
    mode: bool = True,
    force_fingers=None,
    grip_curl: bool = False,
    engine=None,
) -> List[int]:
    """현재 위치에서 target 으로 finger별 선형 보간하며 닫되, 접촉력이 threshold 이상이 된
    finger 는 그 자리에 멈춰(hold) 유지한다. (Jacobian 없이 force-제한 close)

    - 파지/스퀴즈 공용. finger 마다 독립적으로 접근하다가 접촉력에 도달하면 그 finger만 정지.
    - force_fingers: threshold 정지 판정을 적용할 finger 인덱스들. None 이면 전체.
      여기 없는 finger 는 힘과 무관하게 target 까지만 이동(예: 스퀴즈는 thumb 만).
    - grip_curl: True 이면 target 까지 갔는데도 threshold 미달인 finger 의 2,3번 관절을
      스퀴즈처럼 조금씩 더 닫아(curl) threshold 달성을 추가로 시도(파지력 보강).
    - 반환: 마지막으로 도달한 16관절 position (저장/복귀용).
    """
    force_set = set(range(Finger_Num) if force_fingers is None else force_fingers)
    dt = 1.0 / CONTROL_RATE_HZ
    steps = max(1, int(_clamp_duration(duration_sec) * CONTROL_RATE_HZ))
    max_ticks = int(steps * HAND_PRESS_TIMEOUT_FACTOR)

    shm.write_partial(servo_on=(HAND_SAFE_SERVO_ON,), hand_mode=(HAND_SAFE_MODE,))
    msg = shm.read()
    start = [int(msg.j_pos[0][j]) for j in range(Hand_DOF)]
    current = list(start)

    finger_joints = [range(f * JOINTS_PER_FINGER, (f + 1) * JOINTS_PER_FINGER)
                     for f in range(Finger_Num)]
    progress = [0.0] * Finger_Num     # finger별 접근 진행도 (0=start, 1=target)
    settled = [False] * Finger_Num    # 접촉력 ≥ threshold → 그 자리 hold

    for i in range(Finger_Num):
        if i not in force_set:
            progress[i] = 1.0
            for j in finger_joints[i]:
                current[j] = target[j]

    def commit() -> None:
        shm.write_partial(j_tar=(tuple(current),), servo_on=(HAND_SAFE_SERVO_ON,),
                          hand_mode=(HAND_SAFE_MODE,))

    def normal_forces() -> np.ndarray:
        """finger별 접촉 normal force(127점 force_z 합). mode=False/무효 프레임이면 0."""
        tactile, _t, valid, _seq = paxini.read()
        if not mode or not int(valid):
            return np.zeros(Finger_Num, dtype=np.float32)
        return np.nan_to_num(tactile, nan=0.0)[:, :, 2].sum(axis=1)

    for _ in range(max_ticks + 1):
        forces = normal_forces()
        for i in range(Finger_Num):
            # threshold 정지는 force_set 의 finger 에만 적용. 그 외 finger 는 target 까지 이동.
            if settled[i] or (i in force_set and forces[i] >= threshold):
                settled[i] = True                      # 접촉력 도달(또는 이미 도달) → hold
                continue
            progress[i] = min(1.0, progress[i] + 1.0 / steps)
            for j in finger_joints[i]:
                current[j] = round(start[j] + (target[j] - start[j]) * progress[i])
        commit()
        if engine is not None:        # ★ deploy: 스퀴즈 누름 구간 샘플 적재
            engine.add_sample(shm, paxini)
        # 모든 finger 가 hold 되었거나 target 까지 다 이동하면 종료.
        if all(settled[i] or progress[i] >= 1.0 for i in range(Finger_Num)):
            break
        time.sleep(dt)

    # --- 파지 보강 curl: target 까지 갔는데도 threshold 미달인 finger 의 2,3번 관절을
    #     스퀴즈처럼 조금씩 더 닫으며 threshold 달성을 추가로 시도한다. ---
    if grip_curl:
        extra = [0] * Finger_Num          # finger별 추가 curl 누적 count
        curl_max_ticks = max_ticks        # 보강 단계 tick 상한(무한루프 방지)
        for _ in range(curl_max_ticks + 1):
            # 더 닫을 finger: force_set 이면서 아직 미도달이고 curl 한계 미만.
            active = [i for i in force_set
                      if not settled[i] and extra[i] < _GRIP_CURL_MAX_COUNT]
            if not active:
                break
            forces = normal_forces()
            for i in active:
                if forces[i] >= threshold:
                    settled[i] = True     # 추가 curl 로 파지력 달성 → hold
                    continue
                extra[i] = min(_GRIP_CURL_MAX_COUNT, extra[i] + _GRIP_CURL_STEP_COUNT)
                for off in GRIP_CURL_JOINT_OFFSETS:
                    j = i * JOINTS_PER_FINGER + off
                    current[j] = target[j] + extra[i]
            commit()
            time.sleep(dt)

    # threshold 적용 대상(force_set) 중 미도달 finger 만 경고.
    pending = [i for i in force_set if not settled[i]]
    if pending:
        print(f"[move_hand_to_target_until_force] finger {pending} threshold({threshold}N) "
              "미도달 — target 위치 유지.")
    return current


def _set_squeeze_flag(on: bool) -> None:
    """스퀴즈 구간 플래그 파일에 '1'/'0' 기록(로거가 읽어 24_squeeze_on 으로 저장)."""
    try:
        SQUEEZE_FLAG_FILE.write_text("1" if on else "0", encoding="ascii")
    except OSError as exc:
        print(f"[squeeze] flag write 실패: {exc}")


def move_hand_to_squeeze(
    shm: ShmAccess,
    paxini: PaxiniShmReader,
    target_position: List[int],
    duration: float,
    threshold: float,
    return_position: List[int],
    hold_sec: float = SQUEEZE_HOLD_SEC,
    return_duration: float = HAND_SQUEEZE_RETURN_DURATION,
    pre_wait_sec: float = SQUEEZE_PRE_WAIT_SEC,
    engine=None,
) -> None:
    # 스퀴즈 target: thumb(SQUEEZE_FORCE_FINGERS) 관절만 target_position(HAND_SAVE_POINT)으로
    # 닫고, 나머지 손가락은 return_position(=직전 명령값) 그대로 둔다(target==직전 명령
    # → 안 움직임). ※ 실측(j_pos)을 기준으로 잡으면 안 된다 — 쥔 손가락은 실측<명령이라
    # 실측 기반 재명령은 그 손가락을 풀어 버린다.
    squeeze_target = list(return_position)
    for f in SQUEEZE_FORCE_FINGERS:
        for j in range(f * JOINTS_PER_FINGER, (f + 1) * JOINTS_PER_FINGER):
            squeeze_target[j] = target_position[j]

    # 0) 스퀴즈 전 대기: 현재(파지) 자세를 그대로 유지하며 pre_wait_sec 초 대기.
    print(f"  [squeeze] 스퀴즈 전 {pre_wait_sec}s 대기")
    _hold_hand_position(shm, return_position, pre_wait_sec)

    # 1~2 구간동안만 squeeze_on=1 (로거 24_squeeze_on). try/finally 로 항상 0 으로 복구.
    _set_squeeze_flag(True)
    if engine is not None:
        engine.reset()          # ★ deploy: 이 스퀴즈의 샘플만 모으도록 버퍼 비움
    try:
        # 1) 스퀴즈: thumb 만 닫고 나머지 손가락은 return_position 유지, threshold 도달 position 을 받아둔다.
        #    스퀴즈 정지(threshold)는 thumb(SQUEEZE_FORCE_FINGERS) 에만 적용.
        squeezed = move_hand_to_target_until_force(
            shm, paxini, squeeze_target, duration, threshold, mode=True,
            force_fingers=SQUEEZE_FORCE_FINGERS, engine=engine)

        # 2) holding: 스퀴즈 자세 유지.
        print(f"  [squeeze] threshold({threshold}) 도달 → {hold_sec}s holding")
        _hold_hand_position(shm, squeezed, hold_sec, engine=engine, paxini=paxini)
    finally:
        _set_squeeze_flag(False)

    # 3) 스퀴즈 되돌리기: squeeze 를 위해 움직였던 thumb(SQUEEZE_FORCE_FINGERS) 관절만
    #    파지(return_position) 위치로 되돌리고, 물체를 잡고 있는 나머지 finger 는 현재
    #    파지 상태 그대로 둔다. (전체를 재명령하면 다른 finger 도 움직여 물체를 놓칠 수 있음)
    print("  [squeeze] thumb 만 파지 position 으로 복귀 (나머지 finger 는 파지 유지)")
    # move_hand_to(16관절 실측 기준 재명령)를 쓰면 쥐고 있던 finger 도 풀렸다 조여진다.
    # thumb 관절만 보간하고 나머지는 마지막 명령값(squeezed)으로 고정 커밋한다.
    move_fingers_to(shm, return_position, SQUEEZE_FORCE_FINGERS, return_duration,
                    hold_position=squeezed)


def hand_hold_volt_zero(shm: ShmAccess, duration_sec: float) -> None:
    """Hand를 volt 모드로 두고 0V(16개 0)를 duration_sec 동안 유지 (0V 확인용)."""
    duration_sec = _clamp_duration(duration_sec)
    steps = max(1, int(duration_sec * CONTROL_RATE_HZ))
    dt = 1.0 / CONTROL_RATE_HZ
    zero = tuple(HAND_VOLT_ZERO_TARGET)
    for _ in range(steps):
        shm.write_partial(
            servo_on=(HAND_SAVE_SERVO_ON,),
            hand_mode=(HAND_SAVE_MODE,),
            j_tar=(zero,),
        )
        time.sleep(dt)


def run_one_sequence(
    shm: ShmAccess,
    paxini: PaxiniShmReader,
    engine=None,
) -> None:
    """1사이클(손만): Hand안전 → 파지 → 스퀴즈 → 스퀴즈 직후 추론.
    Franka(팔)는 이동하지 않는다(실행 전 자세 고정). '물체 원위치' 단계 없음.
    (deploy_ros2 도 이 함수를 공유 — ROS 버전과 시퀀스 동일)"""
    shm.write_partial(servo_on=(HAND_SAFE_SERVO_ON,), hand_mode=(HAND_SAFE_MODE,))
    print("===================1. 안전 위치 (손만)===================")
    move_hand_to(shm, HAND_SAFE_POSITION, HAND_MOVE_DURATION)

    print("===================2. 파지 위치 (손만)===================")
    # 파지(GRIP_FORCE_THRESHOLD) 도달 position 을 저장 → 스퀴즈 후 이 자세로 복귀.
    # target 까지 갔는데도 GRIP_FORCE_THRESHOLD 미달인 finger 는 2,3번 관절을 더 닫아 파지력 확보.
    grip_position = move_hand_to_target_until_force(
        shm, paxini, HAND_GRIP_POINT, HAND_MOVE_DURATION, GRIP_FORCE_THRESHOLD,
        grip_curl=True)

    print("===================3. 스퀴즈 모션===================")
    # (자코비안 없음) HAND_SAVE_POINT 방향으로 이동 → finger force≥SQUEEZE_FORCE_THRESHOLD(정지)
    # → hold → 파지(grip_position) 복귀.
    print("squeeze motion")
    move_hand_to_squeeze(shm, paxini, HAND_SAVE_POINT, HAND_SQUEEZE_DURATION,
                         SQUEEZE_FORCE_THRESHOLD, grip_position,
                         return_duration=HAND_SQUEEZE_RETURN_DURATION,
                         engine=engine)

    # ★ deploy: 스퀴즈 직후 추론 (절대강성 + 등급 터미널 출력)
    if engine is not None:
        stiffness, cls, cname = engine.infer()
        print("\n" + "=" * 48)
        if stiffness is not None:
            print(f"  [추론 결과] {engine.fruit}")
            print(f"    절대강성 = {stiffness:.3f}")
            print(f"    등급     = {cname}  (class {cls})")
        else:
            print("  [추론 결과] 샘플 부족 — 추론 불가 (스퀴즈가 너무 짧거나 무효 프레임)")
        print("=" * 48 + "\n")
    # (4단계 '물체 원위치' 제거 — 팔 고정, 스퀴즈 후 복귀만 하고 종료)


def _write_marker(marker_path: Path, event: str, demo_id: int) -> None:
    """데모 경계 마커 한 줄을 기록한다.

    형식: ``event,demo_id,t_mono_ns,iso_wall``
    - t_mono_ns: time.monotonic_ns(). 리눅스 CLOCK_MONOTONIC 은 시스템 전역이라
      hdf5_logger 의 ``21_logger_t_mono_ns`` 와 같은 시계 → HDF5 를 이 시각으로
      데모별 구간 슬라이스할 수 있다(split_demos_by_marker.py 참고).
    - iso_wall: 사람이 읽기 위한 벽시계 시각(정렬에는 사용하지 않음).
    """
    with open(marker_path, "a") as f:
        f.write(f"{event},{demo_id},{time.monotonic_ns()},{datetime.now().isoformat()}\n")
        f.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="모션 시퀀스 (실험 11데모, TOP/BOTTOM/middle 선택)")
    parser.add_argument("--TOP", action="store_true", help="Franka 저장점 TOP (기본)")
    parser.add_argument("--BOTTOM", action="store_true", help="Franka 저장점 BOTTOM")
    parser.add_argument("--middle", action="store_true", help="Franka 저장점 middle")
    parser.add_argument(
        "--marker-file",
        default=DEFAULT_MARKER_FILE,
        help=f"데모 시작/끝 마커 파일 경로 (기본: {DEFAULT_MARKER_FILE})",
    )
    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_args()
    print(f"[pose] HAND_GRIP_POINT ← {HAND_GRIP_POINT_FILE} (tar, rad→count): {HAND_GRIP_POINT}")
    print(f"[pose] HAND_SAVE_POINT ← HAND_GRIP_POINT + thumb_3 {_SQUEEZE_EXTRA_COUNT} count (~{SQUEEZE_EXTRA_DEG:g}°): {HAND_SAVE_POINT}")
    marker_path = Path(args.marker_file).resolve()
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    # 마커 파일은 덮어쓰지 않고 이어쓰기만 (시작 시 S, 끝날 때 E 한 줄씩 추가)
    _set_squeeze_flag(False)   # 이전 실행 잔여 플래그 제거(시작 시 0)

    shm = ShmAccess(key=SHM_MSG_KEY)
    if not shm.attach():
        raise SystemExit("SHM attach 실패. C++ 프로세스(Franka+KISTAR)가 먼저 실행 중이어야 합니다.")

    # PaXini tactile SHM(0x3934) — 접촉력 피드백 소스. 없으면(센서 미실행) 힘=0 으로
    # 간주되어 move_hand_to_target_until_force 는 target 까지 그대로 이동한다(안전 fallback).
    paxini = PaxiniShmReader(PAXINI_SHM_KEY)
    if not paxini.attach():
        print(f"[motion_sequence] 경고: PaXini SHM {hex(PAXINI_SHM_KEY)} 없음 — "
              "접촉력 피드백 없이 position 이동만 수행합니다.")

    # ★ deploy: 시작 시 과일을 묻고, 그 과일의 모델+포즈+채널처리를 한꺼번에 적용
    fruit = ask_fruit()
    model_path, pose_file, force_zero = resolve_fruit_config(fruit)  # 모델 없으면 여기서 안내후 종료
    set_pose_for_fruit(pose_file)                                    # 과일별 파지 포즈 로드
    set_thresholds_for_fruit(fruit)                                  # 과일별 파지/스퀴즈 임계값 로드
    engine = StiffnessInferenceEngine(model_path=model_path, fruit=fruit,
                                      label_dir=LABEL_DIR, force_zero=force_zero)
    print(f"[deploy] 추론엔진 준비 완료. 과일={fruit}, 모델={Path(model_path).name}")

    try:
        for demo_id in range(NUM_DEMOS):
            print(f"\n--- 데모 {demo_id}/{NUM_DEMOS - 1} ---")
            print(f"[motion_sequence] S 전송 (demo{demo_id})")
            _write_marker(marker_path, "S", demo_id)
            run_one_sequence(shm, paxini, engine=engine)   # ★ deploy: 엔진 전달
            print(f"[motion_sequence] E 전송 (demo{demo_id})")
            _write_marker(marker_path, "E", demo_id)
            if demo_id < NUM_DEMOS - 1:
                print(f"  전체 프로세스 {PAUSE_BETWEEN_DEMOS_SEC}초 대기 후 다음 데모(S) 시작...")
                time.sleep(PAUSE_BETWEEN_DEMOS_SEC)
        print("\n시퀀스 완료 (11 데모).")
    finally:
        _set_squeeze_flag(False)   # 종료 시 항상 0 으로
        shm.detach()


if __name__ == "__main__":
    main()