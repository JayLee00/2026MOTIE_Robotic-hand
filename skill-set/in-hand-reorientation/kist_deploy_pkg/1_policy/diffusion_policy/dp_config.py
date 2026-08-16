#!/usr/bin/env python3
"""Diffusion Policy 공용 설정 — 학습·평가·배포가 같은 값을 쓰도록 한 곳에 모음.

여기 값을 바꾸면 train / eval_rollout / run 이 모두 따라간다. 단, 학습된 체크포인트는
자기 하이퍼파라미터를 함께 저장하므로 배포 시엔 체크포인트 값이 우선한다(불일치 방지).

obs / action 스펙 (사용자 지정):
    obs    = 03_hand_j_pos(16) + 06_hand_j_kin(12) + 20_paxini_ft(12)
             + 30_fruit_pos(3) + 32_fruit_size(3)              = 46
    action = 04_hand_j_tar(16)  절대 관절 타겟 [count]
"""
from __future__ import annotations

# ── HDF5 키 → obs 구성 ──────────────────────────────────────────────────────
# 2026-08-07 수집분부터 레코더 저장 순서·이름이 바뀌었다. 이름으로 매칭하므로
# 순서는 무관하지만 개명된 키는 반영해야 한다:
#     20_paxini_ft  →  10_hand_paxini_ft
#     21_paxini_raw →  11_hand_paxini_raw   (여기선 미사용)
# 이번 변경: 과일 방향(31_fruit_quat) 을 obs 에 추가 → 46 → 50 차원.
OBS_SPEC = [
    ("03_hand_j_pos",     16),   # 손 현재 관절각 [count]
    ("06_hand_j_kin",     12),   # 손 F/T (손가락당 Ty,Tx,Fz). 중지 3채널은 센서 고장으로 0
    ("10_hand_paxini_ft", 12),   # paxini 촉각 F/T (손가락당 Fz,Fx,Fy)
    ("30_fruit_pos",       3),   # 과일 위치 [m], 카메라 광학 프레임
    ("31_fruit_quat",      4),   # 과일 방향 (회전 위상) — 이번에 추가
    ("32_fruit_size",      3),   # 과일 축 길이 [a,b,c] [m]
]

# 쿼터니언을 손대지 않는 이유: 실측 결과 데모 내 부호 뒤집힘이 0회이고 norm 이 정확히 1,
# 프레임간 변화도 0.12~0.20 으로 이미 연속이다. 흔히 하는 w>=0 정규화는 w 가 음수 구간
# (실측 최소 -0.96)을 지날 때 오히려 인위적 점프를 만든다. 그래서 원본 그대로 넣는다.
# 연속 표현이 필요하면 --rot6d 로 6D 회전표현(R 의 앞 두 열)을 대신 쓴다.
QUAT_KEY = "31_fruit_quat"

OBS_KEYS = [k for k, _ in OBS_SPEC]
OBS_DIM = sum(d for _, d in OBS_SPEC)          # 50 (과일 quat 포함)

# obs 벡터 안에서 각 센서군이 차지하는 구간 — ablation·모달리티 dropout 이 쓴다.
OBS_GROUPS: list[tuple[str, int, int]] = []
_off = 0
for _k, _d in OBS_SPEC:
    OBS_GROUPS.append((_k, _off, _off + _d))
    _off += _d
assert _off == OBS_DIM

ACTION_KEY = "04_hand_j_tar"
ACTION_DIM = 16

# ── 시간 축 ────────────────────────────────────────────────────────────────
SRC_HZ = 100.0            # HDF5 원본 기록 주기
DS_STRIDE = 5             # 윈도우 내부 샘플 간격 → 실효 20Hz
CTRL_HZ = SRC_HZ / DS_STRIDE   # 20.0  (배포 제어 주기 기본값)

OBS_HORIZON = 2           # 관찰 스텝 수 (t-stride, t)
PRED_HORIZON = 16         # 예측 액션 길이 → 16 x 50ms = 0.8s
ACTION_STEPS = 8          # 추론 1회당 실행 스텝 수 → 0.4s

DIFF_STEPS = 100          # DDPM 학습 스텝
INFER_STEPS = 10          # DDIM 추론 스텝

# ── 관절 한계 ──────────────────────────────────────────────────────────────
# 출처: tools/glove_teleop.py HAND_LIMITS — 이 데이터의 action 을 실제로 만든 표.
# 1 count = pi/8192 rad (4096 count = 90deg).
JOINT_LIMITS: list[tuple[int, int]] = [(0, 4096) for _ in range(16)]
JOINT_LIMITS[1] = (-4096, 4096)                 # 엄지 외전
for _i in (4, 8, 12):                           # 검지/중지/약지 외전
    JOINT_LIMITS[_i] = (-1000, 1000)
for _i in (3, 7, 11, 15):                       # 엄지 IP / 검지·중지·약지 DIP
    JOINT_LIMITS[_i] = (-2048, 4096)

# 속도 가드: 학습 데이터의 20Hz 틱당 |Δaction| 실측 p99.9 = p100 = 600 count
# (글러브 MAX_STEP=100/frame 에서 오는 하드 캡). 초당으로 환산해 두면 control_hz 를
# 바꿔도 같은 물리 속도가 유지된다.
MAX_RATE_CPS = 600.0 * CTRL_HZ                  # 12000 count/s

# ── 과일 인식 이상치 위생 처리 ──────────────────────────────────────────────
# live_bbox_gui 가 트랙을 놓치면 중심이 수십 cm 튀거나 크기가 2배로 뜬다.
# 튄 프레임은 직전 유효값으로 홀드한다(레코더가 토픽 끊길 때 하는 동작과 동일).
FRUIT_POS_MAX_NORM = 0.60       # |p| 이 이보다 크면 오검출 [m]
FRUIT_POS_MAX_JUMP = 0.10       # 프레임간 점프가 이보다 크면 오검출 [m]
FRUIT_SIZE_RANGE = (0.02, 0.15)  # 축 길이 상식 범위 [m]

# ── 학습 데이터: 파일별 사용 구간 ───────────────────────────────────────────
# 출처: record/logs/'20260805_lemon epi 20' (사람이 적어둔 가위질 목록)
#   값이 int  → 0 ~ 그 인덱스까지 사용
#   None      → 전체 사용
#   "DELETE"  → 사용 안 함
#   "DEAD"    → 정지 구간(액션 std < 5 count)이라 제외. 노트에는 full 이었지만
#               5.6초 동안 손이 멈춰 있어 '가만히 있기'를 학습시킨다.
DATA_ROOT = "/home/js/Desktop/vive_franka_teleop/record/logs"

# 2026-08-07 수집분 (record/logs/'20260807_lemon epi 20_ver2').
# 이번 노트에는 클램프 숫자가 없다 — 낙하한 에피소드만 빼고 나머지는 전체 사용.
CLAMP_SPEC: dict[str, dict[int, object]] = {
    "exp_20260807_031811_lemon_wo_object.h5": {
        0: None, 1: None, 2: None, 3: None, 4: None,
        5: "DELETE",                                   # 낙하 — 학습 사용 x
        6: None, 7: None, 8: None, 9: None, 10: None, 11: None, 12: None, 13: None,
    },
    "exp_20260807_035756_lemon_wo_object_2.h5": {
        0: "DELETE",                                   # 낙하 — 학습 사용 x
        1: None, 2: None, 3: None, 4: None, 5: None,
        6: None, 7: None, 8: None, 9: None, 10: None, 11: None,
    },
}

# 이전 수집분(2026-08-05). 원본 h5 가 다른 세션에 의해 삭제되어 더 이상 로드 불가.
# 기록용으로 남긴다.
CLAMP_SPEC_20260805_LOST: dict[str, dict[int, object]] = {
    "exp_20260805_020543.h5": {
        0: None, 1: None, 2: None, 3: None, 4: 1900, 5: 2800,
        6: None, 7: 2000, 8: 900, 9: 800, 10: 3000,
    },
    "exp_20260805_022717.h5": {
        0: 1500, 1: None, 2: None, 3: "DEAD", 4: 1900, 5: None, 6: None,
        7: None, 8: 900, 9: 2300, 10: None, 11: None, 12: "DELETE",
    },
}

# ── 분할 ───────────────────────────────────────────────────────────────────
SEED = 42
N_HELD_OUT = 3            # 홀드아웃 데모 수 (open-loop rollout 평가용)
