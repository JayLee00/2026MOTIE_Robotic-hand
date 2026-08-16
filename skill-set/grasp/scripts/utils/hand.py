#!/usr/bin/env python3
"""
손 자세 유틸리티 — configs/hand.yaml 을 읽어 상수 제공.

내보내는 주요 상수:
  HAND_INIT_ANGLES_HW     (rad, HW order)
  HAND_RELEASE_ANGLES_HW  (rad, HW order)
  FIXED_JOINT_ANGLES_DRO  (rad, DRO order)
  DRO_TO_HW               (index map)
  HAND_INIT_ENC           (int, HW order)
  HAND_RELEASE_ENC        (int, HW order)
  RAD_TO_ENC, ENC_TO_RAD
  HAND_STEPS, HAND_PERIOD
  APPROACH_OFFSET_M
"""

import math
from pathlib import Path

import yaml

_YAML_PATH = Path(__file__).resolve().parents[2] / "configs" / "hand.yaml"

# ---------------------------------------------------------------------------
# YAML 로드
# ---------------------------------------------------------------------------

def _load_cfg() -> dict:
    with open(_YAML_PATH) as f:
        return yaml.safe_load(f)


_cfg = _load_cfg()

# ---------------------------------------------------------------------------
# Encoder 변환 상수
# ---------------------------------------------------------------------------

RAD_TO_ENC = 8192.0 / math.pi
ENC_TO_RAD = math.pi / 8192.0


def to_enc(angles_rad: list) -> list:
    """라디안 리스트 → int encoder 리스트."""
    return [int(round(a * RAD_TO_ENC)) for a in angles_rad]


# ---------------------------------------------------------------------------
# YAML에서 각도 읽기 헬퍼
# ---------------------------------------------------------------------------

def _r(degrees: float) -> float:
    """degrees → radians."""
    return math.radians(degrees)


# ---------------------------------------------------------------------------
# 기본 각도 파싱
# ---------------------------------------------------------------------------

_abd      = _r(_cfg["abduction_deg"])
_th_j1_i  = _r(_cfg["hand_init"]["thumb_j1_deg"])
_th_j2_i  = _r(_cfg["hand_init"]["thumb_j2_deg"])
_fi_j3_i  = _r(_cfg["hand_init"]["finger_j3_deg"])

_th_j1_r  = _r(_cfg["hand_release"]["thumb_j1_deg"])
_th_j2_r  = _r(_cfg["hand_release"]["thumb_j2_deg"])
_fi_j3_r  = _r(_cfg["hand_release"]["finger_j3_deg"])

_th_j1_g  = _r(_cfg["hand_grasp"]["thumb_j1_deg"])
_th_j2_g  = _r(_cfg["hand_grasp"]["thumb_j2_deg"])
_fi_bend_g = _r(_cfg["hand_grasp"]["finger_bend_deg"])
# 관절별 override (옵션) — 없으면 finger_bend_deg 를 그대로 사용
_fi_j2_g = _r(_cfg["hand_grasp"].get("finger_j2_deg", _cfg["hand_grasp"]["finger_bend_deg"]))
_fi_j3_g = _r(_cfg["hand_grasp"].get("finger_j3_deg", _cfg["hand_grasp"]["finger_bend_deg"]))
_fi_j4_g = _r(_cfg["hand_grasp"].get("finger_j4_deg", _cfg["hand_grasp"]["finger_bend_deg"]))
# 엄지 3·4번 관절 override (옵션) — 다른 손가락과 독립적으로 조절
_th_j3_g = _r(_cfg["hand_grasp"].get("thumb_j3_deg", _cfg["hand_grasp"]["finger_bend_deg"]))
_th_j4_g = _r(_cfg["hand_grasp"].get("thumb_j4_deg", _cfg["hand_grasp"]["finger_bend_deg"]))

_z = 0.0   # zero

# ---------------------------------------------------------------------------
# 손 자세 배열 (HW order: thumb×4, index×4, middle×4, ring×4)
# ---------------------------------------------------------------------------

HAND_INIT_ANGLES_HW: list[float] = [
    _th_j1_i, _th_j2_i, _z,      _z,      # thumb
    -_abd,    _z,        _fi_j3_i, _z,     # index  j1=-abd, j3=굽힘
     _z,      _z,        _fi_j3_i, _z,     # middle j3=굽힘
     _abd,    _z,        _fi_j3_i, _z,     # ring   j1=+abd, j3=굽힘
]

HAND_RELEASE_ANGLES_HW: list[float] = [
    _th_j1_r, _th_j2_r, _z,      _z,      # thumb
    -_abd,    _z,        _fi_j3_r, _z,     # index  j3=0 (펴짐)
     _z,      _z,        _fi_j3_r, _z,     # middle j3=0
     _abd,    _z,        _fi_j3_r, _z,     # ring   j3=0
]

# ---------------------------------------------------------------------------
# 파지 자세 (DRO internal order: index×4, middle×4, ring×4, thumb×4)
# ---------------------------------------------------------------------------

DRO_TO_HW: list[int] = [
    12, 13, 14, 15,   # thumb  → HW  0-3
     0,  1,  2,  3,   # index  → HW  4-7
     4,  5,  6,  7,   # middle → HW  8-11
     8,  9, 10, 11,   # ring   → HW 12-15
]

FIXED_JOINT_ANGLES_DRO: list[float] = [
    -_abd,     _fi_j2_g, _fi_j3_g, _fi_j4_g,  # index  (0-3)  j2/j3/j4 개별 조절 가능
     _z,       _fi_j2_g, _fi_j3_g, _fi_j4_g,  # middle (4-7)
     _abd,     _fi_j2_g, _fi_j3_g, _fi_j4_g,  # ring   (8-11)
    _th_j1_g, _th_j2_g,  _th_j3_g, _th_j4_g,  # thumb  (12-15) j3/j4 개별 조절 가능
]

# ---------------------------------------------------------------------------
# Encoder 값 (int, HW order)
# ---------------------------------------------------------------------------

HAND_INIT_ENC:    list[int] = to_enc(HAND_INIT_ANGLES_HW)
HAND_RELEASE_ENC: list[int] = to_enc(HAND_RELEASE_ANGLES_HW)

# ---------------------------------------------------------------------------
# 손 모션 파라미터
# ---------------------------------------------------------------------------

HAND_STEPS:  int   = int(_cfg["motion"]["steps"])
HAND_PERIOD: float = float(_cfg["motion"]["period_sec"])

# approach_offset_m → utils/arm.py (configs/arm.yaml)
