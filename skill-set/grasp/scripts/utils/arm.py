#!/usr/bin/env python3
"""
팔 설정 유틸리티 — configs/arm.yaml 을 읽어 상수 제공.

내보내는 주요 상수:
  PLANNING_GROUP, EE_LINK, REF_FRAME, PLANNING_TIME
  HOME_JOINT_NAMES, HOME_JOINT_VALUES
  APPROACH_OFFSET_M, PLACE_Z_DESCENT_M, GRASP_Z_OFFSET_M
  EE_YAW_DEG, EE_X_OFFSET_M, EE_Y_OFFSET_M
  TOP_Z_PCT, Z_TOP_PCT
"""

from pathlib import Path
import yaml

_YAML_PATH = Path(__file__).resolve().parents[2] / "configs" / "arm.yaml"


def _load_cfg() -> dict:
    with open(_YAML_PATH) as f:
        return yaml.safe_load(f)


_cfg = _load_cfg()

# ── MoveIt 계획 ───────────────────────────────────────────────────────────────
PLANNING_GROUP: str   = _cfg["planning"]["group_name"]
EE_LINK:        str   = _cfg["planning"]["ee_link"]
REF_FRAME:      str   = _cfg["planning"]["ref_frame"]
PLANNING_TIME:  float = float(_cfg["planning"]["planning_time_sec"])

# ── HOME 자세 ─────────────────────────────────────────────────────────────────
HOME_JOINT_NAMES:  list[str]   = list(_cfg["home"]["joint_names"])
HOME_JOINT_VALUES: list[float] = list(_cfg["home"]["joint_values"])

# ── Approach / Place / Grasp 오프셋 ──────────────────────────────────────────
APPROACH_OFFSET_M:   float = float(_cfg["approach_offset_m"])
PLACE_Z_DESCENT_M:   float = float(_cfg["place_z_descent_m"])
GRASP_Z_OFFSET_M:    float = float(_cfg["grasp_z_offset_m"])

# ── EE 자세 보정 ──────────────────────────────────────────────────────────────
EE_YAW_DEG:   float = float(_cfg["ee_correction"]["yaw_deg"])
EE_X_OFFSET_M: float = float(_cfg["ee_correction"]["x_offset_m"])
EE_Y_OFFSET_M: float = float(_cfg["ee_correction"]["y_offset_m"])

# ── 포인트 클라우드 ───────────────────────────────────────────────────────────
TOP_Z_PCT: float = float(_cfg["pointcloud"]["top_z_pct"])
Z_TOP_PCT:  float = float(_cfg["pointcloud"]["z_top_pct"])
