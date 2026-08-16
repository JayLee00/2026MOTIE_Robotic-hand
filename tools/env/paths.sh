#!/usr/bin/env bash
# 프로젝트 경로 상수 — 모든 스크립트가 이 파일만 source 하면 위치를 안다.
# (스크립트 위치 기준으로 계산하므로 프로젝트를 통째로 옮겨도 그대로 동작한다.)
#
#   source "<...>/tools/env/paths.sh"

RAS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export RAS_ROOT

# ── skill-set (실행 순서 = 시퀀스 번호) ──────────────────────────────────────
export RAS_SKILL_GRASP="$RAS_ROOT/skill-set/grasp"                        # seq 1 물체 파지
export RAS_SKILL_INHAND="$RAS_ROOT/skill-set/in-hand-reorientation"       # seq 2 손 안 조작
export RAS_SKILL_PHYSICS="$RAS_ROOT/skill-set/inference-physics-property" # seq 3 물성 추론
export RAS_SKILL_PLACE="$RAS_ROOT/skill-set/place"                        # seq 4 물체 내려놓기

# ── ROS2 워크스페이스 (빌드 순서: fr_ws → kistar_ws) ─────────────────────────
export RAS_FR_WS="$RAS_ROOT/tools/ros2/fr_ws"
export RAS_DEX_ROS="$RAS_ROOT/tools/ros2/dex_ros"
export RAS_KISTAR_WS="$RAS_DEX_ROS/isaac-ros/kistar_ws"

# ── 기타 ─────────────────────────────────────────────────────────────────────
export RAS_PIPELINE="$RAS_ROOT/pipeline"
export RAS_TOOLS="$RAS_ROOT/tools"
export RAS_LOGS="$RAS_ROOT/logs"
export RAS_CONDA_BASE="${RAS_CONDA_BASE:-$HOME/miniconda3}"
