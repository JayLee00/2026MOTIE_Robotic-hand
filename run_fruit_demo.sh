#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════════
#  단일 명령어 — 물체 파지 → 손 안 조작 → 물성 추론 → 물체 내려놓기
# ════════════════════════════════════════════════════════════════════════════
#
#   ./run_fruit_demo.sh                          # orange 기본
#   ./run_fruit_demo.sh --fruit kiwi
#   ./run_fruit_demo.sh --fruit kiwi --stiffness-fruit kiwi
#   ./run_fruit_demo.sh --dry-run                # preflight 만 (로봇 미동작)
#   ./run_fruit_demo.sh --help
#
# 이 스크립트는 환경만 잡아주고 pipeline/run_pipeline.py 로 넘긴다.
# 상세 절차/사전 조건은 docs/RUNBOOK.md.
#
# ⚠ 이 명령은 실제 로봇을 움직인다. E-stop 옆 인원 상주 필수.
# ⚠ 사전 조건(제어 PC): Dual_Arm RT 런타임 + control_pc.launch(require_control:=true,
#    sequence_arbiter 포함) + PaXini writer + front RealSense 카메라 발행.

set -o pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ROS2 + 워크스페이스 + DOMAIN_ID=9 (conda 이탈 포함)
# shellcheck source=tools/env/setup_env.sh
source "$HERE/tools/env/setup_env.sh"

exec /usr/bin/python3 -u "$HERE/pipeline/run_pipeline.py" "$@"
