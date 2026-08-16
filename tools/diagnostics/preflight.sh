#!/usr/bin/env bash
# 전체 사전 점검 — 로봇을 움직이지 않는다.
#
#   tools/diagnostics/preflight.sh
#
# 확인 항목: ROS 도메인 · 제어 PC sequence_arbiter · 카메라 3스트림(Control PC 발행) ·
#            MoveIt 트윈 1개 · place 모델 서비스 5종(/health)
#
# 실체는 파이프라인 러너의 --dry-run 이다 (실행 경로와 100% 동일한 점검을 쓰기 위함).
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/run_fruit_demo.sh" --dry-run "$@"
