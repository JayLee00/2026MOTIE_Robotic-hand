#!/usr/bin/env bash
# 통합 실시간 시각화 (과일 6DoF + Paxini 촉각 3D + FT) 실행 래퍼 — env 내부 처리.
#
#   과일 포즈 : record/fruit_overlay.py 의 Overlay 재사용 (/fruit/pose, /fruit/size)
#   촉각      : kist-vtdp-wrapper 의 Scene3D(Open3D) 이식 (/paxini/<side>/raw 1524)
#   FT        : Fz/Tx/Ty x 4 스트립차트 (/hand/<side>/kin 12)
#
# 사용:
#   bash scripts/run_live_viz.sh                 # 실시간
#   bash scripts/run_live_viz.sh --selftest      # ROS 없이 렌더 검증 (PNG 저장)
#   bash scripts/run_live_viz.sh --no-tactile    # 과일만
#   bash scripts/run_live_viz.sh --vmax 2.0      # 촉각 색상 상한 조정
#   그 외 인자는 Visualization/live_viz.py 로 그대로 전달 (--color-topic 등)
#
# 키: q/ESC 종료, s 스냅샷(Visualization/snapshots/)
#
# ⚠️ sudo 로 돌리지 말 것 — root 는 X 디스플레이 접근이 막혀 창이 안 뜬다.
set +u   # ROS setup.bash 가 미설정 변수를 참조하므로 -u 금지
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="$(dirname "$HERE")"
# 2026-08-16 이관 (prime-ws): 이 PC 표준 환경 사용. 패키지의 fastdds_lan_only.xml 은
# 원본 PC(192.168.0.1) 전용 — 여기서 쓰면 DDS 가 전멸한다.
ROBOT_ROOT="$(cd "$PROJ/../../.." && pwd)"    # …/RobotAgentSystem
source "$ROBOT_ROOT/tools/env/setup_env.sh"
export DISPLAY="${DISPLAY:-:1}"

if [[ $EUID -eq 0 ]]; then
    echo "경고: root 로 실행 중입니다. 창이 안 뜨거나 X 접근이 막힐 수 있습니다."
    echo "      일반 사용자로 실행하세요:  bash scripts/run_live_viz.sh"
fi

exec python3 "$PROJ/3_visualization/Visualization/live_viz.py" "$@"
