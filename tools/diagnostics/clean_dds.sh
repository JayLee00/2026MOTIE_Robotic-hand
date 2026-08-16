#!/usr/bin/env bash
# DDS 잔재 정리 — 고아 ROS 프로세스 + 스테일 Fast DDS 공유메모리.
#
#   tools/diagnostics/clean_dds.sh            # 확인만 (아무것도 안 지움)
#   tools/diagnostics/clean_dds.sh --apply    # 실제 정리
#
# 언제 쓰나:
#   · place 서버가 "no move_group node on domain 9" 로 죽을 때
#   · "[RTPS_TRANSPORT_SHM Error] Failed init_port ... open_and_lock_file failed"
#   · 실행을 여러 번 중단한 뒤 디스커버리가 이상할 때
#
# 왜 필요한가: 고아 노드는 DDS participant 를 계속 점유한다. 이 PC 의 Fast DDS 프로파일은
# useBuiltinTransports=false 라 같은 호스트 디스커버리가 participant ID 범위
# (maxInitialPeersRange)에 걸리는데, 고아가 쌓이면 새로 뜨는 move_group 의 ID 가 그 범위를
#넘어가 **나중에 뜨는 프로세스에게 안 보이게** 된다. SHM 세그먼트도 함께 새어 나간다.
# 상세: docs/TROUBLESHOOTING.md
#
# ⚠ 이 스크립트는 **이 PC 의 로컬 ROS 프로세스만** 건드린다. Control PC 는 무관하다.
#   실행 중인 데모가 있으면 먼저 끝낼 것.
set -o pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../env" && pwd)/setup_env.sh" > /dev/null

APPLY=0
[ "${1:-}" = "--apply" ] && APPLY=1

PATTERNS=(
  "[m]oveit_ros_move_group" "[r]obot_state_publisher" "[s]tatic_transform_publisher"
  "[t]rajectory_bridge" "[j]oint_state_publisher" "[r]viz2" "[m]oveit_simple"
  "[v]ision_pipeline.skill_server" "[_]ros2cli_daemon"
)

echo "=== 로컬 ROS 프로세스 ==="
TOTAL=0
for p in "${PATTERNS[@]}"; do
  c=$(pgrep -cf "$p" 2>/dev/null); c=${c:-0}
  [ "$c" -gt 0 ] && { printf '  %-34s %s\n' "${p//[\[\]]/}" "$c"; TOTAL=$((TOTAL+c)); }
done
[ "$TOTAL" -eq 0 ] && echo "  (없음)"

echo
echo "=== Fast DDS 공유메모리 ==="
echo "  /dev/shm 의 fastrtps 파일: $(ls /dev/shm 2>/dev/null | grep -ci fastrtps)"
command -v fastdds >/dev/null && fastdds shm clean 2>&1 | sed 's/^/  /'

if [ "$APPLY" -eq 0 ]; then
  echo
  echo "=== 확인 모드 — 실제 정리는 --apply ==="
  exit 0
fi

echo
echo "=== 정리 ==="
for p in "${PATTERNS[@]}"; do kill $(pgrep -f "$p" 2>/dev/null) 2>/dev/null; done
sleep 5
for p in "${PATTERNS[@]}"; do kill -9 $(pgrep -f "$p" 2>/dev/null) 2>/dev/null; done
sleep 2

LEFT=0
for p in "${PATTERNS[@]}"; do n=$(pgrep -cf "$p" 2>/dev/null); LEFT=$((LEFT + ${n:-0})); done
echo "  남은 ROS 프로세스: $LEFT"

if [ "$LEFT" -eq 0 ]; then
  # 살아 있는 참여자가 없을 때만 안전하게 전량 제거한다.
  # ⚠ 프로세스가 살아 있는 상태에서 지우면 그 프로세스의 SHM 포트가 깨져
  #   "Failed init_port ... open_and_lock_file failed" 가 난다.
  rm -f /dev/shm/fastrtps_* /dev/shm/sem.fastrtps_* 2>/dev/null
  echo "  fastrtps 공유메모리 제거 완료 (남은: $(ls /dev/shm 2>/dev/null | grep -ci fastrtps))"
else
  echo "  ⚠ ROS 프로세스가 남아 있어 공유메모리는 건드리지 않았다 (살아있는 포트를 깨뜨린다)"
fi
command -v fastdds >/dev/null && fastdds shm clean 2>&1 | sed 's/^/  /'
echo
echo "완료. 이제 ./run_fruit_demo.sh 를 다시 실행하면 된다."
