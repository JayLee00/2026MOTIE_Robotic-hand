#!/usr/bin/env bash
# run_task3_demo.sh — (B) deploy_task3_ros2_demo.py 실행 + 결과 GUI 자동 실행.
#
# 이미 파지한 상태에서 스퀴즈 1회 + 3속성 추론. Pick→Inhand→Stiffness→Place
# 시퀀스 체인용이라 dual_arm_msgs + sequence_client 워크스페이스가 필요하다.
#
# 사용:
#   ./run_task3_demo.sh              # 과일(=포즈파일) 번호 입력, GUI 자동 실행
#   ./run_task3_demo.sh --no-gui
#   DUAL_ARM_WS=/다른/경로/install/setup.bash ./run_task3_demo.sh

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# 1) ROS2 Humble + 시스템 python3 (conda 미사용).
source "$HERE/env.sh"

# 2) (B) 전용 의존: dual_arm_msgs + sequence_client.
WS="${DUAL_ARM_WS:-$HOME/Dual_Arm_Hand_Ctrl/ros2/install/setup.bash}"
if [ -f "$WS" ]; then
    source "$WS"
    echo "[run_task3_demo] 워크스페이스 source: $WS"
else
    echo "[run_task3_demo] 경고: 워크스페이스 setup.bash 없음 -> $WS"
    echo "                 dual_arm_msgs/sequence_client import 가 실패합니다."
    echo "                 경로가 다르면:  DUAL_ARM_WS=/경로/install/setup.bash ./run_task3_demo.sh"
fi

echo "[run_task3_demo] python: $(which python3)"
echo "[run_task3_demo] (B) deploy_task3_ros2_demo 실행 (GUI 자동) ..."

exec python3 stiffness_deploy_ros2/launch/deploy_task3_ros2_demo.py "$@"
