#!/usr/bin/env bash
# run_task3_gui.sh — 실제 환경에서 "강성 스퀴즈 배포 + 결과 GUI" 를 한 번에 실행.
#
# deploy_task3_ros2.py 가 결과 GUI(stiffness_gui.py)를 자동으로 별도 프로세스로 띄운다.
# 이 스크립트는 그 실행에 필요한 ROS2 환경(env.sh)과 Dual_Arm 워크스페이스
# (dual_arm_msgs + sequence_client) 를 source 해 주는 역할만 한다.
#
# 사용:
#   ./run_task3_gui.sh              # 실행 후 과일 번호 입력, GUI 자동 실행
#   ./run_task3_gui.sh --no-gui     # GUI 없이 (이미 GUI 를 따로 띄웠을 때)
#   DUAL_ARM_WS=/다른/경로/install/setup.bash ./run_task3_gui.sh   # 워크스페이스 경로 지정
#
# GUI 만 따로 보고 싶으면:  source env.sh && python3 stiffness_deploy_ros2/gui/stiffness_gui.py
# ROS 없는 PC 에서 GUI 확인:  python3 stiffness_deploy_ros2/gui/stiffness_gui.py --demo

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# 1) ROS2 Humble + 시스템 python3 (conda 미사용). rclpy 는 여기서 붙는다.
source "$HERE/env.sh"

# 2) deploy_task3_ros2 전용 의존: dual_arm_msgs + sequence_client 워크스페이스.
WS="${DUAL_ARM_WS:-$HOME/Dual_Arm_Hand_Ctrl/ros2/install/setup.bash}"
if [ -f "$WS" ]; then
    source "$WS"
    echo "[run_task3_gui] 워크스페이스 source: $WS"
else
    echo "[run_task3_gui] 경고: 워크스페이스 setup.bash 없음 -> $WS"
    echo "                dual_arm_msgs/sequence_client import 가 실패할 수 있습니다."
    echo "                경로가 다르면:  DUAL_ARM_WS=/경로/install/setup.bash ./run_task3_gui.sh"
fi

echo "[run_task3_gui] python: $(which python3)"
echo "[run_task3_gui] 배포 실행 (GUI 는 자동 실행됨) ..."

# 3) 배포 실행. GUI 는 deploy_task3_ros2 가 서브프로세스로 띄운다. 인자는 그대로 전달.
exec python3 stiffness_deploy_ros2/launch/deploy_task3_ros2.py "$@"
