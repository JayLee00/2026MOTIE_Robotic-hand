#!/usr/bin/env bash
# run_deploy_gui.sh — 실제 환경에서 "전체 배포(deploy_ros2) + 결과 GUI" 를 한 번에 실행.
#
# deploy_ros2.py 가 결과 GUI(stiffness_gui.py)를 자동으로 별도 프로세스로 띄운다.
# deploy_ros2 는 dual_arm_msgs/sequence_client 에 의존하지 않으므로(std_msgs 만 사용)
# run_task3_gui.sh 와 달리 Dual_Arm 워크스페이스 source 가 필요 없다 — env.sh 만으로 충분.
#
# 사용:
#   ./run_deploy_gui.sh              # 실행 후 과일 번호 입력, GUI 자동 실행
#   ./run_deploy_gui.sh --no-gui     # GUI 없이 (개발용 / 이미 GUI 를 따로 띄웠을 때)
#
# GUI 만 따로 보고 싶으면:  source env.sh && python3 stiffness_deploy_ros2/gui/stiffness_gui.py
# ROS 없는 PC 에서 GUI 확인:  /usr/bin/python3 stiffness_deploy_ros2/gui/stiffness_gui.py --demo

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# ROS2 Humble + 시스템 python3 (conda 미사용). rclpy 는 여기서 붙는다.
source "$HERE/env.sh"

echo "[run_deploy_gui] python: $(which python3)"
echo "[run_deploy_gui] 배포 실행 (GUI 는 자동 실행됨) ..."

# 배포 실행. GUI 는 deploy_ros2 가 서브프로세스로 띄운다. 인자는 그대로 전달.
exec python3 stiffness_deploy_ros2/launch/deploy_ros2.py "$@"
