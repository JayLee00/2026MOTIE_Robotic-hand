#!/usr/bin/env bash
# run_demo.sh — (A) deploy_ros2_demo.py 단독 실행 + 결과 GUI 자동 실행.
#
# 파지부터 직접 하고, 스퀴즈·3속성(무게/크기/강성) 추론을 반복한다.
# 표준 메시지(std_msgs/sensor_msgs)만 쓰므로 env.sh 하나면 충분하다
# (dual_arm_msgs / sequence_client 불필요 — 그건 run_task3_demo.sh 쪽).
#
# 사용:
#   ./run_demo.sh              # 포즈 번호 입력, GUI 자동 실행
#   ./run_demo.sh --no-gui     # GUI 없이(개발용)
#
# GUI 만 따로:            source env.sh && python3 stiffness_deploy_ros2/gui/property_gui.py
# 로봇 없이 GUI 확인:      python3 stiffness_deploy_ros2/gui/property_gui.py --demo

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

source "$HERE/env.sh"

echo "[run_demo] python: $(which python3)"
echo "[run_demo] (A) deploy_ros2_demo 실행 (GUI 자동) ..."

exec python3 stiffness_deploy_ros2/launch/deploy_ros2_demo.py "$@"
