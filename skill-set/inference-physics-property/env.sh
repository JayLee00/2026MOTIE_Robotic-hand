#!/bin/bash
# stiffness_deploy_ros2 실행/테스트 환경 설정.
#
# 사용법:  source env.sh
#
# ★ conda 는 절대 쓰지 않는다 (이 머신의 ~/.bashrc `rs()` 관례와 동일한 이유).
#   conda(conda-forge 빌드) python 으로 시스템 ROS2 Humble 의 rclpy 를 PYTHONPATH 로 얹어
#   쓰면 import 는 되지만, conda 가 번들한 libstdc++ 등 공유 라이브러리가 시스템 RMW/DDS
#   확장 모듈과 ABI 충돌해 종료 시 크래시(core dump)가 재현됨 — deploy_task3_ros2 로 실증.
#   그래서 실제 ROS2 토픽 통신은 반드시 시스템 python3(/usr/bin/python3) 로 실행한다.
#   torch 는 시스템 python 에 `pip install --user` 로 설치돼 있다
#   (numpy/pyyaml 은 apt 시스템 패키지로 이미 충족 — requirements.txt 참고).

# 이 브리지는 Dual_Arm_Hand_Ctrl 워크스페이스를 source 하지 않는다.
# std_msgs/sensor_msgs 표준 메시지만 쓰고 커스텀 msg 패키지에 의존하지 않으므로,
# Dual_Arm_Hand_Ctrl 의 노드가 (동일 ROS_DOMAIN_ID 로) 어디서 떠 있든 DDS 로 토픽만
# 주고받으면 된다 — 로컬 워크스페이스 sourcing 여부와 데이터 경로는 무관함(source 없이도 동일 동작 확인).

# 활성 conda 환경이 있으면 완전히 벗어난다 (PATH 에서 miniconda3 경로 전부 제거).
if command -v conda >/dev/null 2>&1 && [ -n "${CONDA_DEFAULT_ENV}" ]; then
    conda deactivate 2>/dev/null || true
fi
export PATH="$(echo "$PATH" | tr ':' '\n' | grep -v 'miniconda3' | paste -sd: -)"
hash -r

source /opt/ros/humble/setup.bash

export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=9
export ROS_LOCALHOST_ONLY=0

echo "[env.sh] ROS2 humble 환경 준비 완료 (시스템 python, conda 미사용). Dual_Arm_Hand_Ctrl 은 별도 프로세스로 실행 후 토픽으로 연결."
echo "[env.sh] python: $(which python3) ($(python3 --version))"
