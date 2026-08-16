#!/usr/bin/env bash
# ROS2 워크스페이스 2개를 올바른 순서로 빌드한다 (최초 1회 + 소스 변경 시).
#
#   tools/ros2/build.sh              # 둘 다 빌드
#   tools/ros2/build.sh kistar       # kistar_ws 만
#   tools/ros2/build.sh franka       # fr_ws 만
#
# 순서가 중요하다: kistar 의 xacro 가 franka_description 의 robots/common/franka_robot.xacro
# 와 fr3 yaml, 그리고 package://franka_description/meshes/** 를 참조한다.
#
# ⚠ `set -u` 는 쓰지 않는다 — /opt/ros/humble/setup.bash 가 미설정 변수를 참조해서 죽는다.
set -eo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../env" && pwd)/paths.sh"
WHAT="${1:-all}"

# conda 이탈 (colcon/ament 는 시스템 python3 로).
# ⚠ conda 가 PATH 에 남으면 cmake 의 find_package(Python3) 가 conda python 을 잡아
#   rosidl 이 "No module named 'em'" 로 실패한다.
PATH="$(echo "$PATH" | tr ':' '\n' | grep -v -i 'conda' | paste -sd: -)"
export PATH
unset PYTHONPATH PYTHONHOME CONDA_PREFIX CONDA_DEFAULT_ENV || true

source /opt/ros/humble/setup.bash

if [ "$WHAT" = "all" ] || [ "$WHAT" = "franka" ]; then
  echo "=== [1/2] fr_ws (franka_description 등) ==="
  cd "$RAS_FR_WS"
  # 활성 실행 경로가 실제로 요구하는 franka 패키지는 franka_description 하나뿐이다
  # (kistar xacro 가 $(find franka_description) 로 include). 나머지(libfranka/franka_hardware
  # 등)는 이 PC 가 실기를 직접 잡지 않으므로 쓰이지 않지만, 워크스페이스 일관성을 위해 함께
  # 빌드한다 — libfranka 의 C++ 의존(Poco/pinocchio/Eigen/TinyXML2)은 install_apt.sh 가 넣는다.
  #
  # 의존을 넣기 곤란한 환경이면 필요한 것만 빌드해도 트윈은 정상 동작한다:
  #   colcon build --symlink-install --packages-select franka_description
  colcon build --symlink-install
fi

if [ "$WHAT" = "all" ] || [ "$WHAT" = "kistar" ]; then
  echo "=== [2/2] kistar_ws (bringup/description/moveit_config/dual_arm_msgs/sequence_client) ==="
  [ -f "$RAS_FR_WS/install/setup.bash" ] && source "$RAS_FR_WS/install/setup.bash"
  cd "$RAS_KISTAR_WS"
  colcon build --symlink-install
fi

echo
echo "=== 완료 ==="
echo "  source $RAS_ROOT/tools/env/setup_env.sh"
