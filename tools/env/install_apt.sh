#!/usr/bin/env bash
# 최초 1회 — 이 PC 에 부족한 ROS2 Humble 패키지를 설치한다. (sudo 필요)
#
#   tools/env/install_apt.sh              # 필수만
#   tools/env/install_apt.sh --with-camera # + RealSense (카메라를 이 PC 로 되돌릴 때만)
#
# 이미 설치된 것은 apt 가 알아서 건너뛴다.
set -eo pipefail

REQUIRED=(
  # MoveIt — 이 PC 가 디지털 트윈(move_group)을 소유한다
  ros-humble-moveit
  ros-humble-moveit-configs-utils
  ros-humble-moveit-planners-ompl
  ros-humble-moveit-kinematics
  ros-humble-moveit-simple-controller-manager
  ros-humble-moveit-ros-visualization
  # 컨트롤러 / 상태 발행 (트윈 launch 가 참조)
  ros-humble-controller-manager
  ros-humble-joint-state-broadcaster
  ros-humble-joint-trajectory-controller
  ros-humble-joint-state-publisher
  ros-humble-joint-state-publisher-gui
  # URDF
  ros-humble-xacro
  # libfranka 빌드 의존 (fr_ws). 이 PC 는 실기를 직접 잡지 않지만 워크스페이스를 통째로
  # 빌드하려면 필요하다 — 없으면 "Could NOT find Poco" 로 fr_ws 전체가 abort 된다.
  libpoco-dev
  ros-humble-pinocchio
  libeigen3-dev
  libtinyxml2-dev
)

# ⚠ 카메라는 Control PC 가 발행한다. 아래는 카메라를 이 PC 로 되돌릴 때만 필요하다.
CAMERA=(
  ros-humble-realsense2-camera
  ros-humble-realsense2-camera-msgs
)

PKGS=("${REQUIRED[@]}")
for a in "$@"; do
  [ "$a" = "--with-camera" ] && PKGS+=("${CAMERA[@]}")
done

echo "설치 대상 (${#PKGS[@]}개):"
printf '  %s\n' "${PKGS[@]}"
echo
sudo apt update
sudo apt install -y "${PKGS[@]}"

echo
echo "=== 완료 ==="
echo "다음: tools/ros2/build.sh"
