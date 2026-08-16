#!/usr/bin/env bash
# Grasp_fruit MoveIt 컨테이너 시작 스크립트
# 처음 실행 시 이미지 빌드 (~5분), 이후는 캐시 사용으로 수 초 내 시작
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GRASP_FRUIT_DIR="$(dirname "${SCRIPT_DIR}")"

IMAGE_NAME="grasp_fruit_moveit:latest"
KISTAR_WS="/home/kist/HARILAB/dex_ros/isaac-ros/kistar_ws"
# install 내 심볼릭 링크가 원본 빌드 경로(/root/HARILAB/...)를 가리키므로
# 컨테이너 안에서도 동일한 절대 경로로 마운트한다.
KISTAR_WS_IN_CONTAINER="/root/HARILAB/dex_ros/isaac-ros/kistar_ws"

# ── 이미지 빌드 (변경 없으면 캐시 사용) ──────────────────────────────────────
echo "=== Docker 이미지 빌드: ${IMAGE_NAME} ==="
docker build \
    -f "${GRASP_FRUIT_DIR}/docker/Dockerfile.moveit" \
    -t "${IMAGE_NAME}" \
    "${GRASP_FRUIT_DIR}"

# ── 컨테이너 실행 ─────────────────────────────────────────────────────────────
echo ""
echo "=== MoveIt 컨테이너 시작 ==="
echo "    - move_group  (/move_action, /compute_ik, /compute_cartesian_path)"
echo "    - robot_state_publisher"
echo "    - franka_joint_state_relay  (/franka/joint_position → /joint_states)"
echo ""
echo "종료: Ctrl+C"
echo ""

docker run --rm \
    --network=host \
    -e ROS_DOMAIN_ID=9 \
    -e RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
    -e ROS_LOCALHOST_ONLY=0 \
    -v "${KISTAR_WS}:${KISTAR_WS_IN_CONTAINER}:ro" \
    "${IMAGE_NAME}"
