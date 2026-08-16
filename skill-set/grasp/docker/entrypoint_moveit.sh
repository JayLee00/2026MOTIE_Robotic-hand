#!/usr/bin/env bash
# Grasp_fruit MoveIt 컨테이너 진입점
# 역할: franka_kistar_moveit_config 워크스페이스 소싱 → move_group + joint_state_relay 실행
#
# 마운트: -v <host_kistar_ws>:/root/HARILAB/dex_ros/isaac-ros/kistar_ws:ro
#   install 디렉토리 내 심볼릭 링크가 /root/HARILAB/.../build/... 를 가리키므로
#   컨테이너 안에서도 동일한 절대 경로로 마운트해야 링크가 정상 해소된다.
set -eo pipefail

KISTAR_WS="/root/HARILAB/dex_ros/isaac-ros/kistar_ws"

set +u
source /opt/ros/humble/setup.bash
set -u

if [ ! -f "${KISTAR_WS}/install/setup.bash" ]; then
    echo "[ERROR] ${KISTAR_WS}/install/setup.bash 없음."
    echo "        start_moveit.sh 가 올바른 볼륨 마운트 경로를 사용하는지 확인하세요."
    exit 1
fi

set +u
source "${KISTAR_WS}/install/setup.bash"
set -u

export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-9}
export RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}
export ROS_LOCALHOST_ONLY=0

echo "=== Grasp_fruit MoveIt (planning only) ==="
echo "    ROS_DOMAIN_ID=${ROS_DOMAIN_ID}"
echo "    RMW=${RMW_IMPLEMENTATION}"
echo ""

# joint_state_relay 백그라운드 실행 (/franka/joint_position → /joint_states)
python3 /ros2_moveit/franka_joint_state_relay.py &
RELAY_PID=$!
echo "[OK] franka_joint_state_relay PID=${RELAY_PID}"

# move_group + robot_state_publisher 론치
ros2 launch /ros2_moveit/launch_moveit.py &
LAUNCH_PID=$!
echo "[OK] launch_moveit PID=${LAUNCH_PID}"

# 두 프로세스 중 하나가 종료되면 모두 종료
wait -n ${RELAY_PID} ${LAUNCH_PID}
kill ${RELAY_PID} ${LAUNCH_PID} 2>/dev/null || true
