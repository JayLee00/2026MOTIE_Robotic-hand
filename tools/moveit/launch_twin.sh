#!/usr/bin/env bash
# MoveIt 디지털 트윈 + RViz — 이 PC 에서 **정확히 1개**만 띄운다.
#
#   tools/moveit/launch_twin.sh                       # 기본 (실로봇 추종 + RViz)
#   tools/moveit/launch_twin.sh use_rviz:=false       # launch 인자 그대로 전달
#
# ⚠ move_group 이 2개면 /move_action 이 중복되어 **모든 arm move 가 실패**한다.
#   place skill 서버 preflight 도 이 경우 즉시 중단한다.
#
# 이 launch 는 카메라를 띄우지 않는다 — 카메라는 Control PC 담당.
# 실 하드웨어는 잡지 않고(ros2_control:=false / use_fake_hardware:=true) 계획/시각화만 한다.
set -o pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../env" && pwd)/setup_env.sh"

# ── 중복 검사 ────────────────────────────────────────────────────────────────
# ⚠ `ros2 node list` 를 쓰지 않는다: 그것은 ros2 데몬 캐시를 거치므로 데몬이 차갑거나
#   오래된 상태면 **살아 있는 move_group 을 0개로 보고**한다. 그 말을 믿고 여기서 하나 더
#   띄우면 정확히 우리가 막으려는 중복이 만들어진다. rclpy 로 노드 그래프를 직접 본다.
PROBE=$(/usr/bin/python3 - <<'PY' 2>/dev/null
import time
import rclpy
try:
    rclpy.init()
    n = rclpy.create_node("twin_dup_probe")
    time.sleep(2.0)                      # discovery
    found = [f"{ns.rstrip('/')}/{name}"
             for name, ns in n.get_node_names_and_namespaces() if name == "move_group"]
    n.destroy_node(); rclpy.shutdown()
    print("OK " + " ".join(found))
except Exception as e:                   # noqa: BLE001
    print(f"ERR {type(e).__name__}: {e}")
PY
)

case "$PROBE" in
  OK*)
    FOUND=${PROBE#OK}
    FOUND=${FOUND# }
    if [ -n "$FOUND" ]; then
      echo "⚠ move_group 이 이미 실행 중이다: $FOUND"
      echo "  중복 기동 금지 — 기존 것을 그대로 쓰거나, 종료한 뒤 다시 실행할 것."
      exit 1
    fi
    echo "[launch_twin] move_group 없음 확인 → 기동한다"
    ;;
  *)
    echo "⚠ move_group 중복 검사에 실패했다: $PROBE"
    echo "  검사 없이 띄우면 중복 /move_action 위험이 있어 중단한다."
    echo "  수동 확인:  ros2 node list --no-daemon | grep move_group"
    echo "  확인 후 강제로 띄우려면:  SKIP_DUP_CHECK=1 $0 $*"
    [ "${SKIP_DUP_CHECK:-0}" = "1" ] || exit 1
    echo "  SKIP_DUP_CHECK=1 — 검사를 건너뛰고 진행한다"
    ;;
esac

exec ros2 launch franka_kistar_bringup dual_fr3_kistar_moveit.launch.py \
    joint_state_mode:=direct \
    robot_ip:=192.168.0.100 \
    use_rviz:=true \
    "$@"
