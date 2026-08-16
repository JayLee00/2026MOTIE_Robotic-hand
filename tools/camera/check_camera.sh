#!/usr/bin/env bash
# 카메라 스트림 확인 — front RealSense 는 **Control PC 가 발행**한다 (이 PC 에서 띄우지 않음).
#
#   tools/camera/check_camera.sh
#
# 세 스트림이 모두 살아 있어야 파지(seq 1)와 내려놓기(seq 4)가 동작한다.
#
# ⚠ 판정은 **실제 데이터 수신**으로만 한다. `ros2 topic list` 는 ros2 데몬 캐시를 쓰기 때문에
#   데몬이 차갑거나(방금 뜸) 오래된 상태면 살아 있는 토픽을 빈 목록으로 반환한다 — 실제로
#   "목록은 없음, 그런데 30Hz 수신 중"인 자기모순 출력이 나온 적이 있다. 목록은 참고용일 뿐이다.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../env" && pwd)/setup_env.sh"

TOPICS=(
  /front_cam/front/color/image_raw
  /front_cam/front/aligned_depth_to_color/image_raw
  /front_cam/front/color/camera_info
)

echo
echo "=== front_cam 토픽 목록 (참고 — 데몬 우회) ==="
ros2 topic list --no-daemon 2>/dev/null | grep -E '^/front_cam/' | sed 's/^/  /' \
  || echo "  (목록 조회 실패 — 아래 수신 판정을 볼 것)"

echo
echo "=== 실제 수신 판정 (각 4초) ==="
FAILED=()
for t in "${TOPICS[@]}"; do
  rate=$(timeout 6 ros2 topic hz "$t" 2>/dev/null | grep -m1 'average rate' | awk '{print $3}')
  if [ -n "$rate" ]; then
    printf '  OK    %-58s %s Hz\n' "$t" "$rate"
  else
    printf '  없음  %-58s\n' "$t"
    FAILED+=("$t")
  fi
done

echo
if [ ${#FAILED[@]} -eq 0 ]; then
  echo "카메라 정상 — 3개 스트림 모두 수신 중."
  echo "(위 '토픽 목록'이 비어 있어도 수신되고 있으면 정상이다. 데몬 캐시 문제다:"
  echo " 신경 쓰이면  ros2 daemon stop  후 다시 조회할 것.)"
  exit 0
fi

cat <<EOF
⚠ 수신 없음: ${FAILED[*]}

순서대로 확인:
  1. Control PC 에서 카메라가 떠 있는가
       ros2 launch trajectory_receiver control_pc.launch.py require_control:=true
     front 카메라는 이 launch 가 함께 띄운다(camera:=true 기본). 카메라만 필요하면
     나머지를 끈다: state_pub:=false trajectory:=false ee:=false arm_q:=false
                    hand:=false arbiter:=false
     ⚠ 파라미터(네임스페이스 front_cam/front, depth 노출 1500us 고정, temporal_filter,
       align_depth)가 그 launch 에 영구 반영되어 있다. rs_launch.py 를 직접 띄우면
       기본값으로 떠서 토픽 이름도 다르고 트레이 depth 가 대량 소실된다.
  2. ROS_DOMAIN_ID 가 양쪽 다 9 인가
  3. 방화벽이 DDS 를 막고 있지 않은가 (ping 은 되는데 토픽만 안 보이는 전형적 증상)
       sudo ufw status        # 필요 시 서브넷 허용
  4. 같은 서브넷인가 (이 PC 192.168.0.101 / Control PC 192.168.0.100)
  5. aligned_depth 만 없으면 Control PC 의 align_depth.enable 이 꺼진 것이다
EOF
exit 1
