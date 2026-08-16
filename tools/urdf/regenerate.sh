#!/usr/bin/env bash
# 생성 URDF 스냅샷 재생성 + place 사본 동기화.
#
#   tools/urdf/regenerate.sh
#
# 언제 필요한가:
#   - xacro / 메시 / franka_description 을 수정했을 때
#   - 트윈 launch 가 `[urdf_snapshot] ... status='mismatch'` 로 즉시 종료할 때
#     (스냅샷 파일과 사이드카 *.sha256 해시가 어긋난 상태)
#
# 무엇을 하는가:
#   1. dex_ros 의 regenerate_urdf.sh 를 올바른 REPO_ROOT 로 호출
#      (그 스크립트의 REPO_ROOT 자동 계산은 원본 PC 의 디렉토리 깊이를 가정한다 →
#       현재 배치에서는 어긋나므로 env 로 넘겨준다)
#   2. 생성물: dual_fr3_kistar{,_v2}.urdf, table.urdf, ttable.urdf + 각 *.sha256
#   3. place skill 이 hand_fk 로 파싱하는 사본을 정본과 동일하게 맞춘다
set -eo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../env" && pwd)/setup_env.sh" > /dev/null

SCRIPT="$RAS_DEX_ROS/isaac-ros/kistar_ws/src/franka_kistar_bringup/scripts/regenerate_urdf.sh"
[ -f "$SCRIPT" ] || { echo "regenerate_urdf.sh 없음: $SCRIPT"; exit 1; }

echo "=== [1/2] 스냅샷 재생성 ==="
REPO_ROOT="$RAS_DEX_ROS" bash "$SCRIPT"

echo
echo "=== [2/2] place 사본 동기화 ==="
SRC="$RAS_DEX_ROS/isaac-ros/kistar_ws/src/franka_kistar_description/urdf/generated/dual_fr3_kistar_v2.urdf"
DST="$RAS_SKILL_PLACE/dex_ros/isaac-ros/kistar_ws/src/franka_kistar_description/urdf/generated/dual_fr3_kistar_v2.urdf"
if [ -f "$DST" ]; then
  cp "$SRC" "$DST"
  echo "  $DST 갱신"
else
  echo "  WARN: place 사본이 없다 ($DST) — igr_service 가 뜨지 않는다"
fi

echo
echo "=== 검증 ==="
( cd "$(dirname "$SRC")" && sha256sum -c ./*.sha256 )
( cd "$RAS_DEX_ROS/isaac-ros/kistar_ws/src/franka_kistar_bringup/urdf/generated" && sha256sum -c ./*.sha256 )
