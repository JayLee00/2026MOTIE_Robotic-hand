#!/usr/bin/env bash
# 과일 인식(SAM2+FoundationPose) → 통합 시각화 를 순서대로 띄운다.
#
#   1) foundation_pose/run_foundation_pose.sh --no-overlay
#        SAM2 로 과일을 잡고 FoundationPose 가 6DoF 를 추정
#        → /fruit/pose (위치+오리엔테이션), /fruit/size
#        자체 오버레이 창은 끈다 (아래 2)번이 대체하므로 창 중복 방지)
#   2) Visualization/live_viz.py
#        과일 바운딩박스 + XYZ축 + Paxini 촉각 3D(4개) + 손가락 4칸 FT(Tx,Ty,Fz×1000)
#
# 사용:
#   bash scripts/run_fruit_viz_all.sh                 # 기본 lemon
#   bash scripts/run_fruit_viz_all.sh --fruit peach   # 과일 지정
#   bash scripts/run_fruit_viz_all.sh --check         # 전제조건 점검만
#   bash scripts/run_fruit_viz_all.sh --keep-overlay  # 기존 fruit_overlay 창도 함께
#   그 외 인자는 run_foundation_pose.sh 로 전달 (--seg-hz, --no-click, --mesh …)
#
# 종료: Ctrl+C — 이 스크립트가 띄운 것 전부 정리 (docker fp_server 포함)
#
# ⚠️ sudo 로 실행하지 말 것 (root 는 X 접근이 막혀 창이 안 뜬다)
set +u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="$(dirname "$HERE")"

FP_ARGS=()
VIZ_ARGS=()
KEEP_OVERLAY=""
CHECK=""
FRUIT_GIVEN=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --keep-overlay) KEEP_OVERLAY=1 ;;
        --check)        CHECK=1; FP_ARGS+=("--check") ;;
        --fruit)        FRUIT_GIVEN=1; FP_ARGS+=("$1" "$2"); shift ;;
        --no-tactile|--vmax|--color-topic|--info-topic|--pose-topic|--size-topic|--tactile-topic|--ft-topic)
                        # 시각화 쪽 인자
                        if [[ "$1" == "--no-tactile" ]]; then VIZ_ARGS+=("$1")
                        else VIZ_ARGS+=("$1" "$2"); shift; fi ;;
        *)              FP_ARGS+=("$1") ;;
    esac
    shift
done
# --fruit 를 값 없이 주면 run_foundation_pose.sh 가 조용히 lemon 으로 떨어진다.
# 여기서는 명시적으로 lemon 을 넣어 의도가 드러나게 한다.
[[ -z "$FRUIT_GIVEN" ]] && FP_ARGS+=("--fruit" "lemon")
[[ -z "$KEEP_OVERLAY" ]] && FP_ARGS+=("--no-overlay")

if [[ $EUID -eq 0 ]]; then
    echo "경고: root 로 실행 중 — 창이 안 뜰 수 있습니다. 일반 사용자로 실행하세요."
fi

FP_PID=""
cleanup() {
    echo ""
    echo "정리 중..."
    [[ -n "$FP_PID" ]] && kill -INT "$FP_PID" 2>/dev/null
    sleep 2
    [[ -n "$FP_PID" ]] && kill -9 "$FP_PID" 2>/dev/null
    pkill -f "[l]ive_viz.py" 2>/dev/null
    echo "종료"
}
trap cleanup EXIT INT TERM

echo "══ 1) 과일 인식 (SAM2 + FoundationPose) ══"
echo "   인자: ${FP_ARGS[*]}"
bash "$PROJ/foundation_pose/run_foundation_pose.sh" "${FP_ARGS[@]}" &
FP_PID=$!

if [[ -n "$CHECK" ]]; then
    wait "$FP_PID"
    echo "점검만 수행 — 종료"
    exit 0
fi

# /fruit/pose 가 실제로 나올 때까지 기다린다 (SAM2+FP 로드에 시간이 걸린다)
echo ""
echo "══ 2) /fruit/pose 대기 (최대 180초) ══"
# 2026-08-16 이관 (prime-ws): 이 PC 표준 환경 사용. 패키지의 fastdds_lan_only.xml 은
# 원본 PC(192.168.0.1) 전용 — 여기서 쓰면 DDS 가 전멸한다.
ROBOT_ROOT="$(cd "$PROJ/../../.." && pwd)"    # …/RobotAgentSystem
source "$ROBOT_ROOT/tools/env/setup_env.sh"
export DISPLAY="${DISPLAY:-:1}"

ok=""
for i in $(seq 1 36); do
    if ! kill -0 "$FP_PID" 2>/dev/null; then
        echo "✗ 과일 인식 프로세스가 종료됐습니다. 로그: /tmp/fp_*.log"
        exit 1
    fi
    if timeout 5 ros2 topic hz /fruit/pose 2>/dev/null | grep -q "average rate"; then
        ok=1; break
    fi
    printf "   대기 %3ds...\r" $((i * 5))
done
echo ""
if [[ -z "$ok" ]]; then
    echo "✗ /fruit/pose 가 안 나옵니다. 그래도 시각화는 띄웁니다 (카메라만이라도 확인)."
    echo "  확인: tail -30 /tmp/fp_bbox.log  /tmp/fp_node.log"
else
    echo "✓ /fruit/pose 수신 확인"
fi

echo ""
echo "══ 3) 통합 시각화 ══"
echo "   과일 바운딩박스 + XYZ축 | Paxini 촉각 3D x4 | 손가락 4칸 FT(Tx,Ty,Fz×1000)"
echo "   키: q/ESC 종료, s 스냅샷"
exec bash "$HERE/run_live_viz.sh" "${VIZ_ARGS[@]}"
