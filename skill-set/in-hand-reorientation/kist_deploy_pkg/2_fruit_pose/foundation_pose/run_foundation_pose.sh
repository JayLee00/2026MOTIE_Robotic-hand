#!/usr/bin/env bash
# FoundationPose 과일 6DoF 원커맨드 런처  (run_fruit_viz.sh 의 자세추정 대체판)
#
#   [제어 PC realsense]  /front_cam/front/color/image_raw/compressed
#         ▼ republish (compressed→raw)
#   /front_cam/front/color/image_fast
#         ▼
#   fp_ros_node.py (호스트)  ── 첫 프레임만 SAM2 마스크 ──┐
#         │                                              │ TCP :5577
#         │                                              ▼
#         │                          [docker] fp_server.py = FoundationPose
#         ▼                                              │
#   /fruit/pose + /fruit/size  ◄─────────────────────────┘
#         ▼
#   fruit_overlay.py (원본 그대로)  →  화면 오버레이
#
# 기존 run_fruit_viz.sh 와의 차이: 자세를 SAM2 OBB 의 PCA 가 아니라
# FoundationPose 의 CAD 정합으로 뽑는다. 오버레이·토픽은 동일하다.
#
# 사용:  bash foundation_pose/run_foundation_pose.sh
#        bash foundation_pose/run_foundation_pose.sh --check     # 전제조건만
#        bash foundation_pose/run_foundation_pose.sh --compare   # /fruit_fp/* 로 발행(기존과 동시 비교)
#
# 종료: Ctrl+C
set +u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="$(cd "$HERE/.." && pwd)"
NS=/front_cam/front
COLOR_C="$NS/color/image_raw/compressed"
COLOR_FAST="$NS/color/image_fast"
INFO="$NS/color/camera_info"
DEPTH="$NS/aligned_depth_to_color/image_raw"

# 2026-08-16 이관 (prime-ws, RTX 5090): docker(sm_86) 대신 conda env `foundationpose`
# (build_sm120.sh 가 만든 sm_120 빌드)로 fp_server 를 돌린다.
CPY="$HOME/miniconda3/envs/foundationpose/bin/python"
FP_GPU="${FP_GPU:-0}"          # fp_server GPU — VTDP 정책(cuda:1)과 분리
CONTAINER=fp_server            # (docker 아님 — pkill 라벨로만 사용)
PORT=5577
MESH="$HERE/assets/orange.obj"
FPROOT="$HERE/FoundationPose"
PUB_NS=/fruit          # 기본은 드롭인(기존 오버레이가 그대로 받음)

# 이 PC 의 표준 환경 — ROS2 + 워크스페이스 + DOMAIN_ID=9 + DDS 프로파일
# (~/.ros/fastdds_ros2_link.xml). 패키지의 config/fastdds_lan_only.xml 은 원본 PC
# (192.168.0.1) 전용이라 여기서 쓰면 DDS 가 전멸한다 — 절대 교체하지 말 것.
ROBOT_ROOT="$(cd "$PROJ/../../../.." && pwd)"    # …/RobotAgentSystem
source "$ROBOT_ROOT/tools/env/setup_env.sh"
export DISPLAY="${DISPLAY:-:1}"

while [ $# -gt 0 ]; do
  case "$1" in
    --compare) PUB_NS=/fruit_fp ;;
    --mesh)    MESH="$2"; MESH_EXPLICIT=1; shift ;;
    --fruit)   FRUIT="$2"; shift ;;
    --seg-hz)  SEGHZ="$2"; shift ;;
    --reseg-every) RESEG="$2"; shift ;;   # 세그 N프레임마다 자세투영 점으로 재시드
    --no-click) NOCLICK="--no-click" ;;
    --no-overlay) NOOVERLAY=1 ;;   # 자체 fruit_overlay 창을 띄우지 않음
                                   # (Visualization/live_viz.py 로 대체할 때 창 중복 방지)
    --check)   CHECK=1 ;;
  esac
  shift
done
# --fruit 를 주면 카탈로그에서 그 과일의 CAD 를 쓴다 (--mesh 보다 우선순위 낮음)
FRUIT="${FRUIT:-lemon}"
if [ -z "${MESH_EXPLICIT:-}" ] && [ -f "$HERE/fruits.yaml" ]; then
  m=$(/usr/bin/python3 - "$HERE" "$FRUIT" <<'PY' 2>/dev/null
import os, sys, yaml
here, key = sys.argv[1], sys.argv[2]
for c in yaml.safe_load(open(os.path.join(here, "fruits.yaml")))["fruits"]:
    if str(c["id"]) == key or c["name"].lower() == key.lower():
        p = c["mesh"] if os.path.isabs(c["mesh"]) else os.path.join(here, c["mesh"])
        print(p if os.path.isfile(p) else "")
        break
PY
)
  [ -n "$m" ] && MESH="$m"
fi
# 그래도 없으면 assets/ 의 아무 obj
[ -f "$MESH" ] || { alt=$(ls "$HERE"/assets/*.obj 2>/dev/null | head -1); [ -n "$alt" ] && MESH="$alt"; }

hz() { timeout 6 ros2 topic hz "$1" 2>/dev/null | grep -oP 'average rate: [\d.]+' | head -1; }

# ── 이전 실행 잔재 정리 ────────────────────────────────────────────────────
# 이게 없으면 재시작할 때마다 republish 가 하나씩 새고, 같은 컬러 프레임이 N중으로
# 발행돼 시간동기화가 무너진다. 실측으로 15개까지 쌓여 자세가 30Hz→2.5Hz 로
# 주저앉은 적이 있다. 남은 노드/컨테이너도 GPU 와 토픽을 물고 있으므로 같이 치운다.
# pgrep -fc 는 못 찾아도 "0" 을 찍고 exit 1 을 낸다 → `|| echo 0` 을 붙이면 두 줄이
# 나와 산술식이 깨진다. 첫 줄만 취한다.
stale() { pgrep -fc "$1" 2>/dev/null | head -1; }
N_STALE=$(( $(stale "image_transport/republish") + $(stale "fp_ros_node.py") \
          + $(stale "fruit_label_node.py") + $(stale "foundation_pose/fp_server.py") ))
if [ "$N_STALE" -gt 0 ]; then
  echo "── 이전 실행 잔재 정리 ──"
  pkill -f "image_transport/republish.*color/image_fast" 2>/dev/null
  pkill -f "foundation_pose/fp_ros_node.py"   2>/dev/null
  pkill -f "foundation_pose/fruit_label_node.py" 2>/dev/null
  pkill -f "record/fruit_overlay.py"          2>/dev/null
  pkill -f "foundation_pose/fp_server.py"     2>/dev/null
  sleep 2
  echo "  정리 완료 (republish $(stale 'image_transport/republish')개 남음)"
fi

echo "── 0) 준비물 점검 ──"
MISSING=0
if ! "$CPY" -c "import torch, pytorch3d, nvdiffrast" >/dev/null 2>&1; then
  echo "  ✗ conda env 'foundationpose' 불완전 — build_sm120.sh 를 돌리세요"; MISSING=1
else echo "  ✓ conda env foundationpose (sm_120 빌드)"; fi
if [ ! -d "$FPROOT" ]; then echo "  ✗ FoundationPose 저장소 없음: $FPROOT"; MISSING=1
else echo "  ✓ FoundationPose 저장소"; fi
if [ ! -d "$FPROOT/weights" ] || [ -z "$(ls -A "$FPROOT/weights" 2>/dev/null)" ]; then
  echo "  ✗ 가중치 없음: $FPROOT/weights"; MISSING=1
else echo "  ✓ 가중치"; fi
if [ ! -f "$MESH" ]; then echo "  ✗ 메시 없음: $MESH"; MISSING=1
else echo "  ✓ 메시 ($(basename "$MESH"))"; fi
if [ "$MISSING" = 1 ]; then
  echo ""; echo "→ 먼저 준비 스크립트를 돌리세요:  bash $HERE/setup.sh"; exit 1
fi

echo "── 1) 카메라 점검 ──"
C=$(hz "$COLOR_C"); I=$(hz "$INFO"); D=$(hz "$DEPTH")
echo "  color(compressed): ${C:-✗ 없음}"
echo "  camera_info      : ${I:-✗ 없음}"
echo "  depth(aligned)   : ${D:-✗ 없음}"
if [ -z "${C:-}" ] || [ -z "${I:-}" ] || [ -z "${D:-}" ]; then
  echo ""
  echo "✗ 카메라(특히 정렬깊이)가 필요합니다. 제어 PC 에서 realsense 를 띄우세요 (rs)."
  echo "  ros2 param set /front_cam/front align_depth.enable true"
  exit 1
fi
[ "${CHECK:-0}" = 1 ] && { echo "점검만 수행 — 종료"; exit 0; }

PIDS=()
cleanup() {
  echo ""; echo "정리 중..."
  for p in "${PIDS[@]:-}"; do
    kill -- "-$p" 2>/dev/null || kill "$p" 2>/dev/null   # 프로세스 그룹째
  done
  # `ros2 run` 은 래퍼라 그것만 죽이면 실제 republish 자식이 살아남는다. 그렇게 새어나온
  # 게 쌓이면 같은 컬러 프레임이 N중으로 발행돼 시간동기화가 무너지고 자세 주기가
  # 30Hz → 12Hz 로 주저앉는다(실측: 15개 누적, image_fast 51.8Hz).
  pkill -f "image_transport/republish.*$COLOR_FAST" 2>/dev/null
  pkill -f "foundation_pose/fp_server.py" 2>/dev/null
  wait 2>/dev/null; echo "종료"
}
trap cleanup EXIT INT TERM

echo ""
echo "── 2) 컬러 소스: raw 직접 구독 (republish 생략) ──"
# 2026-08-16 이관 수정: 이 PC 에서는 Control PC 가 raw image_raw 를 이미 30Hz 발행한다
# (러너 preflight 가 매번 확인, grasp/place 도 동일 토픽 사용). 게다가 이 PC 에는
# ros-humble-compressed-image-transport 플러그인이 없어 republish 가 즉사한다
# (TransportLoadException — 실측). 구 PC(raw 미발행)용이던 republish 우회를 제거.
COLOR_RAW="$NS/color/image_raw"
R=$(hz "$COLOR_RAW"); echo "  $COLOR_RAW : ${R:-✗ 없음 (Control PC realsense 확인)}"
# ── 구 republish 경로 (봉인 보존 — compressed 플러그인 설치 시에만 유효) ──
# pkill -f "image_transport/republish.*$COLOR_FAST" 2>/dev/null; sleep 1
# setsid ros2 run image_transport republish compressed raw \
#   --ros-args -r "in/compressed:=$COLOR_C" -r "out:=$COLOR_FAST" >/tmp/fp_republish.log 2>&1 &
# PIDS+=($!); sleep 3

echo "── 3) FoundationPose 서버 (conda env foundationpose, GPU$FP_GPU) ──"
pkill -f "foundation_pose/fp_server.py" 2>/dev/null; sleep 1
setsid env CUDA_VISIBLE_DEVICES="$FP_GPU" PYTHONUNBUFFERED=1 \
  "$CPY" "$HERE/fp_server.py" --mesh "$MESH" --port "$PORT" --fp-root "$FPROOT" \
  >/tmp/fp_server.log 2>&1 &
SERVER_PID=$!; PIDS+=($SERVER_PID)
echo "  서버 기동 — 모델 로드 대기(최대 120s, 로그 /tmp/fp_server.log)"
for i in $(seq 1 120); do
  grep -q "listening on" /tmp/fp_server.log 2>/dev/null && break
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "  ✗ fp_server 가 죽었습니다. 로그:"; tail -25 /tmp/fp_server.log; exit 1
  fi
  sleep 1
done
if grep -q "listening on" /tmp/fp_server.log 2>/dev/null; then
  echo "  ✓ fp_server 준비 완료 (:$PORT)"
else
  echo "  ✗ 시간 초과. 로그:"; tail -25 /tmp/fp_server.log; exit 1
fi

echo "── 4) ROS2 브리지 (호스트, SAM2 초기 마스크) ──"
# --reseg-every 기본 15: seg 5Hz × 15프레임 = 3초마다 "손 안(자세투영 점)" 재시드
# (fp_ros_node 자체 기본 30 = 6초 → 데모 요구인 3초 주기로 당김. 클릭 시드는 항상 켬)
/usr/bin/python3 "$HERE/fp_ros_node.py" \
  --server "127.0.0.1:$PORT" --ns "$PUB_NS" --seg-hz "${SEGHZ:-5}" \
  --reseg-every "${RESEG:-15}" ${NOCLICK:-} \
  --color-topic "$COLOR_RAW" --depth-topic "$DEPTH" --info-topic "$INFO" \
  >/tmp/fp_node.log 2>&1 &
PIDS+=($!)
echo "  기동 중... (SAM2 로드 ~10초, 로그 /tmp/fp_node.log)"
sleep 15
grep -E "SAM2 준비|SAM2 마스크|초기 등록|K 수신|접속|실패" /tmp/fp_node.log | tail -5 | sed 's/^/    /'

echo "── 5) 과일 라벨 노드 (/fruit/type) ──"
/usr/bin/python3 "$HERE/fruit_label_node.py" --fruit "$FRUIT" >/tmp/fp_label.log 2>&1 &
PIDS+=($!); sleep 3
grep -E "과일 =|카탈로그|CAD 교체|모르는 과일" /tmp/fp_label.log | tail -3 | sed 's/^/    /'

if [[ -n "${NOOVERLAY:-}" ]]; then
  echo "── 6) fruit_overlay 생략 (--no-overlay) ──"
else
  echo "── 6) fruit_overlay (원본 그대로) ──"
  /usr/bin/python3 "$PROJ/record/fruit_overlay.py" \
    --color-topic "$COLOR_C" --info-topic "$INFO" >/tmp/fp_overlay.log 2>&1 &
  PIDS+=($!); sleep 3
fi

echo ""
echo "── 상태 ──"
echo "  $PUB_NS/pose : $(hz "$PUB_NS/pose" || echo '✗ 아직 (창에서 과일을 클릭하세요)')"
echo "  /fruit/type  : $(timeout 5 ros2 topic echo /fruit/type --once 2>/dev/null | grep -oP 'data: \K\d+' || echo '✗')"
echo ""
echo "창: 'Fruit 6DoF overlay' (q=종료, s=스냅샷)"
echo "로그: /tmp/fp_node.log (브리지), docker logs $CONTAINER (추정), /tmp/fp_overlay.log"
[ "$PUB_NS" = "/fruit_fp" ] && echo "※ --compare 모드: 기존 오버레이는 /fruit/* 를 보므로 이 자세는 안 보입니다."
echo "Ctrl+C 로 전체 종료"
wait
