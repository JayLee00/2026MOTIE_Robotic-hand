#!/usr/bin/env bash
# measure_update_rate.sh — update rate 계측 원-커맨드 러너 (결과 자동 저장).
#
# 무엇을 하나:
#   1) 와이어 rate 모니터(`ros2 topic hz`)를 토픽별로 백그라운드에 띄우고, 매 줄에
#      epoch 타임스탬프를 붙여 파일로 저장한다 (→ 나중에 시계열로 분석 가능).
#   2) 토픽 QoS(`ros2 topic info --verbose`)를 함께 저장한다 (C1 진단용).
#   3) 계측판 배포(deploy_ros2_exp)를 foreground 로 실행하고 stdout 을 tee 로 저장한다
#      (→ [measure] 스퀴즈 계측 블록이 그대로 남는다).
#   4) 종료 시 tools/rate_summary.py 로 summary.md(마크다운 표)를 만든다.
#
# 사용:
#   ./tools/measure_update_rate.sh --label kiwi_baseline        # 계측판 배포 + rate 모니터
#   ./tools/measure_update_rate.sh --label kiwi_curl --forcecurl # 힘-도달 curl 판
#   ./tools/measure_update_rate.sh --monitor --duration 30      # rate 모니터만 (배포는 다른 터미널)
#
# 결과: docs/rate_log/<타임스탬프>_<label>/
#   hz__hand_right_q_target.log   ... 토픽별 rate 시계열 (epoch + ros2 topic hz 원문)
#   info__hand_right_q_target.txt ... 토픽 QoS
#   exp_stdout.log                ... 배포/계측 전체 로그
#   summary.md                    ... 붙여넣기용 마크다운 요약
#
# 감시 토픽 변경:  RATE_TOPICS="/hand/right/q_target /paxini/right/ft" ./tools/measure_update_rate.sh
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

MODE="exp"          # exp | forcecurl | monitor
DUR=30              # --monitor 모드 지속시간(s)
LABEL=""

while [ $# -gt 0 ]; do
    case "$1" in
        --monitor)    MODE="monitor";   shift ;;
        --forcecurl)  MODE="forcecurl"; shift ;;
        --duration)   DUR="$2";         shift 2 ;;
        --label)      LABEL="$2";       shift 2 ;;
        -h|--help)    sed -n '2,30p' "$0"; exit 0 ;;
        *) echo "[measure_rate] 알 수 없는 인자: $1 (--help 참고)" >&2; exit 2 ;;
    esac
done

# ROS2 Humble + 시스템 python3 (conda 미사용) — env.sh 규약 그대로.
# ROS setup.bash 는 미설정 변수를 참조하므로(AMENT_TRACE_SETUP_FILES) source 동안만 -u 해제.
set +u
source "$HERE/env.sh"
set -u

# shellcheck disable=SC2206  # 공백 분리(토픽 목록)가 의도된 동작
TOPICS=(${RATE_TOPICS:-/hand/right/q_target /paxini/right/ft /hand/right/joint_states})

OUT="$HERE/docs/rate_log/$(date +%Y%m%d_%H%M%S)${LABEL:+_$LABEL}"
mkdir -p "$OUT"
echo "[measure_rate] 저장 위치: ${OUT#$HERE/}"

sanitize() { echo "$1" | tr '/' '_'; }

# ── 1) rate 모니터 기동 ───────────────────────────────────────────────
# setsid 로 프로세스 그룹을 분리해, 종료 시 파이프라인(ros2 topic hz)까지 확실히 정리한다.
MON_PIDS=()
for t in "${TOPICS[@]}"; do
    log="$OUT/hz$(sanitize "$t").log"
    # 파일명은 '/'→'_' 로 뭉개지므로(q_target 과 구분 불가) 원본 토픽명을 헤더로 남긴다.
    echo "# topic: $t" > "$log"
    setsid bash -c "
        stdbuf -oL ros2 topic hz '$t' --window 100 2>&1 |
        while IFS= read -r line; do printf '%s %s\n' \"\$(date +%s.%3N)\" \"\$line\"; done
    " >> "$log" &
    MON_PIDS+=("$!")
    echo "[measure_rate] hz 모니터 시작: $t"
done

# ── 1b) 센서 실측 update rate 모니터 ─────────────────────────────────
# `ros2 topic hz` 는 '발행률'(통신)이라 퍼블리셔가 같은 값을 재발행하면 센서가 멈춰도 정상으로
# 보인다. 그래서 '값이 실제로 바뀌는 빈도'를 따로 잰다. 종료 시 TERM 을 받아 리포트를 남긴다.
# monitor 모드는 정해진 시간만, 배포 모드는 배포가 끝날 때까지(TERM 으로 종료).
SENSOR_DUR=3600
[ "$MODE" = "monitor" ] && SENSOR_DUR="$DUR"
setsid python3 -u "$HERE/tools/sensor_update_rate.py" \
    --duration "$SENSOR_DUR" --out "$OUT" > "$OUT/sensor_change.log" 2>&1 &
SENSOR_PID=$!
echo "[measure_rate] 센서 값변화 모니터 시작 (sensor_change.log)"

cleanup() {
    # 센서 모니터는 리포트를 써야 하므로 TERM 후 잠시 기다린다.
    if [ -n "$SENSOR_PID" ]; then
        kill -TERM -- "-$SENSOR_PID" 2>/dev/null || kill -TERM "$SENSOR_PID" 2>/dev/null
        for _ in 1 2 3 4 5 6 7 8 9 10; do
            kill -0 "$SENSOR_PID" 2>/dev/null || break
            sleep 0.5
        done
        SENSOR_PID=""
    fi
    for pid in "${MON_PIDS[@]}"; do
        kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null
    done
    wait 2>/dev/null
}
trap cleanup EXIT INT TERM

# ── 2) QoS 스냅샷 (디스커버리 여유 2s 후) ────────────────────────────
sleep 2
for t in "${TOPICS[@]}"; do
    ros2 topic info "$t" --verbose > "$OUT/info$(sanitize "$t").txt" 2>&1
done

# 재현용 메타데이터 (나중 분석 때 조건 대조).
{
    echo "mode=$MODE"
    echo "label=${LABEL:-none}"
    echo "topics=${TOPICS[*]}"
    echo "date=$(date -Is)"
    echo "git_commit=$(git rev-parse --short HEAD 2>/dev/null)"
    echo "git_dirty=$(git status --porcelain 2>/dev/null | wc -l) files"
    echo "python=$(which python3) ($(python3 --version 2>&1))"
    echo "kernel=$(uname -r)"
    echo "ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-} RMW=${RMW_IMPLEMENTATION:-}"
} > "$OUT/meta.txt"

# ── 3) 본체 실행 ─────────────────────────────────────────────────────
case "$MODE" in
    monitor)
        echo "[measure_rate] 모니터 전용 — ${DUR}s 수집. 다른 터미널에서 배포를 실행하세요."
        sleep "$DUR"
        ;;
    exp|forcecurl)
        script="stiffness_deploy_ros2/launch/deploy_ros2_exp.py"
        [ "$MODE" = "forcecurl" ] && script="stiffness_deploy_ros2/launch/deploy_ros2_exp_forcecurl.py"
        echo "[measure_rate] 실행: $script  (과일 번호를 입력하세요)"
        # -u: 파이프로 묶여도 프롬프트/로그가 즉시 나오도록 (input() 은 그대로 대화형).
        python3 -u "$script" 2>&1 | tee "$OUT/exp_stdout.log"
        ;;
esac

# ── 4) 요약 생성 ─────────────────────────────────────────────────────
cleanup
trap - EXIT INT TERM
python3 "$HERE/tools/rate_summary.py" "$OUT"

echo "[measure_rate] 완료 → ${OUT#$HERE/}/summary.md"
echo "[measure_rate] 요약을 docs/UPDATE_RATE_CHECK.md 의 결과 기록 표에 붙여넣으세요."
