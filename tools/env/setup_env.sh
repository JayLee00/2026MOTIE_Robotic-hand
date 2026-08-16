#!/usr/bin/env bash
# ROS2 실행 환경 준비 — 구 산업부 PC `~/.bashrc` 의 `rs()` 함수를 대체한다.
#
#   source tools/env/setup_env.sh
#
# ⚠ conda 를 반드시 벗어난다: conda(conda-forge) 빌드 python 으로 시스템 ROS2 Humble 의
#   rclpy 를 쓰면 import 는 되지만 libstdc++ 등 공유 라이브러리가 시스템 RMW/DDS 확장
#   모듈과 ABI 충돌해 종료 시 core dump 가 난다. ROS2 토픽 통신은 항상 /usr/bin/python3.
#
# ⚠ ROS_DOMAIN_ID=9 는 **라이브 로봇 버스**다. 실기와 무관한 테스트는 87 등 격리 도메인에서.

# ── conda 완전 이탈 ──────────────────────────────────────────────────────────
if command -v conda >/dev/null 2>&1 && [ -n "${CONDA_DEFAULT_ENV:-}" ]; then
    conda deactivate 2>/dev/null || true
fi
# PATH 에서 conda 계열 디렉토리를 전부 걷어낸다 (miniconda3/anaconda3/condabin/opt/conda ...)
PATH="$(echo "$PATH" | tr ':' '\n' | grep -v -i 'conda' | paste -sd: -)"
export PATH
unset PYTHONPATH PYTHONHOME 2>/dev/null || true

# conda 상태 변수는 **일관되게** 비운다.
# ⚠ CONDA_PREFIX 만 지우고 CONDA_SHLVL 을 1로 남기면 conda 입장에서 모순 상태가 된다
#   (= "env 가 활성인데 그 경로를 모른다"). 그 상태에서 자식 프로세스가 conda activate 를
#   하면 conda 가 이전 prefix 를 deactivate 하려다 None 을 join 해서
#   "TypeError: expected str, bytes or os.PathLike object, not NoneType" 로 죽는다.
#   실제로 place 모델 서비스 기동이 이 이유로 실패했다. SHLVL=0 = "활성 env 없음".
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER 2>/dev/null || true
unset "${!CONDA_PREFIX_@}" 2>/dev/null || true      # CONDA_PREFIX_1, _2, ... (중첩 활성화 잔재)
export CONDA_SHLVL=0
hash -r

# ── 경로 상수 ────────────────────────────────────────────────────────────────
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/paths.sh"

# ── ROS2 + 워크스페이스 오버레이 ─────────────────────────────────────────────
source /opt/ros/humble/setup.bash
[ -f "$RAS_FR_WS/install/setup.bash" ]     && source "$RAS_FR_WS/install/setup.bash"     || echo "[setup_env] WARN: fr_ws 미빌드 ($RAS_FR_WS/install) — tools/ros2/build.sh"
[ -f "$RAS_KISTAR_WS/install/setup.bash" ] && source "$RAS_KISTAR_WS/install/setup.bash" || echo "[setup_env] WARN: kistar_ws 미빌드 ($RAS_KISTAR_WS/install) — tools/ros2/build.sh"

# ── DDS / 도메인 ─────────────────────────────────────────────────────────────
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-9}"     # 9 = 라이브 로봇 버스 (제어 PC와 공유)
export ROS_LOCALHOST_ONLY=0

# ── 모델 가중치: 로컬 HF 캐시만 사용 (네트워크 조회 차단) ────────────────────
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

echo "[setup_env] ROS2 humble / DOMAIN_ID=$ROS_DOMAIN_ID / RMW=$RMW_IMPLEMENTATION"
echo "[setup_env] python: $(command -v python3) ($(python3 --version 2>&1))"
echo "[setup_env] root:   $RAS_ROOT"
