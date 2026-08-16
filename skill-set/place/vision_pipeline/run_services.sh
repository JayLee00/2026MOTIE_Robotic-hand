#!/usr/bin/env bash
# Launch the five place-skill model microservices (each in its own conda env). Ctrl+C stops all.
# AnyPlace + IGR + grid share `anyplace_cu128`; SAM3 -> `sam3`; Molmo -> `molmo`.
#
# 2-GPU split (this PC has 2x RTX 5090 / 32GB each — the stack sums to ~30GB):
#   GPU0 = Molmo alone (~17GB)
#   GPU1 = SAM(~10) + AnyPlace(~2) + IGR(~1)
#   grid = CPU
# molmo_service hardcodes device_map='cuda:0', but CUDA_VISIBLE_DEVICES remaps physical->logical,
# so it still lands on the requested card.
#
# ⚠ `conda activate` is NOT used. This script is normally spawned by the pipeline runner, which
#   first sources tools/env/setup_env.sh to get OUT of conda (ROS2 rclpy must run on the system
#   python). `conda activate` from that state is fragile — it needs a consistent CONDA_SHLVL /
#   CONDA_PREFIX pair and a conda shell function, and it crashed here with
#   "TypeError: expected str, bytes or os.PathLike object, not NoneType" (conda tried to
#   deactivate a prefix that had been unset), leaving `python: command not found`.
#   Calling each env's interpreter by absolute path is equivalent and has no shell state at all.
#   (Verified: none of the three envs has etc/conda/activate.d hooks, so nothing is skipped.)
set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
cd "$REPO"
PY=vision_pipeline/services
pids=()

# 가중치는 로컬 HF 캐시에 이미 있다 — 네트워크 조회 차단(오프라인 결정성).
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

# 환경별 인터프리터 존재 확인 — 없으면 조용히 "python: command not found" 로 흘러가지 않게
# 여기서 즉시, 무엇이 없는지 말하고 죽는다.
for e in molmo sam3 anyplace_cu128; do
  if [ ! -x "$CONDA_BASE/envs/$e/bin/python" ]; then
    echo "[run_services] FATAL: conda env '$e' 의 python 이 없다: $CONDA_BASE/envs/$e/bin/python" >&2
    echo "[run_services]   재생성 방법: tools/env/conda/README.md" >&2
    exit 1
  fi
done

# run <script> <conda-env> <port> [CUDA_VISIBLE_DEVICES]
run() {
  echo "[run_services] $1 :$3 (env=$2 gpu=${4:-cpu})"
  CUDA_VISIBLE_DEVICES="${4:-}" "$CONDA_BASE/envs/$2/bin/python" "$PY/$1" --port "$3" &
  pids+=($!)
}

# NOTE: the front camera streams from the Control PC over ROS2 domain 9 (the ROS backend
# subscribes directly) — no local capture_service. The camera PC must run realsense with
# align_depth.enable:=true.
run molmo_service.py      molmo          8810 0   # bf16 ~17GB — alone on GPU0
run sam_service.py        sam3           8811 1
run anyplace_service.py   anyplace_cu128 8801 1
run igr_service.py        anyplace_cu128 8816 1   # Act-VH IGR completion + PaXini hand cloud (FK)
run grid_service.py       anyplace_cu128 8815     # host image grid (CPU; auto-opens browser)

trap 'echo; echo "[run_services] stopping"; kill ${pids[*]} 2>/dev/null' INT TERM
echo "[run_services] up: molmo:8810 sam:8811 anyplace:8801 igr:8816 grid:8815  (Ctrl+C)"
echo "[run_services] VRAM: GPU0 Molmo ~17GB / GPU1 SAM+AnyPlace+IGR ~13GB"
wait
