#!/usr/bin/env bash
# FoundationPose 준비 스크립트 — 한 번만 돌리면 된다.
#
# 이 순서는 이 PC(RTX 4080 SUPER = sm_89, KIST 망)에서 실제로 통과시킨 절차다.
# 두 가지 함정이 있어서 공식 문서대로 하면 막힌다:
#
#   (A) 공식 이미지의 nvcc 는 CUDA 11.3 이라 sm_89(Ada) 를 모른다.
#       → kaolin 과 nvdiffrast 컴파일이 `nvcc fatal: Unsupported gpu architecture
#         'compute_89'` 로 죽는다. sm_86 으로 강제하면 된다(같은 8.x 라 Ada 에서 실행됨).
#       → 게다가 nvdiffrast 는 ops.py 에서 TORCH_CUDA_ARCH_LIST 를 '' 로 덮어써
#         환경변수가 안 먹는다. 그래서 파일을 직접 패치한다.
#   (B) 이 망에서 drive.google.com 이 막혀 있다(HTTP 000, github/google 은 200).
#       → 공식 가중치 링크를 못 쓴다. HuggingFace 미러에서 받는다.
#
# 결과물: `foundationpose:local` 이미지 (공식 이미지 + 위 패치를 구운 것)
#
# 호스트의 파이썬 환경(torch 2.6+cu124 / SAM2)은 건드리지 않는다.
#
# 사용:  bash foundation_pose/setup.sh
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE=wenbowen123/foundationpose:latest
IMAGE=foundationpose:local
CPY=/opt/conda/envs/my/bin/python
NVDR_OPS=/opt/conda/envs/my/lib/python3.8/site-packages/nvdiffrast/torch/ops.py
FPROOT="$HERE/FoundationPose"
WEIGHTS="$FPROOT/weights"
ASSETS="$HERE/assets"
ARCH=8.6              # sm_86 바이너리는 sm_89(Ada) 에서 그대로 돈다
HF_REPO=gpue/foundationpose-weights

step() { echo ""; echo "── $* ──"; }
ok()   { echo "  ✓ $*"; }
bad()  { echo "  ✗ $*"; }

step "1) 도커 + nvidia 런타임"
command -v docker >/dev/null || { bad "docker 없음"; exit 1; }
if ! docker info 2>/dev/null | grep -q nvidia; then
  bad "docker 에 nvidia 런타임 없음. 설치:"
  echo "      sudo apt-get install -y nvidia-container-toolkit"
  echo "      sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker"
  exit 1
fi
ok "nvidia 런타임"

step "2) 베이스 이미지 (약 20GB)"
if docker image inspect "$BASE" >/dev/null 2>&1; then ok "이미 있음"
else docker pull "$BASE" || { bad "pull 실패"; exit 1; }; ok "완료"; fi

step "3) FoundationPose 저장소"
if [ -f "$FPROOT/run_demo.py" ]; then ok "이미 있음"
else
  # git clone 이 이 망에서 자주 멈춘다 → tarball 이 훨씬 빠르고 안정적
  mkdir -p "$FPROOT"
  curl -sL https://github.com/NVlabs/FoundationPose/archive/refs/heads/main.tar.gz \
    | tar xz -C "$FPROOT" --strip-components=1 || { bad "다운로드 실패"; exit 1; }
  ok "받음"
fi

step "4) 가중치 (HuggingFace 미러 — 구글 드라이브가 막혀 있음)"
if [ -f "$WEIGHTS/2023-10-28-18-33-37/model_best.pth" ] \
   && [ -f "$WEIGHTS/2024-01-11-20-02-45/model_best.pth" ]; then ok "이미 있음"
else
  python3 -c "import huggingface_hub" 2>/dev/null || python3 -m pip install --user -q huggingface_hub
  python3 - "$WEIGHTS" "$HF_REPO" <<'PY' || { bad "가중치 다운로드 실패"; exit 1; }
import sys
from huggingface_hub import snapshot_download
snapshot_download(repo_id=sys.argv[2], local_dir=sys.argv[1])
print("weights ok")
PY
  ok "refiner(68MB) + scorer(190MB)"
fi

step "5) 확장 빌드 + sm_86 패치 → $IMAGE"
if docker image inspect "$IMAGE" >/dev/null 2>&1; then
  ok "이미 있음 (다시 만들려면: docker rmi $IMAGE)"
else
  echo "  (a) mycpp + kaolin 빌드 (sm_$ARCH 강제, 수 분)"
  docker rm -f fp_setup >/dev/null 2>&1
  docker run -d --name fp_setup --gpus all --env NVIDIA_DISABLE_REQUIRE=1 \
    -e TORCH_CUDA_ARCH_LIST="$ARCH" -v /home:/home -w "$FPROOT" "$BASE" \
    bash -lc "source /opt/conda/etc/profile.d/conda.sh && conda activate my &&
              cd $FPROOT/mycpp && mkdir -p build && cd build &&
              cmake .. -DPYTHON_EXECUTABLE=\$(which python) && make -j\$(nproc) &&
              cd /kaolin && rm -rf build *egg* &&
              TORCH_CUDA_ARCH_LIST=$ARCH pip install -e . &&
              sed -i \"s|os.environ\\['TORCH_CUDA_ARCH_LIST'\\] = ''|os.environ['TORCH_CUDA_ARCH_LIST'] = '$ARCH'|\" $NVDR_OPS &&
              $CPY -c 'import nvdiffrast.torch as dr; dr.RasterizeCudaContext(); print(\"NVDIFFRAST OK\")'" \
    >/dev/null 2>&1
  RC=$(docker wait fp_setup)
  if [ "$RC" != 0 ] || ! docker logs fp_setup 2>&1 | grep -q "NVDIFFRAST OK"; then
    bad "빌드 실패 (rc=$RC). 로그:"; docker logs fp_setup 2>&1 | tail -25; exit 1
  fi
  echo "  (b) 이미지로 굽기"
  docker commit fp_setup "$IMAGE" >/dev/null && docker rm -f fp_setup >/dev/null
  ok "$IMAGE 생성"
fi

step "6) 검증"
docker run --rm --gpus all --env NVIDIA_DISABLE_REQUIRE=1 -v /home:/home -w "$FPROOT" \
  "$IMAGE" "$CPY" -c "
import sys; sys.path.insert(0,'.')
import torch, kaolin, mycpp
import nvdiffrast.torch as dr
dr.RasterizeCudaContext()
from estimater import FoundationPose
print('STACK OK', torch.cuda.get_device_name(0))
" 2>&1 | grep -E "STACK OK|Error|error" | tail -3

step "7) 과일 메시"
mkdir -p "$ASSETS"
if [ -f "$ASSETS/orange.obj" ]; then ok "이미 있음"
else python3 "$HERE/make_fruit_mesh.py" --diameter 0.070 -o "$ASSETS/orange.obj" >/dev/null && ok "생성"; fi

echo ""
echo "════════════════════════════════════════════════"
echo " 준비 끝. 실행:"
echo "   bash $HERE/run_foundation_pose.sh"
echo "════════════════════════════════════════════════"
