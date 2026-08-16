#!/usr/bin/env bash
# FoundationPose sm_120 (RTX 5090 / Blackwell) 재빌드 — 이 PC(prime-ws) 전용 원커맨드.
#
# 배포 README §4-C 의 절차를 그대로 스크립트화했다. docker 는 쓰지 않는다
# (호스트에 /usr/local/cuda-12.8 존재 — 기존 20GB 이미지는 torch2.0/sm_86 이라 폐기 대상).
# 재실행 안전: conda env 가 있으면 생성을 건너뛰고 pip 단계는 멱등이다.
set -euo pipefail

HERE="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FP="$HERE/FoundationPose"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda env list | grep -qE '^foundationpose\s' || conda env create -f "$FP/environment.yml"
conda activate foundationpose

export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="12.0+PTX"
nvcc --version | tail -1

python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision
python -m pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git"
python -m pip install --no-build-isolation "git+https://github.com/NVlabs/nvdiffrast.git"

# nvdiffrast 는 런타임 JIT 때 TORCH_CUDA_ARCH_LIST 를 '' 로 덮어써 sm 자동탐지로 돌아간다.
# 5090 에서 확실히 12.0 으로 고정 (배포 README §4-C 지시사항).
OPS="$(python -c "import nvdiffrast, os; print(os.path.join(os.path.dirname(nvdiffrast.__file__), 'torch', 'ops.py'))")"
sed -i "s/os.environ\[.TORCH_CUDA_ARCH_LIST.\] = ''/os.environ['TORCH_CUDA_ARCH_LIST'] = '12.0'/" "$OPS"
grep -n "TORCH_CUDA_ARCH_LIST" "$OPS" | head -3

python -m pip install -r "$FP/requirements.txt"

# 구 PC(py3.8/sm_86) 산출물 제거 후 mycpp 재빌드 — build_all_conda.sh 가 rm -rf build 포함
bash "$FP/build_all_conda.sh"

python - <<'EOF'
import torch
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(), "|", torch.cuda.get_device_name(0))
import pytorch3d, nvdiffrast
print("pytorch3d", pytorch3d.__version__, "/ nvdiffrast import OK")
EOF
ls "$FP/mycpp/build/"*.so 2>/dev/null && echo "mycpp .so OK"
echo "BUILD_SM120_DONE"
