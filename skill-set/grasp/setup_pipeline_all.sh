#!/usr/bin/env bash
# Set up the 'grasp_fruit' conda environment.
#
# Usage:
#   bash setup_pipeline_all.sh              # create env + install everything
#   bash setup_pipeline_all.sh --skip-conda # skip env creation (env already exists)
#
# Requirements:
#   - miniforge3 at /home/kist/miniforge3
#   - GPU: RTX 4090 (CUDA 12.8)

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
CONDA_BASE="${CONDA_BASE:-/home/user/miniconda3}"
ENV_NAME="grasp_fruit"
PYTHON="$CONDA_BASE/envs/$ENV_NAME/bin/python"
PIP="$CONDA_BASE/envs/$ENV_NAME/bin/pip"

# ---------------------------------------------------------------------------
# Step 0: Create conda env
# ---------------------------------------------------------------------------
SKIP_CONDA=false
for arg in "$@"; do [[ "$arg" == "--skip-conda" ]] && SKIP_CONDA=true; done

if [[ "$SKIP_CONDA" == false ]]; then
    echo "========================================"
    echo "  Creating conda env: $ENV_NAME"
    echo "========================================"
    "$CONDA_BASE/bin/conda" env create -f "$REPO_DIR/environment_pipeline_all.yml"
else
    echo "[skip] conda env creation"
fi

# ---------------------------------------------------------------------------
# Step 1: PyTorch (CUDA 12.8)
# ---------------------------------------------------------------------------
echo ""
echo "========================================"
echo "  Installing PyTorch (CUDA 12.8)"
echo "========================================"
$PIP install \
    torch==2.10.0+cu128 \
    torchvision==0.25.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128

# ---------------------------------------------------------------------------
# Step 2: Remaining pip packages
# ---------------------------------------------------------------------------
echo ""
echo "========================================"
echo "  Installing pip packages"
echo "========================================"
$PIP install \
    "transformers>=4.50" \
    accelerate \
    qwen-vl-utils \
    "timm>=1.0.17" \
    einops \
    "ftfy==6.1.1" \
    regex \
    "iopath>=0.1.10" \
    huggingface_hub \
    tqdm \
    "numpy>=1.26,<2" \
    scipy \
    pillow \
    opencv-python \
    matplotlib

# ---------------------------------------------------------------------------
# Step 3: Verify
# ---------------------------------------------------------------------------
echo ""
echo "========================================"
echo "  Verification"
echo "========================================"
$PYTHON - << 'VERIFY_EOF'
import sys
print(f"Python: {sys.version}")

import torch
print(f"torch:  {torch.__version__}  CUDA: {torch.cuda.is_available()}")

from transformers import Qwen2_5_VLForConditionalGeneration
print("Qwen2.5-VL:  OK")

from transformers import Sam3Model, Sam3Processor
print("SAM3:        OK")

import scipy.spatial.transform
print("scipy:       OK")

import cv2
print("opencv:      OK")

print("\nAll checks passed!")
VERIFY_EOF

echo ""
echo "========================================"
echo "  Setup complete:  conda activate $ENV_NAME"
echo "========================================"
