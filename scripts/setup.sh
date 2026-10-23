#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
cd "${REPO_ROOT}"

if [[ "$(uname -s)" != Linux ]]; then
    echo "Training requires Linux and an NVIDIA CUDA GPU." >&2
    exit 1
fi
if ! command -v nvcc >/dev/null 2>&1; then
    echo "Install the CUDA 12.8 toolkit and put nvcc on PATH before running setup." >&2
    exit 1
fi
if ! nvcc --version | grep -q 'release 12\.8'; then
    echo "This setup targets CUDA 12.8. Use a matching toolkit for the pinned PyTorch build." >&2
    exit 1
fi

"${PYTHON_BIN}" -m pip install --upgrade pip setuptools wheel
"${PYTHON_BIN}" -m pip install torch==2.11.0 torchvision==0.26.0 \
    --index-url https://download.pytorch.org/whl/cu128
"${PYTHON_BIN}" -m pip install -r requirements.txt
"${PYTHON_BIN}" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot access CUDA. Check the NVIDIA driver and GPU allocation.")
print(f"PyTorch {torch.__version__}; CUDA {torch.version.cuda}; {torch.cuda.get_device_name(0)}")
PY

git submodule update --init --recursive
RASTERIZER_DIR="${REPO_ROOT}/submodules/diff-gaussian-rasterization"
PATCH_FILE="${REPO_ROOT}/patches/diff-gaussian-rasterization-cstdint.patch"
if git -C "${RASTERIZER_DIR}" apply --reverse --check "${PATCH_FILE}" >/dev/null 2>&1; then
    echo "Rasterizer compatibility patch is already applied."
elif git -C "${RASTERIZER_DIR}" apply --check "${PATCH_FILE}"; then
    git -C "${RASTERIZER_DIR}" apply "${PATCH_FILE}"
else
    echo "Cannot apply the rasterizer compatibility patch to this checkout." >&2
    exit 1
fi

export MAX_JOBS="${MAX_JOBS:-4}"
"${PYTHON_BIN}" -m pip install --no-build-isolation \
    ./submodules/diff-gaussian-rasterization \
    ./submodules/simple-knn \
    ./submodules/fused-ssim
"${PYTHON_BIN}" - <<'PY'
import torch
from diff_gaussian_rasterization import GaussianRasterizer
from simple_knn._C import distCUDA2
from fused_ssim import fused_ssim
assert (torch.ones(1, device="cuda") * 2).item() == 2
print("CUDA extensions imported successfully.")
PY
"${PYTHON_BIN}" -m unittest discover -s tests -v
