#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 4 ]]; then
    echo "Usage: bash scripts/train_eval.sh SCENE_DIR MODEL_DIR [fv|vanilla] [ITERATIONS]" >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
# Resolve user paths before changing directories.
SCENE_DIR="$("${PYTHON_BIN}" -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$1")"
MODEL_DIR="$("${PYTHON_BIN}" -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$2")"
METHOD="${3:-fv}"
ITERATIONS="${4:-7000}"

if [[ "${METHOD}" != fv && "${METHOD}" != vanilla ]]; then
    echo "Method must be fv or vanilla." >&2
    exit 1
fi
if ! [[ "${ITERATIONS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ITERATIONS must be a positive integer." >&2
    exit 1
fi
if [[ ! -d "${SCENE_DIR}/images" || ! -d "${SCENE_DIR}/sparse/0" ]]; then
    echo "Expected images/ and sparse/0/ under ${SCENE_DIR}." >&2
    exit 1
fi
if [[ -e "${MODEL_DIR}" ]]; then
    echo "Output already exists: ${MODEL_DIR}. Choose a new directory." >&2
    exit 1
fi

cd "${REPO_ROOT}"
train_args=(
    -s "${SCENE_DIR}" -m "${MODEL_DIR}" --eval
    --iterations "${ITERATIONS}" --max_gaussians -1
    --test_iterations "${ITERATIONS}" --save_iterations "${ITERATIONS}"
    --disable_viewer
)
if [[ "${METHOD}" == fv ]]; then
    train_args+=(
        --first_variation_adc
        --first_variation_birth_selection q_threshold
        --first_variation_birth_threshold 0.0000001
        --first_variation_max_growth_fraction 0.10
        --first_variation_score_views 16
        --first_variation_score_ema 0.0
        --first_variation_score_clip_quantile 0.0
        --first_variation_death_selection standard_opacity
        --first_variation_seed 0
    )
fi

mkdir -p "${MODEL_DIR}"
{
    echo "git_commit=$(git rev-parse HEAD)"
    echo "method=${METHOD}"
    echo "iterations=${ITERATIONS}"
    echo "seed=0"
    echo "scene=${SCENE_DIR}"
    "${PYTHON_BIN}" -c 'import torch; print(f"torch={torch.__version__}\ncuda={torch.version.cuda}\ngpu={torch.cuda.get_device_name(0)}")'
} > "${MODEL_DIR}/run.txt"
"${PYTHON_BIN}" train.py "${train_args[@]}" 2>&1 | tee "${MODEL_DIR}/train.log"
"${PYTHON_BIN}" render.py -m "${MODEL_DIR}" --iteration "${ITERATIONS}" --skip_train \
    2>&1 | tee "${MODEL_DIR}/render.log"
"${PYTHON_BIN}" metrics.py -m "${MODEL_DIR}" 2>&1 | tee "${MODEL_DIR}/metrics.log"
