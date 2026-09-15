#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/cjc/MMhgnnpro/MMhgnn_utsw}"
UTSW_DATA_ROOT="${UTSW_DATA_ROOT:-/home/cjc/UTSWdata/UTSW-Glioma}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/cjc/utsw_idh_outputs}"
GPU_ID="${GPU_ID:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ "${GPU_ID}" == "7" ]]; then
  echo "GPU 7 is prohibited for UTSW jobs." >&2
  exit 2
fi

cd "${PROJECT_ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

SOURCE_CHECKPOINT="${OUTPUT_ROOT}/missing_aware_aux_affine_e4a_seed42/checkpoints/best.pt"
E4B_OUTPUT="${OUTPUT_ROOT}/frozen_affine_e4b_seed42"

"${PYTHON_BIN}" tools/train_frozen_affine_e4b.py \
  --config configs/utsw_idh/frozen_affine_e4b.yaml \
  --source-checkpoint "${SOURCE_CHECKPOINT}" \
  --data-root "${UTSW_DATA_ROOT}" \
  --output-dir "${E4B_OUTPUT}" \
  --reuse-cache \
  --device cuda

"${PYTHON_BIN}" evaluate.py \
  --config configs/utsw_idh/frozen_affine_e4b.yaml \
  --checkpoint "${E4B_OUTPUT}/checkpoints/best.pt" \
  --data-root "${UTSW_DATA_ROOT}" \
  --output-dir "${E4B_OUTPUT}"

"${PYTHON_BIN}" tools/diagnose_e3_pre_e4.py \
  --config configs/utsw_idh/frozen_affine_e4b.yaml \
  --checkpoint "${E4B_OUTPUT}/checkpoints/best.pt" \
  --data-root "${UTSW_DATA_ROOT}" \
  --output-dir "${E4B_OUTPUT}/diagnostics/post_e4b_validation" \
  --output-prefix e4b_validation \
  --device cuda
