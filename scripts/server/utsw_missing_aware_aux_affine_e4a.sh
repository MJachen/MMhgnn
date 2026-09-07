#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/cjc/MMhgnnpro/MMhgnn_utsw}"
UTSW_DATA_ROOT="${UTSW_DATA_ROOT:-/home/cjc/UTSWdata/UTSW-Glioma}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/cjc/utsw_idh_outputs}"
GPU_ID="${GPU_ID:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "${PROJECT_ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

EXTRA_ARGS=()
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
  EXTRA_ARGS+=(--resume "${RESUME_CHECKPOINT}")
fi

"${PYTHON_BIN}" train.py \
  --config configs/utsw_idh/missing_aware_aux_affine_e4a.yaml \
  --data-root "${UTSW_DATA_ROOT}" \
  --output-dir "${OUTPUT_ROOT}/missing_aware_aux_affine_e4a_seed42" \
  "${EXTRA_ARGS[@]}"

"${PYTHON_BIN}" tools/diagnose_e3_pre_e4.py \
  --config configs/utsw_idh/missing_aware_aux_affine_e4a.yaml \
  --checkpoint "${OUTPUT_ROOT}/missing_aware_aux_affine_e4a_seed42/checkpoints/best.pt" \
  --data-root "${UTSW_DATA_ROOT}" \
  --output-dir "${OUTPUT_ROOT}/missing_aware_aux_affine_e4a_seed42/diagnostics/post_e4a_validation" \
  --output-prefix e4a_validation \
  --device cuda
