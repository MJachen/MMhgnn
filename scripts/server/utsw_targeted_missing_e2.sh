#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

: "${UTSW_DATA_ROOT:?Set UTSW_DATA_ROOT to the copied UTSW-Glioma directory}"
GPU_ID="${GPU_ID:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/utsw_idh/server}"
PYTHON_BIN="${PYTHON_BIN:-python}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

EXTRA_ARGS=()
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
  EXTRA_ARGS+=(--resume "${RESUME_CHECKPOINT}")
fi

"${PYTHON_BIN}" train.py \
  --config configs/utsw_idh/targeted_missing_e2.yaml \
  --data-root "${UTSW_DATA_ROOT}" \
  --output-dir "${OUTPUT_ROOT}/targeted_missing_e2_seed42" \
  "${EXTRA_ARGS[@]}"

