#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

: "${UTSW_DATA_ROOT:?Set UTSW_DATA_ROOT to the copied UTSW-Glioma directory}"
GPU_ID="${GPU_ID:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/utsw_idh/server}"
PYTHON_BIN="${PYTHON_BIN:-python}"
FULL_BASELINE_CHECKPOINT="${FULL_BASELINE_CHECKPOINT:-${OUTPUT_ROOT}/full_baseline_seed42/checkpoints/best.pt}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

if [[ ! -f "${FULL_BASELINE_CHECKPOINT}" ]]; then
  echo "Full-baseline checkpoint not found: ${FULL_BASELINE_CHECKPOINT}" >&2
  exit 2
fi

"${PYTHON_BIN}" evaluate.py \
  --config configs/utsw_idh/missing15.yaml \
  --checkpoint "${FULL_BASELINE_CHECKPOINT}" \
  --data-root "${UTSW_DATA_ROOT}" \
  --output-dir "${OUTPUT_ROOT}/missing15_seed42"
