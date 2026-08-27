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

"${PYTHON_BIN}" scripts/utsw_check_server.py \
  --config configs/utsw_idh/server_smoke.yaml \
  --data-root "${UTSW_DATA_ROOT}"

"${PYTHON_BIN}" scripts/utsw_check_dataset.py \
  --data-root "${UTSW_DATA_ROOT}" \
  --output-json "${OUTPUT_ROOT}/dataset_sanity_report.json"

"${PYTHON_BIN}" scripts/utsw_server_smoke.py \
  --config configs/utsw_idh/server_smoke.yaml \
  --data-root "${UTSW_DATA_ROOT}" \
  --output-dir "${OUTPUT_ROOT}/server_smoke"
