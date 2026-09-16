#!/usr/bin/env bash
set -euo pipefail

TASK_NAME="${1:?usage: utsw_multitask_stage_b.sh <grade_2_vs_34|mgmt>}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/cjc/MMhgnnpro/MMhgnn_utsw}"
UTSW_DATA_ROOT="${UTSW_DATA_ROOT:-/home/cjc/UTSWdata/UTSW-Glioma}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/cjc/utsw_idh_outputs}"
GPU_ID="${GPU_ID:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ "${GPU_ID}" == "7" ]]; then echo "GPU 7 is prohibited for UTSW jobs." >&2; exit 2; fi
case "${TASK_NAME}" in
  grade_2_vs_34) CONFIG="configs/utsw_grade/frozen_affine_e4b.yaml" ;;
  mgmt) CONFIG="configs/utsw_mgmt/frozen_affine_e4b.yaml" ;;
  *) echo "Unsupported task: ${TASK_NAME}" >&2; exit 2 ;;
esac

cd "${PROJECT_ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
TASK_ROOT="${OUTPUT_ROOT}/utsw_multitask/${TASK_NAME}"
MANIFEST="${TASK_ROOT}/task_manifest.csv"
SPLIT="${TASK_ROOT}/frozen_split.json"
FINGERPRINT="${TASK_ROOT}/task_manifest_fingerprint.json"
SOURCE_CHECKPOINT="${TASK_ROOT}/e4a_seed42/checkpoints/best.pt"
E4B_OUTPUT="${TASK_ROOT}/e4b_seed42"

[[ -f "${SOURCE_CHECKPOINT}" ]] || { echo "Missing task-specific E4A checkpoint: ${SOURCE_CHECKPOINT}" >&2; exit 4; }
if [[ -e "${E4B_OUTPUT}/checkpoints/best.pt" ]]; then
  echo "Refusing to overwrite existing E4B checkpoint: ${E4B_OUTPUT}/checkpoints/best.pt" >&2
  exit 3
fi

CACHE_ARGS=()
if [[ "${REUSE_CACHE:-0}" == "1" ]]; then CACHE_ARGS+=(--reuse-cache); fi
"${PYTHON_BIN}" tools/train_frozen_affine_e4b.py \
  --config "${CONFIG}" --source-checkpoint "${SOURCE_CHECKPOINT}" \
  --data-root "${UTSW_DATA_ROOT}" --manifest-csv "${MANIFEST}" --split-json "${SPLIT}" \
  --manifest-fingerprint-json "${FINGERPRINT}" --output-dir "${E4B_OUTPUT}" \
  --device cuda "${CACHE_ARGS[@]}"

"${PYTHON_BIN}" evaluate.py \
  --config "${CONFIG}" --checkpoint "${E4B_OUTPUT}/checkpoints/best.pt" \
  --data-root "${UTSW_DATA_ROOT}" --manifest-csv "${MANIFEST}" --split-json "${SPLIT}" \
  --manifest-fingerprint-json "${FINGERPRINT}" --output-dir "${E4B_OUTPUT}"

"${PYTHON_BIN}" tools/diagnose_e3_pre_e4.py \
  --config "${CONFIG}" --checkpoint "${E4B_OUTPUT}/checkpoints/best.pt" \
  --data-root "${UTSW_DATA_ROOT}" --manifest-csv "${MANIFEST}" --split-json "${SPLIT}" \
  --manifest-fingerprint-json "${FINGERPRINT}" \
  --output-dir "${E4B_OUTPUT}/diagnostics/post_e4b_validation" \
  --output-prefix e4b_validation --device cuda
