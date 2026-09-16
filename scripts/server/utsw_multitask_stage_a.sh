#!/usr/bin/env bash
set -euo pipefail

TASK_NAME="${1:?usage: utsw_multitask_stage_a.sh <grade_2_vs_34|mgmt>}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/cjc/MMhgnnpro/MMhgnn_utsw}"
UTSW_DATA_ROOT="${UTSW_DATA_ROOT:-/home/cjc/UTSWdata/UTSW-Glioma}"
UTSW_METADATA_CSV="${UTSW_METADATA_CSV:-/home/cjc/UTSWdata/UTSW_Glioma_Metadata-2-1.csv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/cjc/utsw_idh_outputs}"
GPU_ID="${GPU_ID:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ "${GPU_ID}" == "7" ]]; then
  echo "GPU 7 is prohibited for UTSW jobs." >&2
  exit 2
fi

case "${TASK_NAME}" in
  grade_2_vs_34) CONFIG="configs/utsw_grade/missing_aware_aux_affine_e4a.yaml" ;;
  mgmt) CONFIG="configs/utsw_mgmt/missing_aware_aux_affine_e4a.yaml" ;;
  *) echo "Unsupported task: ${TASK_NAME}" >&2; exit 2 ;;
esac

cd "${PROJECT_ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader

TASK_ROOT="${OUTPUT_ROOT}/utsw_multitask/${TASK_NAME}"
MANIFEST="${TASK_ROOT}/task_manifest.csv"
SPLIT="${TASK_ROOT}/frozen_split.json"
FINGERPRINT="${TASK_ROOT}/task_manifest_fingerprint.json"
E4A_OUTPUT="${TASK_ROOT}/e4a_seed42"

"${PYTHON_BIN}" tools/audit_utsw_multitask_labels.py \
  --metadata-csv "${UTSW_METADATA_CSV}" \
  --imaging-manifest metadata/utsw_idh_manifest.csv \
  --frozen-split splits/utsw_idh/utsw_idh_debug_split.json \
  --output-root "${OUTPUT_ROOT}/utsw_multitask"

"${PYTHON_BIN}" tools/preflight_utsw_multitask_e4c.py \
  --config "${CONFIG}" --data-root "${UTSW_DATA_ROOT}" \
  --manifest-csv "${MANIFEST}" --split-json "${SPLIT}" \
  --manifest-fingerprint-json "${FINGERPRINT}"

"${PYTHON_BIN}" tests/smoke_utsw_affine_e4a.py \
  --config "${CONFIG}" --data-root "${UTSW_DATA_ROOT}" \
  --manifest-csv "${MANIFEST}" --split-json "${SPLIT}" \
  --manifest-fingerprint-json "${FINGERPRINT}" --device cuda

EXTRA_ARGS=()
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
  EXTRA_ARGS+=(--resume "${RESUME_CHECKPOINT}")
elif [[ -e "${E4A_OUTPUT}/checkpoints/best.pt" ]]; then
  echo "Refusing to overwrite existing E4A checkpoint: ${E4A_OUTPUT}/checkpoints/best.pt" >&2
  exit 3
fi

"${PYTHON_BIN}" train.py \
  --config "${CONFIG}" --data-root "${UTSW_DATA_ROOT}" --output-dir "${E4A_OUTPUT}" \
  --manifest-csv "${MANIFEST}" --split-json "${SPLIT}" \
  --manifest-fingerprint-json "${FINGERPRINT}" "${EXTRA_ARGS[@]}"

"${PYTHON_BIN}" tools/diagnose_e3_pre_e4.py \
  --config "${CONFIG}" --checkpoint "${E4A_OUTPUT}/checkpoints/best.pt" \
  --data-root "${UTSW_DATA_ROOT}" --manifest-csv "${MANIFEST}" --split-json "${SPLIT}" \
  --manifest-fingerprint-json "${FINGERPRINT}" \
  --output-dir "${E4A_OUTPUT}/diagnostics/post_e4a_validation" \
  --output-prefix e4a_validation --device cuda
