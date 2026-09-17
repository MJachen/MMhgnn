#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/cjc/MMhgnnpro/MMhgnn_utsw}"
GPU_ID="${GPU_ID:?GPU_ID must name the physical BraTS GPU}"
PYTHON_BIN="${PYTHON_BIN:-/home/cjc/yes/envs/brats-more-cu126/bin/python}"
TASK_ROOT="${TASK_ROOT:-outputs/cross_task/brats2020_hgg_lgg}"
E4A_OUTPUT="${TASK_ROOT}/e4a_seed42"
E4B_OUTPUT="${TASK_ROOT}/e4b_seed42"
SPLIT_JSON="${TASK_ROOT}/frozen_split.json"
FINGERPRINT_JSON="${TASK_ROOT}/dataset_fingerprint.json"
E4A_CONFIG="configs/brats2020/latest_pipeline_e4a.yaml"
E4B_CONFIG="configs/brats2020/latest_pipeline_e4b.yaml"

if [[ "${GPU_ID}" == "7" ]]; then
  echo "GPU 7 is prohibited." >&2
  exit 2
fi

cd "${PROJECT_ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

[[ -f "${SPLIT_JSON}" ]] || { echo "Missing frozen historical split: ${SPLIT_JSON}" >&2; exit 3; }
[[ ! -e "${E4A_OUTPUT}/checkpoints/best.pt" ]] || { echo "Refusing to overwrite E4A best.pt" >&2; exit 4; }
[[ ! -e "${E4B_OUTPUT}/checkpoints/best.pt" ]] || { echo "Refusing to overwrite E4B best.pt" >&2; exit 5; }

echo "[Step 1/7] BraTS preflight on physical GPU ${GPU_ID}"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
"${PYTHON_BIN}" tools/preflight_brats2020_latest_pipeline.py \
  --e4a-config "${E4A_CONFIG}" --e4b-config "${E4B_CONFIG}" \
  --output-root "${TASK_ROOT}" --device cuda

echo "[Step 2/7] BraTS E4A seed42 from random initialization"
"${PYTHON_BIN}" train.py --config "${E4A_CONFIG}" --output-dir "${E4A_OUTPUT}"

echo "[Step 3/7] Verify BraTS E4A best.pt and expose validation metrics"
[[ -s "${E4A_OUTPUT}/checkpoints/best.pt" ]] || { echo "E4A best.pt missing or empty" >&2; exit 6; }
"${PYTHON_BIN}" -c "import shutil,torch; from pathlib import Path; o=Path('${E4A_OUTPUT}'); c=torch.load(o/'checkpoints/best.pt',map_location='cpu'); e=int(c['epoch']); s=o/'metrics'/f'val_missing15_epoch_{e:03d}.csv'; d=o/'metrics'/'validation_metrics.csv'; assert s.is_file(),s; shutil.copy2(s,d); print({'best_epoch':e,'validation_metrics':str(d)})"

echo "[Step 4/7] BraTS E4B frozen-backbone affine-only recalibration"
"${PYTHON_BIN}" tools/train_frozen_affine_e4b.py \
  --config "${E4B_CONFIG}" \
  --source-checkpoint "${E4A_OUTPUT}/checkpoints/best.pt" \
  --output-dir "${E4B_OUTPUT}" --device cuda

echo "[Step 5/7] Validation-derived ONE global threshold already frozen by E4B"
[[ -s "${E4B_OUTPUT}/metrics/threshold_calibration.json" ]] || { echo "Missing threshold calibration" >&2; exit 7; }

echo "[Step 6/7] Final 15-pattern test already completed after threshold freeze"
[[ -s "${E4B_OUTPUT}/metrics/test_predictions.csv" ]] || { echo "Missing final test predictions" >&2; exit 8; }

echo "[Step 7/7] Validation alignment diagnostic"
"${PYTHON_BIN}" tools/diagnose_e3_pre_e4.py \
  --config "${E4B_CONFIG}" --checkpoint "${E4B_OUTPUT}/checkpoints/best.pt" \
  --output-dir "${E4B_OUTPUT}/metrics" --output-prefix validation --device cuda

echo "BraTS2020 latest E4A->E4B pipeline completed successfully."
