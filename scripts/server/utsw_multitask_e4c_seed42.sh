#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-/home/cjc/MMhgnnpro/MMhgnn_utsw}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/cjc/utsw_idh_outputs}"
PYTHON_BIN="${PYTHON_BIN:-python}"

"${SCRIPT_DIR}/utsw_grade_e4a_seed42.sh"
"${SCRIPT_DIR}/utsw_grade_e4b_seed42.sh"
cd "${PROJECT_ROOT}"
"${PYTHON_BIN}" tools/verify_utsw_multitask_e4c_stage.py \
  --task-root "${OUTPUT_ROOT}/utsw_multitask/grade_2_vs_34"

"${SCRIPT_DIR}/utsw_mgmt_e4a_seed42.sh"
"${SCRIPT_DIR}/utsw_mgmt_e4b_seed42.sh"
cd "${PROJECT_ROOT}"
"${PYTHON_BIN}" tools/verify_utsw_multitask_e4c_stage.py \
  --task-root "${OUTPUT_ROOT}/utsw_multitask/mgmt"

cd "${PROJECT_ROOT}"
"${PYTHON_BIN}" tools/summarize_utsw_multitask_e4c.py \
  --output-root "${OUTPUT_ROOT}" \
  --idh-split-json splits/utsw_idh/utsw_idh_debug_split.json
