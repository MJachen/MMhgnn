#!/usr/bin/env bash
set -euo pipefail

CONFIG="configs/default.yaml"
CHECKPOINT="outputs/demo_run/checkpoints/best.pt"
OUT_ROOT="outputs/server_eval"
RUN_NAME="eval_$(date +%Y%m%d_%H%M%S)"
COMBO=()

usage() {
  cat <<'EOF'
Usage:
  bash scripts/server_evaluate_and_pack.sh \
    --config configs/default.yaml \
    --checkpoint outputs/demo_run/checkpoints/best.pt \
    --run-name default_eval

Optional:
  --out-root outputs/server_eval
  --combo t2 t1ce

Default behavior evaluates all non-empty modality combinations configured in the YAML.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --checkpoint)
      CHECKPOINT="$2"
      shift 2
      ;;
    --out-root)
      OUT_ROOT="$2"
      shift 2
      ;;
    --run-name)
      RUN_NAME="$2"
      shift 2
      ;;
    --combo)
      shift
      while [[ $# -gt 0 && "$1" != --* ]]; do
        COMBO+=("$1")
        shift
      done
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

RUN_DIR="${OUT_ROOT}/${RUN_NAME}"
mkdir -p "${RUN_DIR}"

COMMAND=(python evaluate.py --config "${CONFIG}" --checkpoint "${CHECKPOINT}" --output-dir "${RUN_DIR}")
if [[ ${#COMBO[@]} -gt 0 ]]; then
  COMMAND+=(--combo "${COMBO[@]}")
fi

printf '%q ' "${COMMAND[@]}" > "${RUN_DIR}/command.txt"
printf '\n' >> "${RUN_DIR}/command.txt"

{
  echo "run_name=${RUN_NAME}"
  echo "config=${CONFIG}"
  echo "checkpoint=${CHECKPOINT}"
  echo "out_root=${OUT_ROOT}"
  echo "started_at=$(date -Is)"
  echo "host=$(hostname)"
  git rev-parse HEAD 2>/dev/null | sed 's/^/git_commit=/'
  git status --short 2>/dev/null | sed 's/^/git_status=/'
  python --version 2>&1 | sed 's/^/python=/'
} > "${RUN_DIR}/run_metadata.txt"

echo "Running evaluation:"
cat "${RUN_DIR}/command.txt"

"${COMMAND[@]}" 2>&1 | tee "${RUN_DIR}/evaluate.log"

{
  echo "finished_at=$(date -Is)"
  echo "result_dir=${RUN_DIR}"
  echo "main_metrics_csv=${RUN_DIR}/evaluate_only/metrics.csv"
  echo "main_metrics_json=${RUN_DIR}/evaluate_only/metrics.json"
} >> "${RUN_DIR}/run_metadata.txt"

ARCHIVE="${OUT_ROOT}/${RUN_NAME}.tar.gz"
tar -czf "${ARCHIVE}" -C "${OUT_ROOT}" "${RUN_NAME}"

echo "Evaluation finished."
echo "Result directory: ${RUN_DIR}"
echo "Archive: ${ARCHIVE}"

