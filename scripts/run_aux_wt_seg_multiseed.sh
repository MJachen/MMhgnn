#!/usr/bin/env bash
set -euo pipefail

# Launch three independent training jobs on three physical GPUs while keeping
# the subject split fixed. Override settings through environment variables:
#   ARM=b1 GPU_IDS=0,1,2 SEEDS=42,43,44 PYTHON_BIN=python bash scripts/run_aux_wt_seg_multiseed.sh

ARM="${ARM:-lambda_005}"
GPU_IDS_CSV="${GPU_IDS:-0,1,2}"
SEEDS_CSV="${SEEDS:-42,43,44}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_DATA_CHECK="${SKIP_DATA_CHECK:-0}"

case "$ARM" in
  lambda_005)
    CONFIG="${CONFIG:-configs/experiments/aux_wt_seg_lambda005_multiseed.yaml}"
    ;;
  lambda_000)
    CONFIG="${CONFIG:-configs/experiments/aux_wt_seg_lambda000_multiseed.yaml}"
    ;;
  b1)
    CONFIG="${CONFIG:-configs/experiments/aux_wt_seg_b1_multiseed.yaml}"
    ;;
  *)
    echo "Unsupported ARM=$ARM; expected lambda_005, lambda_000, or b1." >&2
    exit 2
    ;;
esac

RUN_ROOT="${RUN_ROOT:-outputs/aux_wt_seg_multiseed/${ARM}}"
SPLIT_FILE="configs/splits/aux_wt_seg_shared_seed42.json"
IFS=',' read -r -a GPU_IDS_ARRAY <<< "$GPU_IDS_CSV"
IFS=',' read -r -a SEEDS_ARRAY <<< "$SEEDS_CSV"

if [[ ${#GPU_IDS_ARRAY[@]} -ne ${#SEEDS_ARRAY[@]} ]]; then
  echo "GPU_IDS and SEEDS must have the same number of entries." >&2
  exit 2
fi
if [[ ${#SEEDS_ARRAY[@]} -ne 3 ]]; then
  echo "This protocol requires exactly three seeds and three GPUs." >&2
  exit 2
fi
if [[ "$(printf '%s\n' "${GPU_IDS_ARRAY[@]}" | sort -u | wc -l)" -ne 3 ]]; then
  echo "GPU_IDS must name three distinct physical GPUs." >&2
  exit 2
fi
if [[ "$(printf '%s\n' "${SEEDS_ARRAY[@]}" | sort -u | wc -l)" -ne 3 ]]; then
  echo "SEEDS must contain three distinct values." >&2
  exit 2
fi
if [[ ! -f "$CONFIG" || ! -f "$SPLIT_FILE" ]]; then
  echo "Missing config or fixed split: CONFIG=$CONFIG SPLIT_FILE=$SPLIT_FILE" >&2
  exit 2
fi

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/pids"

if [[ "$SKIP_DATA_CHECK" != "1" ]]; then
  echo "Running dataset and fixed-split preflight check..."
  "$PYTHON_BIN" scripts/check_dataset_config.py \
    --config "$CONFIG" \
    --require-split \
    > "$RUN_ROOT/preflight_dataset.json"
fi

git_commit="$(git rev-parse HEAD 2>/dev/null || printf 'unknown')"
split_sha256="$(sha256sum "$SPLIT_FILE" | awk '{print $1}')"

for seed in "${SEEDS_ARRAY[@]}"; do
  run_dir="$RUN_ROOT/seed_${seed}"
  if [[ -d "$run_dir" ]] && [[ -n "$(find "$run_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "Refusing to overwrite non-empty run directory: $run_dir" >&2
    exit 3
  fi
done

for index in "${!SEEDS_ARRAY[@]}"; do
  seed="${SEEDS_ARRAY[$index]}"
  gpu="${GPU_IDS_ARRAY[$index]}"
  run_dir="$RUN_ROOT/seed_${seed}"
  log_file="$RUN_ROOT/logs/seed_${seed}.log"
  pid_file="$RUN_ROOT/pids/seed_${seed}.pid"

  command=(
    "$PYTHON_BIN" -u train.py
    --config "$CONFIG"
    --seed "$seed"
    --output-dir "$run_dir"
  )

  printf 'seed=%s gpu=%s output=%s\n' "$seed" "$gpu" "$run_dir"
  printf 'command:'
  printf ' %q' "${command[@]}"
  printf '\n'

  if [[ "$DRY_RUN" == "1" ]]; then
    continue
  fi

  mkdir -p "$run_dir"
  printf '%s\n' "$git_commit" > "$run_dir/git_commit.txt"
  printf '%s\n' "$split_sha256" > "$run_dir/split_sha256.txt"
  printf 'CUDA_VISIBLE_DEVICES=%q' "$gpu" > "$run_dir/command.txt"
  printf ' %q' "${command[@]}" >> "$run_dir/command.txt"
  printf '\n' >> "$run_dir/command.txt"

  nohup env CUDA_VISIBLE_DEVICES="$gpu" "${command[@]}" \
    > "$log_file" 2>&1 < /dev/null &
  pid=$!
  printf '%s\n' "$pid" > "$pid_file"
  echo "launched seed=$seed on physical GPU=$gpu pid=$pid log=$log_file"
done

if [[ "$DRY_RUN" == "1" ]]; then
  echo "Dry run complete; no training jobs were started."
else
  echo "All three jobs launched. Monitor with:"
  echo "  nvidia-smi"
  echo "  for f in $RUN_ROOT/logs/*.log; do echo \"===== \$f =====\"; tail -n 5 \"\$f\"; done"
  echo "  ps -fp \$(paste -sd, $RUN_ROOT/pids/*.pid)"
fi
