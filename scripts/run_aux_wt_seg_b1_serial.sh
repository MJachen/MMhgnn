#!/usr/bin/env bash
set -euo pipefail

# Run the three B1 seeds sequentially on one physical GPU. This is intended for
# a busy server where only one GPU is available. Wrap this script in nohup if
# the SSH session must be allowed to close.

GPU_ID="${GPU_ID:-7}"
SEEDS_CSV="${SEEDS:-42,43,44}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${CONFIG:-configs/experiments/aux_wt_seg_b1_multiseed.yaml}"
RUN_ROOT="${RUN_ROOT:-outputs/aux_wt_seg_multiseed/b1}"
SKIP_DATA_CHECK="${SKIP_DATA_CHECK:-0}"
SPLIT_FILE="configs/splits/aux_wt_seg_shared_seed42.json"

IFS=',' read -r -a SEEDS_ARRAY <<< "$SEEDS_CSV"
if [[ ${#SEEDS_ARRAY[@]} -ne 3 ]]; then
  echo "B1 protocol requires exactly three seeds." >&2
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

mkdir -p "$RUN_ROOT/logs"
for seed in "${SEEDS_ARRAY[@]}"; do
  run_dir="$RUN_ROOT/seed_${seed}"
  if [[ -d "$run_dir" ]] && [[ -n "$(find "$run_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "Refusing to overwrite non-empty run directory: $run_dir" >&2
    exit 3
  fi
done

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
  log_file="$RUN_ROOT/logs/seed_${seed}.log"
  command=(
    "$PYTHON_BIN" -u train.py
    --config "$CONFIG"
    --seed "$seed"
    --output-dir "$run_dir"
  )

  mkdir -p "$run_dir"
  printf '%s\n' "$git_commit" > "$run_dir/git_commit.txt"
  printf '%s\n' "$split_sha256" > "$run_dir/split_sha256.txt"
  printf 'CUDA_VISIBLE_DEVICES=%q' "$GPU_ID" > "$run_dir/command.txt"
  printf ' %q' "${command[@]}" >> "$run_dir/command.txt"
  printf '\n' >> "$run_dir/command.txt"

  echo "Starting seed=$seed on physical GPU=$GPU_ID; log=$log_file"
  env CUDA_VISIBLE_DEVICES="$GPU_ID" "${command[@]}" > "$log_file" 2>&1
  test -f "$run_dir/metrics/test_metrics.csv"
  printf 'completed\n' > "$run_dir/.complete"
  echo "Completed seed=$seed"
done

echo "All B1 seeds completed successfully."
