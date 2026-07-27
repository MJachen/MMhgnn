# Three-seed auxiliary WT experiment

## Protocol

The experiment tests whether `lambda_seg=0.05` improves the classification task across seeds 42, 43, and 44.

- All seeds use `configs/splits/aux_wt_seg_shared_seed42.json` (292 train / 36 validation / 37 test).
- The subject split is fixed; only initialization, data order, and stochastic training behavior change.
- Each seed writes to its own output directory and runs on one physical GPU.
- The primary checkpoint remains classification-best by validation balanced accuracy because the scientific question is classification improvement.
- A stronger segmentation log value from another epoch must not be used to describe the deployed classification checkpoint.

The tracked split has SHA256:

```text
a4f2a8610d8ef1eb2e1fdc09e1ac833a09e8b32bc407937c87a0106346e0d9f6
```

The launcher verifies the file is present and records its SHA256 in every run directory.

## Pull the branch on the server

```bash
cd ~/MMhgnnpro/MMhgnn
git fetch origin
git checkout feature/aux-wt-seg
git pull --ff-only origin feature/aux-wt-seg
```

Activate the existing project environment, then verify data and the fixed split:

```bash
python scripts/check_dataset_config.py \
  --config configs/experiments/aux_wt_seg_lambda005_multiseed.yaml \
  --require-split
```

## Launch λ=0.05 on three GPUs

The following command maps seed 42→GPU 0, seed 43→GPU 1, and seed 44→GPU 2. The launcher uses `nohup`, records PIDs/commands/commit/split hash, and returns after all jobs start.

```bash
ARM=lambda_005 \
GPU_IDS=0,1,2 \
SEEDS=42,43,44 \
PYTHON_BIN=python \
bash scripts/run_aux_wt_seg_multiseed.sh
```

Equivalent explicit commands are:

```bash
mkdir -p outputs/aux_wt_seg_multiseed/lambda_005/logs

CUDA_VISIBLE_DEVICES=0 nohup python -u train.py \
  --config configs/experiments/aux_wt_seg_lambda005_multiseed.yaml \
  --seed 42 --output-dir outputs/aux_wt_seg_multiseed/lambda_005/seed_42 \
  > outputs/aux_wt_seg_multiseed/lambda_005/logs/seed_42.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python -u train.py \
  --config configs/experiments/aux_wt_seg_lambda005_multiseed.yaml \
  --seed 43 --output-dir outputs/aux_wt_seg_multiseed/lambda_005/seed_43 \
  > outputs/aux_wt_seg_multiseed/lambda_005/logs/seed_43.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python -u train.py \
  --config configs/experiments/aux_wt_seg_lambda005_multiseed.yaml \
  --seed 44 --output-dir outputs/aux_wt_seg_multiseed/lambda_005/seed_44 \
  > outputs/aux_wt_seg_multiseed/lambda_005/logs/seed_44.log 2>&1 &
```

Do not run the launcher and the three explicit commands together.

## Monitor

```bash
nvidia-smi
for f in outputs/aux_wt_seg_multiseed/lambda_005/logs/*.log; do
  echo "===== $f ====="
  tail -n 10 "$f"
done
ps -fp $(paste -sd, outputs/aux_wt_seg_multiseed/lambda_005/pids/*.pid)
```

Completion requires all three files:

```text
outputs/aux_wt_seg_multiseed/lambda_005/seed_<seed>/metrics/test_metrics.csv
```

## Matched λ=0 baseline

Three λ=0.05 seeds alone estimate variability but cannot prove an improvement. Run the matched classification-only arm on the same split after the auxiliary jobs finish:

```bash
ARM=lambda_000 \
GPU_IDS=0,1,2 \
SEEDS=42,43,44 \
PYTHON_BIN=python \
bash scripts/run_aux_wt_seg_multiseed.sh
```

## Aggregate after both arms finish

```bash
python scripts/summarize_aux_wt_seg_multiseed.py \
  --run-root outputs/aux_wt_seg_multiseed/lambda_005 \
  --baseline-root outputs/aux_wt_seg_multiseed/lambda_000 \
  --expected-seeds 42,43,44
```

Primary outputs:

```text
outputs/aux_wt_seg_multiseed/lambda_005/summary/multiseed_report.md
outputs/aux_wt_seg_multiseed/lambda_005/summary/multiseed_summary.csv
outputs/aux_wt_seg_multiseed/lambda_005/summary/multiseed_comparison.csv
```

Keep the lightweight segmentation head only if paired classification improvements are larger than seed variation and directionally consistent. Better segmentation alone is not sufficient.
