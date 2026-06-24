# Seed42 observed-only protocol validation

This protocol must pass before adding modality completion or new fusion logic.

## Train the controlled pair

Run from the repository root on the server:

```bash
git fetch origin
git checkout codex/safe-calibration-diagnostics
git pull --ff-only

CURRICULUM_OUT=outputs/protocol_v2_curriculum_seed42
mkdir -p "${CURRICULUM_OUT}"
CUDA_VISIBLE_DEVICES=0 python train.py \
  --config configs/brats2020_public_missing_curriculum_server.yaml \
  --seed 42 \
  --output-dir "${CURRICULUM_OUT}" \
  2>&1 | tee "${CURRICULUM_OUT}/console.log"

TARGETED_OUT=outputs/protocol_v2_no_t1ce_focus_seed42
mkdir -p "${TARGETED_OUT}"
CUDA_VISIBLE_DEVICES=0 python train.py \
  --config configs/brats2020_public_no_t1ce_focus_server.yaml \
  --seed 42 \
  --output-dir "${TARGETED_OUT}" \
  2>&1 | tee "${TARGETED_OUT}/console.log"
```

`train.py` also writes a structured `train.log`. The separate `console.log` retains progress-bar and stderr output.

## Validate the protocol

```bash
python scripts/compare_protocol_runs.py \
  --curriculum-output outputs/protocol_v2_curriculum_seed42 \
  --targeted-output outputs/protocol_v2_no_t1ce_focus_seed42
```

The command exits with status 0 only when:

- both runs have the same base model-state SHA256;
- targeted fine-tuning changes the model-state SHA256;
- targeted fine-tuning satisfies the joint validation guardrails and is selected;
- selected predictions differ for `t2`, `flair`, `t2_flair`, and `t2_t1_flair`.

Key audit files are under each run's `metrics/` directory:

- `checkpoint_selection.json`
- `checkpoint_hashes.json`
- `prediction_hashes.json`
- `train_history.csv`
- `threshold_calibration.json`
