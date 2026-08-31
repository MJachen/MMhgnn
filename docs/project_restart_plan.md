# Project Restart Plan

> 中文摘要：恢复顺序是先固定 `1471a6b` 协议版本并检查环境/数据，再复现 seed-42 基线，随后依次隔离 loss、calibration、T1 干扰和节点融合因素。所有新实验必须使用不可覆盖的输出目录、相同 split、验证集选阈值以及逐病例配对结果。

## Operating rules

This is an execution plan only. No command in this document was run during the recovery review. Do not start a new training run until the branch decision and output naming convention are confirmed.

Use these controls for every new run:

- Pin the code commit and record `git rev-parse HEAD`.
- Reuse one immutable split per seed across every compared condition.
- Write to a new output directory; never reuse `missing_curriculum_seed42` or `no_t1ce_focus_seed42`.
- Save resolved config, split summary/hash, checkpoint hash, per-case probabilities, selected thresholds, and console/structured logs.
- Select checkpoints and thresholds with validation data only. Open test results only after the condition is frozen.
- Compare paired conditions on identical cases and seeds.

## Proposed configuration matrix

These files do not exist yet. Create them only after approval, preferably under `configs/recovery/`, with `configs/brats2020_public_calib_safe_server.yaml` from commit `1471a6b` as the common base.

| Planned config | Purpose | Key overrides |
|---|---|---|
| `reference_protocol.yaml` | Exact current reference reproduction | Missing curriculum; weighted CE; mask-aware fusion and classifier enabled; penalty 1.0; three-group calibration; targeted fine-tune disabled; fixed split. |
| `unweighted_fixed05.yaml` | Loss/calibration control 1 | Unweighted CE; calibration disabled; threshold 0.5. |
| `weighted_fixed05.yaml` | Control 2 | Weighted CE; calibration disabled; threshold 0.5. |
| `weighted_global.yaml` | Control 3 | Weighted CE; validation global threshold. |
| `weighted_group2.yaml` | Control 4 | Weighted CE; validation thresholds for has-T1ce/no-T1ce. Requires restoring a two-group fitter; only old dispatch compatibility exists now. |
| `weighted_group3.yaml` | Control 5 | Weighted CE; current three-group thresholds. |
| `fusion_shared.yaml` | Existing simplest fusion control | Mask-aware node fusion off; modal edges off; mask-aware classifier off; penalty irrelevant. This uses shared mean+max, not pure mean. |
| `fusion_gate.yaml` | Gate without hand penalty | Mask-aware node fusion on; penalty 0.0; mask-aware classifier off. |
| `fusion_gate_penalty.yaml` | Gate plus T1 penalty | Same as above; penalty 1.0; mask-aware classifier off. |

All loss/calibration comparisons must keep fusion, data sampling, optimizer, checkpoint policy, and split fixed. All fusion comparisons must keep the selected loss/calibration protocol fixed.

## Phase 0: environment and data check

**Goal:** prove that the pinned protocol code, Python environment, data paths, split, and existing checkpoint are readable without training.

**Recommended code state:** `codex/safe-calibration-diagnostics` commit `1471a6b`, because the latest outputs depend on it. Review/merge or pin it before continuing.

**Commands (server):**

```bash
git status --short --branch
git fetch origin
git switch codex/safe-calibration-diagnostics
git rev-parse HEAD

python --version
python -m pip check
python -c "import torch,nibabel,pandas,sklearn,scipy,yaml; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)"
python scripts/check_dataset_config.py --config configs/brats2020_public_calib_safe_server.yaml --show 5
python -m pytest -q

python -c "import torch; p='server_results/outputs/protocol_v2_curriculum_seed42/base_best.pt'; x=torch.load(p,map_location='cpu'); print(sorted(x.keys()))"
```

The final checkpoint command must use the actual server-side checkpoint path if `server_results/` is only a local fetched copy.

**Inputs:** requirements, public BraTS root/label spreadsheet, fixed split JSON, protocol-v2 checkpoint.

**Expected outputs:** exact commit hash, dependency/CUDA versions, 365 valid cases, class distribution 76/289, no unit-test failure, readable checkpoint dictionary.

**Success criteria:** code is pinned to the protocol commit; all 365 cases and required modality/segmentation files resolve; split contains 292/36/37 cases with no overlap; model state loads on CPU and GPU availability is known.

**If it fails, check first:** wrong branch; stale server paths; missing `pytest` from requirements; PyTorch/CUDA binary mismatch; label column mapping; BraTS naming pattern; checkpoint path copied without the actual `.pt` files.

## Phase 1: reproduce the current anatomy-only reference

**Goal:** reproduce the most traceable current result before changing any factor.

**Config:** first use the existing `configs/brats2020_public_missing_curriculum_server.yaml` at commit `1471a6b`. After parity is established, copy its resolved settings into the planned `configs/recovery/reference_protocol.yaml`.

**Command:**

```bash
OUT=outputs/recovery/reference_seed42_$(git rev-parse --short HEAD)
mkdir -p "$OUT"
CUDA_VISIBLE_DEVICES=0 python train.py \
  --config configs/brats2020_public_missing_curriculum_server.yaml \
  --seed 42 \
  --output-dir "$OUT" \
  2>&1 | tee "$OUT/console.log"
```

**Expected outputs:** resolved config, split summary, `base_best.pt`/`best.pt`, train history, threshold calibration, all 15 test-combination metrics and predictions, gate files, hashes, and logs.

**Success criteria:** same fixed split and architecture metadata; no NaN/error; selected model-state hash is reproducible if determinism is exact, otherwise full-modality and core no-T1ce metrics remain within a predeclared tolerance such as absolute BAC `0.03`; all missing modalities have gate weight zero.

**If it fails, check first:** commit/config/split mismatch; nondeterministic CUDA kernels; data preprocessing or package version drift; modality and mask order; checkpoint selection status; unique output path.

Do not treat bitwise hash mismatch alone as model failure across different CUDA stacks. Record it and compare per-case probabilities and metrics.

## Phase 2: reproduce training stability and calibration choices

**Goal:** isolate the contribution of class weighting and threshold policy while holding model predictions fixed wherever possible.

**Order:**

1. unweighted CE + threshold 0.5;
2. weighted CE + threshold 0.5;
3. weighted CE + global validation calibration;
4. weighted CE + two-group validation calibration;
5. weighted CE + three-group validation calibration.

The cheapest rigorous design is:

- Train the unweighted and weighted models once each.
- For the weighted model, fit global/two-group/three-group thresholds offline from the same saved validation probabilities. Do not retrain for calibration-only conditions.
- Keep the test probabilities identical across the four weighted threshold conditions; only decision thresholds may differ.

**Commands after the planned configs/offline calibration script are approved:**

```bash
for CFG in unweighted_fixed05 weighted_fixed05; do
  OUT="outputs/recovery/${CFG}_seed42"
  mkdir -p "$OUT"
  CUDA_VISIBLE_DEVICES=0 python train.py \
    --config "configs/recovery/${CFG}.yaml" \
    --seed 42 --output-dir "$OUT" 2>&1 | tee "$OUT/console.log"
done

python scripts/calibrate_saved_predictions.py \
  --run outputs/recovery/weighted_fixed05_seed42 \
  --modes fixed_05 global grouped_2way_t1ce grouped_3way_t1ce_t1 \
  --output-dir outputs/recovery/weighted_calibration_seed42
```

`scripts/calibrate_saved_predictions.py` is a planned small utility, not present in the current repository. Implementing it is preferable to retraining four identical weighted models.

**Expected outputs:** one common validation/test probability table per trained model; threshold JSON per policy; one metric table joining all five conditions; confusion matrices; prediction-positive rate by combination.

**Success criteria:** class weighting comparison uses identical data/model settings; threshold comparisons use byte-identical probability tables; no test label is read during calibration; thresholds stay within safe bounds; collapse patterns are explicitly reported.

**If it fails, check first:** accidental retraining/config drift; probabilities regenerated from different checkpoints; calibration using logits rather than softmax positive probabilities; single-class subgroups; extreme first-max tie behavior; wrong group/mask order.

## Phase 3: diagnose the no-T1ce/with-T1 failure mode

**Goal:** determine whether poor BAC comes from discrimination failure, a logit shift, subgroup threshold instability, or checkpoint drift.

**Config/model:** use the frozen weighted model from Phase 2. No new training.

**Planned command:**

```bash
python scripts/analyze_calibration_drift.py \
  --run outputs/recovery/weighted_fixed05_seed42 \
  --combos t2_t1 t1_flair t2_t1_flair \
  --reference-combos t2 flair t2_flair \
  --output-dir outputs/recovery/calibration_drift_seed42
```

This analysis utility is also planned, not currently present.

**Expected outputs:** per-case labels/probabilities; positive-rate and confusion table; class-conditional probability histograms/ECDFs; subgroup counts and fallback reasons; validation-to-test threshold/score drift; patient-paired probability differences where combinations share cases.

**Success criteria:** every apparent collapse is classified as ranking failure (AUC), threshold shift (AUC acceptable but BAC poor), or both; unique patient counts are reported separately from combination-expanded counts.

**If it fails, check first:** missing per-case validation predictions; patient IDs not aligned across combinations; group threshold applied to the wrong mask; threshold search on the test set; all-positive/all-negative predictions hidden by aggregate AUC.

## Phase 4: validate the T1-interference hypothesis

**Goal:** perform a true paired inference ablation for `t2_t1`, `t1_flair`, and `t2_t1_flair`.

**Required protocol correction:** do not use the current `build_drop_t1_ablation_report` alone. It remaps to separately evaluated combinations with different group thresholds. Add an evaluator mode that, for each case, runs the same selected checkpoint twice, removes T1 only at the input availability mask/tensor, and applies the same predeclared threshold to both predictions.

**Planned command:**

```bash
python scripts/evaluate_drop_t1_paired.py \
  --config configs/recovery/weighted_group3.yaml \
  --checkpoint outputs/recovery/reference_seed42_<commit>/best.pt \
  --combos t2_t1 t1_flair t2_t1_flair \
  --threshold-policy freeze-original-group \
  --output-dir outputs/recovery/drop_t1_paired_seed42
```

**Inputs:** one checkpoint, same test cases, original tensors/masks, frozen original-group threshold. AUC is threshold-free; BAC/sensitivity/specificity must use exactly the same threshold for each original/drop pair.

**Expected outputs:** paired per-case probabilities, original/drop predictions, AUC/BAC/sensitivity/specificity deltas, bootstrap confidence intervals, and a secondary table showing normal operational group re-dispatch only if clearly labeled.

**Success criteria:** identical model hash and case order; only T1 tensor and availability bit change; paired outputs are complete; the conclusion is based on consistent direction across target combinations and seeds, not one aggregate metric.

**If it fails, check first:** preprocessing changed more than T1; different thresholds were used; cases are misaligned; results are driven by a handful of validation/test negatives. Record that the current simulated ablation intentionally keeps ROI masks fixed because they were built from all physically present modalities before T1 is zeroed; separately test a strict physically-missing-input scenario if that matches deployment.

## Phase 5: validate mask-aware node fusion

**Goal:** measure the incremental effect of learned gating and the fixed T1 penalty.

**Conditions:**

1. Existing simple shared mean+max fusion (`fusion_shared.yaml`);
2. mask-aware gating with penalty `0.0` (`fusion_gate.yaml`);
3. mask-aware gating with penalty `1.0` (`fusion_gate_penalty.yaml`).

Disable the mask-aware classifier in all three conditions. Otherwise classifier-mask bias can be mistaken for a node-fusion effect. If a literal arithmetic-average baseline is required, it needs a small reviewed implementation because the current simple path combines modality mean and max through `shared_projector`.

**Command template:**

```bash
for SEED in 42 43 44; do
  for CFG in fusion_shared fusion_gate fusion_gate_penalty; do
    OUT="outputs/recovery/${CFG}_seed${SEED}"
    mkdir -p "$OUT"
    CUDA_VISIBLE_DEVICES=0 python train.py \
      --config "configs/recovery/${CFG}.yaml" \
      --seed "$SEED" --output-dir "$OUT" 2>&1 | tee "$OUT/console.log"
  done
done
```

**Expected outputs:** nine complete runs, same split per seed across conditions, all 15 combinations, real gate exports for gated conditions, mean/SD and patient-level paired bootstrap CIs.

**Success criteria:** gating improves the predeclared no-T1ce primary metric without unacceptable full-modality loss; the penalty adds benefit beyond gating alone; unavailable gates remain zero; conclusions are consistent across seeds.

**If it fails, check first:** mask-aware classifier accidentally enabled; split differs within a seed; penalty applied in the no-penalty condition; checkpoint policies differ; gate saturation; branch/config hash mismatch.

## Phase 6: choose the next research direction

**Goal:** make one evidence-based decision, without expanding architecture prematurely.

Use this decision rule:

| Evidence | Decision |
|---|---|
| Drop-T1 helps and gating+penalty consistently beats gating alone | Retain a condition-specific T1 reliability prior; then tune/regularize it on validation only. |
| Drop-T1 helps but fixed penalty is unstable | Replace the fixed penalty with a more general learned reliability mechanism, with explicit regularization and ablation. |
| Gating helps but penalty does not | Keep mask-aware gating; remove the hand-coded penalty. |
| Neither helps | Return to the simpler anatomy-only shared fusion baseline and focus on training/calibration robustness. |
| Results vary strongly across seeds | Do not add modules; improve split strategy, uncertainty estimation, and calibration sample efficiency first. |

**Required decision package:** one table of primary/secondary metrics over seeds, paired confidence intervals, calibration/fallback counts, full-modality guardrail results, and a short recommendation tied to the rule above.

## Recommended first real-data action

After Phase 0, rerun the exact protocol-v2 curriculum reference at seed 42 in a new immutable directory. This validates the code/data/environment chain. The first new scientific experiment should then be the paired same-threshold drop-T1 inference ablation because it requires no retraining and directly tests the hypothesis that motivated the penalty.
