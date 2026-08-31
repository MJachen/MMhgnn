# Project Recovery Report

> 中文摘要：项目已经完成单个固定 seed 上可追溯的 anatomy-only 真实数据基线，并完成一次带验证保护条件的 no-T1ce 定向微调尝试。当前尚未证明 drop-T1、mask-aware gating 或固定 T1 penalty 带来受控的真实数据收益，因此下一阶段应先做协议复现和成对消融，而不是继续增加新模型模块。

## Executive conclusion

The project has reached a traceable single-seed real-data baseline for arbitrary missing-modality glioma grading using five fixed anatomy nodes, weighted cross-entropy, mask-aware fusion, and three-group validation calibration. The latest targeted no-T1ce fine-tuning was genuinely executed but rejected by a full-modality guardrail. The next step is controlled validation of the simpler baselines, real-data drop-T1 inference, and mask-fusion/penalty ablations, not a new architecture.

## Repository state

Review date: 2026-07-13.

| Item | Observed state | Evidence |
|---|---|---|
| Current branch | `main`, tracking `origin/main` | `git status --short --branch` |
| Current HEAD | `2d87862` (`Add public BraTS data configs`) | `git log --all` |
| Newer branch | `codex/safe-calibration-diagnostics` at `1471a6b` | local and origin branch refs |
| Tracked modifications/deletions | None | `git status --short` |
| Untracked work | Research notes, literature matrix, reports, `scripts/generate_experiment_report.py`, and `tmp/` | `git status --short` |
| Potentially valuable deleted tracked files | None found | Git status and tracked diff |
| Latest outputs match current HEAD? | No. `protocol_v2_*` used branch commit `1471a6b`, which is two commits ahead of `main` | branch diff, resolved configs, protocol audit files |

The untracked files were not modified. They should be classified and committed or archived later, but they do not currently change runtime behavior.

## Project goal and research logic

### Task

The project is a missing multi-modal brain MRI binary classification system. Inputs are some or all of `T2`, `T1ce`, `T1`, and `FLAIR`, plus a tumor segmentation used to define anatomy. Labels are LGG=`0` and HGG=`1`. Output is `P(HGG)` plus a thresholded class and evaluation artifacts.

### Role of the hypergraph

The hypergraph is not used to infer which modalities are present. It propagates information among five spatially related tumor/context regions after modality features have already been pooled and fused. Fixed hyperedges encode lesion interior, boundary, peri-tumoral progression, and lesion-background context. Evidence: `models/hybrid_hypergraph.py:ANATOMY_HYPEREDGES`, `_build_anatomy_incidence`, and `HybridHypergraphClassifier.forward`.

### Current hypothesis versus early hypothesis

The early design used seven ROIs, prototype nodes, prior/modal/dynamic-KNN branches, and a more complex two-stage graph. The current implementation deliberately contracts the topology to five segmentation-derived anatomy nodes and five fixed anatomy hyperedges. Missing-modality handling is moved into training sampling, a node-level gate, a mask-aware classifier bias, and threshold calibration.

The resulting research chain is:

1. Reduce graph complexity to make comparisons stable and interpretable.
2. Address class imbalance and threshold collapse before adding representation modules.
3. Model combination-dependent logit shifts with grouped calibration.
4. Test whether T1 is harmful specifically when T1ce is absent.
5. If the effect is real, let node-level fusion use the missingness mask and optionally penalize T1 in that condition.
6. Retain only mechanisms that improve controlled real-data comparisons without damaging full-modality performance.

## Current end-to-end pipeline

For a complete diagram, see `docs/current_pipeline.md`.

1. `datasets/brats_dataset.py` scans case folders, reads labels, selects an available-modality combination, loads volumes, and zero-fills unavailable modalities.
2. `build_acp_masks` derives `core`, `boundary`, `peri_inner`, `peri_outer`, and `distal_normal` from `seg > 0`, an internal distance transform, radius-3/radius-7 dilation, and an estimated brain mask.
3. A modality-specific `Light3DBackbone` produces features. Masked mean and max pooling creates one feature per modality and anatomy region.
4. Default mask-aware fusion scores every modality separately for every anatomy node. Missing modalities get zero final weight; T1 receives a fixed score penalty when T1ce is absent.
5. The five fused nodes pass through the fixed anatomy incidence matrix and `HGNNStack`.
6. Attention pooling creates one graph representation; an MLP classifier produces two logits. An optional availability-mask bias head adjusts the logits.
7. Softmax produces `P(HGG)`.
8. Training uses weighted cross-entropy with weights derived only from the training split.
9. The selected checkpoint predicts the validation set for calibration. The default three groups are `has_t1ce`, `no_t1ce_no_t1`, and `no_t1ce_with_t1`.
10. Frozen group thresholds are dispatched to all 15 test combinations. The evaluator saves metrics, per-case predictions, attention, edge/ROI diagnostics, and gating summaries.

## Audit of the five iterations

### 1. Anatomy-only contraction

**State: implemented in `main` and trained on real data.**

- Nodes are exactly `core`, `boundary`, `peri_inner`, `peri_outer`, and `distal_normal` (`datasets/brats_dataset.py:ROI_NAMES`; `configs/default.yaml:model.roi_names`).
- Hyperedges are `[0,1,2]`, `[1,2,3]`, `[2,3,4]`, `[0,2,4]`, and `[0,1,2,3]` (`models/hybrid_hypergraph.py:ANATOMY_HYPEREDGES`).
- Prototype node count is zero, prototype flags must be false, and enabling them raises an exception.
- Dynamic KNN is absent from the current forward path. The old `knn` key remains only as compatibility metadata.
- A legacy local modal-hypergraph fusion path remains. It is not the default, and it is separate from topology among anatomy nodes.
- Anatomy-only is the required architecture, not merely an optional experiment. Default graph config is five nodes/five edges.
- The top section of `README.md` is stale and conflicts with the code. The later anatomy-only README material, code, current configs, and protocol-v2 outputs agree and are more authoritative.

### 2. Class-balanced loss and global calibration

**State: implemented and exercised on real data; controlled incremental comparison is missing.**

- Current loss is `nn.CrossEntropyLoss`, normally with inverse-frequency class weights. BCE and focal loss are not implemented (`train.py:criterion`; `utils/training.py:class_weights_from_records`).
- Weights are calculated from `splits["train"]`, so they are fold/split-specific and do not use validation/test labels.
- The global threshold search optimizes balanced accuracy by default. Main commit `2d87862` searches 201 points over `[0,1]`; branch commit `1471a6b` constrains the range to `[0.05,0.95]` and uses a closest-to-0.5 tie-break for protocol-v2.
- Calibration uses validation probabilities. Test labels are not used to choose thresholds in the inspected training path.
- Results are saved to `metrics/threshold_calibration.json` and embedded in the selected checkpoint.
- Fixed `0.5` remains available when calibration is disabled and appears in historical fixed-threshold evaluation packages.
- No direct test leakage was found. The main uncertainty is repeated use of the same validation patients across combinations and subsequent model-development decisions on this one split.

### 3. Grouped calibration

**State: three-group implementation and real-data execution complete; robustness evidence is single-split.**

- Group assignment is implemented in `utils/metrics.py:calibration_group_from_mask` and dispatched by `threshold_dispatch_for_combo`.
- The current default is `grouped_3way_t1ce_t1` (`configs/default.yaml:calibration.threshold_mode`).
- The older two-group `grouped_has_t1ce` dispatch remains for compatibility, but no current training-time two-group fitting function exists. A YAML mode change alone would fall through to global calibration.
- Main falls back when a group is too small or single-class. Branch `1471a6b` additionally requires minimum total, positive, and negative counts.
- Protocol-v2 thresholds were: has T1ce `0.7205`; no T1ce/no T1 `0.3065`; no T1ce/with T1 `0.3020`. None fell back.
- `no_t1ce_with_t1` falls back in `outputs/acp_demo/demo_summary.json`, but that is a synthetic one-case condition. It did not fall back in the real protocol-v2 validation data.
- Protocol-v2 group counts of 288/108/144 are repeated case-combination observations from 36 unique validation patients. They must not be described as independent patient counts.

### 4. Drop-T1 inference ablation

**State: reporting helper exists; only synthetic output exists; current implementation is not a strict same-threshold ablation.**

- The helper covers `t2_t1 -> t2`, `t1_flair -> flair`, and `t2_t1_flair -> t2_flair`; it also maps `t1 -> none` as not applicable (`utils/metrics.py:DROP_T1_ABLATION_REMAP`).
- It does not retrain and does not directly modify gate weights. It reuses metrics from separately evaluated modality combinations.
- The same selected model can underlie both combinations, but each combination receives its own grouped threshold before the report is built. Thus BAC, sensitivity, and specificity do not isolate the input effect under an identical threshold.
- Only `outputs/acp_demo/drop_t1_ablation*` exists. No corresponding files were found under `server_results` or `privatedata`.
- The demo supports only that the report format and remapping execute. It cannot establish that T1 harms real patients or quantify a real metric change.

### 5. Mask-aware node fusion

**State: implemented and executed on real data; causal benefit is not yet established.**

- One shared `modality_gate` MLP is used for every node/modality pair.
- Inputs include the modality-specific node feature, full availability-mask embedding, and optional node-type embedding.
- Scores are produced separately at each anatomy node.
- Missing scores are set to `-1e9` before softmax, then masked and renormalized; final unavailable weights are zero.
- `no_t1ce_t1_penalty` is a fixed configuration scalar, default `1.0`, triggered only when T1 is available and T1ce is unavailable.
- Real protocol-v2 gates show average T1 below T2/FLAIR for the three target combinations, but this is partly enforced by construction and is not evidence of improved classification.
- Gates are saved per test combination in `test/<combo>/gating_stats.csv`.
- Legacy shared mean+max and local modal-HGNN fusion paths remain. A clean true-average baseline is not implemented as a named option.
- The default mask-aware classifier bias head is an additional confounder when attributing gains to node fusion.

## Git and experiment timeline

| Time / commit | Confirmed change | Problem addressed | Retained now | Matching result |
|---|---|---|---|---|
| 2026-06-16 `949b61e` | Initial squashed project import, including the current five-node code plus older configs/artifacts | Established project codebase | Partly; commit mixes several historical stages | Local demos and historical outputs, with mixed provenance |
| 2026-06-16 `13473ab` | Added server evaluation/packing and result-fetch scripts | Move real-run evidence back to local workspace | Yes | `server_results/*` packages |
| 2026-06-16 `2d87862` | Added public BraTS base/local/server configs and dataset configuration checker | Standardize public dataset paths | Current `main` HEAD | Public BraTS result packages |
| 2026-06-18 `699e608` | Added safer calibration constraints, public curriculum and no-T1ce configs, per-case diagnostics | Prevent extreme threshold/tie behavior and support controlled missingness runs | Only on newer branch | `public_calib_safe_seed42`, protocol-v1 curriculum/targeted runs |
| 2026-06-24 `1471a6b` | Added fixed split, joint validation guardrails, checkpoint/prediction hashes, comparison script, tests, and protocol document | Prove fine-tuning occurred and prevent a weak no-T1ce gain from silently replacing a better base model | Only on newer branch | `protocol_v2_curriculum_seed42`, `protocol_v2_no_t1ce_focus_seed42` |

The five conceptual iterations are not cleanly represented as five commits because the initial commit is a large import. Their actual state was therefore determined from code and artifacts, not commit messages alone.

## Current real-data protocol

- Dataset: public BraTS case folders configured under `/home/cjc/brats2020/brats2021`, labels from `brats2021_label2020.xlsx`.
- Total: 365 cases, 76 LGG and 289 HGG.
- Fixed seed-42 split: train 292 (61/231), validation 36 (7/29), test 37 (8/29).
- No cross-validation.
- Training: missing-modality curriculum; default stage ratios progress from full only, to 70% full/30% single-missing, then 40/30/20/10 full/single/double/triple missing.
- Base checkpoint selection: joint validation code exists in commit `1471a6b`; protocol-v2 selected the base checkpoint.
- Calibration: validation-only, all 15 modality combinations, three threshold groups.
- Test: all 15 non-empty combinations.
- Metrics: ACC, ROC AUC, F1, sensitivity, specificity, balanced accuracy, and confusion counts.
- Uncertainty: no valid current mean/SD or confidence intervals. One seed cannot support them.

## Main risks and inconsistencies

| Severity | Risk | Consequence | Evidence / action |
|---|---|---|---|
| High | Latest protocol code is not on `main` | Re-running `main` can silently use weaker checkpoint selection and calibration behavior | Pin/review branch `1471a6b` before any rerun. |
| High | No real same-threshold drop-T1 ablation | T1-interference hypothesis remains untested | Implement a paired per-case inference path after approval. |
| High | Fusion and penalty have no controlled ablation | Gate exports prove execution but not benefit | Compare shared baseline, gating, and gating+penalty on identical splits/seeds. |
| High | Single fixed split and only 7 validation negatives | Thresholds and guardrails may be unstable | Add multiple seeds or stratified CV and patient-level bootstrap CIs after exact reproduction. |
| Medium | Repeated validation cases across 15 combinations | Group `num_samples` overstates independent information | Report unique patients and use clustered/patient-level uncertainty. |
| High | Simulated missingness is applied after brain/anatomy masks are built from all physically present modalities | ROI masks are combination-invariant, but simulated missing cases retain spatial support from modalities declared absent; real physical absence may be harder | Decide whether segmentation plus a modality-independent brain mask is an explicit deployment assumption, or rebuild support using only available inputs. |
| Medium | Main's drop-T1 helper uses different group thresholds | Threshold metrics confound input removal with decision policy | Force one frozen threshold for paired analysis. |
| Medium | Mask-aware classifier is on with mask-aware fusion | Fusion attribution is confounded | Disable the classifier bias in the clean fusion ablation. |
| Medium | Historical packages omit checkpoints/configs | Some prior results are not independently reproducible | Treat protocol-v2 as the source of truth. |
| Low | README and old ablation configs describe removed topology | New users can restart from an obsolete design | Update documentation only after branch/version decision. |

No modality-order bug was found in the inspected path: dataset tensors use `[t2,t1ce,t1,flair]`, while model/calibration helpers explicitly reorder into `[t1,t1ce,t2,flair]`. This remains an area for tests because two orders coexist.

## Status classification

### A. Completed and exercised on real data

- Public BraTS scanning, labels, fixed seed-42 train/validation/test split.
- Five-node/five-edge anatomy-only model with no prototypes or dynamic KNN.
- Weighted cross-entropy using train-only class frequencies.
- Missing-modality curriculum and evaluation over all 15 combinations.
- Validation-only three-group calibration with test dispatch.
- Mask-aware node fusion, T1 penalty execution, classifier mask bias, and real gate export.
- Auditable protocol-v2 checkpoint selection, model hashes, and prediction hashes on branch `1471a6b`.
- Targeted no-T1ce fine-tuning execution and guardrail-based rejection.

These are stable implementation/execution conclusions. They are not all stable claims of performance benefit.

### B. Implemented but lacking adequate real-data validation

- Benefit of weighted CE relative to unweighted CE.
- Benefit of global, two-group, and three-group calibration under identical model predictions.
- Benefit of mask-aware node fusion over the simpler shared fusion path.
- Benefit and optimal value of `no_t1ce_t1_penalty`.
- Drop-T1 report mechanics; real paired inference is absent.
- Generalization across seeds/splits and subgroup calibration stability.

### C. Not completed

- A strict real-data, same-model, same-threshold drop-T1 ablation with per-case paired outputs.
- Clean fusion ablation: simple baseline versus gating versus gating+penalty, with mask-aware classifier controlled.
- Multi-seed or cross-validation summary, standard deviations, confidence intervals, and paired statistical analysis.
- A consolidated reproducibility manifest binding commit, resolved config, split hash, checkpoint hash, and result hash for every historical run.
- Evidence supporting a new completion module or more general reliability-aware gate.

### D. Paths that should not drive new work

- Seven-ROI/prototype/dynamic-KNN README description and `configs/ablation*` branch combinations.
- `outputs/demo_run`, `outputs/smoke_result.json`, and `outputs/summary_mean_std.csv` for current-model conclusions.
- Synthetic `outputs/acp_demo` metrics as evidence of clinical/real-data performance.
- Protocol-v1 curriculum-versus-targeted comparison, whose selected predictions were identical without a sufficient audit trail.
- Direct reruns from `main` while assuming they reproduce protocol-v2.

## Answer to the recovery question

> The project has completed a traceable single-seed real-data anatomy-only baseline and a guarded no-T1ce fine-tuning attempt; it has not yet demonstrated that drop-T1, mask-aware gating, or the T1 penalty improves performance in a controlled real-data comparison.

## Decisions required before implementation

1. Whether commit `1471a6b` should be reviewed and merged into `main`, or used as a pinned recovery branch.
2. Whether the first clean fusion baseline should be the existing shared mean+max path or a newly implemented true arithmetic-mean path.
3. Whether the next evidence standard is three/five fixed seeds or stratified cross-validation; the latter is more expensive but better suited to only seven validation negatives in the current split.
4. Whether drop-T1 threshold metrics should use the original `no_t1ce_with_t1` threshold for both paired predictions, with operational re-dispatch reported only as a secondary analysis.
