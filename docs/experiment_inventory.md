# Experiment Inventory

> 中文摘要：当前最可信的实验是两个 `protocol_v2_*_seed42` 真实数据运行。基础 curriculum 运行完整可追溯；定向 no-T1ce 微调确实改变了权重，但因全模态 BAC 下降超过保护阈值而未被选中。真实 drop-T1 实验尚不存在，`outputs/acp_demo` 只能证明流程可运行。

## Status legend

- **REAL / complete**: training or evaluation artifacts from a real dataset are present, with enough metadata to interpret the run.
- **REAL / incomplete provenance**: metrics look like a real run, but checkpoint, resolved configuration, split, or log is missing.
- **SYNTHETIC**: generated tensors or fabricated demonstration metrics; useful only for mechanics.
- **FAILED protocol objective**: execution completed, but the candidate did not satisfy its predeclared selection criteria.
- **OBSOLETE**: belongs to the earlier seven-ROI/dynamic-graph design and does not describe the current model.

## Recognized runs and result sets

| Run or artifact | Data / split / seed | Model and training | Calibration / combinations | Artifacts | Status and reproducibility |
|---|---|---|---|---|---|
| `outputs/acp_demo` | One hand-built synthetic case; no split | Current five-node mechanics, one optimizer step; `scripts/demo_acp_smoke.py` | Synthetic three-group and drop-T1 reports | `demo_summary.json`, gate and drop-T1 CSV/JSON | **SYNTHETIC**. Reproducible as a smoke demo, but metric fields in the script are constructed demonstration values and are not scientific evidence. |
| `outputs/demo_run` | Older local/demo dataset metadata; split files present | Seven-ROI prototype/KNN-era resolved config | Incomplete | `resolved_config.json`, split/overlay files; no checkpoint and no complete metrics | **OBSOLETE / incomplete**. Do not use as the current baseline. |
| `outputs/smoke_result.json` | Single real-looking case | Seven-ROI smoke path | None | One JSON | **OBSOLETE / incomplete provenance**. Not comparable to current five-node model. |
| `outputs/summary_mean_std.csv` | Seeds/runs are not traceable from adjacent artifacts | Seven old branch names such as `knn_only`, `modal_knn`, and `prior_modal_knn` | Not fully recorded | Aggregated CSV only | **OBSOLETE / non-reproducible**. No matching checkpoints, resolved configs, and run logs in the package. |
| `privatedata/metrics` | Private three-modality-looking dataset; test appears to contain 24 cases; split absent | Five-node/mask-aware-looking metrics; training history shows 189 training labels | Three-group thresholds; seven `T1/T1ce/T2` combinations | Five metric/history files; no config, checkpoint, split, or console log | **REAL-looking / incomplete provenance**. Do not use for a paper-level comparison until dataset identity and config are recovered. |
| `server_results/brats2020_public_eval` | Public BraTS, seed-42-like split package | Anatomy-only, weighted CE, mask-aware fusion | Fifteen combinations; initial calibration | 261 evaluation files, metrics, gates, one log, one split summary; no checkpoint/resolved config | **REAL evaluation / incomplete package**. Useful as historical evidence, not standalone reproduction. |
| `server_results/public_eval_current_calibrated` | Same public evaluation set | Same selected model as the initial public package | Current-at-that-time calibrated thresholds | 276 files; no checkpoint/resolved config | **REAL evaluation / incomplete package**. Threshold comparison run. |
| `server_results/public_eval_fixed_05` | Same public evaluation set | Same selected model | Fixed threshold `0.5` | 276 files; no checkpoint/resolved config | **REAL evaluation / incomplete package**. Useful only for threshold sensitivity. |
| `server_results/public_eval_fixed_01` | Same public evaluation set | Same selected model | Fixed threshold `0.1` | 276 files; no checkpoint/resolved config | **REAL evaluation / incomplete package**. Useful only for threshold sensitivity. |
| `server_results/public_calib_safe_seed42` | Public BraTS, seed 42 | Anatomy-only and mask-aware model | Safe threshold bounds/tie-breaking from commit `699e608` | 276 files; no checkpoint/resolved config | **REAL evaluation / partially reproducible**. Code exists on the newer branch; package itself lacks the checkpoint. |
| `server_results/missing_curriculum_seed42` | Public BraTS, seed 42 | Missing-modality curriculum | Three-group calibration, all 15 combinations | 276 evaluation files; no checkpoint/resolved config | **REAL / completed, but protocol-v1 evidence is superseded**. |
| `server_results/no_t1ce_focus_seed42` | Public BraTS, seed 42 | Intended targeted no-T1ce fine-tuning | Three-group calibration, all 15 combinations | 552 files due to a duplicated nested output tree; no packaged checkpoint/config | **REAL execution / invalid comparison**. Metrics and predictions were byte-identical to the protocol-v1 curriculum run; this exposed a selection/output protocol problem. High overwrite/nesting risk. |
| `server_results/outputs/protocol_v2_curriculum_seed42` | Public BraTS, fixed seed-42 split: 292/36/37 | Anatomy-only, weighted CE, missing curriculum, no targeted fine-tune | Safe three-group calibration; all 15 combinations | 284 files; `base_best.pt`, `best.pt`, resolved config, split summary, logs, hashes, predictions, gates | **REAL / complete single-seed reference**. Most traceable current result. |
| `server_results/outputs/protocol_v2_no_t1ce_focus_seed42` | Exactly the same fixed split and base initialization | Same base training plus targeted no-T1ce fine-tuning | Joint validation guardrails, then same three-group test evaluation | 285 files; `base_best.pt`, `finetune_best.pt`, `best.pt`, resolved config, logs, hashes | **REAL / execution complete; FAILED protocol objective**. Fine-tuned weights changed, but candidate was rejected and `base_best.pt` was selected. |

No `ERROR`, traceback, NaN, or interrupted-run marker was found in the protocol-v2 logs reviewed. Earlier fetched evaluation packages are not complete training archives, so absence of an error there is weaker evidence.

## Latest controlled protocol

Source files:

- `server_results/outputs/protocol_v2_curriculum_seed42/resolved_config.json`
- `server_results/outputs/protocol_v2_curriculum_seed42/split_summary.json`
- `server_results/outputs/protocol_v2_curriculum_seed42/metrics/*`
- `server_results/outputs/protocol_v2_no_t1ce_focus_seed42/metrics/checkpoint_selection.json`
- `server_results/outputs/protocol_v2_compare_seed42.txt`

### Data and split

| Partition | Total | LGG / class 0 | HGG / class 1 |
|---|---:|---:|---:|
| Train | 292 | 61 | 231 |
| Validation | 36 | 7 | 29 |
| Test | 37 | 8 | 29 |
| Total | 365 | 76 | 289 |

This is one fixed 80/10/10 split at seed 42, not cross-validation. The configured split path is `outputs/protocol/brats2020_public_split_seed42.json`; the fetched run includes a `split_summary.json` but not that original split JSON.

### Complete protocol-v2 test results

The selected curriculum and targeted-run test predictions are identical because the targeted candidate was not eligible and both selected the same base model state. Values below therefore describe the selected base checkpoint once.

| Available modalities | Threshold group | Threshold | AUC | BAC | Sensitivity | Specificity |
|---|---|---:|---:|---:|---:|---:|
| `t2` | no T1ce, no T1 | 0.3065 | 0.677 | 0.500 | 1.000 | 0.000 |
| `t1ce` | has T1ce | 0.7205 | 0.832 | 0.750 | 1.000 | 0.500 |
| `t1` | no T1ce, with T1 | 0.3020 | 0.500 | 0.500 | 1.000 | 0.000 |
| `flair` | no T1ce, no T1 | 0.3065 | 0.409 | 0.556 | 0.862 | 0.250 |
| `t2_t1ce` | has T1ce | 0.7205 | 0.828 | 0.812 | 1.000 | 0.625 |
| `t2_t1` | no T1ce, with T1 | 0.3020 | 0.621 | 0.500 | 1.000 | 0.000 |
| `t2_flair` | no T1ce, no T1 | 0.3065 | 0.534 | 0.550 | 0.724 | 0.375 |
| `t1ce_t1` | has T1ce | 0.7205 | 0.832 | 0.750 | 1.000 | 0.500 |
| `t1ce_flair` | has T1ce | 0.7205 | 0.832 | 0.795 | 0.966 | 0.625 |
| `t1_flair` | no T1ce, with T1 | 0.3020 | 0.418 | 0.608 | 0.966 | 0.250 |
| `t2_t1ce_t1` | has T1ce | 0.7205 | 0.828 | 0.812 | 1.000 | 0.625 |
| `t2_t1ce_flair` | has T1ce | 0.7205 | 0.832 | 0.778 | 0.931 | 0.625 |
| `t2_t1_flair` | no T1ce, with T1 | 0.3020 | 0.599 | 0.616 | 0.483 | 0.750 |
| `t1ce_t1_flair` | has T1ce | 0.7205 | 0.832 | 0.778 | 0.931 | 0.625 |
| `t2_t1ce_t1_flair` | has T1ce | 0.7205 | 0.828 | 0.778 | 0.931 | 0.625 |

Evidence: `server_results/outputs/protocol_v2_curriculum_seed42/metrics/test_metrics.csv`.

The test results show two distinct failure modes:

1. AUC can be above chance while threshold metrics collapse to all-positive predictions (`t2` and `t2_t1` have BAC 0.5, sensitivity 1, specificity 0).
2. Validation estimates are optimistic or unstable relative to the 37-case test set. For example, full-modality validation BAC/AUC were 0.911/0.951, while test BAC/AUC were 0.778/0.828.

### Targeted fine-tuning decision

| Validation quantity | Fine-tuned minus base | Guardrail | Passed |
|---|---:|---:|---|
| Mean no-T1ce AUC | +0.0066 | >= 0 | Yes |
| Mean no-T1ce BAC | +0.0714 | >= 0 | Yes |
| Full-modality BAC | -0.0887 | drop no worse than -0.03 | **No** |

The candidate checkpoint has a different model-state SHA256, proving that fine-tuning occurred. The selected checkpoint remained `base_best.pt`; therefore identical selected predictions are expected and are not evidence that the training code did nothing.

## Evidence-level conclusions

| Question | Current answer |
|---|---|
| Anatomy-only baseline trained on real data? | **Yes**, in the protocol-v2 runs. |
| Class-balanced loss exercised on real data? | **Yes**, weighted CE with train-only inverse-frequency weights. No clean unweighted controlled comparison yet. |
| Three-group calibration exercised on real data? | **Yes**, with no fallback in protocol-v2. Only one split is available. |
| Drop-T1 inference ablation run on real data? | **No**. Only `outputs/acp_demo` contains drop-T1 files. |
| Mask-aware node fusion exercised on real data? | **Yes**, and gates were exported. Its causal gain over simpler fusion is unverified. |
| T1 penalty exercised on real data? | **Yes**. Its benefit is unverified because no penalty-off controlled run exists. |
| Mean and standard deviation available for the current model? | **No**. The current traceable protocol has one seed. The old `summary_mean_std.csv` is architecture-mismatched. |
| Confidence intervals available? | **No**. |

## Result integrity risks

- The protocol-v1 curriculum and targeted outputs were identical, and one targeted directory contains a duplicated nested copy. New runs must use unique immutable output directories.
- Earlier server packages omit checkpoints and resolved configs. They cannot independently establish model/config identity.
- `main` does not contain the protocol code that generated the newest results. Reproduction must pin commit `1471a6b` or merge it deliberately after review.
- Group calibration concatenates repeated predictions of the same validation patients across modality combinations. This is validation-only, so it is not test leakage, but the reported group counts are not independent sample counts.
- Current drop-T1 reporting compares combinations using different grouped thresholds. It cannot isolate the effect of removing T1 on BAC, sensitivity, or specificity.
- No current multi-seed or patient-level bootstrap uncertainty estimates exist.
