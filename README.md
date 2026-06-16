# BraTS HGG/LGG Two-Stage Hybrid Dynamic Hypergraph Demo

这是一个面向 BraTS 数据集的 PyTorch demo，用于在不同模态缺失组合下进行 HGG/LGG 二分类。当前模型已经重构为**两阶段混合动态图超图框架**：先在 ROI 内部完成当前可用模态的动态聚合，再在 7 个共享 ROI 节点上进行 `prior + dynamic KNN` 的 ROI 层推理。

## 1. Two-Stage Hybrid Dynamic Hypergraph

### 阶段 1：模态层聚合

目标是对每个 ROI，在当前可用模态条件下聚合多模态观测，得到该 ROI 的共享表示。

步骤如下：

1. 每个模态分别经过轻量 3D CNN backbone。
2. 对每个 ROI、每个可用模态做 masked average pooling，得到 `x_r^m`。
3. 对单个 ROI 构造局部动态模态超图：
   - `e_r_modal = {ROI_r@m | m in available_modalities}`
   - `e_r_link = {ROI_r_shared} ∪ {ROI_r@m | m in available_modalities}`
4. 用轻量 HGNN 只在该 ROI 内部做传播，输出该 ROI 的共享节点表示 `z_r`。
5. 对 7 个 ROI 分别重复这一过程，得到 `Z_roi = [z_NCR, z_ET, z_TC, z_WT, z_Transition, z_ED_inner, z_ED_outer]`。

这一阶段只表达“同一 ROI 在当前可用模态下有哪些观测以及它们如何聚合”，不做 ROI 与 ROI 之间的大范围传播。

### 阶段 2：ROI 层混合动态图

输入是阶段 1 输出的 7 个共享 ROI 节点。

步骤如下：

1. 在 7 个 ROI 节点上固定构造医学先验超边 `H_prior`。
2. 基于当前样本的 `Z_roi` 动态构造 KNN 超边 `H_knn`。
3. KNN 只在 7 个共享 ROI 节点之间构造，默认使用 `cosine` 或 `euclidean`，`k=2/3`。
4. 组合得到 ROI 层超图：
   - `H_roi = H_prior + lambda_knn * H_knn`
   这里通过给 KNN incidence 加权实现动态修正。
5. 在 ROI 层超图上执行 1 到 2 层 HGNN。
6. 对 7 个 ROI 节点做 attention pooling，输出病例 embedding 和二分类 logits。

这一阶段体现的是：

- `H_prior` 提供稳定医学骨架
- `H_knn` 根据当前样本特征动态变化
- KNN 是对先验骨架的个体化补充，而不是与 prior 平行平均融合的主分支

## 2. Design Intuition

### 模态超边

- 表达同一 ROI 在当前可用模态下的观测关系
- 服务于缺失模态场景
- 属于样本条件动态图

### 先验超边

- 提供具有医学意义的结构骨架
- 提升可解释性
- 只在 ROI 层使用，不做过强干预

### KNN 超边

- 在 ROI 层根据当前 ROI 表示动态构造
- 捕捉当前病例下 ROI 之间的个体化相似关系
- 作为对先验骨架的动态补充

## 3. Data Paths

公开 BraTS 数据集的格式和路径由 YAML 配置控制。公共格式配置为：

```text
configs/brats2020_public_base.yaml
```

本地路径配置：

```text
configs/brats2020_public_local.yaml
```

服务器路径配置：

```text
configs/brats2020_public_server.yaml
```

当前服务器路径：

- `data_root = /home/cjc/brats2020/brats2021/`
- `label_xlsx = /home/cjc/brats2020/brats2021_label2020.xlsx`

当前本地路径：

- `data_root = E:/brats2021`
- `label_xlsx = E:/EXPS/tensorexps/brats2021_label2020.xlsx`

标签文件至少包含两列：

- `BraTS2021`
- `label`

标签约定：

- `1 = HGG`
- `0 = LGG`

## 4. ROI Nodes

默认 7 个 ROI 节点：

1. `NCR = seg == 1`
2. `ET = seg == 4`
3. `TC = NCR or ET`
4. `WT = NCR or ED or ET`
5. `Transition = (dilate(TC, r=2) - TC) ∩ ED`
6. `ED_inner = ED 中距离 TC 边界较近的一半`
7. `ED_outer = ED 中距离 TC 边界较远的一半`

## 5. Config Highlights

当前模型相关配置建议如下：

```yaml
model:
  roi_names: [NCR, ET, TC, WT, Transition, ED_inner, ED_outer]
  stage1_num_layers: 1
  stage2_num_layers: 2
  knn_k: 2
  knn_metric: cosine
  lambda_knn: 1.0
  learnable_lambda_knn: false
  branches:
    use_modal_edges: true
    use_prior_edges: true
    use_knn_edges: true
```

说明：

- `use_modal_edges=false` 时，阶段 1 退化为简单共享初始化，不做 ROI 内模态超图传播。
- `use_prior_edges/use_knn_edges` 控制阶段 2 的 ROI 层图组成。
- 这些开关现在是两阶段框架内的结构开关，不再是三条平行主分支的加权平均。

## 6. Basic Usage

安装依赖：

```bash
pip install -r requirements.txt
```

训练，固定四模态：

```bash
python train.py --config configs/brats2020_public_server.yaml
```

本地公开 BraTS 数据检查：

```powershell
python scripts/check_dataset_config.py --config configs/brats2020_public_local.yaml
```

服务器公开 BraTS 数据检查：

```bash
python scripts/check_dataset_config.py --config configs/brats2020_public_server.yaml
```

训练，随机缺失：

```bash
python train.py --config configs/random_missing.yaml
```

测试：

```bash
python evaluate.py --config configs/default.yaml --checkpoint outputs/demo_run/checkpoints/best.pt
```

指定模态组合测试，例如 `t2 + t1ce`：

```bash
python evaluate.py --config configs/default.yaml --checkpoint outputs/demo_run/checkpoints/best.pt --combo t2 t1ce
```

单病例解释：

```bash
python infer_case.py --config configs/default.yaml --checkpoint outputs/demo_run/checkpoints/best.pt --case-dir /home/cjc/brats2020/brats2021/BraTS2021_00000
```

## 7. Missing Modality Protocols

保留两种运行模式：

- `fixed_combo`
  - 训练和测试都使用指定模态组合
- `random_missing`
  - 训练时随机采样非空模态组合
  - 测试时可指定某个组合，或评估多个组合

模态顺序固定为：

```text
[t2, t1ce, t1, flair]
```

## 8. Explainability Outputs

保留并强化以下解释性输出：

- 每个病例 7 个 ROI 的 importance score
- 测试集平均 ROI importance 柱状图
- ROI dropping 平均概率下降
- 病例级 ROI overlay 图
- 单病例原始 BraTS 区域与 ROI 划分图

这些解释性结果现在主要落在**阶段 2 的 ROI 层节点**上，因此比之前的并联三分支结构更清晰。

## 9. Metrics

输出并保存：

- `ACC`
- `AUC`
- `F1`
- `SEN`
- `SPE`
- `confusion matrix`

## 10. Notes

- 当前框架已经从“prior/modal/knn 三分支并联加权求和”改为“两阶段混合动态图”。
- `cross-patient KNN` 仍为预留接口，当前默认不启用。
- 训练/测试/单病例推理命令保持不变，以尽量减少使用成本。

## Anatomy-Only Baseline, Curriculum Missing Training, and Threshold Calibration

The current default model is an anatomy-only hypergraph baseline. It uses five explicit anatomy/context nodes only:
`core`, `boundary`, `peri_inner`, `peri_outer`, and `distal_normal`.
Prototype nodes and prototype hyperedges are disabled by default and are not used in forward inference, metrics, or visual explanation outputs.
The fixed anatomy hypergraph contains five hyperedges:
`{core,boundary,peri_inner}`, `{boundary,peri_inner,peri_outer}`, `{peri_inner,peri_outer,distal_normal}`, `{core,peri_inner,distal_normal}`, and `{core,boundary,peri_inner,peri_outer}`.

Two training modes are supported through `train.mode`:

- `full_modality_train`: trains with all four modalities available. Testing still evaluates all 15 non-empty modality combinations when `eval.test_all_combos: true`.
- `missing_curriculum_train`: epoch-based curriculum training. Stage 1 uses full modality only, Stage 2 mixes full modality and single-missing batches, and Stage 3 mixes full, single-missing, double-missing, and triple-missing batches. The stage ratios and sampling proportions are configurable in YAML.

Class-balanced loss is controlled by:

```yaml
loss:
  loss_type: "weighted_ce"
  use_class_balanced_loss: true
  class_balance_mode: "inverse_frequency"
```

Training logs print train class counts and the computed CE class weights. After training, the best checkpoint is calibrated on the validation set only. The calibrated threshold is saved to `outputs/.../metrics/threshold_calibration.json` and into the checkpoint under `calibrated_threshold`. `evaluate.py` and `infer_case.py` automatically use this threshold when it is available.

Example commands:

```bash
# full-modality training, followed by 15-combo testing
python train.py --config configs/default.yaml

# missing-modality curriculum training, followed by 15-combo testing
python train.py --config configs/missing_curriculum.yaml

# evaluate a trained checkpoint on all 15 modality combinations
python evaluate.py --config configs/default.yaml --checkpoint outputs/demo_run/checkpoints/best.pt

# evaluate one specified modality combination
python evaluate.py --config configs/default.yaml --checkpoint outputs/demo_run/checkpoints/best.pt --combo t2 t1ce

# single-case inference and ROI visualization
python infer_case.py --config configs/default.yaml --checkpoint outputs/demo_run/checkpoints/best.pt --case-dir /home/cjc/brats2020/brats2021/BraTS2021_00000
```

To select a GPU, either set `CUDA_VISIBLE_DEVICES`, or put `device: "cuda:1"` under `train` in the YAML config.

## Experiment 1: Mask-Aware Classifier and Global Threshold Calibration

Experiment 1 keeps the anatomy-only graph unchanged and adds a minimal mask-aware classifier head to reduce decision-boundary shifts under missing modalities. The model still uses five explicit nodes and five anatomy hyperedges; prototype nodes and prototype hyperedges remain disabled.

The implemented classifier is the bias-head variant:

```text
base_logits = classifier(z_graph)
mask_embed = MLP(modality_mask)
logits = base_logits + bias_head(mask_embed)
```

The classifier mask order is fixed as `[t1, t1ce, t2, flair]`. Internally, the dataset still provides masks in the project modality order, and the model reorders them before the mask MLP so training and testing remain consistent.

Relevant config fields:

```yaml
model:
  use_mask_aware_classifier: true
  mask_head_type: "bias"
  mask_embed_dim: 8
  mask_order: ["t1", "t1ce", "t2", "flair"]

calibration:
  enable_threshold_calibration: true
  threshold_metric: "balanced_accuracy"
  threshold_mode: "global"
  default_threshold: 0.5
```

Training still calibrates a single global threshold on the validation set only and stores it in both `metrics/threshold_calibration.json` and the checkpoint. Evaluation and single-case inference automatically use `calibrated_threshold` when present.

## Targeted No-T1CE Fine-Tuning and Grouped Calibration

The current default training path keeps the full-modality anatomy-only + mask-aware backbone as Stage 1, then optionally runs a short targeted fine-tuning stage. This stage is enabled by `train.enable_targeted_finetune` and samples mostly full-modality anchor batches plus no-t1ce combinations that are clinically important for the current experiment.

Default fine-tune sampling:

```yaml
train:
  enable_targeted_finetune: true
  fine_tune_epochs: 8
  fine_tune_lr_scale: 0.1
  freeze_backbone_in_finetune: false
  fine_tune_sampling_mode: "targeted_no_t1ce"
  fine_tune_ratio_full: 0.7
  fine_tune_ratio_t2_only: 0.1
  fine_tune_ratio_flair_only: 0.1
  fine_tune_ratio_t2_flair: 0.1
  fine_tune_ratio_drop_t1ce_fullcontext: 0.0
```

Calibration supports `threshold_mode: grouped_3way_t1ce_t1`. During validation-only calibration, predictions are split into three groups: `has_t1ce`, `no_t1ce_no_t1`, and `no_t1ce_with_t1`. Testing dispatches thresholds by the evaluated modality combination through the same shared grouping function. If any validation group is too small or single-class, the code falls back to the global validation threshold and records the fallback reason.

## Drop-T1 Inference Ablation

A test-only diagnostic ablation is available through:

```yaml
test:
  enable_drop_t1_inference_ablation: true
  drop_t1_ablation_combos: ["t1", "t2_t1", "t1_flair", "t2_t1_flair"]
```

This does not retrain the model. It keeps the original 15-combo evaluation unchanged and adds a comparison table that asks whether dropping `t1` helps when `t1ce` is absent.

Remap rules:

- `t1` -> `none` (`not_applicable_after_drop_t1`)
- `t2_t1` -> `t2`
- `t1_flair` -> `flair`
- `t2_t1_flair` -> `t2_flair`

Important: the ablation branch reuses the calibration group and threshold of the **ablated combo**, not the original combo. For example, `t2_t1 -> t2` will use the threshold group of `t2`.

Outputs:

- `drop_t1_ablation.csv`
- `drop_t1_ablation.json`
- `drop_t1_ablation_summary.csv`

These files report baseline metrics, ablated metrics, threshold groups, applied thresholds, and deltas for `bal_acc`, `auc`, and `spe`.

