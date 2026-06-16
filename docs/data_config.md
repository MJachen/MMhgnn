# 数据配置说明

本项目的目标是让数据处理由 YAML 配置驱动。只要数据组织格式一致，训练、评估和单病例推理都不需要改 Python 代码。

## 1. 公开 BraTS2020/BraTS2021 格式

公开数据集目录假设为：

```text
<data_root>/
  BraTS2021_00000/
    BraTS2021_00000_t1.nii.gz
    BraTS2021_00000_t1ce.nii.gz
    BraTS2021_00000_t2.nii.gz
    BraTS2021_00000_flair.nii.gz
    BraTS2021_00000_seg.nii.gz
```

标签表需要包含：

```text
BraTS2021
label
```

公开数据格式由这个 base config 固定：

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

两者只区别于：

```yaml
data:
  data_root: ...
  label_xlsx: ...
```

## 2. 私有数据格式

私有数据当前使用：

```text
<data_root>/
  <case_id>/
    t2.nii.gz
    t1ce.nii.gz
    t1.nii.gz
    t2_mask.nii.gz
    t1ce_mask.nii.gz
    t1_mask.nii.gz
```

对应模板：

```text
configs/private_t1_t2_t1ce.example.yaml
```

它通过以下字段表达文件组织，不需要改 Dataset 代码：

```yaml
data:
  modalities: ["t2", "t1ce", "t1"]
  image_filename_pattern: "{mod}.nii.gz"
  mask_filename_pattern: "{mod}_mask.nii.gz"
  tumor_mask_source: "union_modality_masks"
```

## 3. 训练前检查数据配置

本地公开数据集：

```powershell
python scripts/check_dataset_config.py --config configs/brats2020_public_local.yaml
```

服务器公开数据集：

```bash
python scripts/check_dataset_config.py --config configs/brats2020_public_server.yaml
```

私有数据集：

```bash
python scripts/check_dataset_config.py --config configs/private_t1_t2_t1ce.yaml
```

检查输出会列出：

- 有效病例数
- 标签分布
- 前几个病例的模态文件和 seg/mask 文件路径

## 4. 公开数据集训练与测试命令

本地训练：

```powershell
python train.py --config configs/brats2020_public_local.yaml
```

服务器训练：

```bash
python train.py --config configs/brats2020_public_server.yaml
```

服务器固定测试并打包：

```bash
bash scripts/server_evaluate_and_pack.sh \
  --config configs/brats2020_public_server.yaml \
  --checkpoint outputs/brats2020_public_server/checkpoints/best.pt \
  --run-name brats2020_public_eval
```

## 5. 如果换成新数据集

优先只改 YAML：

```yaml
data:
  data_root: "/path/to/images"
  label_xlsx: "/path/to/labels.xlsx"
  case_id_col: "case_id"
  label_col: "label"
  modalities: ["t2", "t1ce", "t1", "flair"]
  image_filename_pattern: "{case_id}_{mod}.nii.gz"
  seg_filename_pattern: "{case_id}_seg.nii.gz"
  tumor_mask_source: "seg"
```

如果新数据集没有单独 seg，而是每个模态有 mask：

```yaml
data:
  image_filename_pattern: "{mod}.nii.gz"
  mask_filename_pattern: "{mod}_mask.nii.gz"
  tumor_mask_source: "union_modality_masks"
```

只有当病例不再是“一例一个文件夹”、标签不再是一行一个病例、或文件需要复杂搜索规则时，才需要扩展 `datasets/brats_dataset.py`。

