# Lightweight Whole Tumor auxiliary segmentation

## Purpose and isolation

The optional branch adds voxel-level Whole Tumor (WT) supervision to the four modality-specific encoders. It reads each encoder's spatial feature map before ROI pooling. The classification path remains the existing path:

`known ROI masks -> five anatomy/context nodes -> five fixed anatomy hyperedges -> HGNN -> classifier`.

Predicted segmentation is never used to create ROI masks, nodes, hyperedges, or any classification input. The classification branch therefore still depends on known ROI masks.

## Tensors and spatial fusion

For each modality `m`, the encoder produces `F^m` with shape `[B, C_encoder, d, h, w]`. An independent `1x1x1 Conv3d -> GroupNorm -> ReLU` maps it to 32 channels. Given availability `a` with shape `[B, 4]`, spatial fusion is:

```text
F_seg = sum_m(a_m F_proj^m) / (sum_m(a_m) + eps)
```

The mask is broadcast to `[B, 4, 1, 1, 1, 1]`. An all-zero availability row raises an error. Missing modalities have exactly zero contribution.

The head is `Conv3d(32,32,3) -> GN -> ReLU -> Conv3d(32,16,3) -> GN -> ReLU -> Conv3d(16,1,1)`. Its low-resolution logits are trilinearly resized to the aligned image/label shape `[B, 1, D, H, W]`. No sigmoid is applied inside the model.

## Target and loss

The dataset already returns the aligned `seg` tensor. For BraTS labels `0/1/2/4`, WT is `Y_WT = 1[seg > 0]`. When `tumor_mask_source: union_modality_masks` is used, `seg` is the binary union of the configured tumor masks. Context ROIs (`peri_inner`, `peri_outer`, and `distal_normal`) are never included in WT.

For each case, soft Dice is computed over spatial voxels and then averaged over the batch:

```text
L_Dice = mean_b(1 - (2 sum(p*y) + eps) / (sum(p) + sum(y) + eps))
L_seg = 0.5 L_BCEWithLogits + 0.5 L_Dice
L_total = L_cls + lambda_seg L_seg
```

Empty targets are finite because of `eps`. Logs preserve the classification metrics and add `classification_loss`, `seg_bce_loss`, `seg_dice_loss`, `segmentation_loss`, `wt_dice`, `lambda_seg`, and total `loss`. Best-checkpoint selection remains `train.save_metric`, a classification metric.

## Baseline mode and checkpoints

Existing configs contain no auxiliary key and therefore remain disabled. The explicit baseline arm is:

```powershell
python train.py --config configs/experiments/aux_wt_seg_lambda000.yaml
```

When disabled, auxiliary modules are not constructed, `seg_logits` is not returned, and the baseline state dictionary and classification forward path are unchanged. When the branch is enabled, `load_checkpoint_state_dict` permits an old baseline checkpoint to omit only `auxiliary_segmentation.*`; all other incompatibilities still fail.

Evaluation helpers and single-case inference explicitly skip the auxiliary forward. The head may also be skipped with `model(batch, return_segmentation=False)`.

## Matched experiments

All arms inherit `configs/default.yaml`, use the same seed and `outputs/aux_wt_seg/shared_splits.json`, and preserve optimizer, scheduler behavior, epochs, classification loss, calibration, checkpoint selection, and all-15-combination evaluation.

```powershell
# Baseline, auxiliary branch disabled
python train.py --config configs/experiments/aux_wt_seg_lambda000.yaml

# WT auxiliary supervision, lambda_seg=0.05
python train.py --config configs/experiments/aux_wt_seg_lambda005.yaml

# WT auxiliary supervision, lambda_seg=0.1
python train.py --config configs/experiments/aux_wt_seg_lambda010.yaml
```

Equivalent wrapper commands are `powershell -File scripts/run_aux_wt_seg.ps1 -LambdaSeg 0`, `0.05`, or `0.1`. Each arm writes to a separate output directory.

## Verification and limitation

Run `python -m unittest tests.test_auxiliary_wt_segmentation -v` and `python scripts/aux_wt_seg_smoke.py`. The smoke is synthetic and does not start full training.

This design supervises representation learning but does not remove the classification path's use of ground-truth-derived ROI masks. It predicts only binary WT, not WT/TC/ET or the five graph ROIs.
