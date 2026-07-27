from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def whole_tumor_target(segmentation: torch.Tensor) -> torch.Tensor:
    """Convert aligned BraTS labels (0/1/2/4 or binary masks) to ``[B, 1, D, H, W]`` WT."""
    if segmentation.ndim == 4:
        segmentation = segmentation.unsqueeze(1)
    elif segmentation.ndim != 5 or segmentation.shape[1] != 1:
        raise ValueError(
            "Segmentation labels must have shape [B, D, H, W] or [B, 1, D, H, W], "
            f"got {tuple(segmentation.shape)}."
        )
    return (segmentation > 0).to(dtype=torch.float32)


def soft_dice_loss(seg_logits: torch.Tensor, wt_target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Per-case soft Dice loss, averaged over the batch."""
    if seg_logits.shape != wt_target.shape:
        raise ValueError(f"Logits and WT target shapes differ: {seg_logits.shape} vs {wt_target.shape}.")
    probabilities = torch.sigmoid(seg_logits)
    probabilities = probabilities.flatten(start_dim=1)
    target = wt_target.to(device=seg_logits.device, dtype=seg_logits.dtype).flatten(start_dim=1)
    intersection = (probabilities * target).sum(dim=1)
    denominator = probabilities.sum(dim=1) + target.sum(dim=1)
    dice_per_case = (2.0 * intersection + eps) / (denominator + eps)
    return (1.0 - dice_per_case).mean()


def auxiliary_segmentation_loss(
    seg_logits: torch.Tensor,
    segmentation: torch.Tensor,
    bce_weight: float = 0.5,
    dice_weight: float = 0.5,
) -> Dict[str, torch.Tensor]:
    if bce_weight < 0 or dice_weight < 0 or bce_weight + dice_weight <= 0:
        raise ValueError("Segmentation loss weights must be non-negative and have a positive sum.")
    wt_target = whole_tumor_target(segmentation).to(device=seg_logits.device, dtype=seg_logits.dtype)
    if seg_logits.shape != wt_target.shape:
        raise ValueError(f"seg_logits shape {seg_logits.shape} does not match WT target {wt_target.shape}.")
    bce = F.binary_cross_entropy_with_logits(seg_logits, wt_target)
    dice = soft_dice_loss(seg_logits, wt_target)
    total = bce_weight * bce + dice_weight * dice
    return {
        "seg_bce_loss": bce,
        "seg_dice_loss": dice,
        "segmentation_loss": total,
        "wt_dice": 1.0 - dice.detach(),
    }
