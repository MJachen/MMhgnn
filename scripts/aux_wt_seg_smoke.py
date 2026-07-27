from __future__ import annotations

import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F

from models import HybridHypergraphClassifier
from utils.config import load_config
from utils.losses import auxiliary_segmentation_loss


def make_batch(shape=(12, 12, 12)):
    images = torch.randn(1, 4, *shape)
    availability = torch.tensor([[1.0, 0.0, 1.0, 1.0]])
    images[:, 1].zero_()
    roi_masks = torch.zeros(1, 5, *shape)
    roi_masks[:, 0, 3:7, 3:7, 3:7] = 1
    roi_masks[:, 1, 2:8, 2:8, 2:8] = 1
    roi_masks[:, 2, 1:9, 1:9, 1:9] = 1
    roi_masks[:, 3, :, :, :6] = 1
    roi_masks[:, 4, :, :, 6:] = 1
    segmentation = torch.zeros(1, *shape)
    segmentation[:, 2:8, 2:8, 2:8] = 4
    return {
        "images": images,
        "label": torch.tensor([1]),
        "seg": segmentation,
        "roi_masks": roi_masks,
        "roi_valid": torch.ones(1, 5),
        "available_modalities": availability,
    }


def main():
    torch.manual_seed(42)
    config = load_config("configs/experiments/aux_wt_seg_lambda005.yaml")
    model = HybridHypergraphClassifier(config).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    batch = make_batch()

    output = model(batch)
    classification_loss = F.cross_entropy(output["logits"], batch["label"])
    segmentation_terms = auxiliary_segmentation_loss(output["seg_logits"], batch["seg"])
    total_loss = classification_loss + config["train"]["lambda_seg"] * segmentation_terms["segmentation_loss"]
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    summary = {
        "classification_logits_shape": list(output["logits"].shape),
        "segmentation_logits_shape": list(output["seg_logits"].shape),
        "classification_loss": float(classification_loss.item()),
        "seg_bce_loss": float(segmentation_terms["seg_bce_loss"].item()),
        "seg_dice_loss": float(segmentation_terms["seg_dice_loss"].item()),
        "segmentation_loss": float(segmentation_terms["segmentation_loss"].item()),
        "total_loss": float(total_loss.item()),
        "num_anatomy_nodes": int(output["roi_shared_features"].shape[1]),
        "num_anatomy_hyperedges": int(output["stage_stats"][0, 0].item()),
        "missing_modality": "t1ce",
        "all_outputs_finite": bool(torch.isfinite(output["logits"]).all() and torch.isfinite(output["seg_logits"]).all()),
        "segmentation_head_has_gradient": any(parameter.grad is not None for parameter in model.auxiliary_segmentation.segmentation_head.parameters()),
    }
    if not all(math.isfinite(summary[key]) for key in ("classification_loss", "seg_bce_loss", "seg_dice_loss", "segmentation_loss", "total_loss")):
        raise RuntimeError("Smoke run produced a non-finite loss.")
    if summary["num_anatomy_nodes"] != 5 or summary["num_anatomy_hyperedges"] != 5:
        raise RuntimeError("The baseline five-node/five-hyperedge classification path changed.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
