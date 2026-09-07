from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.brats_dataset import get_all_modality_combinations
from models import HybridHypergraphClassifier
from utils import load_config, set_seed
from utils.runner import compute_modality_auxiliary_loss, move_batch_to_device
from utils.training import brats_collate_fn, build_datasets, class_weights_from_records


def parse_args():
    parser = argparse.ArgumentParser(description="One-subject UTSW E4A affine forward/backward smoke")
    parser.add_argument("--config", default="configs/utsw_idh/missing_aware_aux_affine_e4a.yaml")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def make_pattern_batch(full_batch, modalities, combo, device):
    availability = torch.tensor(
        [[float(modality in combo) for modality in modalities]], dtype=full_batch["available_modalities"].dtype
    )
    pattern_batch = dict(full_batch)
    pattern_batch["available_modalities"] = availability
    pattern_batch["images"] = full_batch["images"] * availability.view(1, len(modalities), 1, 1, 1)
    pattern_batch["combo"] = [tuple(combo)]
    return move_batch_to_device(pattern_batch, device)


def assert_initialization_and_bias_compatibility(config):
    affine_model = HybridHypergraphClassifier(config).eval()
    base_logits = torch.tensor([-0.75, 1.25])
    for combo in get_all_modality_combinations(affine_model.modalities):
        availability = torch.tensor([float(modality in combo) for modality in affine_model.modalities])
        aligned, _, _, scale, bias = affine_model._apply_mask_aware_classifier(base_logits, availability)
        if not torch.allclose(scale, torch.tensor(1.0), atol=1e-6) or not torch.equal(bias, torch.tensor(0.0)):
            raise AssertionError(f"Non-identity affine initialization for {combo}.")
        if not torch.allclose(aligned, base_logits, atol=1e-6, rtol=0.0):
            raise AssertionError(f"Affine initialization changed logits for {combo}.")

    bias_config = load_config("configs/utsw_idh/missing_aware_aux_e3.yaml")
    bias_model = HybridHypergraphClassifier(bias_config).eval()
    availability = torch.tensor([1.0, 0.0, 1.0, 1.0])
    original = bias_model._apply_mask_aware_bias(base_logits, availability)
    unified = bias_model._apply_mask_aware_classifier(base_logits, availability)
    if not all(torch.equal(old, new) for old, new in zip(original, unified[:3])):
        raise AssertionError("Bias-only path changed under the unified classifier interface.")


def main():
    args = parse_args()
    config = load_config(args.config, overrides={"data": {"root": args.data_root}})
    device = torch.device(args.device if not args.device.startswith("cuda") or torch.cuda.is_available() else "cpu")
    set_seed(int(config["seed"]))
    assert_initialization_and_bias_compatibility(config)

    train_ds, _, _, splits = build_datasets(config)
    train_ds.set_epoch(1, int(config["train"]["epochs"]), config["train"])
    full_batch = brats_collate_fn([train_ds[0]])
    modalities = list(config["data"]["modalities"])
    full_combo = tuple(modalities)
    if tuple(full_batch["combo"][0]) != full_combo:
        raise AssertionError("E3 curriculum stage 1 did not produce the expected full-modality sample.")

    model = HybridHypergraphClassifier(config).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_from_records(splits["train"]).to(device))
    optimizer = AdamW(model.parameters(), lr=config["train"]["lr"], weight_decay=config["train"].get("weight_decay", 0.0))
    patterns = [full_combo, ("t1ce",), ("t2",), ("t1",), ("flair",), ("t1", "flair"), ("t2", "t1ce")]
    reports = []
    for combo in patterns:
        batch = make_pattern_batch(full_batch, modalities, combo, device)
        model.train(True)
        optimizer.zero_grad()
        output = model(batch)
        scale = output["affine_scale"]
        bias = output["affine_bias"]
        if not torch.all(scale > 0) or not torch.isfinite(scale).all() or not torch.isfinite(bias).all():
            raise AssertionError(f"Invalid affine parameters for {combo}.")
        if not torch.isfinite(output["logits"]).all() or not torch.isfinite(output["prob"]).all():
            raise AssertionError(f"Non-finite logits/probability for {combo}.")

        main_loss = criterion(output["logits"], batch["label"])
        aux_loss, _, counts = compute_modality_auxiliary_loss(output, batch, criterion, modalities)
        expected_counts = {modality: int(modality in combo) for modality in modalities}
        if counts != expected_counts:
            raise AssertionError(f"Auxiliary availability mismatch for {combo}: {counts}")
        total_loss = main_loss + float(config["auxiliary"]["lambda_aux"]) * aux_loss
        if not torch.isfinite(total_loss):
            raise AssertionError(f"Non-finite loss for {combo}.")
        total_loss.backward()
        optimizer.step()
        reports.append(
            {
                "pattern": "+".join(combo),
                "scale": float(scale.item()),
                "bias": float(bias.item()),
                "main_loss": float(main_loss.item()),
                "aux_loss": float(aux_loss.item()),
                "total_loss": float(total_loss.item()),
                "backward_optimizer_step": True,
            }
        )
        del batch, output, main_loss, aux_loss, total_loss
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(
        json.dumps(
            {
                "status": "ok",
                "case_id": full_batch["case_id"][0],
                "device": str(device),
                "identity_initialization_all_15_masks": True,
                "bias_only_backward_compatible": True,
                "patterns": reports,
                "peak_cuda_memory_gb": torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else None,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
