from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import HybridHypergraphClassifier
from utils import load_config, set_seed
from utils.runner import compute_modality_auxiliary_loss, move_batch_to_device
from utils.training import brats_collate_fn, build_datasets, class_weights_from_records


def parse_args():
    parser = argparse.ArgumentParser(description="One-subject UTSW E3 forward/backward smoke")
    parser.add_argument("--config", default="configs/utsw_idh/missing_aware_aux_e3.yaml")
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


def assert_disabled_compatibility(e1_config, enabled_config, device):
    disabled_config = copy.deepcopy(e1_config)
    disabled_config["auxiliary"] = {"enabled": False, "lambda_aux": 0.1, "pooling": "mean"}
    torch.manual_seed(2026)
    e1_model = HybridHypergraphClassifier(e1_config).to(device).eval()
    torch.manual_seed(2026)
    disabled_model = HybridHypergraphClassifier(disabled_config).to(device).eval()
    disabled_model.load_state_dict(e1_model.state_dict(), strict=True)
    torch.manual_seed(2026)
    enabled_model = HybridHypergraphClassifier(enabled_config).to(device).eval()
    for name, value in e1_model.state_dict().items():
        if not torch.equal(value, enabled_model.state_dict()[name]):
            raise AssertionError(f"E3 changed same-seed E1 main-path initialization: {name}")

    shape = (8, 8, 8)
    batch = {
        "images": torch.randn(1, 4, *shape, device=device),
        "roi_masks": torch.ones(1, 5, *shape, device=device),
        "roi_valid": torch.ones(1, 5, device=device),
        "available_modalities": torch.ones(1, 4, device=device),
    }
    with torch.no_grad():
        e1_logits = e1_model(batch)["logits"]
        disabled_output = disabled_model(batch)
    if "aux_logits" in disabled_output or not torch.equal(e1_logits, disabled_output["logits"]):
        raise AssertionError("auxiliary.enabled=false is not exactly compatible with E1 main prediction.")
    del e1_model, disabled_model, enabled_model, batch
    if device.type == "cuda":
        torch.cuda.empty_cache()


def main():
    args = parse_args()
    config = load_config(args.config, overrides={"data": {"root": args.data_root}})
    e1_config = load_config("configs/utsw_idh/missing_aware_train.yaml", overrides={"data": {"root": args.data_root}})
    device = torch.device(args.device if not args.device.startswith("cuda") or torch.cuda.is_available() else "cpu")
    set_seed(int(config["seed"]))
    assert_disabled_compatibility(e1_config, config, device)

    train_ds, _, _, splits = build_datasets(config)
    train_ds.set_epoch(1, int(config["train"]["epochs"]), config["train"])
    full_batch = brats_collate_fn([train_ds[0]])
    modalities = list(config["data"]["modalities"])
    full_combo = tuple(modalities)
    if tuple(full_batch["combo"][0]) != full_combo:
        raise AssertionError("E1 curriculum stage 1 did not produce the expected full-modality sample.")

    model = HybridHypergraphClassifier(config).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_from_records(splits["train"]).to(device))
    optimizer = AdamW(model.parameters(), lr=config["train"]["lr"], weight_decay=config["train"].get("weight_decay", 0.0))
    patterns = [
        full_combo,
        ("t1ce",),
        ("t2",),
        ("t1",),
        ("flair",),
        ("t1", "flair"),
    ]
    reports = []
    for combo in patterns:
        batch = make_pattern_batch(full_batch, modalities, combo, device)
        availability = batch["available_modalities"]
        unavailable = availability < 0.5
        if unavailable.any() and torch.count_nonzero(batch["images"][unavailable]).item() != 0:
            raise AssertionError(f"Unavailable modality input is not zero for {combo}.")

        model.train(True)
        optimizer.zero_grad()
        output = model(batch)
        if output["logits"].shape != (1, 2):
            raise AssertionError(f"Invalid main logits shape for {combo}: {tuple(output['logits'].shape)}")
        if set(output["aux_logits"]) != set(modalities):
            raise AssertionError(f"Missing auxiliary head output for {combo}.")
        if any(logits.shape != (1, 2) for logits in output["aux_logits"].values()):
            raise AssertionError(f"Invalid auxiliary logits shape for {combo}.")

        main_loss = criterion(output["logits"], batch["label"])
        aux_loss, per_modality_losses, counts = compute_modality_auxiliary_loss(output, batch, criterion, modalities)
        expected_counts = {modality: int(modality in combo) for modality in modalities}
        if counts != expected_counts:
            raise AssertionError(f"Auxiliary availability mismatch for {combo}: {counts}")
        expected_aux_loss = torch.stack(
            [per_modality_losses[modality] for modality in modalities if modality in combo]
        ).mean()
        if not torch.equal(aux_loss, expected_aux_loss):
            raise AssertionError(f"Auxiliary loss is not the observed-modality mean for {combo}.")
        total_loss = main_loss + 0.1 * aux_loss
        if not torch.allclose(total_loss, main_loss + float(config["auxiliary"]["lambda_aux"]) * aux_loss):
            raise AssertionError(f"Combined loss mismatch for {combo}.")
        total_loss.backward()
        for modality in modalities:
            if modality in combo:
                continue
            for parameter in model.auxiliary_heads[modality].parameters():
                if parameter.grad is not None and torch.count_nonzero(parameter.grad).item() != 0:
                    raise AssertionError(f"Missing modality {modality} received auxiliary gradients in {combo}.")
        optimizer.step()

        classifier_mask = output["classifier_mask"][0]
        for gate_idx, modality in enumerate(model.mask_order):
            if modality not in combo and torch.count_nonzero(output["modality_gates"][0, :, gate_idx]).item() != 0:
                raise AssertionError(f"Unavailable modality gate is nonzero for {modality} in {combo}.")
            expected = float(modality in combo)
            if float(classifier_mask[gate_idx].item()) != expected:
                raise AssertionError(f"Classifier mask mismatch for {modality} in {combo}.")
        reports.append(
            {
                "pattern": "+".join(combo),
                "main_logits_shape": list(output["logits"].shape),
                "aux_logits_shapes": {name: list(value.shape) for name, value in output["aux_logits"].items()},
                "observed_aux_counts": counts,
                "main_loss": float(main_loss.item()),
                "aux_loss": float(aux_loss.item()),
                "total_loss": float(total_loss.item()),
                "backward_optimizer_step": True,
            }
        )
        del batch, output, main_loss, aux_loss, total_loss
        if device.type == "cuda":
            torch.cuda.empty_cache()

    payload = {
        "status": "ok",
        "case_id": full_batch["case_id"][0],
        "device": str(device),
        "auxiliary_disabled_e1_compatible": True,
        "patterns": reports,
        "peak_cuda_memory_gb": torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else None,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
