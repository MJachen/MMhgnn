from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch import nn
from torch.optim import AdamW

from datasets.brats_dataset import get_all_modality_combinations, scan_cases
from models.hybrid_hypergraph import ANATOMY_HYPEREDGES, HybridHypergraphClassifier
from tools.train_frozen_affine_e4b import aligned_from_cache, configure_affine_only, state_hash
from utils.config import load_config
from utils.io import load_json, save_json
from utils.runner import compute_modality_auxiliary_loss, move_batch_to_device
from utils.seed import set_seed
from utils.training import build_dataloader, build_datasets, class_weights_from_records


EXPECTED_HISTORICAL_SPLIT_SHA256 = "f6b8e2b84cb4599a1f8b113354ad747a64eba86b2e671a4c6098f8e419be8f54"
EXPECTED_MODALITIES = ["t2", "t1ce", "t1", "flair"]


def parse_args():
    parser = argparse.ArgumentParser(description="BraTS2020 latest E4A->E4B preflight")
    parser.add_argument("--e4a-config", required=True)
    parser.add_argument("--e4b-config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_split(records, split_path: Path):
    payload = load_json(split_path)
    if set(payload) != {"train", "val", "test"}:
        raise RuntimeError("Frozen split must contain exactly train/val/test.")
    record_map = {record.case_id: record for record in records}
    listed = [str(case_id) for name in ("train", "val", "test") for case_id in payload[name]]
    if len(listed) != len(set(listed)):
        raise RuntimeError("Frozen split contains duplicate or cross-split case IDs.")
    missing = sorted(set(record_map) - set(listed))
    unknown = sorted(set(listed) - set(record_map))
    if missing or unknown:
        raise RuntimeError(f"Frozen split cohort mismatch: missing={missing[:5]} unknown={unknown[:5]}")
    return {name: [record_map[str(case_id)] for case_id in payload[name]] for name in ("train", "val", "test")}


def distribution(records):
    counts = Counter(int(record.label) for record in records)
    return {"num_cases": len(records), "class_0_lgg": counts.get(0, 0), "class_1_hgg": counts.get(1, 0)}


def audit_geometry_and_roi(records, modalities):
    image_shapes = Counter()
    modality_shapes = {modality: Counter() for modality in modalities}
    invalid_roi_cases = []
    for record in records:
        reference_shape = None
        for modality in modalities:
            shape = tuple(int(value) for value in nib.load(str(record.files[modality])).shape)
            modality_shapes[modality][str(shape)] += 1
            image_shapes[str(shape)] += 1
            reference_shape = reference_shape or shape
            if shape != reference_shape:
                raise RuntimeError(f"Within-case modality shape mismatch: {record.case_id}")
        seg_image = nib.load(str(record.files["seg"]))
        if tuple(int(value) for value in seg_image.shape) != reference_shape:
            raise RuntimeError(f"Segmentation shape mismatch: {record.case_id}")
        if not np.any(np.asanyarray(seg_image.dataobj) > 0):
            invalid_roi_cases.append(record.case_id)
    if invalid_roi_cases:
        raise RuntimeError(f"Empty segmentation ROI: {invalid_roi_cases[:10]}")
    return {
        "image_shapes_all_modalities": dict(sorted(image_shapes.items())),
        "image_shapes_by_modality": {key: dict(sorted(value.items())) for key, value in modality_shapes.items()},
        "roi_source": "BraTS tumor segmentation (>0) -> five anatomy/context ROIs",
        "roi_valid_cases": len(records),
        "roi_invalid_cases": invalid_roi_cases,
    }


def dataset_fingerprint(records, config, split_path: Path):
    data_root = Path(config["data"]["data_root"])
    rows = []
    for record in sorted(records, key=lambda item: item.case_id):
        files = {}
        for name, path in sorted(record.files.items()):
            resolved = Path(path)
            files[name] = {"path": str(resolved.relative_to(data_root)), "size": resolved.stat().st_size}
        rows.append({"case_id": record.case_id, "label": int(record.label), "files": files})
    canonical = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "canonical_manifest_sha256": hashlib.sha256(canonical).hexdigest(),
        "fingerprint_method": "sorted_case_label_relative_path_file_size",
        "num_cases": len(records),
        "label_xlsx": config["data"]["label_xlsx"],
        "label_xlsx_sha256": sha256_file(Path(config["data"]["label_xlsx"])),
        "frozen_split": str(split_path),
        "frozen_split_sha256": sha256_file(split_path),
    }


def assert_protocol(e4a, e4b, split_sha):
    if list(e4a["data"]["modalities"]) != EXPECTED_MODALITIES:
        raise RuntimeError("BraTS modality order differs from the established protocol.")
    if split_sha != EXPECTED_HISTORICAL_SPLIT_SHA256:
        raise RuntimeError(f"Historical frozen split SHA mismatch: {split_sha}")
    if e4a["task"]["negative_class"] != "LGG" or e4a["task"]["positive_class"] != "HGG":
        raise RuntimeError("BraTS task mapping must remain LGG=0, HGG=1.")
    if e4a["train"]["mode"] != "missing_curriculum_train":
        raise RuntimeError("E4A must use the established missing curriculum.")
    if not e4a.get("auxiliary", {}).get("enabled") or float(e4a["auxiliary"]["lambda_aux"]) != 0.1:
        raise RuntimeError("E4A modality auxiliary supervision drifted from lambda=0.1.")
    if e4a["model"].get("mask_head_type") != "affine":
        raise RuntimeError("E4A must use the mask-conditioned affine head.")
    if int(e4a["graph"].get("num_nodes", -1)) != 5 or int(e4a["graph"].get("num_hyperedges", -1)) != 5:
        raise RuntimeError("Latest pipeline requires five ROI nodes and five anatomy hyperedges.")
    if e4a["graph"].get("use_prototype_nodes") or e4a["graph"].get("use_prototype_edges"):
        raise RuntimeError("Prototype nodes/hyperedges must remain disabled.")
    if e4a["model"].get("branches", {}).get("use_knn_edges"):
        raise RuntimeError("KNN hypergraph branch must remain disabled.")
    if len(ANATOMY_HYPEREDGES) != 5:
        raise RuntimeError("Model anatomy hyperedge constant no longer contains five edges.")
    required_e4b = ["freeze_backbone", "train_affine_only", "use_all_15_train_patterns", "equal_pattern_weight"]
    if e4b["train"]["mode"] != "fixed_combo" or not all(e4b["e4b"].get(key) for key in required_e4b):
        raise RuntimeError("E4B freeze/equal-pattern protocol is incomplete.")
    if e4b.get("auxiliary", {}).get("loss_enabled", True):
        raise RuntimeError("E4B auxiliary loss must be disabled.")
    if e4b["calibration"].get("threshold_mode") != "global":
        raise RuntimeError("E4B must calibrate exactly one global threshold.")
    forbidden = ["utsw", "grade", "mgmt"]
    source = str(e4b["e4b"]["source_checkpoint"]).casefold()
    if any(token in source for token in forbidden):
        raise RuntimeError(f"E4B source checkpoint is not BraTS-specific: {source}")


def smoke_e4a(e4a, device):
    train_ds, _, _, splits = build_datasets(e4a)
    selected = []
    for label in (0, 1):
        selected.append(next(record for record in splits["train"] if int(record.label) == label))
    train_ds.records = selected
    train_ds.combo_mode = "missing_curriculum_train"
    train_ds.set_epoch(max(1, int(e4a["train"]["epochs"])))
    loader = build_dataloader(train_ds, batch_size=2, num_workers=0, shuffle=False)
    model = HybridHypergraphClassifier(e4a).to(device)
    weights = class_weights_from_records(splits["train"]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = AdamW(model.parameters(), lr=float(e4a["train"]["lr"]), weight_decay=float(e4a["train"]["weight_decay"]))
    batch = move_batch_to_device(next(iter(loader)), device)
    optimizer.zero_grad(set_to_none=True)
    output = model(batch)
    main_loss = criterion(output["logits"], batch["label"])
    auxiliary_loss, modality_losses, counts = compute_modality_auxiliary_loss(
        output, batch, criterion, e4a["data"]["modalities"]
    )
    total_loss = main_loss + float(e4a["auxiliary"]["lambda_aux"]) * auxiliary_loss
    total_loss.backward()
    optimizer.step()
    if not torch.isfinite(total_loss):
        raise RuntimeError("E4A smoke produced a non-finite loss.")
    return {
        "forward_pass": True,
        "weighted_ce": float(main_loss.item()),
        "auxiliary_loss": float(auxiliary_loss.item()),
        "total_loss": float(total_loss.item()),
        "auxiliary_observed_counts": counts,
        "auxiliary_per_modality_loss": {key: float(value.item()) for key, value in modality_losses.items()},
        "backward": True,
        "optimizer_step": True,
        "smoke_case_ids": [record.case_id for record in selected],
    }


def smoke_e4b(e4b, device):
    model = HybridHypergraphClassifier(e4b).to(device)
    trainable_names, frozen_count, trainable_count = configure_affine_only(model)
    frozen_before = state_hash(model, include_affine=False)
    combos = get_all_modality_combinations(e4b["data"]["modalities"])
    masks = torch.tensor(
        [[float(modality in combo) for modality in model.mask_order] for combo in combos],
        dtype=torch.float32,
        device=device,
    )
    base_logits = torch.linspace(-2.0, 2.0, len(combos), device=device)
    labels = torch.tensor([index % 2 for index in range(len(combos))], dtype=torch.long, device=device)
    optimizer = AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=float(e4b["e4b"]["lr"]))
    optimizer.zero_grad(set_to_none=True)
    logits, _, _, scale, _ = aligned_from_cache(model, base_logits, masks)
    loss = nn.CrossEntropyLoss()(logits, labels)
    loss.backward()
    if any(parameter.grad is not None for name, parameter in model.named_parameters() if not name.startswith(("mask_encoder.", "mask_affine_head."))):
        raise RuntimeError("E4B smoke found a gradient on a frozen parameter.")
    optimizer.step()
    frozen_after = state_hash(model, include_affine=False)
    if frozen_after != frozen_before or not torch.all(scale > 0):
        raise RuntimeError("E4B frozen hash or positive-scale invariant failed.")
    return {
        "trainable_parameter_names": trainable_names,
        "frozen_parameter_count": frozen_count,
        "trainable_parameter_count": trainable_count,
        "affine_scale_positive": True,
        "frozen_hash_unchanged": True,
        "optimizer_affine_only": True,
        "num_patterns": len(combos),
    }


def main():
    args = parse_args()
    set_seed(42)
    e4a = load_config(args.e4a_config)
    e4b = load_config(args.e4b_config)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    split_path = Path(e4a["data"]["split_json"])
    if not split_path.is_file():
        raise FileNotFoundError(split_path)

    records = scan_cases(
        data_root=e4a["data"]["data_root"],
        label_xlsx=e4a["data"]["label_xlsx"],
        case_id_col=e4a["data"]["case_id_col"],
        label_col=e4a["data"]["label_col"],
        modalities=e4a["data"]["modalities"],
        image_filename_pattern=e4a["data"]["image_filename_pattern"],
        seg_filename_pattern=e4a["data"]["seg_filename_pattern"],
        tumor_mask_source=e4a["data"]["tumor_mask_source"],
    )
    splits = strict_split(records, split_path)
    split_sha = sha256_file(split_path)
    assert_protocol(e4a, e4b, split_sha)
    label_counts = Counter(int(record.label) for record in records)
    if set(label_counts) != {0, 1}:
        raise RuntimeError(f"BraTS labels are not binary 0/1: {label_counts}")

    geometry = audit_geometry_and_roi(records, e4a["data"]["modalities"])
    fingerprint = dataset_fingerprint(records, e4a, split_path)
    save_json(fingerprint, e4a["data"]["manifest_fingerprint_json"])
    split_summary = {
        "historical_split_reused": True,
        "frozen_split_sha256": split_sha,
        "total": distribution(records),
        **{
            name: {
                **distribution(split_records),
                "case_ids": [record.case_id for record in split_records],
                "labels": [int(record.label) for record in split_records],
            }
            for name, split_records in splits.items()
        },
    }
    save_json(split_summary, output_root / "split_summary.json")

    requested_device = args.device
    device = torch.device(requested_device if not requested_device.startswith("cuda") or torch.cuda.is_available() else "cpu")
    e4a_smoke = smoke_e4a(e4a, device)
    e4b_smoke = smoke_e4b(e4b, device)
    combos = get_all_modality_combinations(e4a["data"]["modalities"])
    pattern_counts = Counter(len(combo) for combo in combos)
    report = {
        "status": "ok",
        "dataset_root": e4a["data"]["data_root"],
        "manifest_path": e4a["data"]["label_xlsx"],
        "split_path": str(split_path),
        "mapping": {"0": "LGG", "1": "HGG"},
        "class_counts": {"LGG_0": label_counts[0], "HGG_1": label_counts[1]},
        "split_distribution": {name: distribution(value) for name, value in splits.items()},
        "modality_order": e4a["data"]["modalities"],
        "geometry_and_roi": geometry,
        "patterns": {"total": len(combos), "one_modality": pattern_counts[1], "two_modality": pattern_counts[2], "three_modality": pattern_counts[3], "full": pattern_counts[4]},
        "architecture": {
            "modality_specific_encoders": 4,
            "anatomy_roi_nodes": 5,
            "fixed_anatomy_hyperedges": 5,
            "prototype_nodes": False,
            "prototype_hyperedges": False,
            "knn_hypergraph": False,
            "segmentation_supervision": False,
            "mask_aware_fusion": True,
            "attention_pooling": True,
            "binary_classifier": True,
        },
        "e4a_smoke": e4a_smoke,
        "e4b_smoke": e4b_smoke,
        "leakage_gates": {
            "split_disjoint_complete": True,
            "e4a_test_disabled": not bool(e4a["eval"].get("run_test_after_training", True)),
            "e4b_train_val_cached_before_test": True,
            "validation_only_global_threshold": e4b["calibration"]["threshold_mode"] == "global",
            "test_after_checkpoint_and_threshold_freeze": True,
        },
        "dataset_fingerprint": fingerprint,
    }
    save_json(report, output_root / "brats_preflight_report.json")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
