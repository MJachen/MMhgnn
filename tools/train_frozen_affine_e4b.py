from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.brats_dataset import get_all_modality_combinations
from models import HybridHypergraphClassifier
from utils import load_config, set_seed, setup_logger
from utils.config import ensure_dir
from utils.io import load_json, save_json
from utils.metrics import calibrate_threshold, compute_binary_metrics, summarize_missing_pattern_metrics
from utils.runner import move_batch_to_device
from utils.training import build_dataloader, build_datasets, class_weights_from_records


AFFINE_PREFIXES = ("mask_encoder.", "mask_affine_head.")


def parse_args():
    parser = argparse.ArgumentParser(description="E4B frozen-backbone mask-affine recalibration")
    parser.add_argument("--config", default="configs/utsw_idh/frozen_affine_e4b.yaml")
    parser.add_argument("--source-checkpoint", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--reuse-cache", action="store_true")
    parser.add_argument("--auc-invariance-smoke", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_hash(model: nn.Module, include_affine: bool) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        is_affine = name.startswith(AFFINE_PREFIXES)
        if include_affine != is_affine:
            continue
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def split_provenance(splits) -> Dict:
    return {
        name: [
            {
                "case_id": record.case_id,
                "patient_id": getattr(record, "patient_id", record.case_id),
                "label": int(record.label),
            }
            for record in records
        ]
        for name, records in splits.items()
    }


def configure_affine_only(model: nn.Module):
    for parameter in model.parameters():
        parameter.requires_grad = False
    for module_name in ["mask_encoder", "mask_affine_head"]:
        module = getattr(model, module_name, None)
        if module is None:
            raise RuntimeError(f"E4B requires model.{module_name}.")
        for parameter in module.parameters():
            parameter.requires_grad = True

    trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    invalid = [name for name in trainable_names if not name.startswith(AFFINE_PREFIXES)]
    if invalid:
        raise RuntimeError(f"Non-affine parameters remain trainable: {invalid}")
    frozen_count = sum(parameter.numel() for parameter in model.parameters() if not parameter.requires_grad)
    trainable_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if trainable_count == 0:
        raise RuntimeError("E4B found no trainable affine parameters.")
    return trainable_names, frozen_count, trainable_count


def validate_source_checkpoint(checkpoint: Dict, config: Dict, splits) -> None:
    source_config = checkpoint.get("config", {})
    if source_config.get("model", {}).get("mask_head_type") != "affine":
        raise RuntimeError("E4B source checkpoint is not an E4A affine checkpoint.")
    if checkpoint.get("selection_metric") != "mean15_bal_acc":
        raise RuntimeError("E4B source checkpoint was not selected by mean15_bal_acc.")
    current_fingerprint = load_json(config["data"]["manifest_fingerprint_json"])["canonical_manifest_sha256"]
    if checkpoint.get("canonical_manifest_sha256") != current_fingerprint:
        raise RuntimeError("E4B source checkpoint manifest fingerprint mismatch.")
    if checkpoint.get("split") != split_provenance(splits):
        raise RuntimeError("E4B source checkpoint does not match the frozen train/validation/test split.")


def classifier_mask(modalities: Sequence[str], mask_order: Sequence[str], combo: Sequence[str]) -> List[float]:
    combo_set = set(combo)
    if not combo_set or not combo_set.issubset(set(modalities)):
        raise ValueError(f"Invalid non-empty modality combination: {combo}")
    return [float(modality in combo_set) for modality in mask_order]


def smoke_records(records):
    selected = []
    for label in [0, 1]:
        match = next((record for record in records if int(record.label) == label), None)
        if match is None:
            raise RuntimeError("E4B real-data smoke requires one subject from each class.")
        selected.append(match)
    return selected


@torch.no_grad()
def extract_base_logit_cache(model, config, device, split_name: str, combos, smoke: bool = False) -> pd.DataFrame:
    if split_name not in {"train", "val", "test"}:
        raise ValueError(f"Unsupported cache split: {split_name}")
    rows = []
    captured: List[torch.Tensor] = []

    def capture_classifier_output(_module, _inputs, output):
        captured.append(output.detach().cpu())

    handle = model.classifier.register_forward_hook(capture_classifier_output)
    model.eval()
    try:
        for combo in combos:
            train_ds, val_ds, test_ds, _ = build_datasets(config, explicit_eval_combo=combo)
            if split_name == "train":
                dataset = train_ds
                dataset.explicit_combo = tuple(combo)
            elif split_name == "val":
                dataset = val_ds
            else:
                dataset = test_ds
            if smoke:
                dataset.records = smoke_records(dataset.records)
            loader = build_dataloader(dataset, 1, config["data"].get("num_workers", 0), shuffle=False)
            for batch in tqdm(loader, desc=f"cache {split_name} {'_'.join(combo)}", leave=False):
                captured.clear()
                output = model(move_batch_to_device(batch, device))
                if len(captured) != len(batch["case_id"]):
                    raise RuntimeError("Classifier hook did not capture exactly one base logit per subject.")
                for index, base_logits in enumerate(captured):
                    binary = float((base_logits[1] - base_logits[0]).item())
                    rows.append(
                        {
                            "case_id": batch["case_id"][index],
                            "split": split_name,
                            "combo": "_".join(combo),
                            "mask": "".join(str(int(value)) for value in classifier_mask(model.modalities, model.mask_order, combo)),
                            "y_true": int(batch["label"][index].item()),
                            "base_logit": binary,
                            "base_prob": float(torch.sigmoid(torch.tensor(binary)).item()),
                        }
                    )
                if not torch.isfinite(output["logits"]).all():
                    raise RuntimeError("Non-finite aligned output encountered during base-logit caching.")
    finally:
        handle.remove()
    frame = pd.DataFrame(rows)
    expected = len(combos) * (2 if smoke else len(dataset.records))
    if len(frame) != expected:
        raise RuntimeError(f"Unexpected {split_name} cache size: expected={expected} actual={len(frame)}")
    return frame


def frame_tensors(frame: pd.DataFrame, device):
    base = torch.tensor(frame["base_logit"].to_numpy(), dtype=torch.float32, device=device)
    masks = torch.tensor([[float(char) for char in value] for value in frame["mask"]], dtype=torch.float32, device=device)
    labels = torch.tensor(frame["y_true"].to_numpy(), dtype=torch.long, device=device)
    return base, masks, labels


def aligned_from_cache(model, base_logits: torch.Tensor, masks: torch.Tensor):
    embedding = model.mask_encoder(masks)
    raw_scale, bias = model.mask_affine_head(embedding).unbind(dim=-1)
    scale = F.softplus(raw_scale) + float(model.mask_affine_eps)
    aligned_binary = scale * base_logits + bias
    logits = torch.stack([-0.5 * aligned_binary, 0.5 * aligned_binary], dim=-1)
    return logits, torch.sigmoid(aligned_binary), aligned_binary, scale, bias


def safe_auc(y_true, scores) -> float:
    return float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) == 2 else float("nan")


def validate_affine_outputs(masks: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor) -> None:
    if not torch.all(scale > 0):
        raise RuntimeError("E4B produced a non-positive affine scale.")
    for mask in torch.unique(masks, dim=0):
        selected = torch.all(masks == mask, dim=1)
        selected_scale = scale[selected]
        selected_bias = bias[selected]
        if not torch.allclose(selected_scale, selected_scale[:1].expand_as(selected_scale), rtol=0.0, atol=0.0):
            raise RuntimeError("Samples with the same modality mask received different affine scales.")
        if not torch.allclose(selected_bias, selected_bias[:1].expand_as(selected_bias), rtol=0.0, atol=0.0):
            raise RuntimeError("Samples with the same modality mask received different affine biases.")


@torch.no_grad()
def evaluate_cache(model, frame: pd.DataFrame, device, modalities, threshold: float = 0.5):
    base, masks, _ = frame_tensors(frame, device)
    _, probability, aligned_binary, scale, bias = aligned_from_cache(model, base, masks)
    validate_affine_outputs(masks, scale, bias)
    evaluated = frame.copy()
    evaluated["y_prob"] = probability.cpu().numpy()
    evaluated["aligned_logit"] = aligned_binary.cpu().numpy()
    evaluated["affine_scale"] = scale.cpu().numpy()
    evaluated["affine_bias"] = bias.cpu().numpy()
    metrics_by_combo = {}
    auc_rows = []
    for combo_name, group in evaluated.groupby("combo", sort=False):
        metrics = compute_binary_metrics(group["y_true"], group["y_prob"], threshold=threshold)
        base_auc = safe_auc(group["y_true"], group["base_logit"])
        aligned_auc = safe_auc(group["y_true"], group["aligned_logit"])
        metrics["auc"] = aligned_auc
        base_probability_auc = safe_auc(group["y_true"], group["base_prob"])
        aligned_probability_auc = safe_auc(group["y_true"], group["y_prob"])
        auc_rows.append(
            {
                "combo": combo_name,
                "base_auc": base_auc,
                "aligned_auc": aligned_auc,
                "delta_auc": aligned_auc - base_auc,
                "base_probability_auc": base_probability_auc,
                "aligned_probability_auc": aligned_probability_auc,
                "probability_delta_auc": aligned_probability_auc - base_probability_auc,
            }
        )
        metrics_by_combo[combo_name] = metrics
    pooled = compute_binary_metrics(evaluated["y_true"], evaluated["y_prob"], threshold=threshold)
    pooled["auc"] = safe_auc(evaluated["y_true"], evaluated["aligned_logit"])
    summary = summarize_missing_pattern_metrics(metrics_by_combo, modalities)
    summary["pooled_bal_acc"] = float(pooled["bal_acc"])
    summary["pooled_auc"] = float(pooled["auc"])
    return evaluated, metrics_by_combo, summary, pd.DataFrame(auc_rows)


def save_cache(frame: pd.DataFrame, cache_dir: Path, split_name: str) -> Dict:
    csv_path = cache_dir / f"{split_name}_base_logits.csv"
    frame.to_csv(csv_path, index=False)
    return {
        "split": split_name,
        "num_rows": int(len(frame)),
        "num_subjects": int(frame["case_id"].nunique()),
        "num_patterns": int(frame["combo"].nunique()),
        "sha256": sha256_file(csv_path),
        "path": str(csv_path),
    }


def validate_cache_frame(
    frame: pd.DataFrame,
    split_name: str,
    case_ids: Iterable[str],
    combos,
    labels_by_case: Dict[str, int] | None = None,
    expected_masks_by_combo: Dict[str, str] | None = None,
) -> None:
    expected_cases = set(case_ids)
    expected_combos = {"_".join(combo) for combo in combos}
    if set(frame["split"].unique()) != {split_name}:
        raise RuntimeError(f"Cache split tag mismatch for {split_name}.")
    if set(frame["case_id"].unique()) != expected_cases:
        raise RuntimeError(f"Cache subject provenance mismatch for {split_name}.")
    if set(frame["combo"].unique()) != expected_combos:
        raise RuntimeError(f"Cache pattern coverage mismatch for {split_name}.")
    if frame.duplicated(["case_id", "combo"]).any():
        raise RuntimeError(f"Duplicate subject-pattern pair in {split_name} cache.")
    counts = frame.groupby("combo")["case_id"].nunique()
    if not (counts == len(expected_cases)).all():
        raise RuntimeError(f"Patterns are not equally represented in {split_name} cache.")
    if labels_by_case is not None:
        actual_labels = frame.groupby("case_id")["y_true"].unique()
        for case_id, values in actual_labels.items():
            if len(values) != 1 or int(values[0]) != int(labels_by_case[case_id]):
                raise RuntimeError(f"Cache label provenance mismatch for {split_name}/{case_id}.")
    if expected_masks_by_combo is not None:
        for combo_name, group in frame.groupby("combo", sort=False):
            actual_masks = set(group["mask"].astype(str))
            if actual_masks != {expected_masks_by_combo[combo_name]}:
                raise RuntimeError(f"Cache modality mask mismatch for {split_name}/{combo_name}.")


def load_reusable_train_val_cache(cache_dir, source_path, source_checkpoint, splits, combos, model, smoke=False):
    if smoke:
        raise RuntimeError("Formal train/validation caches cannot be reused for a two-subject smoke run.")
    provenance_path = cache_dir / "cache_provenance.json"
    train_path = cache_dir / "train_base_logits.csv"
    val_path = cache_dir / "val_base_logits.csv"
    if not all(path.is_file() for path in [provenance_path, train_path, val_path]):
        raise RuntimeError("Requested E4B cache reuse, but train/validation cache artifacts are incomplete.")
    metadata = load_json(provenance_path)
    expected_split_sha = hashlib.sha256(json.dumps(split_provenance(splits), sort_keys=True).encode("utf-8")).hexdigest()
    expected_patterns = ["_".join(combo) for combo in combos]
    required = {
        "source_checkpoint_sha256": sha256_file(source_path),
        "canonical_manifest_sha256": source_checkpoint["canonical_manifest_sha256"],
        "split_provenance_sha256": expected_split_sha,
        "mask_order": list(model.mask_order),
        "modalities": list(model.modalities),
        "patterns": expected_patterns,
        "cached_before_checkpoint_selection": ["train", "val"],
        "test_cache_created_before_checkpoint_and_threshold_freeze": False,
    }
    for key, expected in required.items():
        if metadata.get(key) != expected:
            raise RuntimeError(f"Unsafe E4B cache reuse: provenance mismatch for {key}.")
    if "test" in metadata or (cache_dir / "test_base_logits.csv").exists():
        raise RuntimeError("Unsafe E4B cache reuse: a test cache already exists before checkpoint selection.")

    expected_masks = {
        "_".join(combo): "".join(str(int(value)) for value in classifier_mask(model.modalities, model.mask_order, combo))
        for combo in combos
    }
    frames = {}
    for split_name, csv_path in [("train", train_path), ("val", val_path)]:
        split_metadata = metadata.get(split_name, {})
        actual_sha = sha256_file(csv_path)
        if split_metadata.get("sha256") != actual_sha:
            raise RuntimeError(f"Unsafe E4B cache reuse: SHA-256 mismatch for {split_name} cache.")
        frame = pd.read_csv(csv_path, dtype={"case_id": str, "split": str, "combo": str, "mask": str})
        records = splits[split_name]
        case_ids = [record.case_id for record in records]
        labels_by_case = {record.case_id: int(record.label) for record in records}
        validate_cache_frame(frame, split_name, case_ids, combos, labels_by_case, expected_masks)
        if split_metadata.get("num_rows") != len(frame):
            raise RuntimeError(f"Unsafe E4B cache reuse: row-count mismatch for {split_name} cache.")
        if split_metadata.get("num_subjects") != frame["case_id"].nunique():
            raise RuntimeError(f"Unsafe E4B cache reuse: subject-count mismatch for {split_name} cache.")
        if split_metadata.get("num_patterns") != frame["combo"].nunique():
            raise RuntimeError(f"Unsafe E4B cache reuse: pattern-count mismatch for {split_name} cache.")
        frames[split_name] = frame
    return frames["train"], frames["val"], metadata


def mask_affine_rows(model, combos):
    rows = []
    device = next(model.parameters()).device
    for combo in combos:
        mask = torch.tensor(classifier_mask(model.modalities, model.mask_order, combo), dtype=torch.float32, device=device)
        with torch.no_grad():
            embedding = model.mask_encoder(mask.unsqueeze(0))
            raw_scale, bias = model.mask_affine_head(embedding).unbind(dim=-1)
            scale = F.softplus(raw_scale) + float(model.mask_affine_eps)
        rows.append({"combo": "_".join(combo), "mask": "".join(str(int(v)) for v in mask.tolist()), "scale": float(scale.item()), "bias": float(bias.item())})
    return rows


def train_affine(model, train_frame, val_frame, config, device, criterion, output_dir, frozen_hash_before, source_info):
    e4b = config["e4b"]
    base, masks, labels = frame_tensors(train_frame, "cpu")
    generator = torch.Generator().manual_seed(int(config["seed"]))
    loader = DataLoader(
        TensorDataset(base, masks, labels),
        batch_size=int(e4b.get("cache_batch_size", 256)),
        shuffle=True,
        generator=generator,
    )
    optimizer = AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(e4b.get("lr", config["train"]["lr"])),
        weight_decay=float(e4b.get("weight_decay", config["train"].get("weight_decay", 0.0))),
    )
    best_score = -1.0
    bad_epochs = 0
    history = []
    best_path = output_dir / "checkpoints" / "best.pt"
    auc_tolerance = float(e4b.get("auc_invariance_tolerance", 1e-4))
    max_epochs = 1 if e4b.get("smoke", False) else int(e4b.get("epochs", 100))
    for epoch in range(1, max_epochs + 1):
        model.eval()
        model.mask_encoder.train()
        model.mask_affine_head.train()
        losses = []
        for batch_base, batch_masks, batch_labels in loader:
            batch_base = batch_base.to(device)
            batch_masks = batch_masks.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, _, _, scale, bias = aligned_from_cache(model, batch_base, batch_masks)
            validate_affine_outputs(batch_masks, scale, bias)
            loss = criterion(logits, batch_labels)
            loss.backward()
            for name, parameter in model.named_parameters():
                if not name.startswith(AFFINE_PREFIXES) and parameter.grad is not None:
                    raise RuntimeError(f"Frozen parameter received gradient: {name}")
            optimizer.step()
            losses.append(float(loss.item()))

        frozen_hash_after_epoch = state_hash(model, include_affine=False)
        if frozen_hash_after_epoch != frozen_hash_before:
            raise RuntimeError("Frozen representation hash changed during E4B training.")
        _, val_metrics, val_summary, auc_rows = evaluate_cache(model, val_frame, device, config["data"]["modalities"])
        max_auc_delta = float(auc_rows["delta_auc"].abs().max())
        if max_auc_delta > auc_tolerance:
            raise RuntimeError(f"Per-pattern AUC invariance failed: max |delta|={max_auc_delta:.8f}")
        score = float(val_summary["mean15_bal_acc"])
        improved = score > best_score
        if improved:
            best_score = score
            bad_epochs = 0
        else:
            bad_epochs += 1
        row = {
            "epoch": epoch,
            "weighted_ce": float(np.mean(losses)),
            "mean15_bal_acc": score,
            "mean15_auc": float(val_summary["mean15_auc"]),
            "pooled_bal_acc": float(val_summary["pooled_bal_acc"]),
            "pooled_auc": float(val_summary["pooled_auc"]),
            "max_abs_delta_auc": max_auc_delta,
            "bad_epochs": bad_epochs,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(output_dir / "metrics" / "affine_training_history.csv", index=False)
        save_json(
            {"epoch": epoch, "selection_threshold": 0.5, "per_pattern": val_metrics, "summary": val_summary},
            output_dir / "metrics" / f"val_missing15_epoch_{epoch:03d}.json",
        )
        pd.DataFrame([{"combo": name, **metrics} for name, metrics in val_metrics.items()]).to_csv(
            output_dir / "metrics" / f"val_missing15_epoch_{epoch:03d}.csv", index=False
        )
        pd.DataFrame(val_summary["groups"]).to_csv(
            output_dir / "metrics" / f"val_missing15_epoch_{epoch:03d}_groups.csv", index=False
        )
        auc_rows.to_csv(output_dir / "metrics" / f"val_auc_invariance_epoch_{epoch:03d}.csv", index=False)
        if improved:
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "score": score,
                    "best_score": best_score,
                    "selection_metric": "mean15_bal_acc",
                    "selection_metrics": val_summary,
                    "canonical_manifest_sha256": source_info["canonical_manifest_sha256"],
                    "split": source_info["split"],
                    "e4b": source_info["e4b"],
                    "frozen_parameter_hash": frozen_hash_before,
                    "threshold_calibration": {"enabled": False, "mode": "global", "source": "pending_validation_only_calibration"},
                },
                best_path,
            )
        if bad_epochs >= int(e4b.get("early_stop_patience", 15)):
            break
    return best_path, history


def main():
    args = parse_args()
    overrides = {}
    if args.data_root:
        overrides["data"] = {"root": args.data_root}
    if args.output_dir:
        overrides["output_dir"] = args.output_dir
    config = load_config(args.config, overrides=overrides or None)
    e4b = config.get("e4b", {})
    if not e4b.get("enabled", False):
        raise ValueError("E4B requires e4b.enabled=true.")
    if config["train"].get("mode") != "fixed_combo":
        raise ValueError("E4B cache extraction must not use curriculum training.")
    if not all(bool(e4b.get(key, False)) for key in ["freeze_backbone", "train_affine_only", "use_all_15_train_patterns", "equal_pattern_weight"]):
        raise ValueError("E4B frozen affine-only and equal 15-pattern constraints must all be enabled.")
    if config.get("calibration", {}).get("threshold_mode") != "global":
        raise ValueError("E4B requires exactly one global threshold.")
    if config.get("auxiliary", {}).get("loss_enabled", False):
        raise ValueError("E4B auxiliary loss must remain disabled.")
    e4b["smoke"] = bool(args.smoke)
    set_seed(int(config["seed"]))
    logger = setup_logger(config.get("logging", {}).get("level", "INFO"))
    output_dir = ensure_dir(config["output_dir"])
    cache_dir = ensure_dir(output_dir / "cache")
    metrics_dir = ensure_dir(output_dir / "metrics")
    ensure_dir(output_dir / "checkpoints")
    save_json(config, output_dir / "resolved_config.json")
    requested_device = args.device or config["train"].get("device", "cuda")
    device = torch.device(requested_device if not requested_device.startswith("cuda") or torch.cuda.is_available() else "cpu")

    train_ds, _, _, splits = build_datasets(config)
    model = HybridHypergraphClassifier(config).to(device)
    source_path = Path(args.source_checkpoint or e4b["source_checkpoint"])
    source_checkpoint = torch.load(source_path, map_location=device)
    model.load_state_dict(source_checkpoint["model"])
    validate_source_checkpoint(source_checkpoint, config, splits)
    trainable_names, frozen_count, trainable_count = configure_affine_only(model)
    logger.info("E4B trainable parameters: %s", trainable_names)
    logger.info("E4B frozen parameter count=%d trainable parameter count=%d", frozen_count, trainable_count)
    frozen_hash_before = state_hash(model, include_affine=False)
    affine_hash_before = state_hash(model, include_affine=True)
    combos = get_all_modality_combinations(config["data"]["modalities"])

    train_records = smoke_records(splits["train"]) if args.smoke else splits["train"]
    val_records = smoke_records(splits["val"]) if args.smoke else splits["val"]
    train_case_ids = [record.case_id for record in train_records]
    val_case_ids = [record.case_id for record in val_records]
    if args.reuse_cache:
        train_cache, val_cache, cache_metadata = load_reusable_train_val_cache(
            cache_dir, source_path, source_checkpoint, splits, combos, model, smoke=args.smoke
        )
        logger.info("Safely reused verified E4B train/validation caches.")
    else:
        # Phase A deliberately creates train and validation caches only. Test images are not indexed here.
        train_cache = extract_base_logit_cache(model, config, device, "train", combos, smoke=args.smoke)
        val_cache = extract_base_logit_cache(model, config, device, "val", combos, smoke=args.smoke)
        validate_cache_frame(train_cache, "train", train_case_ids, combos)
        validate_cache_frame(val_cache, "val", val_case_ids, combos)
        cache_metadata = {
            "source_checkpoint": str(source_path),
            "source_checkpoint_sha256": sha256_file(source_path),
            "canonical_manifest_sha256": source_checkpoint["canonical_manifest_sha256"],
            "split_provenance_sha256": hashlib.sha256(json.dumps(split_provenance(splits), sort_keys=True).encode("utf-8")).hexdigest(),
            "mask_order": list(model.mask_order),
            "modalities": list(model.modalities),
            "patterns": ["_".join(combo) for combo in combos],
            "cached_before_checkpoint_selection": ["train", "val"],
            "test_cache_created_before_checkpoint_and_threshold_freeze": False,
            "train": save_cache(train_cache, cache_dir, "train"),
            "val": save_cache(val_cache, cache_dir, "val"),
        }
        save_json(cache_metadata, cache_dir / "cache_provenance.json")

    if args.auc_invariance_smoke:
        if not args.reuse_cache:
            raise RuntimeError("AUC-invariance smoke must use --reuse-cache to avoid 3D MRI cache extraction.")
        _, _, _, auc_rows = evaluate_cache(model, val_cache, device, config["data"]["modalities"])
        report = {
            "status": "ok",
            "cache_reused": True,
            "test_read": False,
            "auc_invariance_tolerance": float(e4b.get("auc_invariance_tolerance", 1e-4)),
            "max_abs_probability_delta_auc": float(auc_rows["probability_delta_auc"].abs().max()),
            "max_abs_logit_delta_auc": float(auc_rows["delta_auc"].abs().max()),
            "positive_scale_all_patterns": True,
            "same_mask_same_scale_bias": True,
            "frozen_hash_unchanged": state_hash(model, include_affine=False) == frozen_hash_before,
        }
        if report["max_abs_logit_delta_auc"] > report["auc_invariance_tolerance"]:
            raise RuntimeError(
                f"Logit-space AUC-invariance smoke failed: max |delta|={report['max_abs_logit_delta_auc']:.8f}"
            )
        save_json(report, metrics_dir / "auc_invariance_smoke.json")
        print(json.dumps(report, indent=2))
        return

    class_weights = class_weights_from_records(splits["train"]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    source_info = {
        "canonical_manifest_sha256": source_checkpoint["canonical_manifest_sha256"],
        "split": source_checkpoint["split"],
        "e4b": {
            "source_checkpoint": str(source_path),
            "source_checkpoint_sha256": cache_metadata["source_checkpoint_sha256"],
            "trainable_parameter_names": trainable_names,
            "frozen_parameter_count": frozen_count,
            "trainable_parameter_count": trainable_count,
            "frozen_hash_before": frozen_hash_before,
            "affine_hash_before": affine_hash_before,
            "cache_provenance": cache_metadata,
        },
    }
    best_path, history = train_affine(
        model, train_cache, val_cache, config, device, criterion, output_dir, frozen_hash_before, source_info
    )
    best_checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(best_checkpoint["model"])
    frozen_hash_after = state_hash(model, include_affine=False)
    affine_hash_after = state_hash(model, include_affine=True)
    if frozen_hash_after != frozen_hash_before:
        raise RuntimeError("Final frozen representation hash differs from the E4A source.")
    if affine_hash_after == affine_hash_before:
        raise RuntimeError("E4B affine parameters did not change.")

    val_predictions, val_metrics, val_summary, auc_rows = evaluate_cache(
        model, val_cache, device, config["data"]["modalities"], threshold=0.5
    )
    calibration = calibrate_threshold(
        val_predictions["y_true"], val_predictions["y_prob"], metric="balanced_accuracy"
    )
    calibration.update(
        {
            "enabled": True,
            "mode": "global",
            "source": "frozen_validation_split_all_15_patterns_pooled",
            "num_patterns": 15,
            "num_predictions": int(len(val_predictions)),
            "calibrated_threshold": float(calibration["threshold"]),
        }
    )
    save_json(calibration, metrics_dir / "threshold_calibration.json")
    pd.DataFrame([{"combo": name, **metrics} for name, metrics in val_metrics.items()]).to_csv(
        metrics_dir / "validation_metrics.csv", index=False
    )
    pd.DataFrame(val_summary["groups"]).to_csv(metrics_dir / "validation_missing_pattern_groups.csv", index=False)
    save_json(val_summary, metrics_dir / "validation_missing_pattern_summary.json")
    auc_rows.to_csv(metrics_dir / "validation_auc_invariance.csv", index=False)
    pd.DataFrame(mask_affine_rows(model, combos)).to_csv(metrics_dir / "mask_affine_parameters.csv", index=False)
    best_checkpoint["calibrated_threshold"] = float(calibration["threshold"])
    best_checkpoint["threshold_calibration"] = calibration
    best_checkpoint["e4b"].update(
        {
            "frozen_hash_after": frozen_hash_after,
            "affine_hash_after": affine_hash_after,
            "max_abs_validation_delta_auc": float(auc_rows["delta_auc"].abs().max()),
            "best_epoch": int(best_checkpoint["epoch"]),
        }
    )
    torch.save(best_checkpoint, best_path)
    save_json(best_checkpoint["e4b"], metrics_dir / "frozen_parameter_audit.json")

    if args.smoke:
        save_json(
            {
                "status": "ok",
                "smoke": True,
                "best_epoch": int(best_checkpoint["epoch"]),
                "global_threshold": float(calibration["threshold"]),
                "validation_mean15_bal_acc": float(val_summary["mean15_bal_acc"]),
                "validation_mean15_auc": float(val_summary["mean15_auc"]),
                "max_abs_validation_delta_auc": float(auc_rows["delta_auc"].abs().max()),
                "frozen_hash_unchanged": frozen_hash_after == frozen_hash_before,
                "affine_hash_changed": affine_hash_after != affine_hash_before,
                "optimizer_affine_only": True,
                "validation_used_for_gradient": False,
                "test_read": False,
            },
            output_dir / "e4b_run_summary.json",
        )
        return

    # Phase B starts only after the best checkpoint and validation-only global threshold are frozen.
    test_cache = extract_base_logit_cache(model, config, device, "test", combos, smoke=args.smoke)
    test_records = smoke_records(splits["test"]) if args.smoke else splits["test"]
    test_case_ids = [record.case_id for record in test_records]
    validate_cache_frame(test_cache, "test", test_case_ids, combos)
    cache_metadata["test"] = save_cache(test_cache, cache_dir, "test")
    cache_metadata["test_cache_created_after_checkpoint_and_threshold_freeze"] = True
    save_json(cache_metadata, cache_dir / "cache_provenance.json")
    test_predictions, test_metrics, test_summary, test_auc_rows = evaluate_cache(
        model, test_cache, device, config["data"]["modalities"], threshold=float(calibration["threshold"])
    )
    test_predictions.to_csv(metrics_dir / "test_predictions.csv", index=False)
    pd.DataFrame([{"combo": name, **metrics} for name, metrics in test_metrics.items()]).to_csv(
        metrics_dir / "test_metrics.csv", index=False
    )
    save_json(test_metrics, metrics_dir / "test_metrics.json")
    pd.DataFrame(test_summary["groups"]).to_csv(metrics_dir / "test_missing_pattern_groups.csv", index=False)
    save_json(test_summary, metrics_dir / "test_missing_pattern_summary.json")
    test_auc_rows.to_csv(metrics_dir / "test_auc_invariance.csv", index=False)
    save_json(
        {
            "status": "ok",
            "smoke": bool(args.smoke),
            "best_epoch": int(best_checkpoint["epoch"]),
            "global_threshold": float(calibration["threshold"]),
            "validation_mean15_bal_acc": float(val_summary["mean15_bal_acc"]),
            "validation_mean15_auc": float(val_summary["mean15_auc"]),
            "test_mean15_bal_acc": float(test_summary["mean15_bal_acc"]),
            "test_mean15_auc": float(test_summary["mean15_auc"]),
            "max_abs_validation_delta_auc": float(auc_rows["delta_auc"].abs().max()),
            "frozen_hash_unchanged": frozen_hash_after == frozen_hash_before,
            "affine_hash_changed": affine_hash_after != affine_hash_before,
            "test_read_after_checkpoint_and_threshold_freeze": True,
        },
        output_dir / "e4b_run_summary.json",
    )


if __name__ == "__main__":
    main()
