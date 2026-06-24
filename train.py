from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW

from datasets.brats_dataset import get_all_modality_combinations
from models import HybridHypergraphClassifier
from utils import load_config, set_seed, setup_logger
from utils.config import ensure_dir
from utils.io import save_json
from utils.metrics import (
    build_drop_t1_ablation_report,
    calibrate_grouped_3way_t1ce_t1,
    calibrate_threshold,
    threshold_dispatch_for_combo,
)
from utils.runner import collect_predictions, evaluate_with_explanations, run_epoch
from utils.selection import (
    combo_name,
    finetune_eligibility,
    is_better_joint_summary,
    summarize_joint_validation,
)
from utils.training import build_dataloader, build_datasets, build_sampler, class_weights_from_records, dump_split_summary
from utils.visualization import save_fusion_weight_history


def parse_args():
    parser = argparse.ArgumentParser(description="BraTS HGG/LGG hybrid hypergraph demo training")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def collect_validation_predictions_for_combos(model, config, device, combos):
    y_true, y_prob, combo_tags = [], [], []
    for combo in combos:
        _, val_ds_combo, _, _ = build_datasets(config, explicit_eval_combo=combo)
        val_loader_combo = build_dataloader(
            val_ds_combo,
            config["train"]["batch_size"],
            config["data"].get("num_workers", 0),
            shuffle=False,
        )
        collected = collect_predictions(model, val_loader_combo, device)
        y_true.extend(collected["y_true"].tolist())
        y_prob.extend(collected["y_prob"].tolist())
        combo_tags.extend(collected.get("combos", [tuple(combo)] * len(collected["y_true"])))
    return {"y_true": y_true, "y_prob": y_prob, "combos": combo_tags}


def build_joint_validation_loaders(config, selection_cfg):
    combos = [
        tuple(selection_cfg["full_combo"]),
        *[tuple(combo) for combo in selection_cfg["no_t1ce_combos"]],
        *[tuple(combo) for combo in selection_cfg.get("monitor_combos", [])],
    ]
    loaders = {}
    for combo in dict.fromkeys(combos):
        _, val_ds, _, _ = build_datasets(config, explicit_eval_combo=combo)
        loaders[combo_name(combo)] = build_dataloader(
            val_ds,
            config["train"]["batch_size"],
            config["data"].get("num_workers", 0),
            shuffle=False,
        )
    return loaders


def evaluate_joint_validation(model, loaders, device, selection_cfg, calibration_cfg):
    model.eval()
    predictions = {}
    for name, loader in loaders.items():
        collected = collect_predictions(model, loader, device)
        predictions[name] = {
            "y_true": collected["y_true"],
            "y_prob": collected["y_prob"],
        }
    return summarize_joint_validation(predictions, selection_cfg, calibration_cfg)


def set_backbone_trainable(model, trainable: bool) -> None:
    if hasattr(model, "backbones"):
        for param in model.backbones.parameters():
            param.requires_grad = trainable


def select_device(device_name: str) -> torch.device:
    if device_name.startswith("cuda") and torch.cuda.is_available():
        return torch.device(device_name)
    return torch.device("cpu")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prediction_probability_sha256(path: str | Path) -> str:
    predictions = pd.read_csv(path)
    required = {"case_id", "prob"}
    if not required.issubset(predictions.columns):
        raise ValueError(f"Prediction file must contain {sorted(required)}: {path}")
    digest = hashlib.sha256()
    for row in predictions.itertuples(index=False):
        digest.update(str(row.case_id).encode("utf-8"))
        digest.update(b"\0")
        digest.update(struct.pack("!d", float(row.prob)))
    return digest.hexdigest()


def model_state_sha256(checkpoint_path: str | Path) -> str:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    digest = hashlib.sha256()
    for key in sorted(checkpoint["model"]):
        tensor = checkpoint["model"][key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def checkpoint_hash_report(ckpt_dir: Path) -> dict:
    report = {}
    for name in ["base_best.pt", "finetune_best.pt", "best.pt"]:
        path = ckpt_dir / name
        if path.exists():
            report[name] = {
                "file_sha256": file_sha256(path),
                "model_state_sha256": model_state_sha256(path),
            }
    return report


def history_to_rows(history):
    rows = []
    for item in history:
        row = {"epoch": item["epoch"], "stage": item.get("stage", "base")}
        for split_name in ["train", "val"]:
            for key, value in item.get(split_name, {}).items():
                row[f"{split_name}_{key}"] = value
        summary = item.get("joint_validation")
        if summary:
            for key in ["selection_score", "no_t1ce_mean_auc", "no_t1ce_mean_bal_acc", "full_auc", "full_bal_acc"]:
                row[f"joint_{key}"] = summary[key]
            for name, metrics in summary["combos"].items():
                for key in ["auc", "bal_acc", "sen", "spe", "threshold"]:
                    row[f"val_{name}_{key}"] = metrics[key]
        eligibility = item.get("finetune_eligibility")
        if eligibility:
            row["finetune_eligible"] = bool(eligibility["eligible"])
            for key, value in eligibility["deltas"].items():
                row[f"finetune_delta_{key}"] = value
        rows.append(row)
    return rows


def save_checkpoint(path: Path, model, config, epoch, score, stage, selection_summary=None, eligibility=None):
    torch.save(
        {
            "model": model.state_dict(),
            "config": config,
            "epoch": epoch,
            "score": float(score),
            "stage": stage,
            "selection_summary": selection_summary,
            "selection_eligibility": eligibility,
        },
        path,
    )


def main():
    args = parse_args()
    overrides = {}
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.output_dir is not None:
        overrides["output_dir"] = args.output_dir
    config = load_config(args.config, overrides=overrides or None)

    output_dir = ensure_dir(config["output_dir"])
    ckpt_dir = ensure_dir(output_dir / "checkpoints")
    metrics_dir = ensure_dir(output_dir / "metrics")
    save_json(config, output_dir / "resolved_config.json")

    set_seed(int(config["seed"]))
    logger = setup_logger(config.get("logging", {}).get("level", "INFO"), output_dir / "train.log")
    device = select_device(config["train"].get("device", "cuda"))
    logger.info("Using device: %s", device)

    train_ds, val_ds, _, splits = build_datasets(config)
    dump_split_summary(splits, str(output_dir))
    sampler = build_sampler(train_ds) if config["train"].get("weighted_sampler", False) else None
    train_loader = build_dataloader(
        train_ds,
        config["train"]["batch_size"],
        config["data"].get("num_workers", 0),
        shuffle=sampler is None,
        sampler=sampler,
    )
    val_loader = build_dataloader(
        val_ds,
        config["train"]["batch_size"],
        config["data"].get("num_workers", 0),
        shuffle=False,
    )

    selection_cfg = config["train"].get("joint_validation", {})
    joint_validation_enabled = bool(selection_cfg.get("enabled", False))
    joint_loaders = build_joint_validation_loaders(config, selection_cfg) if joint_validation_enabled else {}
    calibration_cfg = config.get("calibration", {})

    model = HybridHypergraphClassifier(config).to(device)
    if hasattr(model, "get_classifier_info"):
        logger.info("Mask-aware classifier: %s", json.dumps(model.get_classifier_info(), ensure_ascii=False))
    raw_class_weights = class_weights_from_records(splits["train"])
    class_weights = raw_class_weights.to(device)
    train_counts = pd.Series([record.label for record in splits["train"]]).value_counts().reindex([0, 1], fill_value=0).to_dict()
    logger.info("Train class counts: %s", train_counts)
    logger.info("Class-balanced CE weights: %s", [float(value) for value in raw_class_weights.tolist()])
    loss_cfg = config.get("loss", {})
    use_balanced_loss = bool(loss_cfg.get("use_class_balanced_loss", config["train"].get("weighted_ce", True)))
    loss_type = str(loss_cfg.get("loss_type", "weighted_ce" if use_balanced_loss else "ce"))
    criterion = nn.CrossEntropyLoss(weight=class_weights if use_balanced_loss and loss_type in {"weighted_ce", "ce"} else None)
    optimizer = AdamW(model.parameters(), lr=config["train"]["lr"], weight_decay=config["train"].get("weight_decay", 0.0))

    best_score = -1.0
    best_joint_summary = None
    bad_epochs = 0
    history = []
    metric_name = config["train"].get("save_metric", "auc")
    base_best_path = ckpt_dir / "base_best.pt"
    best_path = ckpt_dir / "best.pt"

    for epoch in range(1, int(config["train"]["epochs"]) + 1):
        if hasattr(train_ds, "set_epoch"):
            train_ds.set_epoch(epoch, int(config["train"]["epochs"]), config["train"])
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            criterion=criterion,
            optimizer=optimizer,
            grad_clip=config["train"].get("grad_clip", 0.0),
            threshold=config["eval"].get("threshold", 0.5),
            desc=f"train {epoch}",
        )
        if joint_validation_enabled:
            joint_summary = evaluate_joint_validation(model, joint_loaders, device, selection_cfg, calibration_cfg)
            val_metrics = joint_summary["combos"][joint_summary["full_combo"]]
            score = float(joint_summary["selection_score"])
            improved = is_better_joint_summary(joint_summary, best_joint_summary)
        else:
            joint_summary = None
            val_metrics = run_epoch(
                model,
                val_loader,
                device,
                criterion=criterion,
                optimizer=None,
                threshold=config["eval"].get("threshold", 0.5),
                desc=f"val {epoch}",
            )
            score = val_metrics.get(metric_name, float("nan"))
            if score != score:
                score = val_metrics.get("bal_acc", 0.0)
            improved = score > best_score

        history.append({"epoch": epoch, "stage": "base", "train": train_metrics, "val": val_metrics, "joint_validation": joint_summary})
        logger.info("Epoch %d | train=%s | val=%s", epoch, json.dumps(train_metrics, ensure_ascii=False), json.dumps(val_metrics, ensure_ascii=False))
        if joint_summary:
            logger.info("Joint validation epoch %d | %s", epoch, json.dumps({key: joint_summary[key] for key in ["selection_score", "no_t1ce_mean_auc", "no_t1ce_mean_bal_acc", "full_auc", "full_bal_acc"]}, ensure_ascii=False))

        if improved:
            best_score = float(score)
            best_joint_summary = joint_summary
            bad_epochs = 0
            save_checkpoint(base_best_path, model, config, epoch, score, "base", joint_summary)
            shutil.copy2(base_best_path, best_path)
            logger.info("Saved new base_best checkpoint with protocol score=%.4f", score)
        else:
            bad_epochs += 1
            if bad_epochs >= int(config["train"].get("early_stop_patience", 8)):
                logger.info("Early stopping triggered at epoch %d", epoch)
                break

    base_checkpoint = torch.load(base_best_path, map_location=device)
    model.load_state_dict(base_checkpoint["model"])
    base_joint_summary = base_checkpoint.get("selection_summary")
    selected_source_path = base_best_path
    selection_report = {
        "status": "base_only",
        "selected_checkpoint": "base_best.pt",
        "base_selection_summary": base_joint_summary,
        "finetune_selection_summary": None,
        "finetune_eligibility": None,
        "joint_validation_enabled": joint_validation_enabled,
    }

    if config["train"].get("enable_targeted_finetune", False):
        train_ds.combo_mode = config["train"].get("fine_tune_sampling_mode", "targeted_no_t1ce")
        fine_tune_epochs = int(config["train"].get("fine_tune_epochs", 8))
        fine_tune_lr = float(config["train"]["lr"]) * float(config["train"].get("fine_tune_lr_scale", 0.1))
        freeze_backbone = bool(config["train"].get("freeze_backbone_in_finetune", False))
        logger.info(
            "Targeted fine-tune enabled | epochs=%d | lr=%.6g | sampling=%s | freeze_backbone=%s",
            fine_tune_epochs,
            fine_tune_lr,
            train_ds.combo_mode,
            freeze_backbone,
        )
        if freeze_backbone:
            set_backbone_trainable(model, False)
        fine_tune_optimizer = AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=fine_tune_lr,
            weight_decay=config["train"].get("weight_decay", 0.0),
        )
        finetune_best_path = ckpt_dir / "finetune_best.pt"
        best_unconstrained_summary = None
        best_unconstrained_score = -1.0
        best_eligible_summary = None
        best_eligible_score = -1.0
        best_eligibility = None
        has_eligible_finetune = False

        for ft_epoch in range(1, fine_tune_epochs + 1):
            if hasattr(train_ds, "set_epoch"):
                train_ds.set_epoch(ft_epoch, fine_tune_epochs, config["train"])
            train_metrics = run_epoch(
                model,
                train_loader,
                device,
                criterion=criterion,
                optimizer=fine_tune_optimizer,
                grad_clip=config["train"].get("grad_clip", 0.0),
                threshold=config["eval"].get("threshold", 0.5),
                desc=f"finetune {ft_epoch}",
            )
            if joint_validation_enabled:
                joint_summary = evaluate_joint_validation(model, joint_loaders, device, selection_cfg, calibration_cfg)
                val_metrics = joint_summary["combos"][joint_summary["full_combo"]]
                score = float(joint_summary["selection_score"])
                eligibility = finetune_eligibility(joint_summary, base_joint_summary, selection_cfg)
                better_unconstrained = is_better_joint_summary(joint_summary, best_unconstrained_summary)
                better_eligible = is_better_joint_summary(joint_summary, best_eligible_summary)
            else:
                joint_summary = None
                val_metrics = run_epoch(
                    model,
                    val_loader,
                    device,
                    criterion=criterion,
                    optimizer=None,
                    threshold=config["eval"].get("threshold", 0.5),
                    desc=f"finetune-val {ft_epoch}",
                )
                score = val_metrics.get(metric_name, float("nan"))
                if score != score:
                    score = val_metrics.get("bal_acc", 0.0)
                eligibility = {"eligible": score > best_score, "checks": {metric_name: score > best_score}, "deltas": {metric_name: score - best_score}}
                better_unconstrained = score > best_unconstrained_score
                better_eligible = score > best_eligible_score

            history.append({
                "epoch": f"finetune_{ft_epoch}",
                "stage": "targeted_finetune",
                "train": train_metrics,
                "val": val_metrics,
                "joint_validation": joint_summary,
                "finetune_eligibility": eligibility,
            })
            logger.info("Fine-tune epoch %d | eligible=%s | val=%s", ft_epoch, eligibility["eligible"], json.dumps(val_metrics, ensure_ascii=False))

            if better_unconstrained:
                best_unconstrained_summary = joint_summary
                best_unconstrained_score = float(score)
                if not has_eligible_finetune:
                    save_checkpoint(finetune_best_path, model, config, f"finetune_{ft_epoch}", score, "targeted_finetune", joint_summary, eligibility)
            if eligibility["eligible"] and better_eligible:
                best_eligible_summary = joint_summary
                best_eligible_score = float(score)
                best_eligibility = eligibility
                has_eligible_finetune = True
                save_checkpoint(finetune_best_path, model, config, f"finetune_{ft_epoch}", score, "targeted_finetune", joint_summary, eligibility)
                logger.info("Saved eligible finetune_best checkpoint with protocol score=%.4f", score)

        if freeze_backbone:
            set_backbone_trainable(model, True)
        train_ds.combo_mode = config["train"].get("mode", "full_modality_train")

        if not finetune_best_path.exists():
            raise RuntimeError("Targeted fine-tune produced no finetune_best.pt checkpoint.")
        finetune_checkpoint = torch.load(finetune_best_path, map_location=device)
        if has_eligible_finetune:
            selected_source_path = finetune_best_path
            selection_report.update({
                "status": "finetune_selected",
                "selected_checkpoint": "finetune_best.pt",
                "finetune_selection_summary": best_eligible_summary,
                "finetune_eligibility": best_eligibility,
            })
        else:
            selection_report.update({
                "status": "no_eligible_finetune",
                "selected_checkpoint": "base_best.pt",
                "finetune_selection_summary": finetune_checkpoint.get("selection_summary"),
                "finetune_eligibility": finetune_checkpoint.get("selection_eligibility"),
                "warning": "No fine-tune epoch satisfied the joint validation guardrails; base_best.pt was selected explicitly.",
            })
            logger.warning(selection_report["warning"])

    shutil.copy2(selected_source_path, best_path)

    save_json(history, metrics_dir / "train_history.json")
    history_df = pd.DataFrame(history_to_rows(history))
    history_df.to_csv(metrics_dir / "train_history.csv", index=False)
    fusion_plot_df = pd.DataFrame({"epoch": history_df["epoch"]})
    for column in ["val_alpha", "val_beta", "val_gamma"]:
        if column in history_df.columns:
            fusion_plot_df[column.replace("val_", "")] = history_df[column]
    if len(fusion_plot_df.columns) > 1:
        save_fusion_weight_history(fusion_plot_df, metrics_dir / "fusion_weight_history.png")

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model"])

    calibrated_threshold = float(config["eval"].get("threshold", calibration_cfg.get("default_threshold", 0.5)))
    calibration_result = {
        "default_threshold": float(calibration_cfg.get("default_threshold", 0.5)),
        "threshold": calibrated_threshold,
        "calibrated_threshold": calibrated_threshold,
        "metric": calibration_cfg.get("threshold_metric", "balanced_accuracy"),
        "calibration_metric": calibration_cfg.get("threshold_metric", "balanced_accuracy"),
        "score": None,
        "enabled": False,
    }
    if calibration_cfg.get("enable_threshold_calibration", True):
        calibration_mode = calibration_cfg.get("threshold_mode", "global")
        if calibration_mode == "grouped_3way_t1ce_t1":
            calibration_combos = get_all_modality_combinations(config["data"]["modalities"])
            val_collected = collect_validation_predictions_for_combos(model, config, device, calibration_combos)
            calibration_result = calibrate_grouped_3way_t1ce_t1(
                val_collected["y_true"],
                val_collected["y_prob"],
                val_collected["combos"],
                metric=calibration_cfg.get("threshold_metric", "balanced_accuracy"),
                default_threshold=float(calibration_cfg.get("default_threshold", 0.5)),
                min_samples=int(calibration_cfg.get("min_samples", 2)),
                min_positive=int(calibration_cfg.get("min_positive", 1)),
                min_negative=int(calibration_cfg.get("min_negative", 1)),
                threshold_min=float(calibration_cfg.get("threshold_min", 0.0)),
                threshold_max=float(calibration_cfg.get("threshold_max", 1.0)),
                tie_break=str(calibration_cfg.get("tie_break", "first")),
            )
        else:
            val_collected = collect_predictions(model, val_loader, device)
            calibration_result = calibrate_threshold(
                val_collected["y_true"],
                val_collected["y_prob"],
                metric=calibration_cfg.get("threshold_metric", "balanced_accuracy"),
                threshold_min=float(calibration_cfg.get("threshold_min", 0.0)),
                threshold_max=float(calibration_cfg.get("threshold_max", 1.0)),
                tie_break=str(calibration_cfg.get("tie_break", "first")),
            )
            calibration_result["mode"] = "global"
        calibration_result["enabled"] = True
        calibrated_threshold = float(calibration_result.get("threshold", calibration_result.get("calibrated_threshold", calibrated_threshold)))
        calibration_result["calibrated_threshold"] = calibrated_threshold
        calibration_result["calibration_metric"] = calibration_result.get("metric", calibration_cfg.get("threshold_metric", "balanced_accuracy"))

    checkpoint["calibrated_threshold"] = calibrated_threshold
    checkpoint["threshold_calibration"] = calibration_result
    checkpoint["mask_aware_classifier"] = model.get_classifier_info() if hasattr(model, "get_classifier_info") else {}
    checkpoint["targeted_finetune"] = {
        "enabled": bool(config["train"].get("enable_targeted_finetune", False)),
        "fine_tune_epochs": int(config["train"].get("fine_tune_epochs", 0)),
        "fine_tune_lr_scale": float(config["train"].get("fine_tune_lr_scale", 0.1)),
        "fine_tune_sampling_mode": config["train"].get("fine_tune_sampling_mode", "targeted_no_t1ce"),
    }
    checkpoint["checkpoint_selection"] = selection_report
    torch.save(checkpoint, best_path)
    shutil.copy2(best_path, selected_source_path)

    checkpoint_hashes = checkpoint_hash_report(ckpt_dir)
    selection_report["checkpoint_hashes"] = checkpoint_hashes
    save_json(selection_report, metrics_dir / "checkpoint_selection.json")
    save_json(checkpoint_hashes, metrics_dir / "checkpoint_hashes.json")
    save_json(calibration_result, metrics_dir / "threshold_calibration.json")

    if calibration_result.get("fallback_info"):
        for group_name, info in calibration_result["fallback_info"].items():
            logger.warning(
                "Calibration fallback | group=%s | reason=%s | applied_threshold=%.4f | target=%s",
                group_name,
                info.get("reason"),
                float(info.get("applied_threshold", calibrated_threshold)),
                info.get("fallback_target", ""),
            )
    logger.info("Checkpoint selection | %s", json.dumps(selection_report, ensure_ascii=False))

    combos = (
        get_all_modality_combinations(config["data"]["modalities"])
        if config["eval"].get("test_all_combos", True)
        else [tuple(config["train"].get("fixed_combo", config["data"]["modalities"]))]
    )
    all_test_metrics = {}
    test_rows = []
    for combo in combos:
        _, _, test_ds_combo, _ = build_datasets(config, explicit_eval_combo=combo)
        test_loader = build_dataloader(
            test_ds_combo,
            config["train"]["batch_size"],
            config["data"].get("num_workers", 0),
            shuffle=False,
        )
        name = combo_name(combo)
        logger.info("Evaluating combo: %s", name)
        threshold_dispatch = threshold_dispatch_for_combo(calibration_result, combo, calibrated_threshold)
        metrics = evaluate_with_explanations(
            model,
            test_loader,
            device,
            config["model"]["roi_names"],
            str(output_dir / "test" / name),
            threshold=threshold_dispatch["applied_threshold"],
            threshold_group=threshold_dispatch["threshold_group"],
            explain_num_cases=config["eval"].get("explain_num_cases", 3),
            roi_drop_enabled=config["eval"].get("roi_drop_enabled", True),
            edge_type_drop_enabled=config["eval"].get("edge_type_drop_enabled", True),
        )
        all_test_metrics[name] = metrics
        test_rows.append({"combo": name, "threshold_group": threshold_dispatch["threshold_group"], "applied_threshold": threshold_dispatch["applied_threshold"], **metrics})
        logger.info("Test metrics for %s: %s", name, json.dumps(metrics, ensure_ascii=False))

    if config.get("test", {}).get("enable_drop_t1_inference_ablation", False):
        ablation_payload = build_drop_t1_ablation_report(
            all_test_metrics,
            combo_names=config.get("test", {}).get("drop_t1_ablation_combos", ["t1", "t2_t1", "t1_flair", "t2_t1_flair"]),
        )
        save_json(ablation_payload, metrics_dir / "drop_t1_ablation.json")
        pd.DataFrame(ablation_payload["rows"]).to_csv(metrics_dir / "drop_t1_ablation.csv", index=False)
        pd.DataFrame(ablation_payload["summary"]).to_csv(metrics_dir / "drop_t1_ablation_summary.csv", index=False)

    save_json(all_test_metrics, metrics_dir / "test_metrics.json")
    pd.DataFrame(test_rows).to_csv(metrics_dir / "test_metrics.csv", index=False)
    prediction_hashes = {
        path.parent.name: {
            "file_sha256": file_sha256(path),
            "probability_sha256": prediction_probability_sha256(path),
        }
        for path in sorted((output_dir / "test").glob("*/predictions.csv"))
    }
    save_json(prediction_hashes, metrics_dir / "prediction_hashes.json")


if __name__ == "__main__":
    main()
