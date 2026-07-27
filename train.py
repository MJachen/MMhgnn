from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW

from datasets.brats_dataset import get_all_modality_combinations
from models import HybridHypergraphClassifier
from utils import load_config, set_seed, setup_logger
from utils.b1_checkpoint import flatten_selection_summary, summarize_combo_metrics
from utils.config import ensure_dir
from utils.io import save_json
from utils.metrics import build_drop_t1_ablation_report, calibrate_grouped_3way_t1ce_t1, calibrate_threshold, compute_binary_metrics, threshold_dispatch_for_combo, threshold_for_combo
from utils.runner import collect_predictions, evaluate_with_explanations, run_epoch
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
        val_loader_combo = build_dataloader(val_ds_combo, config["train"]["batch_size"], config["data"].get("num_workers", 0), shuffle=False)
        collected = collect_predictions(model, val_loader_combo, device)
        y_true.extend(collected["y_true"].tolist())
        y_prob.extend(collected["y_prob"].tolist())
        combo_tags.extend(collected.get("combos", [tuple(combo)] * len(collected["y_true"])))
    return {"y_true": y_true, "y_prob": y_prob, "combos": combo_tags}


def evaluate_b1_checkpoint_selection(
    model,
    val_ds,
    val_loader,
    config,
    device,
    cached_full_metrics=None,
):
    """Evaluate every non-empty modality combo and compute the B1 selection score."""
    model.eval()
    modalities = tuple(config["data"]["modalities"])
    combos = get_all_modality_combinations(modalities)
    threshold = float(
        config.get("train", {})
        .get("checkpoint_selection", {})
        .get("threshold", config["eval"].get("threshold", 0.5))
    )
    original_combo = val_ds.explicit_combo
    combo_metrics = {}
    try:
        for combo in combos:
            name = "_".join(combo)
            if tuple(combo) == modalities and cached_full_metrics is not None:
                metrics = {
                    key: float(cached_full_metrics.get(key, float("nan")))
                    for key in ("acc", "auc", "f1", "sen", "spe", "bal_acc")
                }
            else:
                val_ds.explicit_combo = tuple(combo)
                collected = collect_predictions(model, val_loader, device)
                metrics = compute_binary_metrics(
                    collected["y_true"],
                    collected["y_prob"],
                    threshold=threshold,
                )
            combo_metrics[name] = metrics
    finally:
        val_ds.explicit_combo = original_combo

    selection_cfg = config["train"].get("checkpoint_selection", {})
    return summarize_combo_metrics(
        combo_metrics,
        modalities=modalities,
        weights=selection_cfg.get("weights"),
    )


def persist_selection_history(selection_history, metrics_dir):
    if not selection_history:
        return
    save_json(selection_history, metrics_dir / "checkpoint_selection_history.json")
    rows = []
    for item in selection_history:
        row = {"phase": item["phase"], "epoch": item["epoch"]}
        row.update(flatten_selection_summary(item["summary"]))
        rows.append(row)
    pd.DataFrame(rows).to_csv(
        metrics_dir / "checkpoint_selection_history.csv",
        index=False,
    )


def set_backbone_trainable(model, trainable: bool) -> None:
    if hasattr(model, "backbones"):
        for param in model.backbones.parameters():
            param.requires_grad = trainable


def select_device(device_name: str) -> torch.device:
    if device_name.startswith("cuda") and torch.cuda.is_available():
        return torch.device(device_name)
    return torch.device("cpu")


def main():
    args = parse_args()
    overrides = {}
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.output_dir is not None:
        overrides["output_dir"] = args.output_dir
    config = load_config(args.config, overrides=overrides or None)

    set_seed(int(config["seed"]))
    logger = setup_logger(config.get("logging", {}).get("level", "INFO"))

    output_dir = ensure_dir(config["output_dir"])
    ckpt_dir = ensure_dir(output_dir / "checkpoints")
    metrics_dir = ensure_dir(output_dir / "metrics")
    save_json(config, output_dir / "resolved_config.json")

    device = select_device(config["train"].get("device", "cuda"))
    logger.info("Using device: %s", device)

    train_ds, val_ds, test_ds, splits = build_datasets(config)
    dump_split_summary(splits, str(output_dir))
    sampler = build_sampler(train_ds) if config["train"].get("weighted_sampler", False) else None
    train_loader = build_dataloader(train_ds, config["train"]["batch_size"], config["data"].get("num_workers", 0), shuffle=sampler is None, sampler=sampler)
    val_loader = build_dataloader(val_ds, config["train"]["batch_size"], config["data"].get("num_workers", 0), shuffle=False)

    model = HybridHypergraphClassifier(config).to(device)
    if hasattr(model, "get_classifier_info"):
        logger.info("Mask-aware classifier: %s", json.dumps(model.get_classifier_info(), ensure_ascii=False))
    raw_class_weights = class_weights_from_records(splits["train"])
    class_weights = raw_class_weights.to(device)
    train_counts = pd.Series([record.label for record in splits["train"]]).value_counts().reindex([0, 1], fill_value=0).to_dict()
    logger.info("Train class counts: %s", train_counts)
    logger.info("Class-balanced CE weights: %s", [float(v) for v in raw_class_weights.tolist()])
    loss_cfg = config.get("loss", {})
    use_balanced_loss = bool(loss_cfg.get("use_class_balanced_loss", config["train"].get("weighted_ce", True)))
    loss_type = str(loss_cfg.get("loss_type", "weighted_ce" if use_balanced_loss else "ce"))
    criterion = nn.CrossEntropyLoss(weight=class_weights if use_balanced_loss and loss_type in {"weighted_ce", "ce"} else None)
    optimizer = AdamW(model.parameters(), lr=config["train"]["lr"], weight_decay=config["train"].get("weight_decay", 0.0))
    auxiliary_cfg = config["model"].get("auxiliary_segmentation", {})
    segmentation_config = {
        "enabled": bool(auxiliary_cfg.get("enabled", False)),
        "lambda_seg": float(config["train"].get("lambda_seg", 0.0)),
        "seg_bce_weight": float(config["train"].get("seg_bce_weight", 0.5)),
        "seg_dice_weight": float(config["train"].get("seg_dice_weight", 0.5)),
    }
    logger.info("Auxiliary WT segmentation: %s", json.dumps(segmentation_config, ensure_ascii=False))

    best_score = -1.0
    bad_epochs = 0
    history = []
    metric_name = config["train"].get("save_metric", "auc")
    selection_cfg = config["train"].get("checkpoint_selection", {})
    selection_enabled = bool(selection_cfg.get("enabled", False))
    selection_history = []
    checkpoint_metric_name = "b1_selection_score" if selection_enabled else metric_name
    if selection_enabled:
        logger.info(
            "B1 checkpoint selection enabled | score=%s | groups=%s | threshold=%.3f",
            json.dumps(
                selection_cfg.get(
                    "weights",
                    {
                        "macro_bal_acc": 0.5,
                        "worst_group_bal_acc": 0.3,
                        "full_bal_acc": 0.2,
                    },
                ),
                ensure_ascii=False,
            ),
            "has_t1ce,no_t1ce_no_t1,no_t1ce_with_t1",
            float(selection_cfg.get("threshold", config["eval"].get("threshold", 0.5))),
        )

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
            segmentation_config=segmentation_config,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            device,
            criterion=criterion,
            optimizer=None,
            threshold=config["eval"].get("threshold", 0.5),
            desc=f"val {epoch}",
            segmentation_config=segmentation_config,
        )
        selection_summary = None
        if selection_enabled:
            selection_summary = evaluate_b1_checkpoint_selection(
                model,
                val_ds,
                val_loader,
                config,
                device,
                cached_full_metrics=val_metrics,
            )
            score = float(selection_summary["selection_score"])
            selection_history.append(
                {"phase": "train", "epoch": epoch, "summary": selection_summary}
            )
            persist_selection_history(selection_history, metrics_dir)
        else:
            score = val_metrics.get(metric_name, float("nan"))
            if score != score:
                score = val_metrics.get("bal_acc", 0.0)

        history_item = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        if selection_summary is not None:
            history_item["selection"] = selection_summary
        history.append(history_item)
        logger.info("Epoch %d | train=%s | val=%s", epoch, json.dumps(train_metrics, ensure_ascii=False), json.dumps(val_metrics, ensure_ascii=False))
        if selection_summary is not None:
            logger.info(
                "Epoch %d | B1 selection=%.4f | macro BA=%.4f | worst-group BA=%.4f | full BA=%.4f | groups=%s",
                epoch,
                score,
                float(selection_summary["macro_bal_acc"]),
                float(selection_summary["worst_group_bal_acc"]),
                float(selection_summary["full_bal_acc"]),
                json.dumps(selection_summary["group_bal_acc"], ensure_ascii=False),
            )

        if score > best_score:
            best_score = score
            bad_epochs = 0
            checkpoint_payload = {
                "model": model.state_dict(),
                "config": config,
                "epoch": epoch,
                "score": score,
                "selection_metric": checkpoint_metric_name,
            }
            if selection_summary is not None:
                checkpoint_payload["checkpoint_selection"] = selection_summary
            torch.save(checkpoint_payload, ckpt_dir / "best.pt")
            logger.info("Saved new best checkpoint with %s=%.4f", checkpoint_metric_name, score)
        else:
            bad_epochs += 1
            if bad_epochs >= int(config["train"].get("early_stop_patience", 8)):
                logger.info("Early stopping triggered at epoch %d", epoch)
                break

    if config["train"].get("enable_targeted_finetune", False):
        base_checkpoint = torch.load(ckpt_dir / "best.pt", map_location=device)
        torch.save(base_checkpoint, ckpt_dir / "base_best.pt")
        model.load_checkpoint_state_dict(base_checkpoint["model"])
        train_ds.combo_mode = config["train"].get("fine_tune_sampling_mode", "targeted_no_t1ce")
        fine_tune_epochs = int(config["train"].get("fine_tune_epochs", 8))
        fine_tune_lr = float(config["train"]["lr"]) * float(config["train"].get("fine_tune_lr_scale", 0.1))
        freeze_backbone = bool(config["train"].get("freeze_backbone_in_finetune", False))
        logger.info(
            "Targeted fine-tune enabled | epochs=%d | lr=%.6g | sampling=%s | ratios=%s | freeze_backbone=%s",
            fine_tune_epochs,
            fine_tune_lr,
            train_ds.combo_mode,
            {
                "full": config["train"].get("fine_tune_ratio_full", 0.7),
                "t2_only": config["train"].get("fine_tune_ratio_t2_only", 0.1),
                "flair_only": config["train"].get("fine_tune_ratio_flair_only", 0.1),
                "t2_flair": config["train"].get("fine_tune_ratio_t2_flair", 0.1),
                "drop_t1ce_fullcontext": config["train"].get("fine_tune_ratio_drop_t1ce_fullcontext", 0.0),
            },
            freeze_backbone,
        )
        if freeze_backbone:
            set_backbone_trainable(model, False)
        fine_tune_optimizer = AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=fine_tune_lr,
            weight_decay=config["train"].get("weight_decay", 0.0),
        )
        fine_tune_best = best_score
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
                segmentation_config=segmentation_config,
            )
            val_metrics = run_epoch(
                model,
                val_loader,
                device,
                criterion=criterion,
                optimizer=None,
                threshold=config["eval"].get("threshold", 0.5),
                desc=f"finetune-val {ft_epoch}",
                segmentation_config=segmentation_config,
            )
            selection_summary = None
            if selection_enabled:
                selection_summary = evaluate_b1_checkpoint_selection(
                    model,
                    val_ds,
                    val_loader,
                    config,
                    device,
                    cached_full_metrics=val_metrics,
                )
                score = float(selection_summary["selection_score"])
                selection_history.append(
                    {
                        "phase": "finetune",
                        "epoch": ft_epoch,
                        "summary": selection_summary,
                    }
                )
                persist_selection_history(selection_history, metrics_dir)
            else:
                score = val_metrics.get(metric_name, float("nan"))
                if score != score:
                    score = val_metrics.get("bal_acc", 0.0)
            history_item = {
                "epoch": f"finetune_{ft_epoch}",
                "train": train_metrics,
                "val": val_metrics,
            }
            if selection_summary is not None:
                history_item["selection"] = selection_summary
            history.append(history_item)
            logger.info("Fine-tune epoch %d | train=%s | val=%s", ft_epoch, json.dumps(train_metrics, ensure_ascii=False), json.dumps(val_metrics, ensure_ascii=False))
            if selection_summary is not None:
                logger.info(
                    "Fine-tune epoch %d | B1 selection=%.4f | macro BA=%.4f | worst-group BA=%.4f | full BA=%.4f | groups=%s",
                    ft_epoch,
                    score,
                    float(selection_summary["macro_bal_acc"]),
                    float(selection_summary["worst_group_bal_acc"]),
                    float(selection_summary["full_bal_acc"]),
                    json.dumps(selection_summary["group_bal_acc"], ensure_ascii=False),
                )
            if score > fine_tune_best:
                fine_tune_best = score
                checkpoint_payload = {
                    "model": model.state_dict(),
                    "config": config,
                    "epoch": f"finetune_{ft_epoch}",
                    "score": score,
                    "stage": "targeted_finetune",
                    "selection_metric": checkpoint_metric_name,
                }
                if selection_summary is not None:
                    checkpoint_payload["checkpoint_selection"] = selection_summary
                torch.save(checkpoint_payload, ckpt_dir / "best.pt")
                logger.info("Saved fine-tuned best checkpoint with %s=%.4f", checkpoint_metric_name, score)
        if freeze_backbone:
            set_backbone_trainable(model, True)
        train_ds.combo_mode = config["train"].get("mode", "full_modality_train")

    save_json(history, metrics_dir / "train_history.json")
    history_rows = []
    for item in history:
        row = {"epoch": item["epoch"]}
        for split_name in ["train", "val"]:
            for key, value in item[split_name].items():
                row[f"{split_name}_{key}"] = value
        if "selection" in item:
            for key, value in flatten_selection_summary(item["selection"]).items():
                row[f"selection_{key}"] = value
        history_rows.append(row)
    history_df = pd.DataFrame(history_rows)
    history_df.to_csv(metrics_dir / "train_history.csv", index=False)
    fusion_plot_df = pd.DataFrame({"epoch": history_df["epoch"]})
    for col in ["val_alpha", "val_beta", "val_gamma"]:
        if col in history_df.columns:
            fusion_plot_df[col.replace("val_", "")] = history_df[col]
    if len(fusion_plot_df.columns) > 1:
        save_fusion_weight_history(fusion_plot_df, metrics_dir / "fusion_weight_history.png")

    checkpoint_path = ckpt_dir / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_checkpoint_state_dict(checkpoint["model"])

    calibration_cfg = config.get("calibration", {})
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
            )
        else:
            val_collected = collect_predictions(model, val_loader, device)
            calibration_result = calibrate_threshold(
                val_collected["y_true"],
                val_collected["y_prob"],
                metric=calibration_cfg.get("threshold_metric", "balanced_accuracy"),
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
        torch.save(checkpoint, checkpoint_path)
    if calibration_result.get("fallback_info"):
        for group_name, info in calibration_result["fallback_info"].items():
            logger.warning("Calibration fallback | group=%s | reason=%s | applied_threshold=%.4f | target=%s", group_name, info.get("reason"), float(info.get("applied_threshold", calibrated_threshold)), info.get("fallback_target", ""))
    save_json(calibration_result, metrics_dir / "threshold_calibration.json")
    logger.info(
        "Threshold calibration | mode=%s | default=%.3f | calibrated=%.3f | has_t1ce=%s | no_t1ce_no_t1=%s | no_t1ce_with_t1=%s | metric=%s | score=%s",
        calibration_result.get("mode", "global"),
        float(calibration_result.get("default_threshold", 0.5)),
        calibrated_threshold,
        calibration_result.get("threshold_has_t1ce", ""),
        calibration_result.get("threshold_no_t1ce_no_t1", ""),
        calibration_result.get("threshold_no_t1ce_with_t1", ""),
        calibration_result.get("metric"),
        calibration_result.get("score"),
    )

    combos = get_all_modality_combinations(config["data"]["modalities"]) if config["eval"].get("test_all_combos", True) else [tuple(config["train"].get("fixed_combo", config["data"]["modalities"]))]

    all_test_metrics = {}
    test_rows = []
    for combo in combos:
        _, _, test_ds_combo, _ = build_datasets(config, explicit_eval_combo=combo)
        test_loader = build_dataloader(test_ds_combo, config["train"]["batch_size"], config["data"].get("num_workers", 0), shuffle=False)
        combo_name = "_".join(combo)
        logger.info("Evaluating combo: %s", combo_name)
        threshold_dispatch = threshold_dispatch_for_combo(calibration_result, combo, calibrated_threshold)
        metrics = evaluate_with_explanations(
            model,
            test_loader,
            device,
            config["model"]["roi_names"],
            str(output_dir / "test" / combo_name),
            threshold=threshold_dispatch["applied_threshold"],
            threshold_group=threshold_dispatch["threshold_group"],
            explain_num_cases=config["eval"].get("explain_num_cases", 3),
            roi_drop_enabled=config["eval"].get("roi_drop_enabled", True),
            edge_type_drop_enabled=config["eval"].get("edge_type_drop_enabled", True),
        )
        all_test_metrics[combo_name] = metrics
        test_rows.append({"combo": combo_name, "threshold_group": threshold_dispatch["threshold_group"], "applied_threshold": threshold_dispatch["applied_threshold"], **metrics})
        logger.info("Test metrics for %s: %s", combo_name, json.dumps(metrics, ensure_ascii=False))

    if config.get("test", {}).get("enable_drop_t1_inference_ablation", False):
        ablation_payload = build_drop_t1_ablation_report(
            all_test_metrics if "all_test_metrics" in locals() else all_metrics,
            combo_names=config.get("test", {}).get("drop_t1_ablation_combos", ["t1", "t2_t1", "t1_flair", "t2_t1_flair"]),
        )
        save_json(ablation_payload, metrics_dir / "drop_t1_ablation.json")
        pd.DataFrame(ablation_payload["rows"]).to_csv(metrics_dir / "drop_t1_ablation.csv", index=False)
        pd.DataFrame(ablation_payload["summary"]).to_csv(metrics_dir / "drop_t1_ablation_summary.csv", index=False)
        logger.info("Saved drop-t1 inference ablation with %d rows", len(ablation_payload["rows"]))

    save_json(all_test_metrics, metrics_dir / "test_metrics.json")
    pd.DataFrame(test_rows).to_csv(metrics_dir / "test_metrics.csv", index=False)


if __name__ == "__main__":
    main()
