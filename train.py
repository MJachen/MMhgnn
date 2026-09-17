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
from utils.config import ensure_dir
from utils.curriculum import curriculum_sampling_ratios, reset_bad_epochs_on_stage3_entry, should_early_stop, stage3_status
from utils.io import load_json, save_json
from utils.metrics import build_drop_t1_ablation_report, calibrate_grouped_3way_t1ce_t1, calibrate_threshold, compute_binary_metrics, summarize_missing_pattern_metrics, threshold_dispatch_for_combo, threshold_for_combo
from utils.runner import collect_predictions, evaluate_with_explanations, run_auxiliary_epoch, run_dual_view_epoch, run_epoch
from utils.training import build_dataloader, build_datasets, build_sampler, class_weights_from_records, dump_split_summary, task_uncertainty_summary
from utils.visualization import save_fusion_weight_history


def parse_args():
    parser = argparse.ArgumentParser(description="BraTS/UTSW hybrid hypergraph classification training")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--manifest-csv", type=str, default=None)
    parser.add_argument("--split-json", type=str, default=None)
    parser.add_argument("--manifest-fingerprint-json", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    return parser.parse_args()


def collect_validation_predictions_for_combos(model, config, device, combos, threshold: float = 0.5):
    y_true, y_prob, y_score, combo_tags = [], [], [], []
    metrics_by_combo = {}
    for combo in combos:
        _, val_ds_combo, _, _ = build_datasets(config, explicit_eval_combo=combo)
        val_loader_combo = build_dataloader(val_ds_combo, config["train"]["batch_size"], config["data"].get("num_workers", 0), shuffle=False)
        collected = collect_predictions(model, val_loader_combo, device)
        combo_name = "_".join(combo)
        combo_metrics = compute_binary_metrics(
            collected["y_true"], collected["y_prob"], threshold=threshold, auc_score=collected["y_score"]
        )
        if collected["affine_scales"].size > 0:
            combo_metrics["mean_affine_scale"] = float(collected["affine_scales"].mean())
            combo_metrics["mean_affine_bias"] = float(collected["affine_biases"].mean())
        metrics_by_combo[combo_name] = combo_metrics
        y_true.extend(collected["y_true"].tolist())
        y_prob.extend(collected["y_prob"].tolist())
        y_score.extend(collected["y_score"].tolist())
        combo_tags.extend(collected.get("combos", [tuple(combo)] * len(collected["y_true"])))
    return {
        "y_true": y_true,
        "y_prob": y_prob,
        "y_score": y_score,
        "combos": combo_tags,
        "metrics_by_combo": metrics_by_combo,
        "summary": summarize_missing_pattern_metrics(metrics_by_combo, config["data"]["modalities"]),
    }


def set_backbone_trainable(model, trainable: bool) -> None:
    if hasattr(model, "backbones"):
        for param in model.backbones.parameters():
            param.requires_grad = trainable


def select_device(device_name: str) -> torch.device:
    if device_name.startswith("cuda") and torch.cuda.is_available():
        return torch.device(device_name)
    return torch.device("cpu")


def split_provenance(splits):
    return {
        name: [
            {"case_id": record.case_id, "patient_id": getattr(record, "patient_id", record.case_id), "label": int(record.label)}
            for record in records
        ]
        for name, records in splits.items()
    }


@torch.no_grad()
def mask_affine_parameter_rows(model, modalities, combos):
    if not hasattr(model, "_mask_affine_values"):
        return []
    device = next(model.parameters()).device
    rows = []
    for combo in combos:
        availability = torch.tensor(
            [float(modality in combo) for modality in modalities], device=device, dtype=torch.float32
        )
        mask, scale, bias = model._mask_affine_values(availability)
        scale_value = float(scale.item())
        bias_value = float(bias.item())
        rows.append(
            {
                "combo": "_".join(combo),
                "mask_order": ",".join(model.mask_order),
                "mask": "".join(str(int(value)) for value in mask.detach().cpu().tolist()),
                "scale": scale_value,
                "bias": bias_value,
                "scale_positive": scale_value > 0.0,
                "scale_extreme": scale_value < 1e-3 or scale_value > 10.0,
                "bias_finite": bool(torch.isfinite(bias).item()),
                "subject_independent_mask_function": True,
                "applies_to": "validation_and_test",
            }
        )
    return rows


def checkpoint_payload(model, optimizer, config, epoch, score, best_score, bad_epochs, history, splits, threshold_metadata=None, selection_metrics=None, curriculum_state=None):
    fingerprint_path = config.get("data", {}).get("manifest_fingerprint_json")
    manifest_fingerprint = None
    if fingerprint_path and Path(fingerprint_path).is_file():
        manifest_fingerprint = load_json(fingerprint_path).get("canonical_manifest_sha256")
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": config,
        "epoch": int(epoch),
        "score": float(score),
        "best_score": float(best_score),
        "bad_epochs": int(bad_epochs),
        "history": history,
        "seed": int(config["seed"]),
        "canonical_manifest_sha256": manifest_fingerprint,
        "split": split_provenance(splits),
        "selection_metric": config["train"].get("save_metric", "auc"),
        "selection_metrics": selection_metrics or {},
        "curriculum_state": curriculum_state or {},
        "threshold_calibration": threshold_metadata
        or {
            "enabled": False,
            "mode": "global",
            "default_threshold": float(config.get("calibration", {}).get("default_threshold", 0.5)),
            "calibrated_threshold": float(config["eval"].get("threshold", 0.5)),
            "source": "fixed_config_before_validation_calibration",
        },
    }


def main():
    args = parse_args()
    overrides = {}
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.output_dir is not None:
        overrides["output_dir"] = args.output_dir
    data_overrides = {}
    if args.data_root is not None:
        data_overrides["root"] = args.data_root
    if args.manifest_csv is not None:
        data_overrides["manifest_csv"] = args.manifest_csv
    if args.split_json is not None:
        data_overrides["split_json"] = args.split_json
    if args.manifest_fingerprint_json is not None:
        data_overrides["manifest_fingerprint_json"] = args.manifest_fingerprint_json
    if data_overrides:
        overrides["data"] = data_overrides
    if args.resume is not None:
        overrides["train"] = {"resume": args.resume}
    config = load_config(args.config, overrides=overrides or None)

    missing_val_cfg = config.get("missing_aware_validation", {})
    missing_val_enabled = bool(missing_val_cfg.get("enabled", False))
    dual_view_enabled = bool(config["train"].get("dual_view_enabled", False))
    auxiliary_cfg = config.get("auxiliary", {})
    auxiliary_enabled = bool(auxiliary_cfg.get("enabled", False))
    lambda_aux = float(auxiliary_cfg.get("lambda_aux", 0.1))
    mask_head_type = str(config.get("model", {}).get("mask_head_type", "bias"))
    mask_affine_cfg = config.get("mask_affine", {})
    affine_enabled = bool(config.get("model", {}).get("use_mask_aware_classifier", True) and mask_head_type == "affine")
    if affine_enabled:
        if not bool(mask_affine_cfg.get("enabled", False)):
            raise ValueError("E4A requires mask_affine.enabled=true.")
        if str(mask_affine_cfg.get("scale_parameterization", "softplus")) != "softplus":
            raise ValueError("E4A requires mask_affine.scale_parameterization=softplus.")
        if abs(float(mask_affine_cfg.get("init_scale", 1.0)) - 1.0) > 1e-12:
            raise ValueError("E4A requires mask_affine.init_scale=1.0.")
        if abs(float(mask_affine_cfg.get("init_bias", 0.0))) > 1e-12:
            raise ValueError("E4A requires mask_affine.init_bias=0.0.")
        if not auxiliary_enabled or abs(lambda_aux - 0.1) > 1e-12 or str(auxiliary_cfg.get("pooling", "mean")) != "mean":
            raise ValueError("E4A must retain E3 auxiliary supervision with lambda_aux=0.1 and mean pooling.")
    if auxiliary_enabled:
        if config["train"].get("mode") != "missing_curriculum_train":
            raise ValueError("E3 auxiliary supervision requires train.mode=missing_curriculum_train.")
        if dual_view_enabled:
            raise ValueError("E3 auxiliary supervision cannot be combined with E2 dual-view training.")
        if str(auxiliary_cfg.get("pooling", "mean")) != "mean":
            raise ValueError("E3 auxiliary supervision supports auxiliary.pooling=mean only.")
        if lambda_aux < 0:
            raise ValueError("auxiliary.lambda_aux must be non-negative.")
    if dual_view_enabled and config["train"].get("mode") != "targeted_dual_view_train":
        raise ValueError("Dual-view training requires train.mode=targeted_dual_view_train.")
    if config["train"].get("mode") == "targeted_dual_view_train" and not dual_view_enabled:
        raise ValueError("targeted_dual_view_train requires train.dual_view_enabled=true.")
    lambda_missing = float(config["train"].get("lambda_missing", 1.0))
    if lambda_missing < 0:
        raise ValueError("train.lambda_missing must be non-negative.")
    if missing_val_enabled:
        if config["train"].get("mode") not in {"missing_curriculum_train", "targeted_dual_view_train"}:
            raise ValueError("15-pattern model selection requires a missing-aware training mode.")
        if config.get("calibration", {}).get("threshold_mode", "global") != "global":
            raise ValueError("Missing-aware training permits exactly one global validation-calibrated threshold.")
        if config["train"].get("enable_targeted_finetune", False):
            raise ValueError("Targeted fine-tuning must remain disabled for the missing-aware protocol.")

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

    best_score = -1.0
    bad_epochs = 0
    history = []
    metric_name = config["train"].get("save_metric", "auc")
    if missing_val_enabled and metric_name != "mean15_bal_acc":
        raise ValueError("Missing-aware checkpoint selection must use train.save_metric=mean15_bal_acc.")
    validation_interval = max(int(missing_val_cfg.get("interval_epochs", 5)), 1)
    validation_threshold = float(missing_val_cfg.get("selection_threshold", config["eval"].get("threshold", 0.5)))
    all_modality_combos = get_all_modality_combinations(config["data"]["modalities"])
    latest_selection_score = float("nan")
    latest_selection_metrics = {}
    start_epoch = 1
    resume_path = config["train"].get("resume")
    if resume_path:
        resume_checkpoint = torch.load(resume_path, map_location=device)
        if affine_enabled:
            saved_config = resume_checkpoint.get("config", {})
            if saved_config.get("model", {}).get("mask_head_type") != "affine" or saved_config.get("mask_affine", {}) != mask_affine_cfg:
                raise RuntimeError("Resume checkpoint does not match the E4A mask-affine protocol.")
        model.load_state_dict(resume_checkpoint["model"])
        optimizer.load_state_dict(resume_checkpoint["optimizer"])
        if int(resume_checkpoint.get("seed", config["seed"])) != int(config["seed"]):
            raise RuntimeError("Resume checkpoint seed does not match the requested run seed.")
        if dual_view_enabled:
            saved_train_config = resume_checkpoint.get("config", {}).get("train", {})
            dual_view_resume_keys = [
                "mode",
                "dual_view_enabled",
                "lambda_missing",
                "targeted_ratio_full",
                "targeted_ratio_t1ce_absent_t2_present",
                "targeted_ratio_t2_absent_t1ce_present",
                "targeted_ratio_t1ce_t2_both_absent",
            ]
            mismatched = [key for key in dual_view_resume_keys if saved_train_config.get(key) != config["train"].get(key)]
            if mismatched:
                raise RuntimeError(f"Resume checkpoint does not match the E2 dual-view protocol: {mismatched}")
        if auxiliary_enabled:
            saved_auxiliary_config = resume_checkpoint.get("config", {}).get("auxiliary", {})
            auxiliary_resume_keys = ["enabled", "lambda_aux", "pooling"]
            mismatched = [key for key in auxiliary_resume_keys if saved_auxiliary_config.get(key) != auxiliary_cfg.get(key)]
            if mismatched:
                raise RuntimeError(f"Resume checkpoint does not match the E3 auxiliary protocol: {mismatched}")
        if int(config["train"].get("minimum_stage3_epochs", 0)) > 0:
            saved_train_config = resume_checkpoint.get("config", {}).get("train", {})
            curriculum_resume_keys = [
                "mode",
                "curriculum_stage1_ratio",
                "curriculum_stage2_ratio",
                "curriculum_stage3_ratio",
                "stage2_full_ratio",
                "stage2_single_missing_ratio",
                "stage3_full_ratio",
                "stage3_single_missing_ratio",
                "stage3_double_missing_ratio",
                "stage3_triple_missing_ratio",
                "minimum_stage3_epochs",
            ]
            mismatched = [key for key in curriculum_resume_keys if saved_train_config.get(key) != config["train"].get(key)]
            if mismatched:
                raise RuntimeError(f"Resume checkpoint does not match the E4A-C curriculum protocol: {mismatched}")
        fingerprint_path = config.get("data", {}).get("manifest_fingerprint_json")
        current_fingerprint = load_json(fingerprint_path).get("canonical_manifest_sha256") if fingerprint_path and Path(fingerprint_path).is_file() else None
        saved_fingerprint = resume_checkpoint.get("canonical_manifest_sha256")
        if saved_fingerprint and current_fingerprint and saved_fingerprint != current_fingerprint:
            raise RuntimeError("Resume checkpoint manifest fingerprint does not match the current frozen manifest.")
        saved_split = resume_checkpoint.get("split")
        if saved_split and saved_split != split_provenance(splits):
            raise RuntimeError("Resume checkpoint split does not match the current frozen split.")
        start_epoch = int(resume_checkpoint["epoch"]) + 1
        best_score = float(resume_checkpoint.get("best_score", resume_checkpoint.get("score", -1.0)))
        bad_epochs = int(resume_checkpoint.get("bad_epochs", 0))
        history = list(resume_checkpoint.get("history", []))
        latest_selection_score = float(resume_checkpoint.get("score", float("nan")))
        latest_selection_metrics = dict(resume_checkpoint.get("selection_metrics", {}))
        logger.info("Resumed training from %s at epoch %d", resume_path, start_epoch)

    total_epochs = int(config["train"]["epochs"])
    minimum_stage3_epochs = int(config["train"].get("minimum_stage3_epochs", 0))
    curriculum_stabilization_enabled = (
        config["train"].get("mode") == "missing_curriculum_train" and minimum_stage3_epochs > 0
    )
    previous_stage = None
    if curriculum_stabilization_enabled and start_epoch > 1:
        previous_stage = stage3_status(start_epoch - 1, total_epochs, config["train"], minimum_stage3_epochs)["stage"]
    for epoch in range(start_epoch, total_epochs + 1):
        if curriculum_stabilization_enabled:
            curriculum_state = stage3_status(epoch, total_epochs, config["train"], minimum_stage3_epochs)
            bad_epochs, entered_stage3_now = reset_bad_epochs_on_stage3_entry(
                curriculum_state["stage"], previous_stage, bad_epochs
            )
        else:
            curriculum_state = {
                "stage": "not_applicable",
                "stage3_entered_epoch": None,
                "stage3_epochs_completed": 0,
                "early_stopping_protected": False,
            }
            entered_stage3_now = False
        if entered_stage3_now:
            logger.info("Epoch %d | entered Stage3; reset bad_epochs=0", epoch)
        previous_stage = curriculum_state["stage"]
        if hasattr(train_ds, "set_epoch"):
            train_ds.set_epoch(epoch, int(config["train"]["epochs"]), config["train"])
        sampling_report = None
        if dual_view_enabled:
            dual_result = run_dual_view_epoch(
                model,
                train_loader,
                device,
                criterion=criterion,
                optimizer=optimizer,
                lambda_missing=lambda_missing,
                grad_clip=config["train"].get("grad_clip", 0.0),
                threshold=config["eval"].get("threshold", 0.5),
                desc=f"dual-view train {epoch}",
            )
            train_metrics = dual_result["metrics"]
            sampling_report = dual_result["sampling"]
            save_json(sampling_report, metrics_dir / f"train_sampling_epoch_{epoch:03d}.json")
            logger.info("Epoch %d | targeted sampling=%s", epoch, json.dumps(sampling_report["subgroup_frequencies"], ensure_ascii=False))
        elif auxiliary_enabled:
            train_metrics = run_auxiliary_epoch(
                model,
                train_loader,
                device,
                criterion=criterion,
                optimizer=optimizer,
                lambda_aux=lambda_aux,
                grad_clip=config["train"].get("grad_clip", 0.0),
                threshold=config["eval"].get("threshold", 0.5),
                desc=f"auxiliary train {epoch}",
            )
        else:
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
        if (
            config["train"].get("mode") == "missing_curriculum_train"
            and bool(config["train"].get("record_curriculum_sampling", False))
        ):
            sampling_report = train_ds.curriculum_sampling_report()
            save_json(sampling_report, metrics_dir / f"train_sampling_epoch_{epoch:03d}.json")
        val_metrics = run_epoch(
            model,
            val_loader,
            device,
            criterion=criterion,
            optimizer=None,
            threshold=config["eval"].get("threshold", 0.5),
            desc=f"val {epoch}",
        )
        selection_event = not missing_val_enabled
        if missing_val_enabled and (epoch % validation_interval == 0 or epoch == total_epochs):
            selection_event = True
            missing15_val = collect_validation_predictions_for_combos(
                model,
                config,
                device,
                all_modality_combos,
                threshold=validation_threshold,
            )
            latest_selection_metrics = dict(missing15_val["summary"])
            latest_selection_score = float(latest_selection_metrics["mean15_bal_acc"])
            val_metrics["mean15_bal_acc"] = latest_selection_score
            val_metrics["mean15_auc"] = float(latest_selection_metrics["mean15_auc"])
            validation_payload = {
                "epoch": epoch,
                "selection_threshold": validation_threshold,
                "per_pattern": missing15_val["metrics_by_combo"],
                "summary": missing15_val["summary"],
            }
            save_json(validation_payload, metrics_dir / f"val_missing15_epoch_{epoch:03d}.json")
            pd.DataFrame(
                [{"combo": combo_name, **metrics} for combo_name, metrics in missing15_val["metrics_by_combo"].items()]
            ).to_csv(metrics_dir / f"val_missing15_epoch_{epoch:03d}.csv", index=False)
            pd.DataFrame(missing15_val["summary"]["groups"]).to_csv(
                metrics_dir / f"val_missing15_epoch_{epoch:03d}_groups.csv", index=False
            )
            logger.info(
                "Epoch %d | 15-pattern val mean BAC=%.4f mean AUC=%.4f",
                epoch,
                latest_selection_score,
                float(latest_selection_metrics["mean15_auc"]),
            )

        if missing_val_enabled:
            score = latest_selection_score
        else:
            score = val_metrics.get(metric_name, float("nan"))
            if score != score:
                score = val_metrics.get("bal_acc", 0.0)

        improved = False
        if selection_event:
            improved = score > best_score
            if improved:
                best_score = score
                bad_epochs = 0
            else:
                bad_epochs += validation_interval if missing_val_enabled else 1
        curriculum_state.update(
            {
                "sampling_ratios": curriculum_sampling_ratios(curriculum_state["stage"], config["train"])
                if curriculum_stabilization_enabled else {},
                "bad_epochs": bad_epochs,
            }
        )
        train_metrics.update(
            {
                "curriculum_stage": curriculum_state["stage"],
                "stage3_entered_epoch": curriculum_state["stage3_entered_epoch"],
                "stage3_epochs_completed": curriculum_state["stage3_epochs_completed"],
                "early_stopping_protected": curriculum_state["early_stopping_protected"],
                "bad_epochs": bad_epochs,
            }
        )
        history_item = {"epoch": epoch, "train": train_metrics, "val": val_metrics, "curriculum": curriculum_state}
        if sampling_report is not None:
            history_item["sampling"] = sampling_report
        history.append(history_item)
        if auxiliary_enabled:
            auxiliary_rows = [
                {"epoch": item["epoch"], **{key: value for key, value in item["train"].items() if key in {"main_loss", "aux_loss", "total_loss", "lambda_aux"} or key.startswith("aux_")}}
                for item in history
                if "main_loss" in item["train"]
            ]
            pd.DataFrame(auxiliary_rows).to_csv(metrics_dir / "auxiliary_training_history.csv", index=False)
        logger.info(
            "Epoch %d | curriculum=%s | sampling_ratios=%s | stage3_entered_epoch=%d | stage3_completed=%d | protection=%s | bad_epochs=%d",
            epoch,
            curriculum_state["stage"],
            json.dumps(curriculum_state["sampling_ratios"], ensure_ascii=False),
            curriculum_state["stage3_entered_epoch"] or 0,
            curriculum_state["stage3_epochs_completed"],
            curriculum_state["early_stopping_protected"],
            bad_epochs,
        )
        if sampling_report is not None:
            logger.info("Epoch %d | actual curriculum sampling=%s", epoch, json.dumps(sampling_report["subgroup_frequencies"], ensure_ascii=False))
        logger.info("Epoch %d | train=%s | val=%s", epoch, json.dumps(train_metrics, ensure_ascii=False), json.dumps(val_metrics, ensure_ascii=False))
        payload = checkpoint_payload(
            model,
            optimizer,
            config,
            epoch,
            score,
            best_score,
            bad_epochs,
            history,
            splits,
            selection_metrics=latest_selection_metrics,
            curriculum_state=curriculum_state,
        )
        torch.save(payload, ckpt_dir / "last.pt")
        if improved:
            torch.save(payload, ckpt_dir / "best.pt")
            logger.info("Saved new best checkpoint with %s=%.4f", metric_name, score)
        if should_early_stop(
            selection_event,
            bad_epochs,
            int(config["train"].get("early_stop_patience", 8)),
            curriculum_state["early_stopping_protected"],
        ):
            logger.info("Early stopping triggered at epoch %d", epoch)
            break

    if config["train"].get("enable_targeted_finetune", False):
        base_checkpoint = torch.load(ckpt_dir / "best.pt", map_location=device)
        torch.save(base_checkpoint, ckpt_dir / "base_best.pt")
        model.load_state_dict(base_checkpoint["model"])
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
            )
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
            history.append({"epoch": f"finetune_{ft_epoch}", "train": train_metrics, "val": val_metrics})
            logger.info("Fine-tune epoch %d | train=%s | val=%s", ft_epoch, json.dumps(train_metrics, ensure_ascii=False), json.dumps(val_metrics, ensure_ascii=False))
            if score > fine_tune_best:
                fine_tune_best = score
                torch.save({"model": model.state_dict(), "config": config, "epoch": f"finetune_{ft_epoch}", "score": score, "stage": "targeted_finetune"}, ckpt_dir / "best.pt")
                logger.info("Saved fine-tuned best checkpoint with %s=%.4f", metric_name, score)
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
        history_rows.append(row)
    history_df = pd.DataFrame(history_rows)
    history_df.to_csv(metrics_dir / "train_history.csv", index=False)
    sampling_subgroup_rows = []
    sampling_pattern_rows = []
    for item in history:
        sampling = item.get("sampling")
        if not sampling:
            continue
        for name, count in sampling["subgroup_counts"].items():
            sampling_subgroup_rows.append(
                {
                    "epoch": item["epoch"],
                    "subgroup": name,
                    "count": count,
                    "frequency": sampling["subgroup_frequencies"][name],
                    "configured_ratio": sampling["configured_ratios"][name],
                }
            )
        for name, count in sampling["pattern_counts"].items():
            sampling_pattern_rows.append(
                {"epoch": item["epoch"], "pattern": name, "count": count, "frequency": sampling["pattern_frequencies"][name]}
            )
    if sampling_subgroup_rows:
        pd.DataFrame(sampling_subgroup_rows).to_csv(metrics_dir / "train_sampling_subgroups.csv", index=False)
        pd.DataFrame(sampling_pattern_rows).to_csv(metrics_dir / "train_sampling_patterns.csv", index=False)
    fusion_plot_df = pd.DataFrame({"epoch": history_df["epoch"]})
    for col in ["val_alpha", "val_beta", "val_gamma"]:
        if col in history_df.columns:
            fusion_plot_df[col.replace("val_", "")] = history_df[col]
    if len(fusion_plot_df.columns) > 1:
        save_fusion_weight_history(fusion_plot_df, metrics_dir / "fusion_weight_history.png")

    checkpoint_path = ckpt_dir / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    if affine_enabled:
        affine_rows = mask_affine_parameter_rows(model, config["data"]["modalities"], all_modality_combos)
        pd.DataFrame(affine_rows).to_csv(metrics_dir / "mask_affine_parameters.csv", index=False)
        if any(row["scale_extreme"] or not row["scale_positive"] or not row["bias_finite"] for row in affine_rows):
            logger.warning("Extreme or non-finite mask-affine parameter detected; inspect metrics/mask_affine_parameters.csv")

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
        elif missing_val_enabled and calibration_cfg.get("pool_validation_all_combos", True):
            val_collected = collect_validation_predictions_for_combos(model, config, device, all_modality_combos)
            calibration_result = calibrate_threshold(
                val_collected["y_true"],
                val_collected["y_prob"],
                metric=calibration_cfg.get("threshold_metric", "balanced_accuracy"),
            )
            calibration_result.update(
                {
                    "mode": "global",
                    "source": "frozen_validation_split_all_15_patterns_pooled",
                    "num_patterns": len(all_modality_combos),
                    "num_predictions": len(val_collected["y_true"]),
                }
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
    last_checkpoint_path = ckpt_dir / "last.pt"
    if last_checkpoint_path.exists():
        last_checkpoint = torch.load(last_checkpoint_path, map_location="cpu")
        last_checkpoint["calibrated_threshold"] = calibrated_threshold
        last_checkpoint["threshold_calibration"] = calibration_result
        torch.save(last_checkpoint, last_checkpoint_path)
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

    if not bool(config.get("eval", {}).get("run_test_after_training", True)):
        logger.info("Post-training test evaluation disabled by eval.run_test_after_training=false.")
        return

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
            class_names=config["eval"].get("class_names"),
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
    if len(all_test_metrics) == len(all_modality_combos):
        missing_pattern_summary = summarize_missing_pattern_metrics(all_test_metrics, config["data"]["modalities"])
        if auxiliary_enabled:
            singleton_metrics = {
                modality: {
                    "bal_acc": all_test_metrics.get(modality, {}).get("bal_acc", float("nan")),
                    "auc": all_test_metrics.get(modality, {}).get("auc", float("nan")),
                }
                for modality in config["data"]["modalities"]
            }
            save_json(singleton_metrics, metrics_dir / "test_singleton_modalities.json")
            pd.DataFrame([{"modality": modality, **metrics} for modality, metrics in singleton_metrics.items()]).to_csv(
                metrics_dir / "test_singleton_modalities.csv", index=False
            )
        save_json(missing_pattern_summary, metrics_dir / "test_missing_pattern_summary.json")
        pd.DataFrame(missing_pattern_summary["groups"]).to_csv(metrics_dir / "test_missing_pattern_groups.csv", index=False)
        uncertainty = task_uncertainty_summary(config, splits)
        save_json(uncertainty, metrics_dir / "task_uncertainty.json")
        logger.info(
            "15-pattern test summary | mean BAC=%.4f mean AUC=%.4f full BAC=%.4f full AUC=%.4f",
            float(missing_pattern_summary["mean15_bal_acc"]),
            float(missing_pattern_summary["mean15_auc"]),
            float(missing_pattern_summary["full_modality"]["bal_acc"]),
            float(missing_pattern_summary["full_modality"]["auc"]),
        )


if __name__ == "__main__":
    main()
