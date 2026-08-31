from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from datasets.brats_dataset import get_all_modality_combinations
from models import HybridHypergraphClassifier
from utils import load_config, set_seed, setup_logger
from utils.config import ensure_dir
from utils.io import load_json, save_json
from utils.metrics import build_drop_t1_ablation_report, summarize_missing_pattern_metrics, threshold_dispatch_for_combo, threshold_for_combo
from utils.runner import evaluate_with_explanations
from utils.training import build_dataloader, build_datasets


def parse_args():
    parser = argparse.ArgumentParser(description="BraTS/UTSW hybrid hypergraph evaluation")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--combo", nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--manifest-csv", type=str, default=None)
    parser.add_argument("--split-json", type=str, default=None)
    return parser.parse_args()


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
    if data_overrides:
        overrides["data"] = data_overrides
    config = load_config(args.config, overrides=overrides or None)
    set_seed(int(config["seed"]))
    logger = setup_logger(config.get("logging", {}).get("level", "INFO"))
    device = select_device(config["train"].get("device", "cuda"))
    model = HybridHypergraphClassifier(config).to(device)
    if hasattr(model, "get_classifier_info"):
        logger.info("Mask-aware classifier: %s", json.dumps(model.get_classifier_info(), ensure_ascii=False))

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    calibration_result = checkpoint.get("threshold_calibration", {})
    if config.get("missing_aware_validation", {}).get("enabled", False):
        if calibration_result.get("mode", "global") != "global":
            raise RuntimeError("Missing-aware evaluation requires the single global threshold stored in its checkpoint.")
        fingerprint_path = config.get("data", {}).get("manifest_fingerprint_json")
        current_fingerprint = load_json(fingerprint_path).get("canonical_manifest_sha256") if fingerprint_path and Path(fingerprint_path).is_file() else None
        saved_fingerprint = checkpoint.get("canonical_manifest_sha256")
        if saved_fingerprint and current_fingerprint and saved_fingerprint != current_fingerprint:
            raise RuntimeError("Checkpoint manifest fingerprint does not match the current frozen manifest.")
        _, _, _, current_splits = build_datasets(config)
        saved_split = checkpoint.get("split")
        if saved_split and saved_split != split_provenance(current_splits):
            raise RuntimeError("Checkpoint split does not match the current frozen split.")
    threshold = float(checkpoint.get("calibrated_threshold", config.get("calibration", {}).get("default_threshold", config["eval"].get("threshold", 0.5))))
    logger.info("Using evaluation threshold mode=%s default/global=%.4f", calibration_result.get("mode", "global"), threshold)

    output_dir = ensure_dir(Path(config["output_dir"]) / "evaluate_only")
    combos = [tuple(args.combo)] if args.combo else get_all_modality_combinations(config["data"]["modalities"])
    all_metrics = {}
    rows = []
    for combo in combos:
        _, _, test_ds, _ = build_datasets(config, explicit_eval_combo=combo)
        test_loader = build_dataloader(test_ds, config["train"]["batch_size"], config["data"].get("num_workers", 0), shuffle=False)
        combo_name = "_".join(combo)
        threshold_dispatch = threshold_dispatch_for_combo(calibration_result, combo, threshold)
        metrics = evaluate_with_explanations(
            model,
            test_loader,
            device,
            config["model"]["roi_names"],
            str(output_dir / combo_name),
            threshold=threshold_dispatch["applied_threshold"],
            threshold_group=threshold_dispatch["threshold_group"],
            explain_num_cases=config["eval"].get("explain_num_cases", 3),
            roi_drop_enabled=config["eval"].get("roi_drop_enabled", True),
            edge_type_drop_enabled=config["eval"].get("edge_type_drop_enabled", True),
            class_names=config["eval"].get("class_names"),
        )
        all_metrics[combo_name] = metrics
        rows.append({"combo": combo_name, "threshold_group": threshold_dispatch["threshold_group"], "applied_threshold": threshold_dispatch["applied_threshold"], **metrics})
        logger.info("Metrics for %s: %s", combo_name, json.dumps(metrics, ensure_ascii=False))
    if config.get("test", {}).get("enable_drop_t1_inference_ablation", False):
        ablation_payload = build_drop_t1_ablation_report(
            all_test_metrics if "all_test_metrics" in locals() else all_metrics,
            combo_names=config.get("test", {}).get("drop_t1_ablation_combos", ["t1", "t2_t1", "t1_flair", "t2_t1_flair"]),
        )
        save_json(ablation_payload, output_dir / "drop_t1_ablation.json")
        pd.DataFrame(ablation_payload["rows"]).to_csv(output_dir / "drop_t1_ablation.csv", index=False)
        pd.DataFrame(ablation_payload["summary"]).to_csv(output_dir / "drop_t1_ablation_summary.csv", index=False)
        logger.info("Saved drop-t1 inference ablation with %d rows", len(ablation_payload["rows"]))

    save_json(all_metrics, output_dir / "metrics.json")
    pd.DataFrame(rows).to_csv(output_dir / "metrics.csv", index=False)
    if len(all_metrics) == len(get_all_modality_combinations(config["data"]["modalities"])):
        missing_pattern_summary = summarize_missing_pattern_metrics(all_metrics, config["data"]["modalities"])
        save_json(missing_pattern_summary, output_dir / "missing_pattern_summary.json")
        pd.DataFrame(missing_pattern_summary["groups"]).to_csv(output_dir / "missing_pattern_groups.csv", index=False)
        logger.info(
            "15-pattern summary | mean BAC=%.4f mean AUC=%.4f full BAC=%.4f full AUC=%.4f",
            float(missing_pattern_summary["mean15_bal_acc"]),
            float(missing_pattern_summary["mean15_auc"]),
            float(missing_pattern_summary["full_modality"]["bal_acc"]),
            float(missing_pattern_summary["full_modality"]["auc"]),
        )


if __name__ == "__main__":
    main()
