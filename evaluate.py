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
from utils.io import save_json
from utils.metrics import build_drop_t1_ablation_report, threshold_dispatch_for_combo, threshold_for_combo
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


if __name__ == "__main__":
    main()
