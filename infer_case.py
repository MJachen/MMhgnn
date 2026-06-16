from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from datasets.brats_dataset import BraTSClassificationDataset, CaseRecord
from models import HybridHypergraphClassifier
from utils import load_config, set_seed, setup_logger
from utils.config import ensure_dir
from utils.metrics import threshold_dispatch_for_combo, threshold_for_combo
from utils.runner import choose_display_image, move_batch_to_device, roi_drop_analysis
from utils.visualization import save_case_overlay, save_case_partition_visualization, save_roi_barplot


def parse_args():
    parser = argparse.ArgumentParser(description="Infer a single BraTS case")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--case-dir", type=str, required=True)
    parser.add_argument("--combo", nargs="*", default=None)
    return parser.parse_args()


def select_device(device_name: str) -> torch.device:
    if device_name.startswith("cuda") and torch.cuda.is_available():
        return torch.device(device_name)
    return torch.device("cpu")


def build_case_dataset(config, case_dir: str, combo=None):
    case_dir = Path(case_dir)
    case_id = case_dir.name
    data_cfg = config["data"]
    modalities = list(data_cfg["modalities"])
    image_filename_pattern = data_cfg.get("image_filename_pattern", "{case_id}_{mod}.nii.gz")
    tumor_mask_source = data_cfg.get("tumor_mask_source", "seg")
    files = {mod: case_dir / image_filename_pattern.format(case_id=case_id, mod=mod) for mod in modalities}
    if tumor_mask_source == "seg":
        seg_pattern = data_cfg.get("seg_filename_pattern", "{case_id}_seg.nii.gz")
        files["seg"] = case_dir / seg_pattern.format(case_id=case_id, mod="seg")
    elif tumor_mask_source == "union_modality_masks":
        mask_pattern = data_cfg.get("mask_filename_pattern", "{mod}_mask.nii.gz")
        for mod in modalities:
            files[f"{mod}_mask"] = case_dir / mask_pattern.format(case_id=case_id, mod=mod)
    else:
        raise ValueError(f"Unsupported tumor_mask_source: {tumor_mask_source}")
    record = CaseRecord(case_id=case_id, case_dir=case_dir, label=0, files=files)
    return BraTSClassificationDataset(
        [record],
        target_shape=config["data"].get("target_shape"),
        crop_mode=config["data"].get("crop_mode", "wt_bbox"),
        bbox_margin=int(config["data"].get("bbox_margin", 6)),
        combo_mode="fixed_combo",
        fixed_combo=combo or config["train"].get("fixed_combo", config["data"]["modalities"]),
        all_modalities=config["data"]["modalities"],
        random_seed=config["seed"],
        explicit_combo=combo or config["train"].get("fixed_combo", config["data"]["modalities"]),
        q_core=float(config["model"].get("q_core", 0.4)),
        peri_inner_radius=int(config["model"].get("peri_inner_radius", 3)),
        peri_outer_radius=int(config["model"].get("peri_outer_radius", 7)),
        allow_full_resolution_input=bool(config["model"].get("allow_full_resolution_input", False)),
        tumor_mask_source=tumor_mask_source,
    )


def main():
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config["seed"]))
    logger = setup_logger(config.get("logging", {}).get("level", "INFO"))
    device = select_device(config["train"].get("device", "cuda"))

    model = HybridHypergraphClassifier(config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    calibration_result = checkpoint.get("threshold_calibration", {})

    combo = tuple(args.combo) if args.combo else tuple(config["train"].get("fixed_combo", config["data"]["modalities"]))
    base_threshold = float(checkpoint.get("calibrated_threshold", config.get("calibration", {}).get("default_threshold", config["eval"].get("threshold", 0.5))))
    threshold_dispatch = threshold_dispatch_for_combo(calibration_result, combo, base_threshold)
    threshold = float(threshold_dispatch["applied_threshold"])
    dataset = build_case_dataset(config, args.case_dir, combo=combo)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    batch = next(iter(loader))
    batch = move_batch_to_device(batch, device)

    with torch.no_grad():
        output = model(batch)
    prob = float(output["prob"][0].item())
    pred = int(prob >= threshold)
    roi_attn = output["roi_attention"][0].detach().cpu().numpy()
    case_id = batch["case_id"][0]

    out_dir = ensure_dir(Path(config["output_dir"]) / "single_case" / case_id)
    save_roi_barplot(config["model"]["roi_names"], roi_attn, f"ROI importance: {case_id}", out_dir / "roi_importance.png")
    image_3d = choose_display_image(batch["images"][0], batch["available_modalities"][0])
    seg_3d = batch["seg"][0].detach().cpu().numpy()
    roi_masks = {config["model"]["roi_names"][i]: batch["roi_masks"][0, i].detach().cpu().numpy() for i in range(len(config["model"]["roi_names"]))}
    save_case_overlay(image_3d, roi_masks, {k: float(v) for k, v in zip(config["model"]["roi_names"], roi_attn)}, out_dir / "overlay.png")
    save_case_partition_visualization(image_3d, seg_3d, roi_masks, out_dir / "roi_partition.png")

    deltas = roi_drop_analysis(model, loader, device, config["model"]["roi_names"])
    result = {
        "case_id": case_id,
        "combo": list(combo),
        "prob_hgg": prob,
        "pred_label": pred,
        "threshold_group": threshold_dispatch["threshold_group"],
        "threshold": threshold,
        "roi_attention": {k: float(v) for k, v in zip(config["model"]["roi_names"], roi_attn)},
        "roi_drop_delta_prob": {k: float(v) for k, v in zip(config["model"]["roi_names"], deltas[0])},
    }
    with open(out_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info("Single-case inference saved to %s", out_dir)


if __name__ == "__main__":
    main()
