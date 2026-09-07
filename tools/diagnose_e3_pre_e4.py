from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.brats_dataset import get_all_modality_combinations
from models import HybridHypergraphClassifier
from utils import load_config, set_seed
from utils.io import load_json
from utils.metrics import compute_binary_metrics
from utils.runner import move_batch_to_device
from utils.training import brats_collate_fn, build_datasets


SUBGROUP_RULES = {
    "t1ce_absent": lambda combo: "t1ce" not in combo,
    "t2_absent": lambda combo: "t2" not in combo,
    "t1ce_t2_both_present": lambda combo: "t1ce" in combo and "t2" in combo,
    "t1ce_t2_both_absent": lambda combo: "t1ce" not in combo and "t2" not in combo,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Read-only pre-E4 diagnostics for an E3 checkpoint")
    parser.add_argument("--config", default="configs/utsw_idh/missing_aware_aux_e3.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-prefix", default="e3_validation")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def threshold_metrics(y_true, y_prob, threshold):
    metrics = compute_binary_metrics(y_true, y_prob, threshold=float(threshold))
    return {
        "threshold": float(threshold),
        "balanced_accuracy": float(metrics["bal_acc"]),
        "sensitivity": float(metrics["sen"]),
        "specificity": float(metrics["spe"]),
        "accuracy": float(metrics["acc"]),
    }


def threshold_curve(y_true, y_prob):
    return pd.DataFrame([threshold_metrics(y_true, y_prob, threshold) for threshold in np.linspace(0.0, 1.0, 201)])


def distribution_row(values, level, group, y_true):
    values = np.asarray(values, dtype=float)
    return {
        "level": level,
        "group": group,
        "y_true": int(y_true),
        "n": int(len(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "min": float(np.min(values)),
        "p05": float(np.percentile(values, 5)),
        "p25": float(np.percentile(values, 25)),
        "median": float(np.median(values)),
        "p75": float(np.percentile(values, 75)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def safe_auc(y_true, y_prob):
    return float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) == 2 else float("nan")


def optimal_threshold_row(y_true, y_prob):
    curve = threshold_curve(y_true, y_prob)
    best = float(curve["balanced_accuracy"].max())
    optimal = curve[np.isclose(curve["balanced_accuracy"], best, rtol=0.0, atol=1e-12)].iloc[0]
    return {
        "optimal_threshold": float(optimal["threshold"]),
        "optimal_bal_acc": best,
        "auc": safe_auc(y_true, y_prob),
        "sensitivity": float(optimal["sensitivity"]),
        "specificity": float(optimal["specificity"]),
        "accuracy": float(optimal["accuracy"]),
        "num_tied_optimal_thresholds": int(
            np.isclose(curve["balanced_accuracy"], best, rtol=0.0, atol=1e-12).sum()
        ),
    }


def validate_checkpoint_provenance(checkpoint, config, splits):
    if checkpoint.get("selection_metric") != "mean15_bal_acc":
        raise RuntimeError("Checkpoint was not selected by mean15_bal_acc.")
    threshold_metadata = checkpoint.get("threshold_calibration", {})
    if threshold_metadata.get("mode") != "global":
        raise RuntimeError("Checkpoint does not contain one global threshold.")
    if threshold_metadata.get("source") != "frozen_validation_split_all_15_patterns_pooled":
        raise RuntimeError("Checkpoint threshold provenance is not pooled frozen validation.")
    if int(threshold_metadata.get("num_patterns", -1)) != 15:
        raise RuntimeError("Checkpoint threshold metadata does not report 15 validation patterns.")

    current_fingerprint = load_json(config["data"]["manifest_fingerprint_json"])["canonical_manifest_sha256"]
    if checkpoint.get("canonical_manifest_sha256") != current_fingerprint:
        raise RuntimeError("Checkpoint manifest fingerprint does not match the current manifest.")
    expected_val = [
        {
            "case_id": record.case_id,
            "patient_id": getattr(record, "patient_id", record.case_id),
            "label": int(record.label),
        }
        for record in splits["val"]
    ]
    if checkpoint.get("split", {}).get("val") != expected_val:
        raise RuntimeError("Checkpoint validation split does not match the frozen validation split.")


def make_pattern_batch(full_batch, modalities, combo, device):
    availability = torch.tensor(
        [[float(modality in combo) for modality in modalities]], dtype=full_batch["available_modalities"].dtype
    )
    pattern_batch = dict(full_batch)
    pattern_batch["available_modalities"] = availability
    pattern_batch["images"] = full_batch["images"] * availability.view(1, len(modalities), 1, 1, 1)
    pattern_batch["combo"] = [tuple(combo)]
    return move_batch_to_device(pattern_batch, device)


def save_plots(pooled, threshold_df, subject_df, output_dir, output_prefix):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, color in [(0, "#4C78A8"), (1, "#E45756")]:
        values = pooled.loc[pooled["y_true"] == label, "y_prob"]
        ax.hist(values, bins=30, alpha=0.55, density=True, label=f"y={label}", color=color)
    ax.set(xlabel="Main positive-class probability", ylabel="Density", title="Pooled validation scores (15 patterns)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"{output_prefix}_pooled_score_histogram.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(threshold_df["threshold"], threshold_df["balanced_accuracy"], color="#2F4B7C")
    checkpoint_rows = threshold_df[threshold_df["is_checkpoint_threshold"]]
    if not checkpoint_rows.empty:
        ax.scatter(checkpoint_rows["threshold"], checkpoint_rows["balanced_accuracy"], color="#D62728", zorder=3)
    ax.set(xlabel="Threshold", ylabel="Balanced accuracy", title="Pooled validation threshold scan")
    fig.tight_layout()
    fig.savefig(output_dir / f"{output_prefix}_threshold_bac_curve.png", dpi=180)
    plt.close(fig)

    combo_order = sorted(pooled["combo"].unique(), key=lambda value: (value.count("_") + 1, value))
    box_data = [pooled.loc[pooled["combo"] == combo, "y_prob"].to_numpy() for combo in combo_order]
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.boxplot(box_data, tick_labels=combo_order, showfliers=False)
    ax.tick_params(axis="x", rotation=55)
    ax.set(xlabel="Modality pattern", ylabel="Main positive-class probability", title="Validation score by pattern")
    fig.tight_layout()
    fig.savefig(output_dir / f"{output_prefix}_pattern_score_boxplot.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(subject_df["range_prob"], bins=20, color="#72B7B2", edgecolor="white")
    ax.set(xlabel="Within-subject probability range", ylabel="Subjects", title="Cross-pattern prediction range")
    fig.tight_layout()
    fig.savefig(output_dir / f"{output_prefix}_subject_cross_pattern_range_histogram.png", dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    checkpoint_path = Path(args.checkpoint).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_prefix = str(args.output_prefix)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_hash_before = sha256_file(checkpoint_path)

    config = load_config(args.config, overrides={"data": {"root": args.data_root}})
    set_seed(int(config["seed"]))
    device = torch.device(args.device if not args.device.startswith("cuda") or torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    modalities = list(config["data"]["modalities"])
    combos = get_all_modality_combinations(modalities)
    if len(combos) != 15 or any(not combo for combo in combos):
        raise RuntimeError("Expected exactly 15 non-empty modality patterns.")

    _, val_dataset, _, splits = build_datasets(config, explicit_eval_combo=modalities)
    validate_checkpoint_provenance(checkpoint, config, splits)
    model = HybridHypergraphClassifier(config).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    pooled_rows = []
    auxiliary_rows = []
    with torch.inference_mode():
        for subject_index in tqdm(range(len(val_dataset)), desc="validation subjects"):
            full_batch = brats_collate_fn([val_dataset[subject_index]])
            case_id = full_batch["case_id"][0]
            y_true = int(full_batch["label"].item())
            for combo in combos:
                batch = make_pattern_batch(full_batch, modalities, combo, device)
                output = model(batch)
                logits = output["logits"][0]
                combo_name = "_".join(combo)
                pooled_rows.append(
                    {
                        "case_id": case_id,
                        "combo": combo_name,
                        "y_true": y_true,
                        "y_prob": float(output["prob"][0].item()),
                        "logit_positive": float(logits[1].item()),
                        "binary_logit": float((logits[1] - logits[0]).item()),
                    }
                )
                if len(combo) == 1:
                    modality = combo[0]
                    aux_logits = output["aux_logits"][modality][0]
                    auxiliary_rows.append(
                        {
                            "case_id": case_id,
                            "modality": modality,
                            "y_true": y_true,
                            "y_prob": float(torch.softmax(aux_logits, dim=0)[1].item()),
                            "logit_positive": float(aux_logits[1].item()),
                            "binary_logit": float((aux_logits[1] - aux_logits[0]).item()),
                        }
                    )
                del batch, output
            del full_batch

    pooled = pd.DataFrame(pooled_rows)
    auxiliary = pd.DataFrame(auxiliary_rows)
    expected_predictions = len(val_dataset) * len(combos)
    if len(pooled) != expected_predictions or pooled["case_id"].nunique() != len(val_dataset):
        raise RuntimeError("Incomplete pooled validation predictions.")
    pooled.to_csv(output_dir / f"{output_prefix}_pooled_predictions.csv", index=False)

    checkpoint_threshold = float(
        checkpoint.get("calibrated_threshold", checkpoint.get("threshold_calibration", {}).get("threshold", 0.5))
    )
    threshold_df = threshold_curve(pooled["y_true"], pooled["y_prob"])
    best_pooled_bac = float(threshold_df["balanced_accuracy"].max())
    threshold_df["is_optimal"] = np.isclose(
        threshold_df["balanced_accuracy"], best_pooled_bac, rtol=0.0, atol=1e-12
    )
    threshold_df["is_checkpoint_threshold"] = np.isclose(
        threshold_df["threshold"], checkpoint_threshold, rtol=0.0, atol=1e-12
    )
    threshold_df.to_csv(output_dir / f"{output_prefix}_threshold_curve.csv", index=False)
    checkpoint_threshold_metrics = threshold_metrics(pooled["y_true"], pooled["y_prob"], checkpoint_threshold)
    checkpoint_threshold_is_optimal = bool(
        np.isclose(checkpoint_threshold_metrics["balanced_accuracy"], best_pooled_bac, rtol=0.0, atol=1e-12)
    )

    distribution_rows = []
    for combo_name, frame in pooled.groupby("combo", sort=False):
        for label, label_frame in frame.groupby("y_true"):
            distribution_rows.append(distribution_row(label_frame["y_prob"], "pattern", combo_name, label))
    for subgroup_name, predicate in SUBGROUP_RULES.items():
        selected = pooled[pooled["combo"].map(lambda value: predicate(set(value.split("_"))))]
        for label, label_frame in selected.groupby("y_true"):
            distribution_rows.append(distribution_row(label_frame["y_prob"], "subgroup", subgroup_name, label))
    distribution_df = pd.DataFrame(distribution_rows)
    distribution_df.to_csv(output_dir / f"{output_prefix}_score_distribution.csv", index=False)

    pattern_threshold_rows = []
    for combo_name, frame in pooled.groupby("combo", sort=False):
        pattern_threshold_rows.append(
            {"combo": combo_name, **optimal_threshold_row(frame["y_true"].to_numpy(), frame["y_prob"].to_numpy())}
        )
    pattern_thresholds = pd.DataFrame(pattern_threshold_rows)
    pattern_thresholds.to_csv(output_dir / f"{output_prefix}_pattern_thresholds.csv", index=False)

    subject_rows = []
    for (case_id, y_true), frame in pooled.groupby(["case_id", "y_true"], sort=False):
        subject_rows.append(
            {
                "case_id": case_id,
                "y_true": int(y_true),
                "mean_prob": float(frame["y_prob"].mean()),
                "std_prob": float(frame["y_prob"].std(ddof=1)),
                "min_prob": float(frame["y_prob"].min()),
                "max_prob": float(frame["y_prob"].max()),
                "range_prob": float(frame["y_prob"].max() - frame["y_prob"].min()),
                "mean_logit": float(frame["binary_logit"].mean()),
                "std_logit": float(frame["binary_logit"].std(ddof=1)),
                "min_logit": float(frame["binary_logit"].min()),
                "max_logit": float(frame["binary_logit"].max()),
                "range_logit": float(frame["binary_logit"].max() - frame["binary_logit"].min()),
            }
        )
    subject_df = pd.DataFrame(subject_rows)
    subject_df.to_csv(output_dir / f"{output_prefix}_subject_cross_pattern_variance.csv", index=False)
    top10 = subject_df.nlargest(10, "range_prob")
    top10.to_csv(output_dir / f"{output_prefix}_subject_cross_pattern_top10.csv", index=False)
    subject_summary = {
        "mean_within_subject_std_prob": float(subject_df["std_prob"].mean()),
        "median_within_subject_std_prob": float(subject_df["std_prob"].median()),
        "mean_range_prob": float(subject_df["range_prob"].mean()),
        "median_range_prob": float(subject_df["range_prob"].median()),
        "mean_std_logit": float(subject_df["std_logit"].mean()),
        "median_std_logit": float(subject_df["std_logit"].median()),
        "mean_range_logit": float(subject_df["range_logit"].mean()),
        "median_range_logit": float(subject_df["range_logit"].median()),
    }

    full_combo_name = "_".join(modalities)
    full_scores = pooled.loc[pooled["combo"] == full_combo_name, ["case_id", "y_true", "y_prob", "binary_logit"]].rename(
        columns={"y_prob": "full_prob", "binary_logit": "full_binary_logit"}
    )
    subgroup_shift_rows = []
    for subgroup_name, predicate in {"full": lambda combo: combo == set(modalities), **SUBGROUP_RULES}.items():
        selected = pooled[pooled["combo"].map(lambda value: predicate(set(value.split("_"))))]
        per_subject = selected.groupby(["case_id", "y_true"], as_index=False).agg(
            subgroup_mean_prob=("y_prob", "mean"), subgroup_mean_logit=("binary_logit", "mean")
        )
        joined = per_subject.merge(full_scores, on=["case_id", "y_true"], validate="one_to_one")
        joined["prob_shift_from_full"] = joined["subgroup_mean_prob"] - joined["full_prob"]
        joined["logit_shift_from_full"] = joined["subgroup_mean_logit"] - joined["full_binary_logit"]
        for label_name, frame in [("all", joined), ("0", joined[joined["y_true"] == 0]), ("1", joined[joined["y_true"] == 1])]:
            subgroup_shift_rows.append(
                {
                    "subgroup": subgroup_name,
                    "y_true": label_name,
                    "n_subjects": int(len(frame)),
                    "mean_prob": float(frame["subgroup_mean_prob"].mean()),
                    "mean_prob_shift_from_full": float(frame["prob_shift_from_full"].mean()),
                    "mean_abs_prob_shift_from_full": float(frame["prob_shift_from_full"].abs().mean()),
                    "mean_binary_logit": float(frame["subgroup_mean_logit"].mean()),
                    "mean_logit_shift_from_full": float(frame["logit_shift_from_full"].mean()),
                    "mean_abs_logit_shift_from_full": float(frame["logit_shift_from_full"].abs().mean()),
                }
            )
    subgroup_shift_df = pd.DataFrame(subgroup_shift_rows)
    subgroup_shift_df.to_csv(output_dir / f"{output_prefix}_subgroup_score_shift.csv", index=False)

    aux_vs_main_rows = []
    for modality in modalities:
        main_frame = pooled[pooled["combo"] == modality]
        aux_frame = auxiliary[auxiliary["modality"] == modality]
        for predictor, frame in [("main_singleton", main_frame), ("auxiliary_head", aux_frame)]:
            optimum = optimal_threshold_row(frame["y_true"].to_numpy(), frame["y_prob"].to_numpy())
            aux_vs_main_rows.append(
                {
                    "modality": modality,
                    "predictor": predictor,
                    "auc": optimum["auc"],
                    "bac_at_0_5": threshold_metrics(frame["y_true"], frame["y_prob"], 0.5)["balanced_accuracy"],
                    "bac_at_checkpoint_global_threshold": threshold_metrics(
                        frame["y_true"], frame["y_prob"], checkpoint_threshold
                    )["balanced_accuracy"],
                    "validation_optimal_threshold": optimum["optimal_threshold"],
                    "validation_optimal_bal_acc": optimum["optimal_bal_acc"],
                }
            )
    aux_vs_main_df = pd.DataFrame(aux_vs_main_rows)
    aux_vs_main_df.to_csv(output_dir / f"{output_prefix}_aux_vs_main_singleton.csv", index=False)

    save_plots(pooled, threshold_df, subject_df, output_dir, output_prefix)
    checkpoint_hash_after = sha256_file(checkpoint_path)
    if checkpoint_hash_after != checkpoint_hash_before:
        raise RuntimeError("Checkpoint changed during a read-only diagnostic run.")

    optimal_thresholds = pattern_thresholds["optimal_threshold"]
    summary = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256_before": checkpoint_hash_before,
        "checkpoint_sha256_after": checkpoint_hash_after,
        "checkpoint_unchanged": True,
        "config": str(Path(args.config).resolve()),
        "data_root": str(Path(args.data_root).resolve()),
        "device": str(device),
        "validation_only": True,
        "test_dataset_indexed_or_inferred": False,
        "num_validation_subjects": int(len(val_dataset)),
        "num_patterns": int(len(combos)),
        "num_pooled_predictions": int(len(pooled)),
        "checkpoint_threshold": checkpoint_threshold,
        "checkpoint_threshold_metrics": checkpoint_threshold_metrics,
        "pooled_optimal_bal_acc": best_pooled_bac,
        "pooled_optimal_thresholds": threshold_df.loc[threshold_df["is_optimal"], "threshold"].tolist(),
        "checkpoint_threshold_is_optimal_or_tied": checkpoint_threshold_is_optimal,
        "pattern_optimal_threshold_min": float(optimal_thresholds.min()),
        "pattern_optimal_threshold_max": float(optimal_thresholds.max()),
        "pattern_optimal_threshold_range": float(optimal_thresholds.max() - optimal_thresholds.min()),
        "pattern_optimal_threshold_std": float(optimal_thresholds.std(ddof=1)),
        "subject_cross_pattern_summary": subject_summary,
        "top10_cross_pattern_range_subjects": top10.to_dict(orient="records"),
        "subgroup_score_shift": subgroup_shift_df.to_dict(orient="records"),
        "aux_vs_main_singleton": aux_vs_main_df.to_dict(orient="records"),
        "formal_threshold_updated": False,
        "model_or_checkpoint_modified": False,
        "optimizer_created_or_used": False,
    }
    with (output_dir / f"{output_prefix}_diagnostic_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
