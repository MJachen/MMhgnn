from __future__ import annotations

from typing import Dict, Mapping, Sequence

import numpy as np

from utils.metrics import calibrate_threshold, compute_binary_metrics


def combo_name(combo: Sequence[str]) -> str:
    return "_".join(combo)


def _safe_mean(values) -> float:
    finite = [float(value) for value in values if np.isfinite(value)]
    return float(np.mean(finite)) if finite else 0.0


def _calibrate(y_true, y_prob, calibration_cfg: Mapping) -> Dict:
    return calibrate_threshold(
        y_true,
        y_prob,
        metric=calibration_cfg.get("threshold_metric", "balanced_accuracy"),
        threshold_min=float(calibration_cfg.get("threshold_min", 0.0)),
        threshold_max=float(calibration_cfg.get("threshold_max", 1.0)),
        tie_break=str(calibration_cfg.get("tie_break", "first")),
    )


def summarize_joint_validation(
    predictions_by_combo: Mapping[str, Mapping[str, Sequence[float]]],
    selection_cfg: Mapping,
    calibration_cfg: Mapping,
) -> Dict:
    """Build a threshold-safe validation summary used for checkpoint selection."""
    full_name = combo_name(selection_cfg["full_combo"])
    no_t1ce_names = [combo_name(combo) for combo in selection_cfg["no_t1ce_combos"]]
    required = [full_name, *no_t1ce_names]
    missing = [name for name in required if name not in predictions_by_combo]
    if missing:
        raise ValueError(f"Missing joint-validation predictions for: {missing}")

    no_t1ce_true = np.concatenate([np.asarray(predictions_by_combo[name]["y_true"]) for name in no_t1ce_names])
    no_t1ce_prob = np.concatenate([np.asarray(predictions_by_combo[name]["y_prob"]) for name in no_t1ce_names])
    no_t1ce_calibration = _calibrate(no_t1ce_true, no_t1ce_prob, calibration_cfg)

    full_payload = predictions_by_combo[full_name]
    full_calibration = _calibrate(full_payload["y_true"], full_payload["y_prob"], calibration_cfg)

    combo_metrics = {}
    combo_thresholds = {}
    for name, payload in predictions_by_combo.items():
        if name == full_name:
            threshold = float(full_calibration["threshold"])
        elif name in no_t1ce_names:
            threshold = float(no_t1ce_calibration["threshold"])
        else:
            threshold = float(_calibrate(payload["y_true"], payload["y_prob"], calibration_cfg)["threshold"])
        combo_metrics[name] = compute_binary_metrics(payload["y_true"], payload["y_prob"], threshold=threshold)
        combo_metrics[name]["threshold"] = threshold
        combo_thresholds[name] = threshold

    no_t1ce_mean_auc = _safe_mean(combo_metrics[name]["auc"] for name in no_t1ce_names)
    no_t1ce_mean_bal_acc = _safe_mean(combo_metrics[name]["bal_acc"] for name in no_t1ce_names)
    full_auc = float(combo_metrics[full_name]["auc"])
    full_bal_acc = float(combo_metrics[full_name]["bal_acc"])

    weights = selection_cfg.get("weights", {})
    selection_score = (
        float(weights.get("no_t1ce_mean_auc", 0.4)) * no_t1ce_mean_auc
        + float(weights.get("no_t1ce_mean_bal_acc", 0.4)) * no_t1ce_mean_bal_acc
        + float(weights.get("full_auc", 0.1)) * full_auc
        + float(weights.get("full_bal_acc", 0.1)) * full_bal_acc
    )
    return {
        "selection_score": float(selection_score),
        "no_t1ce_mean_auc": no_t1ce_mean_auc,
        "no_t1ce_mean_bal_acc": no_t1ce_mean_bal_acc,
        "full_auc": full_auc,
        "full_bal_acc": full_bal_acc,
        "full_combo": full_name,
        "no_t1ce_combos": no_t1ce_names,
        "combo_thresholds": combo_thresholds,
        "combos": combo_metrics,
    }


def joint_summary_rank(summary: Mapping) -> tuple[float, float, float, float]:
    return (
        float(summary["selection_score"]),
        float(summary["no_t1ce_mean_auc"]),
        float(summary["no_t1ce_mean_bal_acc"]),
        float(summary["full_bal_acc"]),
    )


def is_better_joint_summary(candidate: Mapping, current: Mapping | None) -> bool:
    return current is None or joint_summary_rank(candidate) > joint_summary_rank(current)


def finetune_eligibility(candidate: Mapping, base: Mapping, selection_cfg: Mapping) -> Dict:
    min_auc_delta = float(selection_cfg.get("min_no_t1ce_auc_delta", 0.0))
    min_bal_acc_delta = float(selection_cfg.get("min_no_t1ce_bal_acc_delta", 0.0))
    max_full_drop = float(selection_cfg.get("max_full_bal_acc_drop", 0.03))
    checks = {
        "no_t1ce_auc": float(candidate["no_t1ce_mean_auc"]) >= float(base["no_t1ce_mean_auc"]) + min_auc_delta,
        "no_t1ce_bal_acc": float(candidate["no_t1ce_mean_bal_acc"]) >= float(base["no_t1ce_mean_bal_acc"]) + min_bal_acc_delta,
        "full_bal_acc_guardrail": float(candidate["full_bal_acc"]) >= float(base["full_bal_acc"]) - max_full_drop,
    }
    return {
        "eligible": all(checks.values()),
        "checks": checks,
        "deltas": {
            "no_t1ce_mean_auc": float(candidate["no_t1ce_mean_auc"]) - float(base["no_t1ce_mean_auc"]),
            "no_t1ce_mean_bal_acc": float(candidate["no_t1ce_mean_bal_acc"]) - float(base["no_t1ce_mean_bal_acc"]),
            "full_bal_acc": float(candidate["full_bal_acc"]) - float(base["full_bal_acc"]),
        },
        "limits": {
            "min_no_t1ce_auc_delta": min_auc_delta,
            "min_no_t1ce_bal_acc_delta": min_bal_acc_delta,
            "max_full_bal_acc_drop": max_full_drop,
        },
    }
