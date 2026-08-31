from __future__ import annotations

from typing import Dict, Iterable, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score

MASK_ORDER = ["t1", "t1ce", "t2", "flair"]
CALIBRATION_GROUPS_3WAY = ["has_t1ce", "no_t1ce_no_t1", "no_t1ce_with_t1"]


def compute_binary_metrics(y_true, y_prob, threshold: float = 0.5) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred) if len(y_true) > 0 else 0.0
    f1 = f1_score(y_true, y_pred, zero_division=0) if len(y_true) > 0 else 0.0

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sen = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spe = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    bal_acc = (sen + spe) / 2.0

    auc = float("nan")
    auc_warning = ""
    if len(np.unique(y_true)) >= 2:
        auc = roc_auc_score(y_true, y_prob)
    else:
        auc_warning = "AUC skipped because only one class is present in y_true."

    return {
        "acc": float(acc),
        "auc": float(auc) if not np.isnan(auc) else float("nan"),
        "f1": float(f1),
        "sen": float(sen),
        "spe": float(spe),
        "bal_acc": float(bal_acc),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "auc_warning": auc_warning,
    }


MISSING_PATTERN_METRICS = ("bal_acc", "auc", "acc", "sen", "spe")


def summarize_missing_pattern_metrics(metrics_by_combo: Dict[str, Dict], modalities: Sequence[str]) -> Dict[str, object]:
    """Summarize per-pattern metrics with equal weight for each non-empty modality pattern."""
    rows = []
    for combo_name, metrics in metrics_by_combo.items():
        combo = tuple(part for part in combo_name.split("_") if part)
        rows.append({"combo": combo_name, "modalities": combo, "metrics": metrics})

    def aggregate(selected_rows):
        aggregate_row = {"num_patterns": len(selected_rows)}
        for metric_name in MISSING_PATTERN_METRICS:
            values = np.asarray([row["metrics"].get(metric_name, float("nan")) for row in selected_rows], dtype=float)
            finite = values[np.isfinite(values)]
            aggregate_row[f"mean_{metric_name}"] = float(finite.mean()) if len(finite) > 0 else float("nan")
        return aggregate_row

    grouped = []
    for observed_count in range(1, len(modalities) + 1):
        selected = [row for row in rows if len(row["modalities"]) == observed_count]
        grouped.append({"group_type": "observed_modalities", "group": str(observed_count), **aggregate(selected)})

    subgroup_rules = {
        "t1ce_absent": lambda combo: "t1ce" not in combo,
        "t2_absent": lambda combo: "t2" not in combo,
        "t1ce_t2_both_present": lambda combo: "t1ce" in combo and "t2" in combo,
        "t1ce_t2_both_absent": lambda combo: "t1ce" not in combo and "t2" not in combo,
    }
    for group_name, predicate in subgroup_rules.items():
        selected = [row for row in rows if predicate(set(row["modalities"]))]
        grouped.append({"group_type": "shortcut_subgroup", "group": group_name, **aggregate(selected)})

    overall = aggregate(rows)
    full_combo_name = "_".join(modalities)
    full_metrics = metrics_by_combo.get(full_combo_name, {})
    return {
        "num_patterns": overall["num_patterns"],
        "mean15_bal_acc": overall["mean_bal_acc"],
        "mean15_auc": overall["mean_auc"],
        "mean15_metrics": overall,
        "full_modality_combo": full_combo_name,
        "full_modality": {name: full_metrics.get(name, float("nan")) for name in MISSING_PATTERN_METRICS},
        "groups": grouped,
    }


def calibrate_threshold(y_true, y_prob, metric: str = "balanced_accuracy", num_steps: int = 201) -> Dict[str, float]:
    """Search a global decision threshold on validation predictions only."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    if len(y_true) == 0:
        return {
            "threshold": 0.5,
            "metric": metric,
            "score": 0.0,
            "default_threshold": 0.5,
            "warning": "No validation samples available for threshold calibration.",
        }

    thresholds = np.linspace(0.0, 1.0, num_steps)
    best_threshold = 0.5
    best_score = -1.0
    for threshold in thresholds:
        metrics = compute_binary_metrics(y_true, y_prob, threshold=float(threshold))
        if metric in {"balanced_accuracy", "bal_acc"}:
            score = metrics["bal_acc"]
        elif metric in {"youden", "youden_j"}:
            score = metrics["sen"] + metrics["spe"] - 1.0
        elif metric == "f1":
            score = metrics["f1"]
        else:
            raise ValueError(f"Unsupported calibration metric: {metric}")
        if score > best_score:
            best_score = float(score)
            best_threshold = float(threshold)

    return {
        "threshold": best_threshold,
        "metric": metric,
        "score": best_score,
        "default_threshold": 0.5,
        "warning": "",
    }


def modality_group_from_mask(mask: Sequence[float] | np.ndarray) -> str:
    mask = np.asarray(mask).astype(int).tolist()
    t1, t1ce, _, _ = mask
    if t1ce == 1:
        return "has_t1ce"
    if t1 == 0:
        return "no_t1ce_no_t1"
    return "no_t1ce_with_t1"


def modality_group_from_combo(combo: Iterable[str]) -> str:
    combo_set = set(combo)
    mask = [1 if name in combo_set else 0 for name in MASK_ORDER]
    return modality_group_from_mask(mask)


def calibrate_grouped_3way_t1ce_t1(y_true, y_prob, combos, metric: str = "balanced_accuracy", default_threshold: float = 0.5, min_samples: int = 2) -> Dict[str, object]:
    """Calibrate three validation-only thresholds using t1ce/t1 visibility groups."""
    global_result = calibrate_threshold(y_true, y_prob, metric=metric)
    global_threshold = float(global_result.get("threshold", default_threshold))
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    group_to_indices = {name: [] for name in CALIBRATION_GROUPS_3WAY}
    for idx, combo in enumerate(combos):
        group_to_indices[modality_group_from_combo(combo)].append(idx)

    result: Dict[str, object] = {
        "enabled": True,
        "mode": "grouped_3way_t1ce_t1",
        "metric": metric,
        "calibration_metric": metric,
        "default_threshold": float(default_threshold),
        "global_threshold": global_threshold,
        "threshold": global_threshold,
        "calibrated_threshold": global_threshold,
        "threshold_has_t1ce": global_threshold,
        "threshold_no_t1ce_no_t1": global_threshold,
        "threshold_no_t1ce_with_t1": global_threshold,
        "fallback_info": {},
        "groups": {},
    }

    key_map = {
        "has_t1ce": "threshold_has_t1ce",
        "no_t1ce_no_t1": "threshold_no_t1ce_no_t1",
        "no_t1ce_with_t1": "threshold_no_t1ce_with_t1",
    }

    for name in CALIBRATION_GROUPS_3WAY:
        indices = group_to_indices[name]
        group_info = {
            "num_samples": len(indices),
            "num_positive": 0,
            "num_negative": 0,
            "threshold": global_threshold,
            "fallback": False,
            "warning": "",
            "fallback_target": "",
        }
        if indices:
            labels = y_true[indices]
            group_info["num_positive"] = int((labels == 1).sum())
            group_info["num_negative"] = int((labels == 0).sum())
        if len(indices) < min_samples or (indices and len(np.unique(y_true[indices])) < 2):
            group_info["fallback"] = True
            group_info["warning"] = "Fallback to global threshold because validation group has too few samples or a single class."
            group_info["fallback_target"] = "global_threshold"
            result["fallback_info"][name] = {
                "reason": group_info["warning"],
                "applied_threshold": global_threshold,
                "fallback_target": "global_threshold",
            }
        else:
            group_result = calibrate_threshold(y_true[indices], y_prob[indices], metric=metric)
            group_info["threshold"] = float(group_result["threshold"])
            group_info["score"] = float(group_result["score"])
            result[key_map[name]] = float(group_result["threshold"])
        result["groups"][name] = group_info
    return result


def threshold_dispatch_for_combo(calibration_result: Dict, combo, default_threshold: float = 0.5) -> Dict[str, object]:
    if not calibration_result:
        return {"threshold_group": "global", "applied_threshold": float(default_threshold)}
    mode = calibration_result.get("mode") or calibration_result.get("threshold_mode")
    if mode == "grouped_3way_t1ce_t1":
        group = modality_group_from_combo(combo)
        key_map = {
            "has_t1ce": "threshold_has_t1ce",
            "no_t1ce_no_t1": "threshold_no_t1ce_no_t1",
            "no_t1ce_with_t1": "threshold_no_t1ce_with_t1",
        }
        threshold = float(calibration_result.get(key_map[group], calibration_result.get("global_threshold", default_threshold)))
        return {"threshold_group": group, "applied_threshold": threshold}
    if mode == "grouped_has_t1ce":
        group = "has_t1ce" if "t1ce" in set(combo) else "no_t1ce"
        threshold = float(calibration_result.get("threshold_has_t1ce" if group == "has_t1ce" else "threshold_no_t1ce", calibration_result.get("global_threshold", default_threshold)))
        return {"threshold_group": group, "applied_threshold": threshold}
    threshold = float(calibration_result.get("threshold", calibration_result.get("calibrated_threshold", default_threshold)))
    return {"threshold_group": "global", "applied_threshold": threshold}


def threshold_for_combo(calibration_result: Dict, combo, default_threshold: float = 0.5) -> float:
    return float(threshold_dispatch_for_combo(calibration_result, combo, default_threshold)["applied_threshold"])


DROP_T1_ABLATION_REMAP = {
    "t1": None,
    "t2_t1": "t2",
    "t1_flair": "flair",
    "t2_t1_flair": "t2_flair",
}


def canonical_combo_name(combo) -> str:
    if isinstance(combo, str):
        return combo
    return "_".join(combo)


def drop_t1_ablation_remap(combo_name: str):
    return DROP_T1_ABLATION_REMAP.get(combo_name)


def build_drop_t1_ablation_report(metrics_by_combo: Dict[str, Dict], combo_names: Sequence[str] | None = None):
    combo_names = list(combo_names or DROP_T1_ABLATION_REMAP.keys())
    rows = []
    summary_rows = []
    for original_combo in combo_names:
        ablated_combo = drop_t1_ablation_remap(original_combo)
        baseline = metrics_by_combo.get(original_combo, {})
        row = {
            "original_combo": original_combo,
            "ablated_combo": ablated_combo or "none",
            "status": "ok",
        }
        for prefix, source in [("baseline", baseline)]:
            row[f"{prefix}_threshold_group"] = source.get("threshold_group", "")
            row[f"{prefix}_applied_threshold"] = source.get("applied_threshold", source.get("threshold", ""))
            for key in ["acc", "auc", "f1", "sen", "spe", "bal_acc"]:
                row[f"{prefix}_{key}"] = source.get(key, float("nan"))

        if ablated_combo is None:
            row["status"] = "not_applicable_after_drop_t1"
            rows.append(row)
            continue

        ablation = metrics_by_combo.get(ablated_combo)
        if not ablation:
            row["status"] = "ablated_combo_not_evaluated"
            rows.append(row)
            continue

        row["ablation_threshold_group"] = ablation.get("threshold_group", "")
        row["ablation_applied_threshold"] = ablation.get("applied_threshold", ablation.get("threshold", ""))
        for key in ["acc", "auc", "f1", "sen", "spe", "bal_acc"]:
            row[f"ablation_{key}"] = ablation.get(key, float("nan"))

        for key in ["auc", "spe", "bal_acc"]:
            baseline_value = row.get(f"baseline_{key}", float("nan"))
            ablation_value = row.get(f"ablation_{key}", float("nan"))
            row[f"delta_{key}"] = (ablation_value - baseline_value) if (baseline_value == baseline_value and ablation_value == ablation_value) else float("nan")

        row["improved_bal_acc"] = bool(row.get("delta_bal_acc", 0.0) > 0) if row.get("delta_bal_acc", float("nan")) == row.get("delta_bal_acc", float("nan")) else False
        row["improved_auc"] = bool(row.get("delta_auc", 0.0) > 0) if row.get("delta_auc", float("nan")) == row.get("delta_auc", float("nan")) else False
        row["improved_spe"] = bool(row.get("delta_spe", 0.0) > 0) if row.get("delta_spe", float("nan")) == row.get("delta_spe", float("nan")) else False
        row["improved_after_drop_t1"] = bool(row["improved_bal_acc"] or row["improved_auc"] or row["improved_spe"])
        rows.append(row)
        summary_rows.append({
            "original_combo": original_combo,
            "ablated_combo": ablated_combo,
            "improved_after_drop_t1": row["improved_after_drop_t1"],
            "delta_bal_acc": row.get("delta_bal_acc", float("nan")),
            "delta_auc": row.get("delta_auc", float("nan")),
            "delta_spe": row.get("delta_spe", float("nan")),
        })
    return {"rows": rows, "summary": summary_rows}
