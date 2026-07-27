from __future__ import annotations

import math
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np

from utils.metrics import CALIBRATION_GROUPS_3WAY, modality_group_from_combo


DEFAULT_SELECTION_WEIGHTS = {
    "macro_bal_acc": 0.5,
    "worst_group_bal_acc": 0.3,
    "full_bal_acc": 0.2,
}


def combo_name(combo: Iterable[str]) -> str:
    return "_".join(combo)


def _finite_mean(values: Sequence[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(finite)) if finite else float("nan")


def summarize_combo_metrics(
    combo_metrics: Mapping[str, Mapping[str, float]],
    modalities: Sequence[str],
    weights: Mapping[str, float] | None = None,
) -> Dict[str, object]:
    """Summarize 15-combo validation metrics for B1 checkpoint selection."""
    modalities = tuple(modalities)
    expected_combo_count = (2 ** len(modalities)) - 1
    if len(combo_metrics) != expected_combo_count:
        raise ValueError(
            f"Expected {expected_combo_count} non-empty modality combinations, "
            f"got {len(combo_metrics)}."
        )

    configured_weights = dict(DEFAULT_SELECTION_WEIGHTS)
    if weights is not None:
        configured_weights.update({key: float(value) for key, value in weights.items()})
    if any(value < 0 for value in configured_weights.values()):
        raise ValueError(f"Checkpoint selection weights must be non-negative: {configured_weights}")
    weight_total = sum(configured_weights.values())
    if weight_total <= 0:
        raise ValueError("At least one checkpoint selection weight must be positive.")
    normalized_weights = {
        key: value / weight_total for key, value in configured_weights.items()
    }

    group_values = {
        metric: {group: [] for group in CALIBRATION_GROUPS_3WAY}
        for metric in ("bal_acc", "auc")
    }
    all_values = {metric: [] for metric in ("bal_acc", "auc")}
    for name, metrics in combo_metrics.items():
        combo = tuple(name.split("_"))
        group = modality_group_from_combo(combo)
        for metric in ("bal_acc", "auc"):
            value = float(metrics.get(metric, float("nan")))
            all_values[metric].append(value)
            group_values[metric][group].append(value)

    group_means = {
        metric: {
            group: _finite_mean(group_values[metric][group])
            for group in CALIBRATION_GROUPS_3WAY
        }
        for metric in ("bal_acc", "auc")
    }
    full_name = combo_name(modalities)
    if full_name not in combo_metrics:
        raise ValueError(f"Full-modality validation row is missing: {full_name}")

    macro_bal_acc = _finite_mean(all_values["bal_acc"])
    finite_group_ba = [
        value for value in group_means["bal_acc"].values() if math.isfinite(value)
    ]
    worst_group_bal_acc = min(finite_group_ba) if finite_group_ba else float("nan")
    full_bal_acc = float(combo_metrics[full_name]["bal_acc"])
    selection_score = (
        normalized_weights["macro_bal_acc"] * macro_bal_acc
        + normalized_weights["worst_group_bal_acc"] * worst_group_bal_acc
        + normalized_weights["full_bal_acc"] * full_bal_acc
    )

    finite_group_auc = [
        value for value in group_means["auc"].values() if math.isfinite(value)
    ]
    return {
        "selection_score": float(selection_score),
        "macro_bal_acc": float(macro_bal_acc),
        "worst_group_bal_acc": float(worst_group_bal_acc),
        "full_bal_acc": float(full_bal_acc),
        "macro_auc": float(_finite_mean(all_values["auc"])),
        "worst_group_auc": float(min(finite_group_auc)) if finite_group_auc else float("nan"),
        "full_auc": float(combo_metrics[full_name].get("auc", float("nan"))),
        "group_bal_acc": group_means["bal_acc"],
        "group_auc": group_means["auc"],
        "weights": normalized_weights,
        "combo_metrics": {name: dict(metrics) for name, metrics in combo_metrics.items()},
    }


def flatten_selection_summary(summary: Mapping[str, object]) -> Dict[str, float]:
    row = {
        key: float(summary[key])
        for key in (
            "selection_score",
            "macro_bal_acc",
            "worst_group_bal_acc",
            "full_bal_acc",
            "macro_auc",
            "worst_group_auc",
            "full_auc",
        )
    }
    for metric_key in ("group_bal_acc", "group_auc"):
        values = summary.get(metric_key, {})
        suffix = "bal_acc" if metric_key == "group_bal_acc" else "auc"
        for group in CALIBRATION_GROUPS_3WAY:
            row[f"{group}_{suffix}"] = float(values.get(group, float("nan")))
    return row
