from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize same-split IDH/Grade/MGMT E4A->E4B results.")
    parser.add_argument("--output-root", default="/home/cjc/utsw_idh_outputs")
    parser.add_argument("--idh-split-json", default="splits/utsw_idh/utsw_idh_debug_split.json")
    return parser.parse_args()


def group_values(summary, group_type, group):
    row = next(item for item in summary["groups"] if item["group_type"] == group_type and str(item["group"]) == str(group))
    return float(row["mean_bal_acc"]), float(row["mean_auc"])


def task_counts(task_dir: Path, idh_split: Path | None):
    if task_dir.name in {"grade_2_vs_34", "mgmt"}:
        audit = load_json(task_dir / "split_summary.json")["split_summary"]
        return audit
    split = load_json(idh_split)
    result = {}
    for name in ["train", "val", "test"]:
        raw = split["counts"][name]
        result[name] = {
            "cases": int(raw["cases"]),
            "class0": int(raw.get("class0", raw.get("idh_wildtype"))),
            "class1": int(raw.get("class1", raw.get("idh_mutant"))),
        }
    return result


def make_row(task, display_name, run_dir: Path, task_dir: Path, idh_split: Path | None):
    summary = load_json(run_dir / "metrics" / "test_missing_pattern_summary.json")
    calibration = load_json(run_dir / "metrics" / "threshold_calibration.json")
    counts = task_counts(task_dir, idh_split)
    row = {
        "task": task,
        "display_name": display_name,
        "N_train": counts["train"]["cases"],
        "N_val": counts["val"]["cases"],
        "N_test": counts["test"]["cases"],
        "train_class0": counts["train"]["class0"],
        "train_class1": counts["train"]["class1"],
        "val_class0": counts["val"]["class0"],
        "val_class1": counts["val"]["class1"],
        "test_class0": counts["test"]["class0"],
        "test_class1": counts["test"]["class1"],
        "minority_test_n": min(counts["test"]["class0"], counts["test"]["class1"]),
        "global_threshold": float(calibration.get("threshold", calibration.get("calibrated_threshold", 0.5))),
        "mean15_BAC": float(summary["mean15_bal_acc"]),
        "mean15_AUC": float(summary["mean15_auc"]),
        "full_BAC": float(summary["full_modality"]["bal_acc"]),
        "full_AUC": float(summary["full_modality"]["auc"]),
        "benchmark_scope": "same_split_exploratory_transfer",
    }
    row["small_test_subgroup_warning"] = row["minority_test_n"] < 10
    for count in [1, 2, 3]:
        bac, auc = group_values(summary, "observed_modalities", count)
        row[f"modal{count}_BAC"] = bac
        row[f"modal{count}_AUC"] = auc
    for source, prefix in [
        ("t1ce_absent", "T1CE_absent"),
        ("t2_absent", "T2_absent"),
        ("t1ce_t2_both_present", "both_present"),
        ("t1ce_t2_both_absent", "both_absent"),
    ]:
        bac, auc = group_values(summary, "shortcut_subgroup", source)
        row[f"{prefix}_BAC"] = bac
        row[f"{prefix}_AUC"] = auc
    return row


def main():
    args = parse_args()
    root = Path(args.output_root)
    multitask = root / "utsw_multitask"
    specs = [
        ("idh", "IDH", root / "frozen_affine_e4b_seed42", root, Path(args.idh_split_json)),
        ("grade_2_vs_34", "Grade 2 vs 3/4", multitask / "grade_2_vs_34" / "e4b_seed42", multitask / "grade_2_vs_34", None),
        ("mgmt", "MGMT", multitask / "mgmt" / "e4b_seed42", multitask / "mgmt", None),
    ]
    rows = [make_row(*spec) for spec in specs]
    output = multitask / "e4c_cross_task_same_split_summary.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(output)


if __name__ == "__main__":
    main()
