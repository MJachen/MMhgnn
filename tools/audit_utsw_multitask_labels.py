from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.io import load_json, save_json
from utils.utsw_tasks import normalize_task_label


TASKS = {
    "idh": {"column": "IDH", "negative": "wildtype", "positive": "mutant"},
    "grade_2_vs_34": {"column": "Tumor Grade", "negative": "grade_2", "positive": "grade_3_4"},
    "mgmt": {"column": "MGMT", "negative": "unmethylated", "positive": "methylated"},
}


def parse_args():
    parser = argparse.ArgumentParser(description="Audit UTSW IDH, grade, and MGMT labels against the frozen imaging cohort.")
    parser.add_argument("--metadata-csv", required=True)
    parser.add_argument("--imaging-manifest", default="metadata/utsw_idh_manifest.csv")
    parser.add_argument("--frozen-split", default="splits/utsw_idh/utsw_idh_debug_split.json")
    parser.add_argument("--output-root", default="outputs/utsw_multitask")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bool_series(values: pd.Series) -> pd.Series:
    return values.astype(str).str.strip().str.casefold().eq("true")


def audit_task(task_name, metadata, manifest, split_assignment, source_path, output_root):
    task = TASKS[task_name]
    source_column = task["column"]
    rows = []
    manifest_by_case = manifest.set_index("case_id", drop=False)
    for _, source in metadata.iterrows():
        case_id = str(source["Subject ID"]).strip()
        image_row = manifest_by_case.loc[case_id]
        raw_value = source[source_column]
        normalized, binary_label, label_reason = normalize_task_label(task_name, raw_value)
        imaging_eligible = str(image_row["eligible"]).strip().casefold() == "true"
        eligible = imaging_eligible and binary_label is not None
        reasons = []
        if not imaging_eligible:
            reasons.append(str(image_row["exclusion_reason"]).strip() or "imaging_ineligible")
        if binary_label is None:
            reasons.append(label_reason)
        row = {
            "case_id": case_id,
            "source_column": source_column,
            "raw_label": raw_value,
            "normalized_label": normalized,
            "binary_label": binary_label,
            "imaging_eligible": imaging_eligible,
            "eligible": eligible,
            "exclusion_reason": ";".join(reason for reason in reasons if reason),
            "frozen_split": split_assignment.get(case_id, ""),
        }
        if task_name == "grade_2_vs_34":
            row.update({"raw_grade": raw_value, "normalized_grade": normalized})
        elif task_name == "mgmt":
            row.update({"raw_mgmt": raw_value, "normalized_mgmt": normalized})
        else:
            row.update({"raw_idh": raw_value, "normalized_idh": normalized})
        rows.append(row)

    audit = pd.DataFrame(rows)
    task_dir = output_root / task_name
    task_dir.mkdir(parents=True, exist_ok=True)
    audit.to_csv(task_dir / "label_audit.csv", index=False)
    usable = audit.loc[audit["eligible"]].copy()
    split_summary = {}
    for split_name in ["train", "val", "test"]:
        split_frame = usable.loc[usable["frozen_split"].eq(split_name)]
        split_summary[split_name] = {
            "cases": int(len(split_frame)),
            "class0": int(split_frame["binary_label"].eq(0).sum()),
            "class1": int(split_frame["binary_label"].eq(1).sum()),
            "has_both_classes": bool(split_frame["binary_label"].nunique() == 2),
        }
        if task_name == "grade_2_vs_34":
            split_summary[split_name].update(
                {
                    "grade2": int(split_frame["normalized_label"].eq(2).sum()),
                    "grade3": int(split_frame["normalized_label"].eq(3).sum()),
                    "grade4": int(split_frame["normalized_label"].eq(4).sum()),
                    "grade3_4": int(split_frame["normalized_label"].isin([3, 4]).sum()),
                }
            )

    raw_counts = metadata[source_column].astype(str).value_counts(dropna=False).to_dict()
    summary = {
        "task": task_name,
        "source_file": str(source_path),
        "source_file_sha256": sha256_file(source_path),
        "source_column": source_column,
        "raw_unique_values": {str(key): int(value) for key, value in raw_counts.items()},
        "raw_missing_or_empty": int(metadata[source_column].astype(str).str.strip().eq("").sum()),
        "metadata_rows": int(len(metadata)),
        "imaging_eligible_cases": int(audit["imaging_eligible"].sum()),
        "usable_cases": int(len(usable)),
        "excluded_cases": int(len(audit) - len(usable)),
        "class_distribution": {
            "class0": int(usable["binary_label"].eq(0).sum()),
            "class1": int(usable["binary_label"].eq(1).sum()),
        },
        "binary_mapping": {"0": task["negative"], "1": task["positive"]},
        "frozen_split_reused": True,
        "split_summary": split_summary,
        "split_has_both_classes": all(item["has_both_classes"] for item in split_summary.values()),
    }
    if task_name == "grade_2_vs_34":
        summary["grade_counts"] = {
            "grade2": int(usable["normalized_label"].eq(2).sum()),
            "grade3": int(usable["normalized_label"].eq(3).sum()),
            "grade4": int(usable["normalized_label"].eq(4).sum()),
            "grade3_4": int(usable["normalized_label"].isin([3, 4]).sum()),
        }
    save_json(summary, task_dir / "split_summary.json")
    return summary


def main():
    args = parse_args()
    source_path = Path(args.metadata_csv)
    metadata = pd.read_csv(source_path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    manifest = pd.read_csv(args.imaging_manifest, dtype=str, keep_default_na=False)
    required_source = {"Subject ID", *(task["column"] for task in TASKS.values())}
    missing_source = sorted(required_source.difference(metadata.columns))
    if missing_source:
        raise RuntimeError(f"UTSW source metadata is missing required task columns: {missing_source}")
    if metadata["Subject ID"].duplicated().any():
        raise RuntimeError("UTSW source metadata contains duplicate Subject ID values.")
    source_ids = set(metadata["Subject ID"].astype(str).str.strip())
    manifest_ids = set(manifest["case_id"].astype(str).str.strip())
    if source_ids != manifest_ids:
        raise RuntimeError(
            f"UTSW metadata/manifest subject mismatch: metadata_only={sorted(source_ids-manifest_ids)[:10]} "
            f"manifest_only={sorted(manifest_ids-source_ids)[:10]}"
        )
    split = load_json(args.frozen_split)
    split_assignment = {}
    for split_name in ["train", "val", "test"]:
        for case_id in split[split_name]:
            if case_id in split_assignment:
                raise RuntimeError(f"Frozen split leakage for case {case_id}.")
            split_assignment[case_id] = split_name
    eligible_ids = set(manifest.loc[_bool_series(manifest["eligible"]), "case_id"])
    if set(split_assignment) != eligible_ids:
        raise RuntimeError("Frozen split does not exactly cover the imaging-eligible cohort.")

    output_root = Path(args.output_root)
    summaries = {
        name: audit_task(name, metadata, manifest, split_assignment, source_path, output_root) for name in TASKS
    }
    save_json(summaries, output_root / "label_cohort_split_audit.json")
    print(json.dumps(summaries, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
