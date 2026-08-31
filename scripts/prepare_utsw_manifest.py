from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.utsw import UTSW_MODALITIES, canonical_manifest_fingerprint, load_utsw_metadata, sha256_file


MRI_FILES = {
    "non_ants": {
        "t2": "brain_t2.nii.gz",
        "t1ce": "brain_t1ce.nii.gz",
        "t1": "brain_t1.nii.gz",
        "flair": "brain_flair.nii.gz",
    },
    "ants": {
        "t2": "brain_t2_ants.nii.gz",
        "t1ce": "brain_t1ce_ants.nii.gz",
        "t1": "brain_t1_ants.nii.gz",
        "flair": "brain_fl_ants.nii.gz",
    },
}
SEG_FILENAME = "tumorseg_FeTS.nii.gz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a frozen UTSW IDH eligibility manifest.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--metadata-csv", required=True)
    parser.add_argument("--geometry-qc-csv", required=True)
    parser.add_argument("--mri-version", choices=sorted(MRI_FILES), required=True)
    parser.add_argument("--manifest-csv", default="metadata/utsw_idh_manifest.csv")
    parser.add_argument("--summary-json", default="metadata/utsw_dataset_summary.json")
    parser.add_argument("--exclusions-csv", default="metadata/utsw_idh_exclusions.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root).resolve()
    metadata_path = Path(args.metadata_csv).resolve()
    geometry_path = Path(args.geometry_qc_csv).resolve()
    metadata = load_utsw_metadata(metadata_path)
    qc = pd.read_csv(geometry_path, dtype=str, keep_default_na=False)

    expected_rows = len(metadata) * len(UTSW_MODALITIES) * 2
    if len(qc) != expected_rows:
        raise ValueError(f"Geometry QC row count mismatch: expected {expected_rows}, found {len(qc)}")
    selected_qc = qc.loc[qc["mri_version"].eq(args.mri_version)].copy()
    duplicated = selected_qc.duplicated(["case_id", "modality"], keep=False)
    if duplicated.any():
        raise ValueError("Geometry QC contains duplicate case/modality rows for the selected MRI version.")

    case_qc = selected_qc.groupby("case_id", sort=False).agg(
        geometry_qc_status=("qc_status", lambda values: "pass" if all(value == "pass" for value in values) else "fail"),
        geometry_failure_reason=("failure_reason", lambda values: ";".join(sorted({value for value in values if value}))),
        geometry_qc_modalities=("modality", "nunique"),
    )

    rows = []
    filenames = MRI_FILES[args.mri_version]
    for _, source in metadata.iterrows():
        case_id = source["Subject ID"]
        case_dir = data_root / case_id
        paths = {modality: case_dir / filenames[modality] for modality in UTSW_MODALITIES}
        seg_path = case_dir / SEG_FILENAME
        missing_modalities = [modality for modality, path in paths.items() if not path.exists()]
        geometry = case_qc.loc[case_id] if case_id in case_qc.index else None

        reasons = []
        if source["label_exclusion_reason"]:
            reasons.append(source["label_exclusion_reason"])
        if not case_dir.is_dir():
            reasons.append("missing_case_directory")
        if missing_modalities:
            reasons.append("missing_mri:" + ",".join(missing_modalities))
        if not seg_path.exists():
            reasons.append("missing_segmentation")
        if geometry is None or int(geometry["geometry_qc_modalities"]) != len(UTSW_MODALITIES):
            reasons.append("missing_geometry_qc_rows")
            geometry_status = "fail"
            geometry_reason = "missing_geometry_qc_rows"
        else:
            geometry_status = str(geometry["geometry_qc_status"])
            geometry_reason = str(geometry["geometry_failure_reason"])
            if geometry_status != "pass":
                reasons.append("geometry_qc_failed:" + (geometry_reason or "unspecified"))

        label_value = source["normalized_idh_label"]
        normalized_label = "" if pd.isna(label_value) else int(label_value)
        eligible = len(reasons) == 0
        rows.append(
            {
                "case_id": case_id,
                "patient_id": case_id,
                "patient_id_source": "Subject ID",
                "t2_path": f"{case_id}/{filenames['t2']}",
                "t1ce_path": f"{case_id}/{filenames['t1ce']}",
                "t1_path": f"{case_id}/{filenames['t1']}",
                "flair_path": f"{case_id}/{filenames['flair']}",
                "segmentation_path": f"{case_id}/{SEG_FILENAME}",
                "mri_source_version": args.mri_version,
                "segmentation_source": "FeTS",
                "original_idh_label": source["original_idh_label"],
                "normalized_idh_label": normalized_label,
                "geometry_qc_status": geometry_status,
                "geometry_failure_reason": geometry_reason,
                "eligible": eligible,
                "exclusion_reason": ";".join(reasons),
                "split": "",
                "fold": "",
                "operation_status": source.get("Operation Status", ""),
                "scanner_make": source.get("Scanner Make", ""),
                "scanner_model": source.get("Scanner Model", ""),
                "scanner_strength": source.get("Scanner Strength", ""),
            }
        )

    manifest = pd.DataFrame(rows)
    manifest_path = Path(args.manifest_csv)
    summary_path = Path(args.summary_json)
    exclusions_path = Path(args.exclusions_csv)
    for path in [manifest_path, summary_path, exclusions_path]:
        path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)
    manifest.loc[~manifest["eligible"], ["case_id", "original_idh_label", "geometry_qc_status", "exclusion_reason"]].to_csv(exclusions_path, index=False)

    eligible = manifest.loc[manifest["eligible"]].copy()
    labels = eligible["normalized_idh_label"].astype(int)
    summary = {
        "data_root": "runtime-configured",
        "metadata_source_filename": metadata_path.name,
        "geometry_qc_csv": Path(args.geometry_qc_csv).as_posix(),
        "mri_source_version": args.mri_version,
        "segmentation_source": "FeTS",
        "label_mapping": {"IDH-wildtype": 0, "IDH-mutant": 1},
        "raw_case_directories": int(sum(path.is_dir() for path in data_root.iterdir())),
        "metadata_rows": int(len(metadata)),
        "metadata_unique_subjects": int(metadata["Subject ID"].nunique()),
        "metadata_with_normalized_idh": int(metadata["normalized_idh_label"].notna().sum()),
        "idh_wildtype_raw": int(metadata["normalized_idh_label"].eq(0).sum()),
        "idh_mutant_raw": int(metadata["normalized_idh_label"].eq(1).sum()),
        "idh_unknown_or_excluded": int(metadata["normalized_idh_label"].isna().sum()),
        "four_mri_complete_cases": int(
            manifest["case_id"].map(lambda case_id: all((data_root / case_id / filenames[modality]).exists() for modality in UTSW_MODALITIES)).sum()
        ),
        "segmentation_present_cases": int(manifest["case_id"].map(lambda case_id: (data_root / case_id / SEG_FILENAME).exists()).sum()),
        "geometry_qc_pass_cases": int(manifest["geometry_qc_status"].eq("pass").sum()),
        "geometry_qc_fail_cases": int(manifest["geometry_qc_status"].ne("pass").sum()),
        "eligible_cases": int(len(eligible)),
        "eligible_idh_wildtype": int(labels.eq(0).sum()),
        "eligible_idh_mutant": int(labels.eq(1).sum()),
        "excluded_cases": int((~manifest["eligible"]).sum()),
        "metadata_sha256": sha256_file(metadata_path),
        "geometry_qc_sha256": sha256_file(geometry_path),
        "manifest_sha256": sha256_file(manifest_path),
        "canonical_manifest_sha256": canonical_manifest_fingerprint(manifest),
        "patient_id_note": "No separate patient identifier is present in the supplied CSV; Subject ID is used as patient_id.",
    }
    with open(summary_path, "w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
