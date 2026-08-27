from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.utsw import (
    UTSW_FINGERPRINT_COLUMNS,
    UTSW_MODALITIES,
    canonical_manifest_fingerprint,
    sha256_file,
    validate_relative_data_path,
)


PATH_COLUMNS = [f"{modality}_path" for modality in UTSW_MODALITIES] + ["segmentation_path"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze a platform-neutral UTSW manifest and cohort fingerprint.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--manifest-csv", default="metadata/utsw_idh_manifest.csv")
    parser.add_argument("--geometry-qc-csv", default="metadata/utsw_geometry_qc.csv")
    parser.add_argument("--summary-json", default="metadata/utsw_dataset_summary.json")
    parser.add_argument("--original-summary-json", default="metadata/utsw_dataset_summary_original.json")
    parser.add_argument("--fingerprint-json", default="metadata/utsw_idh_manifest_fingerprint.json")
    parser.add_argument("--inventory-csv", default="metadata/utsw_idh_file_inventory.csv")
    parser.add_argument("--split-json", default="splits/utsw_idh/utsw_idh_debug_split.json")
    return parser.parse_args()


def _relative_path(value: object, data_root: Path) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError("Cannot freeze an empty UTSW data path.")
    candidate = Path(text)
    if candidate.is_absolute():
        try:
            relative = candidate.resolve().relative_to(data_root)
        except ValueError as exc:
            raise ValueError(f"Path is outside data root: {candidate}") from exc
        portable = PurePosixPath(*relative.parts)
    else:
        portable = validate_relative_data_path(text.replace("\\", "/"))
    return portable.as_posix()


def _assert_exact_case(path: Path, data_root: Path) -> None:
    relative = path.relative_to(data_root)
    current = data_root
    for part in relative.parts:
        exact_names = {child.name for child in current.iterdir()}
        if part not in exact_names:
            raise FileNotFoundError(f"Linux case-sensitive path mismatch under {current}: expected {part}")
        current = current / part


def _canonical_inventory_hash(inventory: pd.DataFrame) -> str:
    ordered = inventory.sort_values(["case_id", "modality"], kind="stable").to_dict(orient="records")
    payload = json.dumps(ordered, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sanitize_summary(summary: dict, geometry_path: Path, split_path: Path) -> dict:
    summary = dict(summary)
    summary.pop("metadata_csv", None)
    summary["data_root"] = "runtime-configured"
    summary["geometry_qc_csv"] = geometry_path.as_posix()
    summary["debug_split_json"] = split_path.as_posix()
    summary["label_mapping"] = {"IDH-wildtype": 0, "IDH-mutant": 1}
    return summary


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root).expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"UTSW data root does not exist: {data_root}")

    manifest_path = Path(args.manifest_csv)
    geometry_path = Path(args.geometry_qc_csv)
    summary_path = Path(args.summary_json)
    original_summary_path = Path(args.original_summary_json)
    fingerprint_path = Path(args.fingerprint_json)
    inventory_path = Path(args.inventory_csv)
    split_path = Path(args.split_json)

    manifest = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    for column in PATH_COLUMNS:
        manifest[column] = manifest[column].map(lambda value: _relative_path(value, data_root))

    eligible = manifest.loc[manifest["eligible"].str.casefold().eq("true")].copy()
    if len(eligible) != 618:
        raise RuntimeError(f"Frozen UTSW cohort must contain 618 eligible cases, found {len(eligible)}")
    labels = eligible["normalized_idh_label"].astype(int)
    if {0: int(labels.eq(0).sum()), 1: int(labels.eq(1).sum())} != {0: 442, 1: 176}:
        raise RuntimeError("Frozen UTSW label distribution changed from 442 IDH-WT / 176 IDH-mutant.")

    inventory_rows = []
    for _, row in eligible.iterrows():
        for modality, column in [(m, f"{m}_path") for m in UTSW_MODALITIES] + [("segmentation", "segmentation_path")]:
            relative = validate_relative_data_path(row[column])
            absolute = data_root.joinpath(*relative.parts)
            if not absolute.is_file():
                raise FileNotFoundError(f"Missing frozen UTSW file: {absolute}")
            _assert_exact_case(absolute, data_root)
            inventory_rows.append(
                {
                    "case_id": row["case_id"],
                    "modality": modality,
                    "relative_path": relative.as_posix(),
                    "size_bytes": int(absolute.stat().st_size),
                }
            )

    manifest.to_csv(manifest_path, index=False, lineterminator="\n")
    inventory = pd.DataFrame(inventory_rows)
    inventory.to_csv(inventory_path, index=False, lineterminator="\n")

    if geometry_path.exists():
        geometry = pd.read_csv(geometry_path, dtype=str, keep_default_na=False)
        for column in ["mri_path", "seg_path"]:
            if column in geometry.columns:
                geometry[column] = geometry[column].map(lambda value: _relative_path(value, data_root))
        geometry.to_csv(geometry_path, index=False, lineterminator="\n")

    canonical_hash = canonical_manifest_fingerprint(manifest)
    fingerprint = {
        "schema_version": 1,
        "dataset": "utsw_idh",
        "label_mapping": {"0": "IDH-WT", "1": "IDH-mutant"},
        "eligible_cases": 618,
        "eligible_idh_wildtype": 442,
        "eligible_idh_mutant": 176,
        "mri_source_version": "non_ants",
        "segmentation_source": "FeTS",
        "fingerprint_columns": UTSW_FINGERPRINT_COLUMNS,
        "canonical_manifest_sha256": canonical_hash,
        "manifest_file_sha256": sha256_file(manifest_path),
        "file_inventory_rows": int(len(inventory)),
        "file_inventory_canonical_sha256": _canonical_inventory_hash(inventory),
        "file_inventory_file_sha256": sha256_file(inventory_path),
    }
    fingerprint_path.write_text(json.dumps(fingerprint, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary = _sanitize_summary(summary, geometry_path, split_path)
    summary.update(
        {
            "geometry_qc_sha256": sha256_file(geometry_path) if geometry_path.exists() else None,
            "manifest_sha256": sha256_file(manifest_path),
            "canonical_manifest_sha256": canonical_hash,
            "manifest_fingerprint_json": fingerprint_path.as_posix(),
            "file_inventory_csv": inventory_path.as_posix(),
            "file_inventory_canonical_sha256": fingerprint["file_inventory_canonical_sha256"],
        }
    )
    if not original_summary_path.exists():
        original_summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    split = json.loads(split_path.read_text(encoding="utf-8"))
    split["canonical_manifest_sha256"] = canonical_hash
    split["label_mapping"] = {"0": "IDH-WT", "1": "IDH-mutant"}
    split_path.write_text(json.dumps(split, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(fingerprint, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
