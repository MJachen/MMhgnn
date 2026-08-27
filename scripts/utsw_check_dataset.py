from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import nibabel as nib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.brats_dataset import ROI_NAMES, build_acp_masks, estimate_brain_mask
from datasets.utsw_dataset import _assert_same_geometry, load_utsw_records, split_utsw_records
from utils.utsw import UTSW_MODALITIES, canonical_manifest_fingerprint, resolve_data_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify the frozen UTSW cohort after copying it to a server.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--manifest-csv", default="metadata/utsw_idh_manifest.csv")
    parser.add_argument("--fingerprint-json", default="metadata/utsw_idh_manifest_fingerprint.json")
    parser.add_argument("--inventory-csv", default="metadata/utsw_idh_file_inventory.csv")
    parser.add_argument("--split-json", default="splits/utsw_idh/utsw_idh_debug_split.json")
    parser.add_argument("--geometry-samples", type=int, default=10)
    parser.add_argument("--roi-samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def _canonical_inventory_hash(inventory: pd.DataFrame) -> str:
    columns = ["case_id", "modality", "relative_path", "size_bytes"]
    normalized = inventory.loc[:, columns].copy()
    normalized["size_bytes"] = normalized["size_bytes"].astype(int)
    records = normalized.sort_values(["case_id", "modality"], kind="stable").to_dict(orient="records")
    payload = json.dumps(records, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root).expanduser().resolve()
    manifest = pd.read_csv(args.manifest_csv, dtype=str, keep_default_na=False)
    fingerprint = json.loads(Path(args.fingerprint_json).read_text(encoding="utf-8"))
    actual_fingerprint = canonical_manifest_fingerprint(manifest)
    if actual_fingerprint != fingerprint["canonical_manifest_sha256"]:
        raise RuntimeError("Portable manifest fingerprint differs from the frozen cohort fingerprint.")

    records = load_utsw_records(args.manifest_csv, data_root, args.fingerprint_json)
    labels = pd.Series([record.label for record in records]).value_counts().reindex([0, 1], fill_value=0)
    expected_counts = {
        0: int(fingerprint["eligible_idh_wildtype"]),
        1: int(fingerprint["eligible_idh_mutant"]),
    }
    if len(records) != int(fingerprint["eligible_cases"]) or labels.to_dict() != expected_counts:
        raise RuntimeError(f"Server cohort changed: cases={len(records)} labels={labels.to_dict()}")

    split_utsw_records(records, args.split_json, require_full_cohort=True)
    inventory = pd.read_csv(args.inventory_csv, dtype={"case_id": str, "modality": str, "relative_path": str, "size_bytes": int})
    if len(inventory) != int(fingerprint["file_inventory_rows"]):
        raise RuntimeError("UTSW file inventory row count changed.")
    if _canonical_inventory_hash(inventory) != fingerprint["file_inventory_canonical_sha256"]:
        raise RuntimeError("UTSW file inventory fingerprint changed.")
    for row in inventory.itertuples(index=False):
        path = resolve_data_path(data_root, row.relative_path)
        if not path.is_file():
            raise FileNotFoundError(f"Missing copied UTSW file: {path}")
        if path.stat().st_size != int(row.size_bytes):
            raise RuntimeError(f"Copied UTSW file size mismatch: {path}")

    rng = np.random.default_rng(args.seed)
    sampled = rng.choice(np.asarray(records, dtype=object), size=min(args.geometry_samples, len(records)), replace=False).tolist()
    roi_case_ids = {record.case_id for record in sampled[: min(args.roi_samples, len(sampled))]}
    geometry_rows = []
    roi_rows = []
    for record in sampled:
        seg_image = nib.load(str(record.files["seg"]))
        seg = np.asarray(seg_image.get_fdata(), dtype=np.float32)
        volumes = {}
        for modality in UTSW_MODALITIES:
            image = nib.load(str(record.files[modality]))
            _assert_same_geometry(seg_image, image, record.case_id, modality)
            volumes[modality] = np.asarray(image.get_fdata(), dtype=np.float32)
        geometry_rows.append({"case_id": record.case_id, "status": "pass"})
        if record.case_id in roi_case_ids:
            masks, valid = build_acp_masks(seg, estimate_brain_mask(volumes), q_core=0.4, r1=3, r2=7)
            if not np.all(valid == 1):
                raise RuntimeError(f"ROI generation produced an invalid node for {record.case_id}: {valid.tolist()}")
            roi_rows.append(
                {
                    "case_id": record.case_id,
                    "roi_voxels": {name: int(masks[name].sum()) for name in ROI_NAMES},
                    "roi_valid": valid.tolist(),
                }
            )

    report = {
        "status": "pass",
        "eligible_cases": len(records),
        "idh_wildtype": int(labels[0]),
        "idh_mutant": int(labels[1]),
        "canonical_manifest_sha256": actual_fingerprint,
        "files_checked": int(len(inventory)),
        "geometry_samples": geometry_rows,
        "roi_samples": roi_rows,
    }
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
