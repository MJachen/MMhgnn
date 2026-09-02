from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import nibabel as nib
import numpy as np
import pandas as pd
import torch

from datasets.brats_dataset import (
    ALL_MODALITIES,
    ROI_NAMES,
    BraTSClassificationDataset,
    build_acp_masks,
    estimate_brain_mask,
    zscore_normalize,
)
from utils.io import load_json
from utils.utsw import UTSW_MODALITIES, canonical_manifest_fingerprint, resolve_data_path


@dataclass
class UTSWCaseRecord:
    case_id: str
    patient_id: str
    case_dir: Path
    label: int
    files: Dict[str, Path]


def load_utsw_records(
    manifest_csv: str | Path,
    data_root: str | Path,
    fingerprint_json: str | Path | None = None,
) -> List[UTSWCaseRecord]:
    manifest = pd.read_csv(manifest_csv, dtype=str, keep_default_na=False)
    required = {
        "case_id",
        "patient_id",
        "normalized_idh_label",
        "eligible",
        "geometry_qc_status",
        "mri_source_version",
        "segmentation_source",
        "t2_path",
        "t1ce_path",
        "t1_path",
        "flair_path",
        "segmentation_path",
    }
    missing = sorted(required.difference(manifest.columns))
    if missing:
        raise ValueError(f"UTSW manifest is missing required columns: {missing}")

    eligible = manifest.loc[manifest["eligible"].str.casefold().eq("true")].copy()
    if eligible.empty:
        raise RuntimeError("UTSW manifest contains no eligible cases.")
    if not eligible["geometry_qc_status"].eq("pass").all():
        raise RuntimeError("Eligible UTSW records must all pass geometry QC.")
    if eligible["mri_source_version"].nunique() != 1:
        raise RuntimeError("UTSW eligible cohort mixes MRI source versions.")
    if eligible["segmentation_source"].nunique() != 1 or eligible["segmentation_source"].iloc[0] != "FeTS":
        raise RuntimeError("UTSW first-version cohort must use FeTS segmentation uniformly.")
    if eligible["case_id"].duplicated().any() or eligible["patient_id"].duplicated().any():
        raise RuntimeError("UTSW eligible manifest contains duplicate case_id or patient_id values.")

    if fingerprint_json:
        payload = load_json(fingerprint_json)
        actual = canonical_manifest_fingerprint(manifest)
        expected = str(payload["canonical_manifest_sha256"])
        if actual != expected:
            raise RuntimeError(f"UTSW manifest fingerprint mismatch: expected={expected} actual={actual}")

    records = []
    for _, row in eligible.iterrows():
        files = {modality: resolve_data_path(data_root, row[f"{modality}_path"]) for modality in UTSW_MODALITIES}
        files["seg"] = resolve_data_path(data_root, row["segmentation_path"])
        missing_files = [str(path) for path in files.values() if not path.exists()]
        if missing_files:
            raise FileNotFoundError(f"Frozen UTSW eligible record {row['case_id']} has missing files: {missing_files}")
        records.append(
            UTSWCaseRecord(
                case_id=row["case_id"],
                patient_id=row["patient_id"],
                case_dir=files["t2"].parent,
                label=int(row["normalized_idh_label"]),
                files=files,
            )
        )
    return records


def split_utsw_records(
    records: List[UTSWCaseRecord],
    split_json: str | Path,
    require_full_cohort: bool = True,
    expected_manifest_fingerprint: str | None = None,
):
    payload = load_json(split_json)
    if expected_manifest_fingerprint and payload.get("canonical_manifest_sha256") != expected_manifest_fingerprint:
        raise RuntimeError("UTSW split was not generated from the active frozen manifest fingerprint.")
    record_map = {record.case_id: record for record in records}
    expected = set(record_map)
    split_ids = {name: list(payload[name]) for name in ["train", "val", "test"]}
    for name, values in split_ids.items():
        if len(values) != len(set(values)):
            raise RuntimeError(f"UTSW split {name} contains duplicate case IDs.")
    observed = set().union(*(set(values) for values in split_ids.values()))
    if require_full_cohort and observed != expected:
        missing = sorted(expected.difference(observed))
        extra = sorted(observed.difference(expected))
        raise RuntimeError(f"UTSW split/manifest mismatch: missing={missing[:10]} extra={extra[:10]}")
    if not require_full_cohort and not observed.issubset(expected):
        extra = sorted(observed.difference(expected))
        raise RuntimeError(f"UTSW subset split contains unknown cases: {extra[:10]}")
    for left, right in [("train", "val"), ("train", "test"), ("val", "test")]:
        overlap = set(split_ids[left]).intersection(split_ids[right])
        if overlap:
            raise RuntimeError(f"UTSW case leakage between {left} and {right}: {sorted(overlap)[:10]}")
    splits = {name: [record_map[case_id] for case_id in split_ids[name]] for name in split_ids}
    patient_sets = {name: {record.patient_id for record in values} for name, values in splits.items()}
    for left, right in [("train", "val"), ("train", "test"), ("val", "test")]:
        overlap = patient_sets[left].intersection(patient_sets[right])
        if overlap:
            raise RuntimeError(f"UTSW patient leakage between {left} and {right}: {sorted(overlap)[:10]}")
    declared_counts = payload.get("counts", {})
    for name, values in splits.items():
        actual = {
            "cases": len(values),
            "idh_wildtype": sum(record.label == 0 for record in values),
            "idh_mutant": sum(record.label == 1 for record in values),
        }
        if declared_counts.get(name) != actual:
            raise RuntimeError(f"UTSW split {name} count metadata mismatch: declared={declared_counts.get(name)} actual={actual}")
    return splits


def _load_image(path: Path):
    image = nib.load(str(path))
    data = np.asarray(image.get_fdata(), dtype=np.float32)
    return image, data


def _assert_same_geometry(reference, candidate, case_id: str, name: str, atol: float = 1e-3) -> None:
    if tuple(candidate.shape[:3]) != tuple(reference.shape[:3]):
        raise RuntimeError(f"{case_id} {name}: shape differs from FeTS segmentation.")
    reference_spacing = reference.header.get_zooms()[:3]
    candidate_spacing = candidate.header.get_zooms()[:3]
    if not np.allclose(candidate_spacing, reference_spacing, rtol=1e-5, atol=1e-5):
        raise RuntimeError(f"{case_id} {name}: spacing differs from FeTS segmentation.")
    if nib.aff2axcodes(candidate.affine) != nib.aff2axcodes(reference.affine):
        raise RuntimeError(f"{case_id} {name}: orientation differs from FeTS segmentation.")
    if not np.allclose(candidate.affine, reference.affine, rtol=1e-5, atol=atol):
        raise RuntimeError(f"{case_id} {name}: affine differs from FeTS segmentation.")


class UTSWClassificationDataset(BraTSClassificationDataset):
    """UTSW adapter that preserves the existing model-facing BraTS tensor contract."""

    def __init__(self, records: List[UTSWCaseRecord], *args, **kwargs):
        modalities = list(kwargs.get("all_modalities") or ALL_MODALITIES)
        if modalities != UTSW_MODALITIES:
            raise ValueError(f"UTSW modality order must remain {UTSW_MODALITIES}, found {modalities}")
        target_shape = kwargs.get("target_shape")
        allow_full_resolution = bool(kwargs.get("allow_full_resolution_input", False))
        if target_shape is not None or not allow_full_resolution:
            raise ValueError("UTSW v1 requires native 1 mm full-resolution input: target_shape=null and allow_full_resolution_input=true.")
        super().__init__(records, *args, **kwargs)

    def __getitem__(self, index: int):
        record = self.records[index]
        combo, missing_combo, targeted_subgroup = self._training_view_spec()

        seg_image, seg = _load_image(record.files["seg"])
        spacing = tuple(float(value) for value in seg_image.header.get_zooms()[:3])
        if not np.allclose(spacing, (1.0, 1.0, 1.0), rtol=1e-5, atol=1e-5):
            raise RuntimeError(f"{record.case_id}: UTSW v1 expected 1 mm isotropic FeTS space, found {spacing}")
        if int((seg > 0).sum()) == 0:
            raise RuntimeError(f"{record.case_id}: FeTS segmentation is empty.")

        modality_full: Dict[str, np.ndarray] = {}
        for modality in self.all_modalities:
            image, volume = _load_image(record.files[modality])
            _assert_same_geometry(seg_image, image, record.case_id, modality)
            modality_full[modality] = volume

        brain_mask = estimate_brain_mask(modality_full)
        acp_masks, roi_valid = build_acp_masks(
            seg,
            brain_mask,
            q_core=self.q_core,
            r1=self.peri_inner_radius,
            r2=self.peri_outer_radius,
        )
        image_dict = {modality: zscore_normalize(modality_full[modality]) for modality in self.all_modalities}
        available_mask = np.asarray([1.0 if modality in combo else 0.0 for modality in self.all_modalities], dtype=np.float32)
        image_shape = seg.shape
        images = np.stack(
            [image_dict[modality] if modality in combo else np.zeros(image_shape, dtype=np.float32) for modality in self.all_modalities],
            axis=0,
        )
        roi_stack = np.stack([acp_masks[name].astype(np.float32) for name in ROI_NAMES], axis=0)
        sample = {
            "case_id": record.case_id,
            "patient_id": record.patient_id,
            "images": torch.from_numpy(images),
            "label": torch.tensor(record.label, dtype=torch.long),
            "seg": torch.from_numpy(seg.astype(np.float32)),
            "roi_masks": torch.from_numpy(roi_stack),
            "roi_valid": torch.from_numpy(roi_valid),
            "available_modalities": torch.from_numpy(available_mask),
            "combo": combo,
        }
        if missing_combo is not None:
            missing_mask = np.asarray([1.0 if modality in missing_combo else 0.0 for modality in self.all_modalities], dtype=np.float32)
            sample["missing_available_modalities"] = torch.from_numpy(missing_mask)
            sample["missing_combo"] = missing_combo
            sample["targeted_subgroup"] = targeted_subgroup
        return sample
