from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import sys
from typing import Dict, Iterable, Tuple

import nibabel as nib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.utsw import load_utsw_metadata


MRI_FILES: Dict[str, Dict[str, str]] = {
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
    parser = argparse.ArgumentParser(description="Audit UTSW MRI/FeTS geometry and foreground correspondence.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--metadata-csv", required=True)
    parser.add_argument("--output-csv", default="metadata/utsw_geometry_qc.csv")
    parser.add_argument("--inside-brain-threshold", type=float, default=0.99)
    parser.add_argument("--affine-atol", type=float, default=1e-3)
    parser.add_argument("--bounds-atol-mm", type=float, default=1e-3)
    return parser.parse_args()


def _shape_text(shape: Iterable[int]) -> str:
    return "x".join(str(int(value)) for value in shape)


def _vector_text(values: Iterable[float]) -> str:
    return "|".join(f"{float(value):.6g}" for value in values)


def _matrix_text(matrix: np.ndarray | None) -> str:
    if matrix is None:
        return ""
    return _vector_text(np.asarray(matrix, dtype=float).reshape(-1))


def _physical_bounds(image: nib.spatialimages.SpatialImage) -> Tuple[np.ndarray, np.ndarray]:
    corners = np.asarray(
        list(itertools.product(*[(0.0, float(size - 1)) for size in image.shape[:3]])),
        dtype=float,
    )
    world = nib.affines.apply_affine(image.affine, corners)
    return world.min(axis=0), world.max(axis=0)


def _form_info(image: nib.spatialimages.SpatialImage, form_name: str):
    matrix, code = getattr(image, f"get_{form_name}")(coded=True)
    return matrix, int(code or 0)


def _matrix_consistent(left: np.ndarray | None, right: np.ndarray | None, atol: float):
    if left is None or right is None:
        return None
    return bool(np.allclose(left, right, rtol=1e-5, atol=atol))


def _safe_bool_text(value) -> str:
    if value is None:
        return "not_comparable"
    return "true" if bool(value) else "false"


def audit_case(case_dir: Path, inside_threshold: float, affine_atol: float, bounds_atol: float):
    case_id = case_dir.name
    seg_path = case_dir / SEG_FILENAME
    seg_relative_path = f"{case_id}/{SEG_FILENAME}"
    if not seg_path.exists():
        for version, files in MRI_FILES.items():
            for modality, filename in files.items():
                yield {
                    "case_id": case_id,
                    "mri_version": version,
                    "modality": modality,
                    "mri_path": f"{case_id}/{filename}",
                    "seg_path": seg_relative_path,
                    "qc_status": "fail",
                    "failure_reason": "missing_segmentation",
                }
        return

    try:
        seg_image = nib.load(str(seg_path))
        seg_data = np.asanyarray(seg_image.dataobj)
    except Exception as exc:
        for version, files in MRI_FILES.items():
            for modality, filename in files.items():
                yield {
                    "case_id": case_id,
                    "mri_version": version,
                    "modality": modality,
                    "mri_path": f"{case_id}/{filename}",
                    "seg_path": seg_relative_path,
                    "qc_status": "fail",
                    "failure_reason": f"unreadable_segmentation:{type(exc).__name__}:{exc}",
                }
        return

    seg_foreground = np.isfinite(seg_data) & (seg_data > 0)
    seg_voxels = int(seg_foreground.sum())
    seg_unique = np.unique(seg_data[np.isfinite(seg_data)]).tolist()
    seg_spacing = tuple(float(value) for value in seg_image.header.get_zooms()[:3])
    seg_orientation = "".join(nib.aff2axcodes(seg_image.affine))
    seg_qform, seg_qcode = _form_info(seg_image, "qform")
    seg_sform, seg_scode = _form_info(seg_image, "sform")
    seg_bounds_min, seg_bounds_max = _physical_bounds(seg_image)

    for version, files in MRI_FILES.items():
        for modality, filename in files.items():
            mri_path = case_dir / filename
            base = {
                "case_id": case_id,
                "mri_version": version,
                "modality": modality,
                "mri_path": f"{case_id}/{filename}",
                "seg_path": seg_relative_path,
                "seg_shape": _shape_text(seg_image.shape[:3]),
                "seg_spacing": _vector_text(seg_spacing),
                "seg_orientation": seg_orientation,
                "seg_affine": _matrix_text(seg_image.affine),
                "seg_qform_code": seg_qcode,
                "seg_sform_code": seg_scode,
                "seg_qform": _matrix_text(seg_qform),
                "seg_sform": _matrix_text(seg_sform),
                "seg_bounds_min_mm": _vector_text(seg_bounds_min),
                "seg_bounds_max_mm": _vector_text(seg_bounds_max),
                "seg_unique_labels": json.dumps(seg_unique, ensure_ascii=False),
                "seg_foreground_voxels": seg_voxels,
            }
            if not mri_path.exists():
                yield {**base, "qc_status": "fail", "failure_reason": "missing_mri"}
                continue

            try:
                image = nib.load(str(mri_path))
                data = np.asanyarray(image.dataobj)
            except Exception as exc:
                yield {
                    **base,
                    "qc_status": "fail",
                    "failure_reason": f"unreadable_mri:{type(exc).__name__}:{exc}",
                }
                continue

            spacing = tuple(float(value) for value in image.header.get_zooms()[:3])
            orientation = "".join(nib.aff2axcodes(image.affine))
            qform, qcode = _form_info(image, "qform")
            sform, scode = _form_info(image, "sform")
            bounds_min, bounds_max = _physical_bounds(image)
            affine_consistent = bool(np.allclose(image.affine, seg_image.affine, rtol=1e-5, atol=affine_atol))
            shape_consistent = tuple(image.shape[:3]) == tuple(seg_image.shape[:3])
            spacing_consistent = bool(np.allclose(spacing, seg_spacing, rtol=1e-5, atol=1e-5))
            orientation_consistent = orientation == seg_orientation
            qform_consistent = _matrix_consistent(qform, seg_qform, affine_atol)
            sform_consistent = _matrix_consistent(sform, seg_sform, affine_atol)
            bounds_delta = float(
                max(
                    np.max(np.abs(bounds_min - seg_bounds_min)),
                    np.max(np.abs(bounds_max - seg_bounds_max)),
                )
            )
            bounds_consistent = bounds_delta <= bounds_atol

            inside_fraction = float("nan")
            brain_voxels = 0
            if shape_consistent:
                brain = np.isfinite(data) & (np.abs(data) > 1e-6)
                brain_voxels = int(brain.sum())
                if seg_voxels > 0:
                    inside_fraction = float(np.logical_and(seg_foreground, brain).sum() / seg_voxels)

            failures = []
            if not shape_consistent:
                failures.append("shape_mismatch")
            if not spacing_consistent:
                failures.append("spacing_mismatch")
            if not orientation_consistent:
                failures.append("orientation_mismatch")
            if not affine_consistent:
                failures.append("affine_mismatch")
            if not bounds_consistent:
                failures.append("physical_bounds_mismatch")
            if seg_voxels == 0:
                failures.append("empty_segmentation")
            if not np.isfinite(inside_fraction):
                failures.append("foreground_overlap_not_computable")
            elif inside_fraction < inside_threshold:
                failures.append("segmentation_outside_brain")
            if brain_voxels == 0:
                failures.append("empty_mri_foreground")

            yield {
                **base,
                "mri_shape": _shape_text(image.shape[:3]),
                "mri_spacing": _vector_text(spacing),
                "mri_orientation": orientation,
                "mri_affine": _matrix_text(image.affine),
                "mri_qform_code": qcode,
                "mri_sform_code": scode,
                "mri_qform": _matrix_text(qform),
                "mri_sform": _matrix_text(sform),
                "mri_bounds_min_mm": _vector_text(bounds_min),
                "mri_bounds_max_mm": _vector_text(bounds_max),
                "shape_consistency": shape_consistent,
                "spacing_consistency": spacing_consistent,
                "orientation_consistency": orientation_consistent,
                "affine_consistency": affine_consistent,
                "qform_consistency": _safe_bool_text(qform_consistent),
                "sform_consistency": _safe_bool_text(sform_consistent),
                "physical_bounds_max_abs_diff_mm": bounds_delta,
                "physical_bounds_consistency": bounds_consistent,
                "mri_foreground_voxels": brain_voxels,
                "foreground_overlap": inside_fraction,
                "seg_inside_brain_fraction": inside_fraction,
                "seg_consistency": len(failures) == 0,
                "qc_status": "pass" if not failures else "fail",
                "failure_reason": ";".join(failures),
            }


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root).resolve()
    metadata = load_utsw_metadata(args.metadata_csv)
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for case_id in metadata["Subject ID"].tolist():
        rows.extend(audit_case(data_root / case_id, args.inside_brain_threshold, args.affine_atol, args.bounds_atol_mm))
    qc = pd.DataFrame(rows)
    qc.to_csv(output_path, index=False)

    summary = (
        qc.groupby(["mri_version", "qc_status"], dropna=False)
        .size()
        .unstack(fill_value=0)
        .to_dict(orient="index")
    )
    case_pass = (
        qc.assign(is_pass=qc["qc_status"].eq("pass"))
        .groupby(["case_id", "mri_version"])["is_pass"]
        .all()
        .groupby("mri_version")
        .sum()
        .astype(int)
        .to_dict()
    )
    print(json.dumps({"output_csv": str(output_path), "row_summary": summary, "all_modalities_pass_cases": case_pass}, indent=2))


if __name__ == "__main__":
    main()
