from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.brats_dataset import ROI_NAMES, build_acp_masks, estimate_brain_mask
from utils.utsw import resolve_data_path


COLORS = {
    "core": (0.85, 0.10, 0.10),
    "boundary": (1.00, 0.55, 0.00),
    "peri_inner": (0.10, 0.75, 0.20),
    "peri_outer": (0.10, 0.40, 0.95),
    "distal_normal": (0.60, 0.15, 0.80),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the five fixed UTSW anatomy/context ROI nodes.")
    parser.add_argument("--manifest-csv", default="metadata/utsw_idh_manifest.csv")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", default="outputs/utsw_idh/roi_qc")
    parser.add_argument("--overlay-count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--q-core", type=float, default=0.4)
    parser.add_argument("--peri-inner-radius", type=int, default=3)
    parser.add_argument("--peri-outer-radius", type=int, default=7)
    return parser.parse_args()


def _load(path: str) -> np.ndarray:
    return np.asarray(nib.load(path).get_fdata(), dtype=np.float32)


def _normalized_slice(image: np.ndarray) -> np.ndarray:
    values = image[np.isfinite(image) & (image != 0)]
    if values.size == 0:
        return np.zeros_like(image, dtype=np.float32)
    low, high = np.percentile(values, [1, 99])
    if high <= low:
        return np.zeros_like(image, dtype=np.float32)
    return np.clip((image - low) / (high - low), 0, 1)


def save_overlay(case_id: str, flair: np.ndarray, seg: np.ndarray, masks, output_path: Path) -> None:
    tumor = seg > 0
    slice_scores = tumor.sum(axis=(0, 1))
    slice_index = int(np.argmax(slice_scores))
    base = _normalized_slice(flair[:, :, slice_index]).T
    rgb = np.repeat(base[..., None], 3, axis=-1)
    alpha = 0.28
    for name in ROI_NAMES:
        mask = masks[name][:, :, slice_index].T.astype(bool)
        color = np.asarray(COLORS[name], dtype=np.float32)
        rgb[mask] = (1.0 - alpha) * rgb[mask] + alpha * color

    fig, axis = plt.subplots(figsize=(8, 8))
    axis.imshow(rgb, origin="lower")
    axis.contour(tumor[:, :, slice_index].T, levels=[0.5], colors=["white"], linewidths=0.7)
    axis.set_title(f"{case_id} | axial index {slice_index} | FeTS + five ROI")
    axis.axis("off")
    legend = [plt.Line2D([0], [0], color=COLORS[name], lw=5, label=name) for name in ROI_NAMES]
    axis.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.manifest_csv, dtype=str, keep_default_na=False)
    eligible = manifest.loc[manifest["eligible"].str.casefold().eq("true")].copy()
    output_dir = Path(args.output_dir)
    overlay_dir = output_dir / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    overlay_ids = set(rng.choice(eligible["case_id"].to_numpy(), size=min(args.overlay_count, len(eligible)), replace=False).tolist())
    rows = []
    invariant_failures = []
    data_root = Path(args.data_root).expanduser().resolve()
    for _, record in eligible.iterrows():
        seg = _load(resolve_data_path(data_root, record["segmentation_path"]))
        volumes = {
            modality: _load(resolve_data_path(data_root, record[f"{modality}_path"]))
            for modality in ["t2", "t1ce", "t1", "flair"]
        }
        brain_mask = estimate_brain_mask(volumes)
        masks, valid = build_acp_masks(
            seg,
            brain_mask,
            q_core=args.q_core,
            r1=args.peri_inner_radius,
            r2=args.peri_outer_radius,
        )
        tumor = seg > 0
        checks = {
            "core_inside_tumor": not np.any(np.logical_and(masks["core"], ~tumor)),
            "boundary_inside_tumor": not np.any(np.logical_and(masks["boundary"], ~tumor)),
            "core_boundary_partition_tumor": np.array_equal(np.logical_or(masks["core"], masks["boundary"]), tumor),
            "peri_inner_outside_tumor": not np.any(np.logical_and(masks["peri_inner"], tumor)),
            "peri_outer_outside_inner_and_tumor": not np.any(np.logical_and(masks["peri_outer"], np.logical_or(tumor, masks["peri_inner"]))),
            "distal_disjoint_lesion_context": not np.any(
                np.logical_and(
                    masks["distal_normal"],
                    np.logical_or.reduce([tumor, masks["peri_inner"], masks["peri_outer"]]),
                )
            ),
        }
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            invariant_failures.append({"case_id": record["case_id"], "failed_checks": failed})
        row = {"case_id": record["case_id"], "label": int(record["normalized_idh_label"]), "all_invariants_pass": not failed}
        for index, name in enumerate(ROI_NAMES):
            row[f"{name}_voxels"] = int(masks[name].sum())
            row[f"{name}_valid"] = float(valid[index])
        rows.append(row)
        if record["case_id"] in overlay_ids:
            save_overlay(record["case_id"], volumes["flair"], seg, masks, overlay_dir / f"{record['case_id']}_roi_overlay.png")

    per_case = pd.DataFrame(rows)
    per_case.to_csv(output_dir / "roi_qc_per_case.csv", index=False)
    summary = {
        "eligible_cases": int(len(per_case)),
        "q_core": args.q_core,
        "peri_inner_radius_voxels_at_1mm": args.peri_inner_radius,
        "peri_outer_radius_voxels_at_1mm": args.peri_outer_radius,
        "invariant_failure_cases": int(len(invariant_failures)),
        "invariant_failures": invariant_failures,
        "overlay_cases": sorted(overlay_ids),
        "nodes": {},
    }
    for name in ROI_NAMES:
        values = per_case[f"{name}_voxels"].astype(float)
        valid = per_case[f"{name}_valid"].astype(float)
        summary["nodes"][name] = {
            "mean_volume_mm3": float(values.mean()),
            "median_volume_mm3": float(values.median()),
            "min_volume_mm3": float(values.min()),
            "max_volume_mm3": float(values.max()),
            "empty_rate": float(values.eq(0).mean()),
            "roi_valid_rate": float(valid.mean()),
        }
    with open(output_dir / "roi_qc_summary.json", "w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if invariant_failures:
        raise RuntimeError(f"ROI invariants failed for {len(invariant_failures)} UTSW cases.")


if __name__ == "__main__":
    main()
