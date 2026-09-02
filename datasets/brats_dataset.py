from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import binary_dilation, distance_transform_edt, zoom
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from utils.io import load_json, save_json


ALL_MODALITIES = ["t2", "t1ce", "t1", "flair"]
ROI_NAMES = ["core", "boundary", "peri_inner", "peri_outer", "distal_normal"]


@dataclass
class CaseRecord:
    case_id: str
    case_dir: Path
    label: int
    files: Dict[str, Path]


def get_all_modality_combinations(modalities: Sequence[str] | None = None) -> List[Tuple[str, ...]]:
    modalities = list(modalities or ALL_MODALITIES)
    combos: List[Tuple[str, ...]] = []
    for r in range(1, len(modalities) + 1):
        combos.extend(itertools.combinations(modalities, r))
    return combos


def _resolve_label_column(df: pd.DataFrame, column_name: str | None, column_index: int | None, field_name: str) -> str:
    if column_name and column_name in df.columns:
        return column_name
    if column_index is not None:
        column_index = int(column_index)
        if column_index < 0 or column_index >= len(df.columns):
            raise ValueError(f"{field_name}_col_index={column_index} is out of range for columns {df.columns.tolist()}")
        return str(df.columns[column_index])
    raise ValueError(f"xlsx must contain column '{column_name}' or provide {field_name}_col_index.")


def scan_cases(
    data_root: str,
    label_xlsx: str,
    case_id_col: str | None,
    label_col: str | None,
    modalities: Sequence[str] | None = None,
    image_filename_pattern: str = "{case_id}_{mod}.nii.gz",
    seg_filename_pattern: str | None = "{case_id}_seg.nii.gz",
    mask_filename_pattern: str | None = None,
    tumor_mask_source: str = "seg",
    case_id_col_index: int | None = None,
    label_col_index: int | None = None,
) -> List[CaseRecord]:
    data_root = Path(data_root)
    df = pd.read_excel(label_xlsx)
    modalities = list(modalities or ALL_MODALITIES)
    case_id_col = _resolve_label_column(df, case_id_col, case_id_col_index, "case_id")
    label_col = _resolve_label_column(df, label_col, label_col_index, "label")

    label_map = {str(row[case_id_col]): int(row[label_col]) for _, row in df.iterrows()}
    records: List[CaseRecord] = []

    for case_dir in sorted([p for p in data_root.iterdir() if p.is_dir()]):
        case_id = case_dir.name
        if case_id not in label_map:
            continue
        files = {}
        missing = False
        for mod in modalities:
            path = case_dir / image_filename_pattern.format(case_id=case_id, mod=mod)
            files[mod] = path
            if not path.exists():
                missing = True
        if tumor_mask_source == "seg":
            if not seg_filename_pattern:
                raise ValueError("tumor_mask_source='seg' requires seg_filename_pattern.")
            seg_path = case_dir / seg_filename_pattern.format(case_id=case_id, mod="seg")
            files["seg"] = seg_path
            if not seg_path.exists():
                missing = True
        elif tumor_mask_source == "union_modality_masks":
            if not mask_filename_pattern:
                raise ValueError("tumor_mask_source='union_modality_masks' requires mask_filename_pattern.")
            for mod in modalities:
                mask_path = case_dir / mask_filename_pattern.format(case_id=case_id, mod=mod)
                files[f"{mod}_mask"] = mask_path
                if not mask_path.exists():
                    missing = True
        else:
            raise ValueError(f"Unsupported tumor_mask_source: {tumor_mask_source}")
        if missing:
            continue
        records.append(CaseRecord(case_id=case_id, case_dir=case_dir, label=label_map[case_id], files=files))
    if not records:
        raise RuntimeError("No valid cases found. Please check data_root, label_xlsx, file patterns, and tumor mask source.")
    return records


def create_data_splits(records: List[CaseRecord], split_ratio=(0.8, 0.1, 0.1), seed: int = 42, split_json: str | None = None):
    if split_json and Path(split_json).exists():
        split_data = load_json(split_json)
        record_map = {r.case_id: r for r in records}
        return {k: [record_map[cid] for cid in v if cid in record_map] for k, v in split_data.items()}

    train_ratio, val_ratio, test_ratio = split_ratio
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("split_ratio must sum to 1.0")

    case_ids = [r.case_id for r in records]
    labels = [r.label for r in records]
    train_ids, temp_ids, train_y, temp_y = train_test_split(
        case_ids,
        labels,
        test_size=(1.0 - train_ratio),
        random_state=seed,
        stratify=labels,
    )
    val_size = val_ratio / (val_ratio + test_ratio)
    val_ids, test_ids = train_test_split(temp_ids, test_size=(1.0 - val_size), random_state=seed, stratify=temp_y)
    record_map = {r.case_id: r for r in records}
    split = {
        "train": [record_map[cid] for cid in train_ids],
        "val": [record_map[cid] for cid in val_ids],
        "test": [record_map[cid] for cid in test_ids],
    }
    if split_json:
        save_json({k: [r.case_id for r in v] for k, v in split.items()}, split_json)
    return split


def _load_nifti(path: Path) -> np.ndarray:
    return np.asarray(nib.load(str(path)).get_fdata(), dtype=np.float32)


def zscore_normalize(volume: np.ndarray) -> np.ndarray:
    mask = volume != 0
    if mask.sum() == 0:
        return np.zeros_like(volume, dtype=np.float32)
    values = volume[mask]
    std = values.std()
    if std < 1e-6:
        std = 1.0
    out = np.zeros_like(volume, dtype=np.float32)
    out[mask] = (values - values.mean()) / std
    return out


def crop_or_pad_to_shape(volume: np.ndarray, target_shape: Sequence[int]) -> np.ndarray:
    target_shape = np.asarray(target_shape, dtype=int)
    result = volume
    for axis in range(3):
        size = result.shape[axis]
        target = target_shape[axis]
        if size > target:
            start = (size - target) // 2
            end = start + target
            slicer = [slice(None)] * 3
            slicer[axis] = slice(start, end)
            result = result[tuple(slicer)]
        elif size < target:
            before = (target - size) // 2
            after = target - size - before
            pads = [(0, 0), (0, 0), (0, 0)]
            pads[axis] = (before, after)
            result = np.pad(result, pads, mode="constant")
    return result


def resize_to_shape(volume: np.ndarray, target_shape: Sequence[int], order: int) -> np.ndarray:
    zoom_factors = [t / s for t, s in zip(target_shape, volume.shape)]
    return zoom(volume, zoom=zoom_factors, order=order)


def align_volume_to_shape(volume: np.ndarray, target_shape: Sequence[int], order: int) -> np.ndarray:
    if tuple(volume.shape) == tuple(target_shape):
        return volume.astype(np.float32)
    return resize_to_shape(volume, target_shape, order=order).astype(np.float32)


def compute_bbox(mask: np.ndarray, margin: int = 6):
    coords = np.argwhere(mask > 0)
    if len(coords) == 0:
        return tuple(slice(0, s) for s in mask.shape)
    mins = np.maximum(coords.min(axis=0) - margin, 0)
    maxs = np.minimum(coords.max(axis=0) + margin + 1, mask.shape)
    return tuple(slice(int(lo), int(hi)) for lo, hi in zip(mins, maxs))


def estimate_brain_mask(modality_volumes: Dict[str, np.ndarray]) -> np.ndarray:
    masks = [(volume != 0) for volume in modality_volumes.values()]
    if not masks:
        raise ValueError("No modality volumes provided to estimate brain mask.")
    brain_mask = np.logical_or.reduce(masks)
    return brain_mask.astype(bool)


def build_acp_masks(seg: np.ndarray, brain_mask: np.ndarray, q_core: float = 0.4, r1: int = 3, r2: int = 7):
    tumor = seg > 0
    if tumor.sum() == 0:
        zero = np.zeros_like(tumor, dtype=bool)
        masks = {name: zero.copy() for name in ROI_NAMES}
        valid = np.zeros(len(ROI_NAMES), dtype=np.float32)
        return masks, valid

    internal_dist = distance_transform_edt(tumor)
    tumor_coords = np.argwhere(tumor)
    tumor_values = internal_dist[tumor]
    n_core = max(1, int(np.ceil(len(tumor_values) * q_core)))
    topk_idx = np.argpartition(tumor_values, -n_core)[-n_core:]
    core = np.zeros_like(tumor, dtype=bool)
    core_coords = tumor_coords[topk_idx]
    core[tuple(core_coords.T)] = True
    if core.sum() == 0:
        center_idx = np.argmax(tumor_values)
        center_coord = tumor_coords[center_idx]
        core[tuple(center_coord)] = True

    boundary = np.logical_and(tumor, np.logical_not(core))
    if boundary.sum() == 0 and tumor.sum() > 1:
        # keep at least one voxel for boundary when possible
        farthest_idx = np.argmin(tumor_values)
        boundary_coord = tumor_coords[farthest_idx]
        boundary[tuple(boundary_coord)] = True
        core[tuple(boundary_coord)] = False

    dilated_r1 = binary_dilation(tumor, iterations=r1)
    dilated_r2 = binary_dilation(tumor, iterations=r2)
    peri_inner = np.logical_and(dilated_r1, np.logical_not(tumor))
    peri_outer = np.logical_and(dilated_r2, np.logical_not(dilated_r1))
    distal_normal = np.logical_and(brain_mask, np.logical_not(dilated_r2))

    masks = {
        "core": core,
        "boundary": boundary,
        "peri_inner": peri_inner,
        "peri_outer": peri_outer,
        "distal_normal": distal_normal,
    }
    valid = np.asarray([float(masks[name].sum() > 0) for name in ROI_NAMES], dtype=np.float32)
    return masks, valid


class BraTSClassificationDataset(Dataset):
    def __init__(
        self,
        records: List[CaseRecord],
        target_shape: Sequence[int] | None,
        crop_mode: str = "wt_bbox",
        bbox_margin: int = 6,
        combo_mode: str = "fixed_combo",
        fixed_combo: Sequence[str] | None = None,
        all_modalities: Sequence[str] | None = None,
        random_seed: int = 42,
        explicit_combo: Sequence[str] | None = None,
        q_core: float = 0.4,
        peri_inner_radius: int = 3,
        peri_outer_radius: int = 7,
        allow_full_resolution_input: bool = False,
        tumor_mask_source: str = "seg",
    ):
        self.records = records
        self.target_shape = tuple(target_shape) if target_shape is not None else None
        self.crop_mode = crop_mode
        self.bbox_margin = bbox_margin
        self.combo_mode = combo_mode
        self.fixed_combo = tuple(fixed_combo) if fixed_combo else tuple(ALL_MODALITIES)
        self.all_modalities = list(all_modalities or ALL_MODALITIES)
        self.random_seed = random_seed
        self.random = random.Random(random_seed)
        self.current_epoch = 1
        self.total_epochs = 1
        self.curriculum_config = {}
        self.explicit_combo = tuple(explicit_combo) if explicit_combo else None
        self.all_combos = get_all_modality_combinations(self.all_modalities)
        self.q_core = q_core
        self.peri_inner_radius = peri_inner_radius
        self.peri_outer_radius = peri_outer_radius
        self.allow_full_resolution_input = allow_full_resolution_input
        self.tumor_mask_source = tumor_mask_source

    def __len__(self) -> int:
        return len(self.records)

    def set_epoch(self, epoch: int, total_epochs: int | None = None, curriculum_config: Dict | None = None) -> None:
        """Update epoch-aware modality sampling for curriculum training."""
        self.current_epoch = max(int(epoch), 1)
        if total_epochs is not None:
            self.total_epochs = max(int(total_epochs), 1)
        if curriculum_config is not None:
            self.curriculum_config = dict(curriculum_config)
        # Make curriculum sampling reproducible while still changing by epoch.
        self.random.seed(self.random_seed + self.current_epoch * 9973)

    def _combos_by_visible_count(self, visible_count: int) -> List[Tuple[str, ...]]:
        return [tuple(c) for c in itertools.combinations(self.all_modalities, visible_count)]

    def _sample_curriculum_combo(self) -> Tuple[str, ...]:
        cfg = self.curriculum_config
        progress = (self.current_epoch - 1) / max(self.total_epochs, 1)
        stage1_end = float(cfg.get("curriculum_stage1_ratio", 0.3))
        stage2_end = stage1_end + float(cfg.get("curriculum_stage2_ratio", 0.4))
        full_combo = tuple(self.all_modalities)

        if progress < stage1_end:
            return full_combo

        roll = self.random.random()
        if progress < stage2_end:
            full_ratio = float(cfg.get("stage2_full_ratio", 0.7))
            if roll < full_ratio:
                return full_combo
            return tuple(self.random.choice(self._combos_by_visible_count(len(self.all_modalities) - 1)))

        buckets = [
            ("full", float(cfg.get("stage3_full_ratio", 0.4))),
            ("single", float(cfg.get("stage3_single_missing_ratio", 0.3))),
            ("double", float(cfg.get("stage3_double_missing_ratio", 0.2))),
            ("triple", float(cfg.get("stage3_triple_missing_ratio", 0.1))),
        ]
        total = max(sum(weight for _, weight in buckets), 1e-8)
        cumulative = 0.0
        for name, weight in buckets:
            cumulative += weight / total
            if roll <= cumulative:
                if name == "full":
                    return full_combo
                if name == "single":
                    return tuple(self.random.choice(self._combos_by_visible_count(3)))
                if name == "double":
                    return tuple(self.random.choice(self._combos_by_visible_count(2)))
                return tuple(self.random.choice(self._combos_by_visible_count(1)))
        return full_combo


    def _sample_targeted_no_t1ce_combo(self) -> Tuple[str, ...]:
        cfg = self.curriculum_config
        candidates = [
            (tuple(self.all_modalities), float(cfg.get("fine_tune_ratio_full", 0.7))),
            (("t2",), float(cfg.get("fine_tune_ratio_t2_only", 0.1))),
            (("flair",), float(cfg.get("fine_tune_ratio_flair_only", 0.1))),
            (("t2", "flair"), float(cfg.get("fine_tune_ratio_t2_flair", 0.1))),
            (("t2", "t1", "flair"), float(cfg.get("fine_tune_ratio_drop_t1ce_fullcontext", 0.0))),
        ]
        valid_candidates = [(tuple(m for m in combo if m in self.all_modalities), max(weight, 0.0)) for combo, weight in candidates]
        total = sum(weight for combo, weight in valid_candidates if combo)
        if total <= 0:
            return tuple(self.all_modalities)
        roll = self.random.random()
        cumulative = 0.0
        for combo, weight in valid_candidates:
            if not combo:
                continue
            cumulative += weight / total
            if roll <= cumulative:
                return combo
        return tuple(self.all_modalities)

    def targeted_missing_groups(self) -> Dict[str, List[Tuple[str, ...]]]:
        """Return the four legal E2 sampling groups drawn from the existing non-empty combinations."""
        full_combo = tuple(self.all_modalities)
        groups = {
            "full": [full_combo],
            "t1ce_absent_t2_present": [],
            "t2_absent_t1ce_present": [],
            "t1ce_t2_both_absent": [],
        }
        for combo in self.all_combos:
            combo_set = set(combo)
            if combo == full_combo:
                continue
            if "t1ce" not in combo_set and "t2" in combo_set:
                groups["t1ce_absent_t2_present"].append(tuple(combo))
            elif "t2" not in combo_set and "t1ce" in combo_set:
                groups["t2_absent_t1ce_present"].append(tuple(combo))
            elif "t1ce" not in combo_set and "t2" not in combo_set:
                groups["t1ce_t2_both_absent"].append(tuple(combo))
        return groups

    def _targeted_subgroup_for_combo(self, combo: Sequence[str]) -> str:
        combo = tuple(combo)
        for group_name, candidates in self.targeted_missing_groups().items():
            if combo in candidates:
                return group_name
        raise ValueError(f"Combo {combo} does not belong to an E2 targeted sampling subgroup.")

    def _sample_targeted_missing_combo(self) -> Tuple[str, Tuple[str, ...]]:
        groups = self.targeted_missing_groups()
        cfg = self.curriculum_config
        weighted_groups = [
            ("full", float(cfg.get("targeted_ratio_full", 0.30))),
            ("t1ce_absent_t2_present", float(cfg.get("targeted_ratio_t1ce_absent_t2_present", 0.25))),
            ("t2_absent_t1ce_present", float(cfg.get("targeted_ratio_t2_absent_t1ce_present", 0.25))),
            ("t1ce_t2_both_absent", float(cfg.get("targeted_ratio_t1ce_t2_both_absent", 0.20))),
        ]
        if any(weight < 0 for _, weight in weighted_groups):
            raise ValueError("E2 targeted sampling ratios must be non-negative.")
        if any(weight > 0 and not groups[name] for name, weight in weighted_groups):
            raise ValueError("E2 targeted sampling requested a subgroup with no legal non-empty modality patterns.")
        total = sum(weight for _, weight in weighted_groups)
        if total <= 0:
            raise ValueError("At least one E2 targeted sampling ratio must be positive.")

        roll = self.random.random()
        cumulative = 0.0
        selected_group = weighted_groups[-1][0]
        for group_name, weight in weighted_groups:
            cumulative += weight / total
            if roll <= cumulative:
                selected_group = group_name
                break
        return selected_group, tuple(self.random.choice(groups[selected_group]))

    def _choose_combo(self) -> Tuple[str, ...]:
        if self.explicit_combo is not None:
            return self.explicit_combo
        if self.combo_mode in {"full_modality_train", "fixed_combo"}:
            return self.fixed_combo
        if self.combo_mode == "missing_curriculum_train":
            return self._sample_curriculum_combo()
        if self.combo_mode == "targeted_no_t1ce":
            return self._sample_targeted_no_t1ce_combo()
        if self.combo_mode == "targeted_dual_view_train":
            return self._sample_targeted_missing_combo()[1]
        if self.combo_mode == "random_missing":
            return tuple(self.random.choice(self.all_combos))
        return self.fixed_combo

    def _training_view_spec(self) -> Tuple[Tuple[str, ...], Tuple[str, ...] | None, str | None]:
        sampled_combo = self._choose_combo()
        if self.combo_mode != "targeted_dual_view_train":
            return sampled_combo, None, None
        return tuple(self.all_modalities), sampled_combo, self._targeted_subgroup_for_combo(sampled_combo)

    def _maybe_resize(self, volume: np.ndarray, order: int) -> np.ndarray:
        if self.target_shape is None:
            return volume.astype(np.float32)
        volume = resize_to_shape(volume, self.target_shape, order=order)
        volume = crop_or_pad_to_shape(volume, self.target_shape)
        return volume.astype(np.float32)

    def __getitem__(self, index: int):
        record = self.records[index]
        combo, missing_combo, targeted_subgroup = self._training_view_spec()

        modality_full: Dict[str, np.ndarray] = {}
        for modality in self.all_modalities:
            path = record.files.get(modality)
            if path is None or not path.exists():
                modality_full[modality] = None
            else:
                modality_full[modality] = _load_nifti(path)
        reference_volume = next((volume for volume in modality_full.values() if volume is not None), None)
        if reference_volume is None:
            raise RuntimeError(f"Case {record.case_id} does not contain any readable modality volume.")
        reference_shape = reference_volume.shape
        for modality in self.all_modalities:
            if modality_full[modality] is None:
                modality_full[modality] = np.zeros_like(reference_volume, dtype=np.float32)
            else:
                modality_full[modality] = align_volume_to_shape(modality_full[modality], reference_shape, order=1)

        if self.tumor_mask_source == "seg":
            seg_full = align_volume_to_shape(_load_nifti(record.files["seg"]), reference_shape, order=0)
        elif self.tumor_mask_source == "union_modality_masks":
            mask_volumes = []
            for modality in self.all_modalities:
                mask_path = record.files.get(f"{modality}_mask")
                if mask_path is None or not mask_path.exists():
                    mask_volumes.append(np.zeros_like(reference_volume, dtype=bool))
                else:
                    aligned_mask = align_volume_to_shape(_load_nifti(mask_path), reference_shape, order=0) > 0
                    mask_volumes.append(aligned_mask)
            seg_full = np.logical_or.reduce(mask_volumes).astype(np.float32)
        else:
            raise ValueError(f"Unsupported tumor_mask_source: {self.tumor_mask_source}")

        brain_mask = estimate_brain_mask(modality_full)
        tumor_mask = seg_full > 0
        use_full = self.allow_full_resolution_input and self.target_shape is None
        if use_full:
            bbox = tuple(slice(0, s) for s in seg_full.shape)
        else:
            crop_mask = tumor_mask if tumor_mask.sum() > 0 else brain_mask
            margin = max(self.bbox_margin, self.peri_outer_radius + 1)
            bbox = compute_bbox(crop_mask, margin=margin)

        seg = seg_full[bbox]
        brain_mask = brain_mask[bbox]
        modality_crop = {m: modality_full[m][bbox] for m in self.all_modalities}
        acp_masks, roi_valid = build_acp_masks(seg, brain_mask, q_core=self.q_core, r1=self.peri_inner_radius, r2=self.peri_outer_radius)

        image_dict: Dict[str, np.ndarray] = {}
        for modality in self.all_modalities:
            volume = zscore_normalize(modality_crop[modality])
            image_dict[modality] = self._maybe_resize(volume, order=1)

        roi_resized = {}
        for roi_name, roi_mask in acp_masks.items():
            roi_resized[roi_name] = self._maybe_resize(roi_mask.astype(np.float32), order=0)

        seg_resized = self._maybe_resize(seg.astype(np.float32), order=0)
        available_mask = np.asarray([1.0 if m in combo else 0.0 for m in self.all_modalities], dtype=np.float32)
        image_shape = next(iter(image_dict.values())).shape
        images = np.stack([image_dict[m] if m in combo else np.zeros(image_shape, dtype=np.float32) for m in self.all_modalities], axis=0)
        roi_stack = np.stack([roi_resized[name] for name in ROI_NAMES], axis=0)

        sample = {
            "case_id": record.case_id,
            "images": torch.from_numpy(images),
            "label": torch.tensor(record.label, dtype=torch.long),
            "seg": torch.from_numpy(seg_resized),
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
