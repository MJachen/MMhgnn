from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.data._utils.collate import default_collate

from datasets.brats_dataset import BraTSClassificationDataset, create_data_splits, scan_cases
from utils.io import save_json


def build_datasets(config: Dict, explicit_eval_combo: Sequence[str] | None = None):
    data_cfg = config["data"]
    train_cfg = config["train"]
    model_cfg = config["model"]
    records = scan_cases(
        data_root=data_cfg["data_root"],
        label_xlsx=data_cfg["label_xlsx"],
        case_id_col=data_cfg["case_id_col"],
        label_col=data_cfg["label_col"],
        modalities=data_cfg.get("modalities"),
        image_filename_pattern=data_cfg.get("image_filename_pattern", "{case_id}_{mod}.nii.gz"),
        seg_filename_pattern=data_cfg.get("seg_filename_pattern", "{case_id}_seg.nii.gz"),
        mask_filename_pattern=data_cfg.get("mask_filename_pattern"),
        tumor_mask_source=data_cfg.get("tumor_mask_source", "seg"),
        case_id_col_index=data_cfg.get("case_id_col_index"),
        label_col_index=data_cfg.get("label_col_index"),
    )
    split_json = data_cfg.get("split_json")
    if split_json is None:
        split_json = str(Path(config["output_dir"]) / "splits.json")
    splits = create_data_splits(records, tuple(data_cfg["split_ratio"]), config["seed"], split_json=split_json)

    dataset_kwargs = dict(
        target_shape=data_cfg.get("target_shape"),
        crop_mode=data_cfg.get("crop_mode", "wt_bbox"),
        bbox_margin=int(data_cfg.get("bbox_margin", 6)),
        fixed_combo=train_cfg.get("fixed_combo", data_cfg["modalities"]),
        all_modalities=data_cfg["modalities"],
        random_seed=config["seed"],
        q_core=float(model_cfg.get("q_core", 0.4)),
        peri_inner_radius=int(model_cfg.get("peri_inner_radius", 3)),
        peri_outer_radius=int(model_cfg.get("peri_outer_radius", 7)),
        allow_full_resolution_input=bool(model_cfg.get("allow_full_resolution_input", False)),
        tumor_mask_source=data_cfg.get("tumor_mask_source", "seg"),
    )
    train_ds = BraTSClassificationDataset(splits["train"], combo_mode=train_cfg["mode"], **dataset_kwargs)
    val_ds = BraTSClassificationDataset(
        splits["val"],
        combo_mode="fixed_combo",
        explicit_combo=explicit_eval_combo or train_cfg.get("fixed_combo", data_cfg["modalities"]),
        **dataset_kwargs,
    )
    test_ds = BraTSClassificationDataset(
        splits["test"],
        combo_mode="fixed_combo",
        explicit_combo=explicit_eval_combo or train_cfg.get("fixed_combo", data_cfg["modalities"]),
        **dataset_kwargs,
    )
    return train_ds, val_ds, test_ds, splits


def build_sampler(dataset) -> WeightedRandomSampler:
    labels = [record.label for record in dataset.records]
    class_counts = np.bincount(labels, minlength=2)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = [class_weights[label] for label in labels]
    return WeightedRandomSampler(weights=torch.as_tensor(sample_weights, dtype=torch.double), num_samples=len(sample_weights), replacement=True)


def _pad_tensor_list(values):
    ndim = values[0].dim()
    max_shape = [max(v.shape[i] for v in values) for i in range(ndim)]
    padded = []
    for value in values:
        pad = []
        for i in reversed(range(ndim)):
            pad.extend([0, max_shape[i] - value.shape[i]])
        padded.append(F.pad(value, pad))
    return torch.stack(padded, dim=0)


def brats_collate_fn(batch):
    collated = {}
    for key in batch[0].keys():
        values = [item[key] for item in batch]
        if key in {"case_id", "combo"}:
            collated[key] = values
        elif torch.is_tensor(values[0]) and values[0].dim() >= 3:
            collated[key] = _pad_tensor_list(values)
        else:
            collated[key] = default_collate(values)
    return collated


def build_dataloader(dataset, batch_size: int, num_workers: int = 0, shuffle: bool = False, sampler=None):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=brats_collate_fn,
    )


def class_weights_from_records(records) -> torch.Tensor:
    labels = [record.label for record in records]
    counts = np.bincount(labels, minlength=2)
    weights = len(labels) / np.maximum(counts, 1) / 2.0
    return torch.tensor(weights, dtype=torch.float32)


def dump_split_summary(splits, output_dir: str):
    summary = {k: {"num_cases": len(v), "labels": [r.label for r in v], "case_ids": [r.case_id for r in v]} for k, v in splits.items()}
    save_json(summary, Path(output_dir) / "split_summary.json")
