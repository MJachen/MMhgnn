from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import load_config


def parse_args():
    parser = argparse.ArgumentParser(description="Check dataset paths and file patterns from a YAML config.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--show", type=int, default=5, help="Number of example cases to print.")
    parser.add_argument(
        "--require-split",
        action="store_true",
        help="Fail unless data.split_json exists and is disjoint, duplicate-free, and fully represented in the dataset.",
    )
    return parser.parse_args()


def resolve_column(df: pd.DataFrame, column_name: str | None, column_index: int | None, field_name: str) -> str:
    if column_name and column_name in df.columns:
        return column_name
    if column_index is not None:
        column_index = int(column_index)
        if column_index < 0 or column_index >= len(df.columns):
            raise ValueError(f"{field_name}_col_index={column_index} is out of range for columns {df.columns.tolist()}")
        return str(df.columns[column_index])
    raise ValueError(f"Label file must contain column '{column_name}' or provide {field_name}_col_index.")


def scan_configured_cases(data_cfg):
    data_root = Path(data_cfg["data_root"])
    label_xlsx = Path(data_cfg["label_xlsx"])
    modalities = list(data_cfg.get("modalities") or ["t2", "t1ce", "t1", "flair"])
    image_pattern = data_cfg.get("image_filename_pattern", "{case_id}_{mod}.nii.gz")
    seg_pattern = data_cfg.get("seg_filename_pattern", "{case_id}_seg.nii.gz")
    mask_pattern = data_cfg.get("mask_filename_pattern")
    tumor_mask_source = data_cfg.get("tumor_mask_source", "seg")

    if not data_root.exists():
        raise FileNotFoundError(f"data_root does not exist: {data_root}")
    if not label_xlsx.exists():
        raise FileNotFoundError(f"label_xlsx does not exist: {label_xlsx}")

    df = pd.read_excel(label_xlsx)
    case_id_col = resolve_column(df, data_cfg.get("case_id_col"), data_cfg.get("case_id_col_index"), "case_id")
    label_col = resolve_column(df, data_cfg.get("label_col"), data_cfg.get("label_col_index"), "label")
    label_map = {str(row[case_id_col]): int(row[label_col]) for _, row in df.iterrows()}

    valid = []
    skipped = []
    for case_dir in sorted([path for path in data_root.iterdir() if path.is_dir()]):
        case_id = case_dir.name
        if case_id not in label_map:
            skipped.append({"case_id": case_id, "reason": "case_id_not_in_label_file"})
            continue

        files = {}
        missing_files = []
        for mod in modalities:
            path = case_dir / image_pattern.format(case_id=case_id, mod=mod)
            files[mod] = path
            if not path.exists():
                missing_files.append(str(path))

        if tumor_mask_source == "seg":
            if not seg_pattern:
                raise ValueError("tumor_mask_source='seg' requires seg_filename_pattern.")
            seg_path = case_dir / seg_pattern.format(case_id=case_id, mod="seg")
            files["seg"] = seg_path
            if not seg_path.exists():
                missing_files.append(str(seg_path))
        elif tumor_mask_source == "union_modality_masks":
            if not mask_pattern:
                raise ValueError("tumor_mask_source='union_modality_masks' requires mask_filename_pattern.")
            for mod in modalities:
                mask_path = case_dir / mask_pattern.format(case_id=case_id, mod=mod)
                files[f"{mod}_mask"] = mask_path
                if not mask_path.exists():
                    missing_files.append(str(mask_path))
        else:
            raise ValueError(f"Unsupported tumor_mask_source: {tumor_mask_source}")

        if missing_files:
            skipped.append({"case_id": case_id, "reason": "missing_files", "missing_files": missing_files[:10]})
            continue

        valid.append({"case_id": case_id, "case_dir": case_dir, "label": label_map[case_id], "files": files})

    return valid, skipped


def validate_split_file(data_cfg, records, require_split: bool = False):
    split_value = data_cfg.get("split_json")
    if not split_value:
        if require_split:
            raise ValueError("data.split_json is required for this experiment but is not configured.")
        return {"configured": False, "exists": False}

    split_path = Path(split_value)
    if not split_path.is_absolute():
        split_path = PROJECT_ROOT / split_path
    split_path = split_path.resolve()
    if not split_path.exists():
        if require_split:
            raise FileNotFoundError(f"Required split file does not exist: {split_path}")
        return {"configured": True, "exists": False, "path": str(split_path)}

    with split_path.open("r", encoding="utf-8") as handle:
        split_data = json.load(handle)
    required_keys = {"train", "val", "test"}
    if set(split_data) != required_keys:
        raise ValueError(f"Split file must contain exactly {sorted(required_keys)}, got {sorted(split_data)}")

    record_ids = {record["case_id"] for record in records}
    duplicate_ids = {
        split_name: sorted(case_id for case_id, count in Counter(case_ids).items() if count > 1)
        for split_name, case_ids in split_data.items()
    }
    overlaps = {
        "train_val": sorted(set(split_data["train"]) & set(split_data["val"])),
        "train_test": sorted(set(split_data["train"]) & set(split_data["test"])),
        "val_test": sorted(set(split_data["val"]) & set(split_data["test"])),
    }
    missing_ids = {
        split_name: sorted(set(case_ids) - record_ids)
        for split_name, case_ids in split_data.items()
    }
    if any(duplicate_ids.values()) or any(overlaps.values()) or any(missing_ids.values()):
        raise ValueError(
            "Invalid fixed split: "
            f"duplicates={duplicate_ids}, overlaps={overlaps}, missing_ids={missing_ids}"
        )

    return {
        "configured": True,
        "exists": True,
        "path": str(split_path),
        "counts": {split_name: len(case_ids) for split_name, case_ids in split_data.items()},
        "total_unique_cases": len(set().union(*(set(case_ids) for case_ids in split_data.values()))),
        "duplicates": duplicate_ids,
        "overlaps": overlaps,
        "missing_ids": missing_ids,
    }


def main():
    args = parse_args()
    config = load_config(args.config)
    data_cfg = config["data"]
    records, skipped = scan_configured_cases(data_cfg)
    if not records:
        raise RuntimeError("No valid cases found. Check data_root, label_xlsx, file patterns, and tumor mask source.")

    label_counts = Counter(record["label"] for record in records)
    split_check = validate_split_file(data_cfg, records, require_split=args.require_split)
    payload = {
        "config": str(Path(args.config)),
        "data_root": data_cfg["data_root"],
        "label_xlsx": data_cfg["label_xlsx"],
        "dataset_format": data_cfg.get("dataset_format", ""),
        "modalities": data_cfg.get("modalities"),
        "num_valid_cases": len(records),
        "num_skipped_cases": len(skipped),
        "label_counts": dict(sorted(label_counts.items())),
        "split_check": split_check,
        "examples": [
            {
                "case_id": record["case_id"],
                "case_dir": str(record["case_dir"]),
                "label": record["label"],
                "files": {key: str(value) for key, value in record["files"].items()},
            }
            for record in records[: max(args.show, 0)]
        ],
        "skipped_examples": skipped[: max(args.show, 0)],
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
