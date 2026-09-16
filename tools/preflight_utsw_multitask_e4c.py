from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import load_config
from utils.training import build_datasets


EXPECTED = {
    "grade_2_vs_34": {"train": (492, 83, 409), "val": (61, 10, 51), "test": (59, 7, 52)},
    "mgmt": {"train": (216, 123, 93), "val": (30, 13, 17), "test": (31, 27, 4)},
}


def parse_args():
    parser = argparse.ArgumentParser(description="Read-only E4C task preflight.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--reference-config", default="configs/utsw_idh/missing_aware_aux_affine_e4a.yaml")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--manifest-csv", required=True)
    parser.add_argument("--split-json", required=True)
    parser.add_argument("--manifest-fingerprint-json", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    overrides = {
        "data": {
            "root": args.data_root,
            "manifest_csv": args.manifest_csv,
            "split_json": args.split_json,
            "manifest_fingerprint_json": args.manifest_fingerprint_json,
        }
    }
    config = load_config(args.config, overrides=overrides)
    reference = load_config(args.reference_config)
    protected_sections = [
        "model", "graph", "fusion", "mask_affine", "auxiliary", "train",
        "loss", "missing_aware_validation", "calibration",
    ]
    unequal = [name for name in protected_sections if config.get(name) != reference.get(name)]
    if unequal:
        raise RuntimeError(f"Task config changed protected E4A sections: {unequal}")
    if config["data"].get("label_column") != "normalized_task_label":
        raise RuntimeError("Task config must use normalized_task_label from utils/utsw_tasks.py audit artifacts.")
    train_ds, val_ds, test_ds, splits = build_datasets(config)
    task = config["task"]["name"]
    observed = {}
    for name, records in splits.items():
        values = (len(records), sum(item.label == 0 for item in records), sum(item.label == 1 for item in records))
        observed[name] = {"cases": values[0], "class0": values[1], "class1": values[2]}
        if values != EXPECTED[task][name]:
            raise RuntimeError(f"{task} {name} distribution mismatch: expected={EXPECTED[task][name]} observed={values}")
    patient_sets = {name: {item.patient_id for item in records} for name, records in splits.items()}
    leakage = any(patient_sets[a] & patient_sets[b] for a, b in [("train", "val"), ("train", "test"), ("val", "test")])
    if leakage:
        raise RuntimeError("Patient leakage detected during task preflight.")
    print(json.dumps({
        "status": "ok",
        "task": task,
        "protected_e4a_sections_equal": True,
        "split_distribution": observed,
        "patient_leakage": False,
        "dataset_lengths": {"train": len(train_ds), "val": len(val_ds), "test": len(test_ds)},
        "test_images_read": False,
    }, indent=2))


if __name__ == "__main__":
    main()
