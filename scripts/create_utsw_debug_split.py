from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.utsw import canonical_manifest_fingerprint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a patient-level stratified UTSW debug split.")
    parser.add_argument("--manifest-csv", default="metadata/utsw_idh_manifest.csv")
    parser.add_argument("--output-json", default="splits/utsw_idh/utsw_idh_debug_split.json")
    parser.add_argument("--summary-json", default="metadata/utsw_dataset_summary.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    ratios = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(ratios - 1.0) > 1e-8:
        raise ValueError("train/val/test ratios must sum to 1.0")

    manifest_path = Path(args.manifest_csv)
    manifest = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    eligible = manifest.loc[manifest["eligible"].str.lower().eq("true")].copy()
    eligible["normalized_idh_label"] = eligible["normalized_idh_label"].astype(int)
    if eligible["patient_id"].duplicated().any():
        raise ValueError("Multiple eligible scans share patient_id; grouped split logic must be used explicitly.")

    train, temporary = train_test_split(
        eligible,
        test_size=1.0 - args.train_ratio,
        random_state=args.seed,
        stratify=eligible["normalized_idh_label"],
    )
    relative_test = args.test_ratio / (args.val_ratio + args.test_ratio)
    val, test = train_test_split(
        temporary,
        test_size=relative_test,
        random_state=args.seed,
        stratify=temporary["normalized_idh_label"],
    )
    split_frames = {"train": train, "val": val, "test": test}

    patient_sets = {name: set(frame["patient_id"]) for name, frame in split_frames.items()}
    for left, right in [("train", "val"), ("train", "test"), ("val", "test")]:
        overlap = patient_sets[left].intersection(patient_sets[right])
        if overlap:
            raise RuntimeError(f"Patient leakage between {left} and {right}: {sorted(overlap)[:10]}")
    covered = set().union(*patient_sets.values())
    if covered != set(eligible["patient_id"]):
        raise RuntimeError("Debug split does not cover the full eligible cohort exactly once.")
    for name, frame in split_frames.items():
        if frame["normalized_idh_label"].nunique() != 2:
            raise RuntimeError(f"Split {name} does not contain both IDH classes.")

    split_payload = {
        "dataset": "utsw_idh",
        "purpose": "debug_feasibility_full_modality_baseline",
        "seed": args.seed,
        "ratios": {"train": args.train_ratio, "val": args.val_ratio, "test": args.test_ratio},
        "patient_id_source": "Subject ID",
        "manifest_sha256_before_split_assignment": sha256_file(manifest_path),
        "canonical_manifest_sha256": canonical_manifest_fingerprint(manifest),
        "label_mapping": {"0": "IDH-WT", "1": "IDH-mutant"},
        "train": train["case_id"].tolist(),
        "val": val["case_id"].tolist(),
        "test": test["case_id"].tolist(),
        "counts": {
            name: {
                "cases": int(len(frame)),
                "idh_wildtype": int(frame["normalized_idh_label"].eq(0).sum()),
                "idh_mutant": int(frame["normalized_idh_label"].eq(1).sum()),
            }
            for name, frame in split_frames.items()
        },
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as stream:
        json.dump(split_payload, stream, indent=2, ensure_ascii=False)

    assignment = {
        case_id: split_name
        for split_name, frame in split_frames.items()
        for case_id in frame["case_id"].tolist()
    }
    manifest["split"] = manifest["case_id"].map(assignment).fillna("")
    manifest.to_csv(manifest_path, index=False)

    summary_path = Path(args.summary_json)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["debug_split"] = split_payload["counts"]
    summary["debug_split_seed"] = args.seed
    summary["debug_split_json"] = output_path.as_posix()
    summary["manifest_sha256"] = sha256_file(manifest_path)
    summary["canonical_manifest_sha256"] = canonical_manifest_fingerprint(manifest)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(split_payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
