from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create deterministic subset splits for UTSW server plumbing tests.")
    parser.add_argument("--manifest-csv", default="metadata/utsw_idh_manifest.csv")
    parser.add_argument("--debug-split-json", default="splits/utsw_idh/utsw_idh_debug_split.json")
    parser.add_argument("--output-dir", default="splits/utsw_idh")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _sample(ids: list[str], labels: dict[str, int], count: int, rng: np.random.Generator) -> list[str]:
    if count % 2:
        raise ValueError("Balanced server subset sizes must be even.")
    selected = []
    for label in [0, 1]:
        candidates = np.asarray([case_id for case_id in ids if labels[case_id] == label])
        chosen = rng.choice(candidates, size=count // 2, replace=False).tolist()
        selected.extend(chosen)
    rng.shuffle(selected)
    return selected


def _write_split(path: Path, purpose: str, splits: dict[str, list[str]], labels: dict[str, int], fingerprint: str, seed: int) -> None:
    sets = {name: set(values) for name, values in splits.items()}
    for left, right in [("train", "val"), ("train", "test"), ("val", "test")]:
        overlap = sets[left].intersection(sets[right])
        if overlap:
            raise RuntimeError(f"Subset split leakage between {left} and {right}: {sorted(overlap)}")
    payload = {
        "dataset": "utsw_idh",
        "purpose": purpose,
        "seed": seed,
        "patient_id_source": "Subject ID",
        "canonical_manifest_sha256": fingerprint,
        **splits,
        "counts": {
            name: {
                "cases": len(values),
                "idh_wildtype": sum(labels[case_id] == 0 for case_id in values),
                "idh_mutant": sum(labels[case_id] == 1 for case_id in values),
            }
            for name, values in splits.items()
        },
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.manifest_csv, dtype=str, keep_default_na=False)
    eligible = manifest.loc[manifest["eligible"].str.casefold().eq("true")]
    labels = dict(zip(eligible["case_id"], eligible["normalized_idh_label"].astype(int)))
    debug = json.loads(Path(args.debug_split_json).read_text(encoding="utf-8"))
    fingerprint = str(debug["canonical_manifest_sha256"])
    rng = np.random.default_rng(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    smoke = {name: _sample(debug[name], labels, 2, rng) for name in ["train", "val", "test"]}
    tiny = {
        "train": _sample(debug["train"], labels, 12, rng),
        "val": _sample(debug["val"], labels, 4, rng),
        "test": _sample(debug["test"], labels, 4, rng),
    }
    _write_split(output_dir / "utsw_idh_server_smoke_split.json", "server_pipeline_smoke_only", smoke, labels, fingerprint, args.seed)
    _write_split(output_dir / "utsw_idh_tiny_overfit_split.json", "tiny_set_overfit_only", tiny, labels, fingerprint, args.seed)
    print(json.dumps({"server_smoke": {k: len(v) for k, v in smoke.items()}, "tiny_overfit": {k: len(v) for k, v in tiny.items()}}, indent=2))


if __name__ == "__main__":
    main()
