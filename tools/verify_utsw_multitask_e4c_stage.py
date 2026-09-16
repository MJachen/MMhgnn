from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Verify one completed task-specific E4A->E4B stage.")
    parser.add_argument("--task-root", required=True)
    return parser.parse_args()


def main():
    root = Path(parse_args().task_root)
    required = [
        root / "e4a_seed42/checkpoints/best.pt",
        root / "e4a_seed42/metrics/threshold_calibration.json",
        root / "e4a_seed42/metrics/test_metrics.csv",
        root / "e4a_seed42/metrics/test_missing_pattern_groups.csv",
        root / "e4a_seed42/diagnostics/post_e4a_validation/e4a_validation_diagnostic_summary.json",
        root / "e4b_seed42/checkpoints/best.pt",
        root / "e4b_seed42/metrics/threshold_calibration.json",
        root / "e4b_seed42/metrics/test_metrics.csv",
        root / "e4b_seed42/metrics/test_missing_pattern_groups.csv",
        root / "e4b_seed42/metrics/validation_auc_invariance.csv",
        root / "e4b_seed42/metrics/mask_affine_parameters.csv",
        root / "e4b_seed42/diagnostics/post_e4b_validation/e4b_validation_diagnostic_summary.json",
        root / "e4b_seed42/e4b_run_summary.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Incomplete E4C task artifacts: {missing}")
    with (root / "e4b_seed42/e4b_run_summary.json").open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    checks = {
        "status_ok": summary.get("status") == "ok",
        "frozen_hash_unchanged": summary.get("frozen_hash_unchanged") is True,
        "test_after_freeze": summary.get("test_read_after_checkpoint_and_threshold_freeze") is True,
        "one_global_threshold": isinstance(summary.get("global_threshold"), (int, float)),
    }
    if not all(checks.values()):
        raise RuntimeError(f"E4C stage integrity checks failed: {checks}")
    print(json.dumps({"status": "ok", "task_root": str(root), **checks}, indent=2))


if __name__ == "__main__":
    main()
