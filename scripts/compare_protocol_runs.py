from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_COMBOS = ["t2", "flair", "t2_flair", "t2_t1_flair"]


def parse_args():
    parser = argparse.ArgumentParser(description="Validate the seed42 observed-only protocol pair.")
    parser.add_argument("--curriculum-output", required=True)
    parser.add_argument("--targeted-output", required=True)
    parser.add_argument("--combos", nargs="*", default=DEFAULT_COMBOS)
    return parser.parse_args()


def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Required protocol artifact is missing: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def main():
    args = parse_args()
    curriculum = Path(args.curriculum_output)
    targeted = Path(args.targeted_output)
    curriculum_hashes = load_json(curriculum / "metrics" / "checkpoint_hashes.json")
    targeted_hashes = load_json(targeted / "metrics" / "checkpoint_hashes.json")
    curriculum_predictions = load_json(curriculum / "metrics" / "prediction_hashes.json")
    targeted_predictions = load_json(targeted / "metrics" / "prediction_hashes.json")
    targeted_selection = load_json(targeted / "metrics" / "checkpoint_selection.json")

    curriculum_base = curriculum_hashes["base_best.pt"]["model_state_sha256"]
    targeted_base = targeted_hashes["base_best.pt"]["model_state_sha256"]
    targeted_finetune = targeted_hashes["finetune_best.pt"]["model_state_sha256"]
    prediction_checks = {}
    for combo in args.combos:
        curriculum_entry = curriculum_predictions.get(combo, {})
        targeted_entry = targeted_predictions.get(combo, {})
        curriculum_probability_hash = curriculum_entry.get("probability_sha256")
        targeted_probability_hash = targeted_entry.get("probability_sha256")
        prediction_checks[combo] = (
            curriculum_probability_hash is not None
            and targeted_probability_hash is not None
            and curriculum_probability_hash != targeted_probability_hash
        )
    checks = {
        "shared_base_weights_match": curriculum_base == targeted_base,
        "finetune_weights_differ_from_base": targeted_finetune != targeted_base,
        "finetune_selected": targeted_selection.get("status") == "finetune_selected",
        "selected_predictions_differ": all(prediction_checks.values()),
    }
    payload = {
        "passed": all(checks.values()),
        "checks": checks,
        "prediction_checks": prediction_checks,
        "curriculum_base_model_sha256": curriculum_base,
        "targeted_base_model_sha256": targeted_base,
        "targeted_finetune_model_sha256": targeted_finetune,
        "targeted_selection_status": targeted_selection.get("status"),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    raise SystemExit(0 if payload["passed"] else 1)


if __name__ == "__main__":
    main()
