from __future__ import annotations

from typing import Dict


def curriculum_stage(epoch: int, total_epochs: int, config: Dict) -> str:
    """Return the configured curriculum stage using one-based epoch numbering."""
    epoch = max(int(epoch), 1)
    total_epochs = max(int(total_epochs), 1)
    stage1_epochs = round(total_epochs * float(config.get("curriculum_stage1_ratio", 0.3)))
    stage2_epochs = round(total_epochs * float(config.get("curriculum_stage2_ratio", 0.4)))
    if epoch <= stage1_epochs:
        return "stage1"
    if epoch <= stage1_epochs + stage2_epochs:
        return "stage2"
    return "stage3"


def curriculum_sampling_ratios(stage: str, config: Dict) -> Dict[str, float]:
    if stage == "stage1":
        return {"full": 1.0, "single_missing": 0.0, "double_missing": 0.0, "triple_missing": 0.0}
    if stage == "stage2":
        return {
            "full": float(config.get("stage2_full_ratio", 0.7)),
            "single_missing": float(config.get("stage2_single_missing_ratio", 0.3)),
            "double_missing": 0.0,
            "triple_missing": 0.0,
        }
    if stage == "stage3":
        return {
            "full": float(config.get("stage3_full_ratio", 0.4)),
            "single_missing": float(config.get("stage3_single_missing_ratio", 0.3)),
            "double_missing": float(config.get("stage3_double_missing_ratio", 0.2)),
            "triple_missing": float(config.get("stage3_triple_missing_ratio", 0.1)),
        }
    raise ValueError(f"Unsupported curriculum stage: {stage}")


def first_stage3_epoch(total_epochs: int, config: Dict) -> int:
    for epoch in range(1, max(int(total_epochs), 1) + 1):
        if curriculum_stage(epoch, total_epochs, config) == "stage3":
            return epoch
    return max(int(total_epochs), 1) + 1


def stage3_status(epoch: int, total_epochs: int, config: Dict, minimum_stage3_epochs: int) -> Dict:
    stage = curriculum_stage(epoch, total_epochs, config)
    entered_epoch = first_stage3_epoch(total_epochs, config)
    completed = max(int(epoch) - entered_epoch + 1, 0) if stage == "stage3" else 0
    return {
        "stage": stage,
        "stage3_entered_epoch": entered_epoch,
        "stage3_epochs_completed": completed,
        "early_stopping_protected": completed < max(int(minimum_stage3_epochs), 0),
    }


def should_early_stop(selection_event: bool, bad_epochs: int, patience: int, protected: bool) -> bool:
    return bool(selection_event and not protected and int(bad_epochs) >= int(patience))


def reset_bad_epochs_on_stage3_entry(current_stage: str, previous_stage: str | None, bad_epochs: int):
    entered = current_stage == "stage3" and previous_stage != "stage3"
    return (0 if entered else int(bad_epochs)), entered
