from __future__ import annotations

from typing import Optional, Tuple

from utils.utsw import normalize_idh_label


MISSING_LABELS = {"", "na", "n/a", "unknown", "not tested", "indeterminate", "equivocal", "missing"}


def _normalized_text(value: object) -> Tuple[str, str]:
    raw = "" if value is None else str(value).strip()
    normalized = raw.casefold().replace("_", " ").replace("-", " ")
    return raw, " ".join(normalized.split())


def normalize_grade_label(value: object) -> Tuple[Optional[int], Optional[int], str]:
    """Return normalized WHO grade, binary label, and exclusion reason."""
    raw, normalized = _normalized_text(value)
    mappings = {
        1: {"1", "1.0", "i", "grade 1", "grade i", "who grade 1", "who grade i"},
        2: {"2", "2.0", "ii", "grade 2", "grade ii", "who grade 2", "who grade ii"},
        3: {"3", "3.0", "iii", "grade 3", "grade iii", "who grade 3", "who grade iii"},
        4: {"4", "4.0", "iv", "grade 4", "grade iv", "who grade 4", "who grade iv"},
    }
    if normalized in MISSING_LABELS:
        return None, None, f"excluded_grade_{normalized.replace(' ', '_') or 'empty'}"
    for grade, accepted in mappings.items():
        if normalized in accepted:
            if grade == 1:
                return 1, None, "excluded_grade_1_outside_task"
            return grade, 0 if grade == 2 else 1, ""
    return None, None, f"excluded_grade_ambiguous:{raw}"


def normalize_mgmt_label(value: object) -> Tuple[Optional[str], Optional[int], str]:
    """Return normalized MGMT status, binary label, and exclusion reason."""
    raw, normalized = _normalized_text(value)
    if normalized in MISSING_LABELS:
        return None, None, f"excluded_mgmt_{normalized.replace(' ', '_') or 'empty'}"
    if normalized == "unmethylated":
        return "unmethylated", 0, ""
    if normalized == "methylated":
        return "methylated", 1, ""
    return None, None, f"excluded_mgmt_ambiguous:{raw}"


def normalize_task_label(task_name: str, value: object):
    if task_name == "idh":
        label, reason = normalize_idh_label(value)
        normalized = None if label is None else ("wildtype" if label == 0 else "mutant")
        return normalized, label, reason
    if task_name == "grade_2_vs_34":
        return normalize_grade_label(value)
    if task_name == "mgmt":
        return normalize_mgmt_label(value)
    raise ValueError(f"Unsupported UTSW task: {task_name}")
