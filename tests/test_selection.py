from __future__ import annotations

import unittest

from utils.selection import finetune_eligibility, is_better_joint_summary, summarize_joint_validation


SELECTION_CONFIG = {
    "full_combo": ["t2", "t1ce", "t1", "flair"],
    "no_t1ce_combos": [["t2"], ["flair"], ["t2", "flair"]],
    "weights": {
        "no_t1ce_mean_auc": 0.4,
        "no_t1ce_mean_bal_acc": 0.4,
        "full_auc": 0.1,
        "full_bal_acc": 0.1,
    },
    "min_no_t1ce_auc_delta": 0.0,
    "min_no_t1ce_bal_acc_delta": 0.0,
    "max_full_bal_acc_drop": 0.03,
}

CALIBRATION_CONFIG = {
    "threshold_metric": "balanced_accuracy",
    "threshold_min": 0.05,
    "threshold_max": 0.95,
    "tie_break": "closest_to_0.5",
}


def payload(probabilities):
    return {"y_true": [0, 0, 1, 1], "y_prob": probabilities}


class JointSelectionTests(unittest.TestCase):
    def test_summary_uses_all_required_combos(self):
        predictions = {
            "t2_t1ce_t1_flair": payload([0.1, 0.2, 0.8, 0.9]),
            "t2": payload([0.2, 0.3, 0.7, 0.8]),
            "flair": payload([0.15, 0.35, 0.65, 0.85]),
            "t2_flair": payload([0.1, 0.25, 0.75, 0.9]),
            "t2_t1_flair": payload([0.2, 0.4, 0.6, 0.8]),
        }
        summary = summarize_joint_validation(predictions, SELECTION_CONFIG, CALIBRATION_CONFIG)
        self.assertEqual(summary["no_t1ce_combos"], ["t2", "flair", "t2_flair"])
        self.assertAlmostEqual(summary["no_t1ce_mean_auc"], 1.0)
        self.assertAlmostEqual(summary["no_t1ce_mean_bal_acc"], 1.0)
        self.assertIn("t2_t1_flair", summary["combos"])

    def test_finetune_requires_auc_bac_and_full_guardrail(self):
        base = {
            "selection_score": 0.75,
            "no_t1ce_mean_auc": 0.70,
            "no_t1ce_mean_bal_acc": 0.68,
            "full_auc": 0.85,
            "full_bal_acc": 0.80,
        }
        candidate = {
            "selection_score": 0.80,
            "no_t1ce_mean_auc": 0.75,
            "no_t1ce_mean_bal_acc": 0.72,
            "full_auc": 0.84,
            "full_bal_acc": 0.78,
        }
        eligibility = finetune_eligibility(candidate, base, SELECTION_CONFIG)
        self.assertTrue(eligibility["eligible"])
        self.assertTrue(is_better_joint_summary(candidate, base))

        candidate["full_bal_acc"] = 0.75
        eligibility = finetune_eligibility(candidate, base, SELECTION_CONFIG)
        self.assertFalse(eligibility["eligible"])
        self.assertFalse(eligibility["checks"]["full_bal_acc_guardrail"])


if __name__ == "__main__":
    unittest.main()
