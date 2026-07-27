from __future__ import annotations

from collections import Counter
import unittest

import pandas as pd
import torch

from datasets.brats_dataset import (
    ALL_MODALITIES,
    BraTSClassificationDataset,
    get_all_modality_combinations,
)
from scripts.summarize_b1_experiment import evaluate_retention
from train import evaluate_b1_checkpoint_selection
from utils.b1_checkpoint import summarize_combo_metrics
from utils.config import load_config


class B1ProtocolTests(unittest.TestCase):
    def make_empty_dataset(self):
        return BraTSClassificationDataset(
            records=[],
            target_shape=None,
            combo_mode="balanced_all_combos",
            all_modalities=ALL_MODALITIES,
            random_seed=42,
        )

    def test_balanced_sampling_covers_all_combos_nearly_equally(self):
        dataset = self.make_empty_dataset()
        dataset.set_epoch(3, total_epochs=20, curriculum_config={})
        first_order = [dataset._choose_combo(index) for index in range(37)]
        counts = Counter(first_order)

        self.assertEqual(set(counts), set(get_all_modality_combinations(ALL_MODALITIES)))
        self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)

        dataset.set_epoch(3, total_epochs=20, curriculum_config={})
        self.assertEqual(
            first_order,
            [dataset._choose_combo(index) for index in range(37)],
        )
        dataset.set_epoch(4, total_epochs=20, curriculum_config={})
        self.assertNotEqual(
            first_order[:15],
            [dataset._choose_combo(index) for index in range(15)],
        )

    def test_checkpoint_score_uses_macro_worst_group_and_full_ba(self):
        combo_metrics = {}
        for combo in get_all_modality_combinations(ALL_MODALITIES):
            name = "_".join(combo)
            combo_set = set(combo)
            if "t1ce" in combo_set:
                value = 0.8
            elif "t1" in combo_set:
                value = 0.4
            else:
                value = 0.6
            combo_metrics[name] = {"bal_acc": value, "auc": value + 0.05}
        combo_metrics["_".join(ALL_MODALITIES)] = {"bal_acc": 0.9, "auc": 0.95}

        summary = summarize_combo_metrics(combo_metrics, ALL_MODALITIES)

        self.assertAlmostEqual(summary["worst_group_bal_acc"], 0.4)
        self.assertAlmostEqual(summary["full_bal_acc"], 0.9)
        expected = (
            0.5 * summary["macro_bal_acc"]
            + 0.3 * summary["worst_group_bal_acc"]
            + 0.2 * summary["full_bal_acc"]
        )
        self.assertAlmostEqual(summary["selection_score"], expected)

    def test_minimal_b1_validation_pipeline_runs_all_15_combos(self):
        class MutableValidationDataset:
            explicit_combo = tuple(ALL_MODALITIES)

        dataset = MutableValidationDataset()

        class ValidationLoader:
            def __init__(self, mutable_dataset):
                self.dataset = mutable_dataset

            def __iter__(self):
                combo = tuple(self.dataset.explicit_combo)
                yield {
                    "label": torch.tensor([0, 1]),
                    "available_modalities": torch.tensor(
                        [
                            [float(name in combo) for name in ALL_MODALITIES],
                            [float(name in combo) for name in ALL_MODALITIES],
                        ]
                    ),
                    "case_id": ["case_0", "case_1"],
                    "combo": [combo, combo],
                }

            def __len__(self):
                return 1

        class TinyClassifier(torch.nn.Module):
            def forward(self, batch, branch_override=None, return_segmentation=False):
                probabilities = torch.tensor(
                    [0.2, 0.8],
                    device=batch["label"].device,
                )
                return {
                    "prob": probabilities,
                    "logits": torch.stack(
                        [1.0 - probabilities, probabilities],
                        dim=1,
                    ),
                    "roi_attention": torch.ones(2, 5, device=probabilities.device)
                    / 5.0,
                }

        config = load_config("configs/experiments/aux_wt_seg_b1_multiseed.yaml")
        summary = evaluate_b1_checkpoint_selection(
            TinyClassifier(),
            dataset,
            ValidationLoader(dataset),
            config,
            torch.device("cpu"),
        )

        self.assertEqual(len(summary["combo_metrics"]), 15)
        self.assertAlmostEqual(summary["macro_bal_acc"], 1.0)
        self.assertAlmostEqual(summary["worst_group_bal_acc"], 1.0)
        self.assertAlmostEqual(summary["selection_score"], 1.0)
        self.assertEqual(dataset.explicit_combo, tuple(ALL_MODALITIES))

    def test_retention_rule_requires_all_four_conditions(self):
        passing = pd.DataFrame(
            {
                "delta_macro_bal_acc": [0.01, 0.02, 0.03],
                "delta_macro_auc": [0.01, 0.01, 0.02],
                "delta_critical_bal_acc": [-0.005, 0.0, 0.005],
                "delta_critical_auc": [-0.004, 0.0, 0.004],
                "delta_full_bal_acc": [0.03, 0.04, 0.05],
                "delta_full_auc": [0.02, 0.03, 0.04],
            }
        )
        decision = evaluate_retention(passing)
        self.assertTrue(decision["retain_b1"])

        failing = passing.copy()
        failing.loc[0, "delta_macro_bal_acc"] = -0.001
        self.assertFalse(evaluate_retention(failing)["retain_b1"])


if __name__ == "__main__":
    unittest.main()
