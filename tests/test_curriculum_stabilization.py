from __future__ import annotations

import unittest

from datasets.brats_dataset import BraTSClassificationDataset
from utils.curriculum import curriculum_stage, reset_bad_epochs_on_stage3_entry, should_early_stop, stage3_status


CONFIG = {
    "curriculum_stage1_ratio": 0.10,
    "curriculum_stage2_ratio": 0.20,
    "curriculum_stage3_ratio": 0.70,
}


class CurriculumStabilizationTests(unittest.TestCase):
    def test_exact_100_epoch_boundaries(self):
        expected = {10: "stage1", 11: "stage2", 30: "stage2", 31: "stage3"}
        self.assertEqual({epoch: curriculum_stage(epoch, 100, CONFIG) for epoch in expected}, expected)

    def test_stage3_entry_and_completed_epochs(self):
        before = stage3_status(30, 100, CONFIG, minimum_stage3_epochs=20)
        entered = stage3_status(31, 100, CONFIG, minimum_stage3_epochs=20)
        self.assertEqual(before["stage3_epochs_completed"], 0)
        self.assertEqual(entered["stage3_entered_epoch"], 31)
        self.assertEqual(entered["stage3_epochs_completed"], 1)
        bad_epochs, was_reset = reset_bad_epochs_on_stage3_entry(entered["stage"], before["stage"], 15)
        self.assertTrue(was_reset)
        self.assertEqual(bad_epochs, 0)

    def test_early_stop_protection_and_release(self):
        pre_stage3 = stage3_status(30, 100, CONFIG, minimum_stage3_epochs=20)
        protected = stage3_status(49, 100, CONFIG, minimum_stage3_epochs=20)
        released = stage3_status(50, 100, CONFIG, minimum_stage3_epochs=20)
        self.assertTrue(pre_stage3["early_stopping_protected"])
        self.assertFalse(should_early_stop(True, 15, 15, pre_stage3["early_stopping_protected"]))
        self.assertTrue(protected["early_stopping_protected"])
        self.assertFalse(should_early_stop(True, 15, 15, protected["early_stopping_protected"]))
        self.assertFalse(released["early_stopping_protected"])
        self.assertTrue(should_early_stop(True, 15, 15, released["early_stopping_protected"]))

    def test_protection_does_not_block_best_checkpoint_decision(self):
        state = stage3_status(35, 100, CONFIG, minimum_stage3_epochs=20)
        score, best_score = 0.73, 0.70
        improved = score > best_score
        self.assertTrue(state["early_stopping_protected"])
        self.assertTrue(improved)

    def test_stage3_sampling_is_seeded_and_uses_all_nonempty_buckets(self):
        sampling_config = {
            **CONFIG,
            "stage3_full_ratio": 0.15,
            "stage3_single_missing_ratio": 0.30,
            "stage3_double_missing_ratio": 0.35,
            "stage3_triple_missing_ratio": 0.20,
        }
        dataset = BraTSClassificationDataset([], target_shape=None, combo_mode="missing_curriculum_train", random_seed=42)
        dataset.set_epoch(31, total_epochs=100, curriculum_config=sampling_config)
        first = [dataset._sample_curriculum_combo() for _ in range(2000)]
        dataset.set_epoch(31, total_epochs=100, curriculum_config=sampling_config)
        second = [dataset._sample_curriculum_combo() for _ in range(2000)]
        self.assertEqual(first, second)
        self.assertEqual({len(combo) for combo in first}, {1, 2, 3, 4})


if __name__ == "__main__":
    unittest.main()
