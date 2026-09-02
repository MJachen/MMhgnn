from __future__ import annotations

import unittest
from collections import Counter

from datasets.brats_dataset import ALL_MODALITIES, BraTSClassificationDataset, get_all_modality_combinations


TARGETED_CONFIG = {
    "targeted_ratio_full": 0.30,
    "targeted_ratio_t1ce_absent_t2_present": 0.25,
    "targeted_ratio_t2_absent_t1ce_present": 0.25,
    "targeted_ratio_t1ce_t2_both_absent": 0.20,
}


def make_dataset(seed=42):
    dataset = BraTSClassificationDataset(
        [],
        target_shape=(8, 8, 8),
        combo_mode="targeted_dual_view_train",
        all_modalities=ALL_MODALITIES,
        random_seed=seed,
    )
    dataset.set_epoch(3, total_epochs=100, curriculum_config=TARGETED_CONFIG)
    return dataset


def test_targeted_groups_are_nonempty_existing_patterns():
    dataset = make_dataset()
    groups = dataset.targeted_missing_groups()
    assert {name: len(combos) for name, combos in groups.items()} == {
        "full": 1,
        "t1ce_absent_t2_present": 4,
        "t2_absent_t1ce_present": 4,
        "t1ce_t2_both_absent": 3,
    }
    existing = set(get_all_modality_combinations(ALL_MODALITIES))
    sampled_candidates = {combo for combos in groups.values() for combo in combos}
    assert () not in sampled_candidates
    assert sampled_candidates.issubset(existing)


def test_targeted_sampling_is_seed_reproducible_and_reaches_all_groups():
    first = make_dataset(seed=42)
    second = make_dataset(seed=42)
    first_sequence = [first._sample_targeted_missing_combo() for _ in range(1000)]
    second_sequence = [second._sample_targeted_missing_combo() for _ in range(1000)]
    assert first_sequence == second_sequence
    assert {group for group, _ in first_sequence} == set(first.targeted_missing_groups())


def test_targeted_sampling_matches_ratios_and_is_uniform_within_group():
    dataset = make_dataset(seed=42)
    draws = [dataset._sample_targeted_missing_combo() for _ in range(20000)]
    subgroup_counts = Counter(group for group, _ in draws)
    pattern_counts = Counter((group, combo) for group, combo in draws)
    expected = {
        "full": 0.30,
        "t1ce_absent_t2_present": 0.25,
        "t2_absent_t1ce_present": 0.25,
        "t1ce_t2_both_absent": 0.20,
    }
    for group, ratio in expected.items():
        assert abs(subgroup_counts[group] / len(draws) - ratio) < 0.02
        counts = [pattern_counts[(group, combo)] for combo in dataset.targeted_missing_groups()[group]]
        assert min(counts) > 0
        assert max(counts) / min(counts) < 1.25


class TargetedMissingSamplingTests(unittest.TestCase):
    def test_groups(self):
        test_targeted_groups_are_nonempty_existing_patterns()

    def test_reproducibility_and_coverage(self):
        test_targeted_sampling_is_seed_reproducible_and_reaches_all_groups()

    def test_ratios_and_within_group_uniformity(self):
        test_targeted_sampling_matches_ratios_and_is_uniform_within_group()


if __name__ == "__main__":
    unittest.main()
