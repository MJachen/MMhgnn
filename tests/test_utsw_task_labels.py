from __future__ import annotations

import unittest

from utils.utsw_tasks import normalize_grade_label, normalize_mgmt_label


class UTSWTaskLabelTests(unittest.TestCase):
    def test_grade_2_vs_34_mapping(self):
        for value in ["2", "II", "Grade 2", "WHO grade 2"]:
            self.assertEqual(normalize_grade_label(value), (2, 0, ""))
        for value in ["3", "III", "Grade 3", "WHO grade 3"]:
            self.assertEqual(normalize_grade_label(value), (3, 1, ""))
        for value in ["4", "IV", "Grade 4", "WHO grade 4"]:
            self.assertEqual(normalize_grade_label(value), (4, 1, ""))

    def test_grade_1_and_unknown_are_excluded(self):
        self.assertEqual(normalize_grade_label("Grade 1")[:2], (1, None))
        for value in ["", "NA", "unknown", "NOS", "indeterminate", "grade high"]:
            self.assertIsNone(normalize_grade_label(value)[1])

    def test_mgmt_mapping_is_explicit_only(self):
        self.assertEqual(normalize_mgmt_label("unmethylated"), ("unmethylated", 0, ""))
        self.assertEqual(normalize_mgmt_label("methylated"), ("methylated", 1, ""))
        for value in ["", "unknown", "equivocal", "positive", "IDH mutant"]:
            self.assertIsNone(normalize_mgmt_label(value)[1])


if __name__ == "__main__":
    unittest.main()
