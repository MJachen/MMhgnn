from __future__ import annotations

import unittest
from pathlib import Path

from utils.config import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ProtocolConfigTests(unittest.TestCase):
    def load(self, name):
        return load_config(str(PROJECT_ROOT / "configs" / name))

    def test_curriculum_and_targeted_are_explicitly_separated(self):
        curriculum = self.load("brats2020_public_missing_curriculum_server.yaml")
        targeted = self.load("brats2020_public_no_t1ce_focus_server.yaml")
        self.assertFalse(curriculum["train"]["enable_targeted_finetune"])
        self.assertTrue(targeted["train"]["enable_targeted_finetune"])
        self.assertEqual(curriculum["data"]["split_json"], targeted["data"]["split_json"])

    def test_joint_validation_covers_protocol_combos(self):
        config = self.load("brats2020_public_missing_curriculum_server.yaml")
        joint = config["train"]["joint_validation"]
        self.assertTrue(joint["enabled"])
        self.assertEqual(joint["full_combo"], ["t2", "t1ce", "t1", "flair"])
        self.assertEqual(joint["no_t1ce_combos"], [["t2"], ["flair"], ["t2", "flair"]])
        self.assertIn(["t2", "t1", "flair"], joint["monitor_combos"])


if __name__ == "__main__":
    unittest.main()
