from __future__ import annotations

import unittest

import torch

from datasets.brats_dataset import get_all_modality_combinations
from models import HybridHypergraphClassifier
from train import mask_affine_parameter_rows
from utils.config import load_config


class MaskAffineTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config("configs/utsw_idh/missing_aware_aux_affine_e4a.yaml")
        torch.manual_seed(42)
        self.model = HybridHypergraphClassifier(self.config).eval()

    def test_identity_initialization_for_all_masks(self):
        base_logits = torch.tensor([-1.25, 0.75])
        for combo in get_all_modality_combinations(self.model.modalities):
            availability = torch.tensor([float(modality in combo) for modality in self.model.modalities])
            logits, _, _, scale, bias = self.model._apply_mask_aware_classifier(base_logits, availability)
            self.assertAlmostEqual(float(scale.item()), 1.0, places=6)
            self.assertEqual(float(bias.item()), 0.0)
            self.assertTrue(torch.allclose(logits, base_logits, atol=1e-6, rtol=0.0))
        rows = mask_affine_parameter_rows(
            self.model, self.model.modalities, get_all_modality_combinations(self.model.modalities)
        )
        self.assertEqual(len(rows), 15)
        self.assertTrue(all(row["scale_positive"] and not row["scale_extreme"] for row in rows))
        self.assertTrue(all(abs(row["scale"] - 1.0) < 1e-6 and row["bias"] == 0.0 for row in rows))

    def test_positive_scale_and_binary_logit_equation(self):
        with torch.no_grad():
            self.model.mask_affine_head.weight.zero_()
            self.model.mask_affine_head.bias.copy_(torch.tensor([-100.0, 2.0]))
        base_logits = torch.tensor([-0.4, 0.7])
        availability = torch.tensor([1.0, 0.0, 1.0, 0.0])
        logits, _, _, scale, bias = self.model._apply_mask_aware_classifier(base_logits, availability)
        self.assertGreater(float(scale.item()), 0.0)
        self.assertTrue(torch.isfinite(logits).all())
        expected_binary_logit = scale * (base_logits[1] - base_logits[0]) + bias
        self.assertTrue(torch.allclose(logits[1] - logits[0], expected_binary_logit, atol=1e-6))

    def test_bias_head_keeps_original_path_exactly(self):
        bias_config = load_config("configs/utsw_idh/missing_aware_aux_e3.yaml")
        torch.manual_seed(123)
        bias_model = HybridHypergraphClassifier(bias_config).eval()
        base_logits = torch.tensor([0.2, -0.8])
        availability = torch.tensor([1.0, 0.0, 1.0, 1.0])
        original = bias_model._apply_mask_aware_bias(base_logits, availability)
        unified = bias_model._apply_mask_aware_classifier(base_logits, availability)
        for original_value, unified_value in zip(original, unified[:3]):
            self.assertTrue(torch.equal(original_value, unified_value))
        self.assertIsNone(unified[3])
        self.assertIsNone(unified[4])


if __name__ == "__main__":
    unittest.main()
