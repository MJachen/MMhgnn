from __future__ import annotations

import copy
import math
import unittest

import torch
import torch.nn.functional as F

from models.hybrid_hypergraph import ANATOMY_HYPEREDGES, AuxiliaryWTSegmentationBranch, HybridHypergraphClassifier
from utils.config import load_config
from utils.losses import auxiliary_segmentation_loss
from utils.runner import run_epoch


def make_config(auxiliary_enabled: bool):
    config = copy.deepcopy(load_config("configs/default.yaml"))
    config["model"].update(
        {
            "backbone_channels": [2, 4],
            "backbone_out_dim": 4,
            "roi_hidden_dim": 8,
            "hgnn_hidden_dim": 8,
            "classifier_hidden_dim": 8,
            "mask_embed_dim": 4,
            "dropout": 0.0,
            "auxiliary_segmentation": {
                "enabled": auxiliary_enabled,
                "target": "wt",
                "projection_channels": 8,
                "hidden_channels": [8, 4],
                "output_channels": 1,
                "fusion": "masked_mean",
                "group_norm_groups": 4,
            },
        }
    )
    config["fusion"].update(
        {"mask_embed_dim": 4, "node_type_embed_dim": 2, "gating_hidden_dim": 8}
    )
    return config


def make_batch(batch_size: int = 1, shape=(8, 8, 8), availability=None):
    torch.manual_seed(7)
    images = torch.randn(batch_size, 4, *shape)
    if availability is None:
        availability = torch.ones(batch_size, 4)
    availability = availability.float()
    for batch_idx in range(batch_size):
        for modality_idx in range(4):
            if availability[batch_idx, modality_idx] < 0.5:
                images[batch_idx, modality_idx].zero_()

    roi_masks = torch.zeros(batch_size, 5, *shape)
    roi_masks[:, 0, 2:5, 2:5, 2:5] = 1.0
    roi_masks[:, 1, 1:6, 1:6, 1:6] = 1.0
    roi_masks[:, 2, 1:7, 1:7, 1:7] = 1.0
    roi_masks[:, 3, :, :, :4] = 1.0
    roi_masks[:, 4, :, :, 4:] = 1.0
    segmentation = torch.zeros(batch_size, *shape)
    segmentation[:, 2:6, 2:6, 2:6] = 4.0
    return {
        "images": images,
        "label": torch.arange(batch_size, dtype=torch.long) % 2,
        "seg": segmentation,
        "roi_masks": roi_masks,
        "roi_valid": torch.ones(batch_size, 5),
        "available_modalities": availability,
    }


def has_finite_nonzero_gradient(module: torch.nn.Module) -> bool:
    return any(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        and bool(torch.any(parameter.grad != 0))
        for parameter in module.parameters()
    )


class AuxiliaryWTSegmentationTests(unittest.TestCase):
    def test_baseline_disabled_regression(self):
        config_without_auxiliary_key = make_config(False)
        config_without_auxiliary_key["model"].pop("auxiliary_segmentation")
        config_explicitly_disabled = make_config(False)

        torch.manual_seed(11)
        legacy_style_model = HybridHypergraphClassifier(config_without_auxiliary_key).eval()
        torch.manual_seed(11)
        disabled_model = HybridHypergraphClassifier(config_explicitly_disabled).eval()
        batch = make_batch()

        with torch.no_grad():
            expected = legacy_style_model(batch)
            actual = disabled_model(batch)

        self.assertIsNone(disabled_model.auxiliary_segmentation)
        self.assertFalse(any(key.startswith("auxiliary_segmentation.") for key in disabled_model.state_dict()))
        self.assertEqual(list(legacy_style_model.state_dict()), list(disabled_model.state_dict()))
        torch.testing.assert_close(actual["logits"], expected["logits"], rtol=0, atol=0)
        self.assertEqual(tuple(actual["logits"].shape), (1, 2))
        self.assertNotIn("seg_logits", actual)
        self.assertEqual(len(ANATOMY_HYPEREDGES), 5)
        self.assertEqual(actual["roi_shared_features"].shape[1], 5)

    def test_auxiliary_forward_shapes(self):
        model = HybridHypergraphClassifier(make_config(True)).eval()
        batch = make_batch(batch_size=2)
        with torch.no_grad():
            output = model(batch)
        self.assertEqual(tuple(output["logits"].shape), (2, 2))
        self.assertEqual(tuple(output["seg_logits"].shape), (2, 1, 8, 8, 8))
        self.assertTrue(torch.isfinite(output["seg_logits"]).all())
        self.assertTrue(torch.all(output["stage_stats"][:, 0] == 5))

    def test_total_loss_backward_reaches_both_branches(self):
        model = HybridHypergraphClassifier(make_config(True)).train()
        batch = make_batch()
        output = model(batch)
        classification_loss = F.cross_entropy(output["logits"], batch["label"])
        segmentation_terms = auxiliary_segmentation_loss(output["seg_logits"], batch["seg"])
        total_loss = classification_loss + 0.1 * segmentation_terms["segmentation_loss"]
        total_loss.backward()

        self.assertIsNotNone(model.auxiliary_segmentation)
        self.assertTrue(has_finite_nonzero_gradient(model.auxiliary_segmentation.segmentation_head))
        for modality in model.modalities:
            self.assertTrue(has_finite_nonzero_gradient(model.auxiliary_segmentation.projections[modality]))
            self.assertTrue(has_finite_nonzero_gradient(model.backbones[modality]))
        self.assertTrue(has_finite_nonzero_gradient(model.anatomy_hypergraph))
        self.assertTrue(has_finite_nonzero_gradient(model.classifier))

    def test_missing_modality_masked_mean_and_all_zero_guard(self):
        projected = torch.stack(
            [torch.full((1, 2, 2, 2, 2), value) for value in (1.0, 100.0, 3.0, 5.0)],
            dim=1,
        )
        availability = torch.tensor([[1.0, 0.0, 1.0, 1.0]])
        fused = AuxiliaryWTSegmentationBranch.masked_mean_fusion(projected, availability)
        changed_missing = projected.clone()
        changed_missing[:, 1] = 1e6
        fused_after_change = AuxiliaryWTSegmentationBranch.masked_mean_fusion(changed_missing, availability)

        torch.testing.assert_close(fused, torch.full_like(fused, 3.0), rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(fused, fused_after_change, rtol=0, atol=0)
        self.assertTrue(torch.isfinite(fused).all())
        with self.assertRaisesRegex(ValueError, "all-zero mask"):
            AuxiliaryWTSegmentationBranch.masked_mean_fusion(projected, torch.zeros_like(availability))

    def test_segmentation_loss_is_finite_for_normal_tiny_and_empty_targets(self):
        targets = []
        normal = torch.zeros(2, 4, 4, 4)
        normal[:, 1:3, 1:3, 1:3] = 2
        targets.append(normal)
        tiny = torch.zeros(2, 4, 4, 4)
        tiny[:, 0, 0, 0] = 4
        targets.append(tiny)
        targets.append(torch.zeros(2, 4, 4, 4))

        for target in targets:
            with self.subTest(foreground_voxels=int((target > 0).sum())):
                logits = torch.zeros(2, 1, 4, 4, 4, requires_grad=True)
                terms = auxiliary_segmentation_loss(logits, target)
                for value in terms.values():
                    self.assertTrue(torch.isfinite(value).all())
                terms["segmentation_loss"].backward()
                self.assertIsNotNone(logits.grad)
                self.assertTrue(torch.isfinite(logits.grad).all())

    def test_old_baseline_checkpoint_loads_in_both_modes(self):
        baseline_model = HybridHypergraphClassifier(make_config(False)).eval()
        baseline_state = copy.deepcopy(baseline_model.state_dict())
        baseline_model.load_checkpoint_state_dict(baseline_state)

        auxiliary_model = HybridHypergraphClassifier(make_config(True)).eval()
        incompatible = auxiliary_model.load_checkpoint_state_dict(baseline_state)
        self.assertTrue(incompatible.missing_keys)
        self.assertTrue(all(key.startswith("auxiliary_segmentation.") for key in incompatible.missing_keys))
        self.assertFalse(incompatible.unexpected_keys)

        batch = make_batch()
        with torch.no_grad():
            baseline_logits = baseline_model(batch)["logits"]
            auxiliary_logits = auxiliary_model(batch, return_segmentation=False)["logits"]
        torch.testing.assert_close(auxiliary_logits, baseline_logits, rtol=0, atol=0)

    def test_training_loop_reports_separate_auxiliary_losses(self):
        model = HybridHypergraphClassifier(make_config(True))
        batch = make_batch()

        class OneBatchLoader:
            dataset = [None]

            def __iter__(self):
                return iter([batch])

            def __len__(self):
                return 1

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        metrics = run_epoch(
            model,
            OneBatchLoader(),
            torch.device("cpu"),
            criterion=torch.nn.CrossEntropyLoss(),
            optimizer=optimizer,
            segmentation_config={
                "enabled": True,
                "lambda_seg": 0.05,
                "seg_bce_weight": 0.5,
                "seg_dice_weight": 0.5,
            },
            desc="aux-test",
        )
        for key in (
            "classification_loss",
            "seg_bce_loss",
            "seg_dice_loss",
            "segmentation_loss",
            "wt_dice",
            "lambda_seg",
            "loss",
        ):
            self.assertIn(key, metrics)
            self.assertTrue(math.isfinite(metrics[key]))
        self.assertAlmostEqual(
            metrics["loss"],
            metrics["classification_loss"] + 0.05 * metrics["segmentation_loss"],
            places=6,
        )


if __name__ == "__main__":
    unittest.main()
