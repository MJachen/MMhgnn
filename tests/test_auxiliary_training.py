from __future__ import annotations

import copy
import unittest

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import Dataset

from models import HybridHypergraphClassifier
from utils.config import load_config
from utils.runner import compute_modality_auxiliary_loss, run_auxiliary_epoch
from utils.training import build_dataloader


MODALITIES = ["t2", "t1ce", "t1", "flair"]


class TinyAuxiliaryDataset(Dataset):
    def __init__(self):
        self.masks = [
            torch.tensor([1.0, 1.0, 1.0, 1.0]),
            torch.tensor([0.0, 1.0, 0.0, 0.0]),
            torch.tensor([1.0, 0.0, 0.0, 0.0]),
            torch.tensor([0.0, 0.0, 1.0, 1.0]),
        ]

    def __len__(self):
        return len(self.masks)

    def __getitem__(self, index):
        mask = self.masks[index]
        return {
            "images": torch.ones(4, 2, 2, 2) * mask[:, None, None, None],
            "label": torch.tensor(index % 2, dtype=torch.long),
            "available_modalities": mask,
        }


class TinyAuxiliaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.modalities = MODALITIES
        self.encoder = nn.Linear(4, 3)
        self.classifier = nn.Linear(3, 2)
        self.auxiliary_heads = nn.ModuleDict({modality: nn.Linear(3, 2) for modality in MODALITIES})

    def forward(self, batch):
        features = self.encoder(batch["images"].mean(dim=(2, 3, 4)))
        logits = self.classifier(features)
        return {
            "logits": logits,
            "prob": torch.softmax(logits, dim=1)[:, 1],
            "aux_logits": {modality: self.auxiliary_heads[modality](features) for modality in MODALITIES},
        }


class AuxiliaryTrainingTests(unittest.TestCase):
    def test_auxiliary_disabled_preserves_e1_state_and_prediction(self):
        base_config = load_config("configs/utsw_idh/missing_aware_train.yaml")
        disabled_config = copy.deepcopy(base_config)
        disabled_config["auxiliary"] = {"enabled": False, "lambda_aux": 0.1, "pooling": "mean"}
        torch.manual_seed(123)
        base_model = HybridHypergraphClassifier(base_config).eval()
        torch.manual_seed(123)
        disabled_model = HybridHypergraphClassifier(disabled_config).eval()
        disabled_model.load_state_dict(base_model.state_dict(), strict=True)
        batch = {
            "images": torch.randn(1, 4, 8, 8, 8),
            "roi_masks": torch.ones(1, 5, 8, 8, 8),
            "roi_valid": torch.ones(1, 5),
            "available_modalities": torch.ones(1, 4),
        }
        with torch.no_grad():
            base_output = base_model(batch)
            disabled_output = disabled_model(batch)
        self.assertNotIn("aux_logits", disabled_output)
        self.assertTrue(torch.equal(base_output["logits"], disabled_output["logits"]))

    def test_observed_modalities_only_and_loss_formula(self):
        model = TinyAuxiliaryClassifier()
        batch = {
            "images": torch.ones(1, 4, 2, 2, 2),
            "label": torch.tensor([1]),
            "available_modalities": torch.tensor([[0.0, 1.0, 0.0, 0.0]]),
        }
        output = model(batch)
        criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.7, 1.3]))
        auxiliary_loss, modality_losses, counts = compute_modality_auxiliary_loss(output, batch, criterion, MODALITIES)
        self.assertEqual(counts, {"t2": 0, "t1ce": 1, "t1": 0, "flair": 0})
        self.assertIsNone(modality_losses["t2"])
        self.assertTrue(torch.equal(auxiliary_loss, modality_losses["t1ce"]))
        main_loss = criterion(output["logits"], batch["label"])
        total_loss = main_loss + 0.1 * auxiliary_loss
        self.assertTrue(torch.allclose(total_loss, main_loss + 0.1 * modality_losses["t1ce"]))

    def test_runner_updates_parameters_and_reports_counts(self):
        dataset = TinyAuxiliaryDataset()
        loader = build_dataloader(dataset, batch_size=2, num_workers=0, shuffle=False)
        model = TinyAuxiliaryClassifier()
        before = copy.deepcopy(model.state_dict())
        metrics = run_auxiliary_epoch(
            model,
            loader,
            torch.device("cpu"),
            criterion=nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.0])),
            optimizer=AdamW(model.parameters(), lr=1e-3),
            lambda_aux=0.1,
            desc="auxiliary unit",
        )
        self.assertAlmostEqual(metrics["total_loss"], metrics["main_loss"] + 0.1 * metrics["aux_loss"], places=6)
        self.assertEqual(metrics["aux_t2_count"], 2)
        self.assertEqual(metrics["aux_t1ce_count"], 2)
        self.assertEqual(metrics["aux_t1_count"], 2)
        self.assertEqual(metrics["aux_flair_count"], 2)
        self.assertTrue(any(not torch.equal(before[name], value) for name, value in model.state_dict().items()))


if __name__ == "__main__":
    unittest.main()
