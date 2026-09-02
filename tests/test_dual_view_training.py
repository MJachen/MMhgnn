from __future__ import annotations

import unittest

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import Dataset

from utils.runner import run_dual_view_epoch
from utils.training import build_dataloader


class TinyDualViewDataset(Dataset):
    groups = {
        "full": [("t2", "t1ce", "t1", "flair")],
        "t1ce_absent_t2_present": [("t2",)],
        "t2_absent_t1ce_present": [("t1ce",)],
        "t1ce_t2_both_absent": [("t1",)],
    }

    def __init__(self):
        self.curriculum_config = {
            "targeted_ratio_full": 0.30,
            "targeted_ratio_t1ce_absent_t2_present": 0.25,
            "targeted_ratio_t2_absent_t1ce_present": 0.25,
            "targeted_ratio_t1ce_t2_both_absent": 0.20,
        }

    def __len__(self):
        return len(self.groups)

    def targeted_missing_groups(self):
        return self.groups

    def __getitem__(self, index):
        subgroup = list(self.groups)[index]
        missing_combo = self.groups[subgroup][0]
        modalities = ["t2", "t1ce", "t1", "flair"]
        return {
            "case_id": f"case_{index}",
            "images": torch.ones(4, 2, 2, 2) * (index + 1),
            "label": torch.tensor(index % 2, dtype=torch.long),
            "seg": torch.ones(2, 2, 2),
            "roi_masks": torch.ones(5, 2, 2, 2),
            "roi_valid": torch.ones(5),
            "available_modalities": torch.ones(4),
            "combo": tuple(modalities),
            "missing_available_modalities": torch.tensor([float(name in missing_combo) for name in modalities]),
            "missing_combo": missing_combo,
            "targeted_subgroup": subgroup,
        }


class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Linear(2, 2)

    def forward(self, batch):
        features = torch.stack(
            [batch["images"].mean(dim=(1, 2, 3, 4)), batch["available_modalities"].mean(dim=1)], dim=1
        )
        logits = self.classifier(features)
        return {
            "logits": logits,
            "prob": torch.softmax(logits, dim=1)[:, 1],
            "stage_stats": torch.tensor([[5.0, 1.0, 0.0]], device=logits.device).repeat(logits.shape[0], 1),
        }


def test_dual_view_runner_combines_losses_and_records_sampling():
    dataset = TinyDualViewDataset()
    loader = build_dataloader(dataset, batch_size=2, num_workers=0, shuffle=False)
    model = TinyClassifier()
    before = model.classifier.weight.detach().clone()
    result = run_dual_view_epoch(
        model,
        loader,
        torch.device("cpu"),
        criterion=nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.0])),
        optimizer=AdamW(model.parameters(), lr=1e-3),
        lambda_missing=1.0,
        desc="dual-view unit",
    )
    metrics = result["metrics"]
    assert abs(metrics["loss"] - metrics["loss_full"] - metrics["loss_missing"]) < 1e-6
    assert result["sampling"]["num_samples"] == len(dataset)
    assert all(count == 1 for count in result["sampling"]["subgroup_counts"].values())
    assert not torch.equal(before, model.classifier.weight.detach())


class DualViewTrainingTests(unittest.TestCase):
    def test_runner(self):
        test_dual_view_runner_combines_losses_and_records_sampling()


if __name__ == "__main__":
    unittest.main()
