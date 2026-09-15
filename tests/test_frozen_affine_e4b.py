from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.optim import AdamW

from datasets.brats_dataset import get_all_modality_combinations
from models import HybridHypergraphClassifier
from tools.train_frozen_affine_e4b import (
    AFFINE_PREFIXES,
    aligned_from_cache,
    classifier_mask,
    configure_affine_only,
    evaluate_cache,
    load_reusable_train_val_cache,
    sha256_file,
    split_provenance,
    state_hash,
    smoke_records,
    validate_cache_frame,
)
from utils import load_config


class FrozenAffineE4BTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config("configs/utsw_idh/frozen_affine_e4b.yaml")
        self.model = HybridHypergraphClassifier(self.config)
        self.trainable_names, self.frozen_count, self.trainable_count = configure_affine_only(self.model)

    def test_only_affine_alignment_parameters_are_trainable(self):
        self.assertGreater(self.frozen_count, self.trainable_count)
        self.assertTrue(self.trainable_names)
        self.assertTrue(all(name.startswith(AFFINE_PREFIXES) for name in self.trainable_names))

    def test_optimizer_step_changes_only_affine_hash(self):
        frozen_before = state_hash(self.model, include_affine=False)
        affine_before = state_hash(self.model, include_affine=True)
        optimizer = AdamW([p for p in self.model.parameters() if p.requires_grad], lr=5e-4)
        criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.7, 1.7]))
        base = torch.tensor([-1.0, -0.2, 0.4, 1.3])
        masks = torch.tensor([[1, 1, 1, 1], [0, 1, 1, 0], [1, 0, 0, 0], [0, 0, 1, 1]], dtype=torch.float32)
        labels = torch.tensor([0, 1, 0, 1])
        logits, _, _, scale, _ = aligned_from_cache(self.model, base, masks)
        self.assertTrue(torch.all(scale > 0))
        loss = criterion(logits, labels)
        loss.backward()
        self.assertTrue(all(parameter.grad is None for name, parameter in self.model.named_parameters() if not name.startswith(AFFINE_PREFIXES)))
        optimizer.step()
        self.assertEqual(frozen_before, state_hash(self.model, include_affine=False))
        self.assertNotEqual(affine_before, state_hash(self.model, include_affine=True))

    def test_positive_scale_all_15_masks(self):
        for combo in get_all_modality_combinations(self.config["data"]["modalities"]):
            mask = torch.tensor([classifier_mask(self.model.modalities, self.model.mask_order, combo)], dtype=torch.float32)
            _, _, _, scale, _ = aligned_from_cache(self.model, torch.tensor([0.2]), mask)
            self.assertGreater(float(scale.item()), 0.0)

    def test_same_mask_has_identical_scale_and_bias(self):
        mask = torch.tensor([[1, 0, 1, 0]] * 5, dtype=torch.float32)
        _, _, _, scale, bias = aligned_from_cache(self.model, torch.linspace(-2.0, 2.0, 5), mask)
        self.assertTrue(torch.equal(scale, scale[:1].expand_as(scale)))
        self.assertTrue(torch.equal(bias, bias[:1].expand_as(bias)))

    def test_extreme_logits_show_probability_saturation_but_logit_auc_is_invariant(self):
        base_logit = torch.tensor([-30.0, -20.0, 10.0, 20.0, 30.0], dtype=torch.float32)
        aligned_logit = 2.0 * base_logit + 30.0
        labels = np.asarray([0, 1, 0, 1, 0])
        base_logit_auc = roc_auc_score(labels, base_logit.numpy())
        aligned_logit_auc = roc_auc_score(labels, aligned_logit.numpy())
        base_probability_auc = roc_auc_score(labels, torch.sigmoid(base_logit).numpy())
        aligned_probability_auc = roc_auc_score(labels, torch.sigmoid(aligned_logit).numpy())
        self.assertEqual(base_logit_auc, aligned_logit_auc)
        self.assertNotEqual(base_probability_auc, aligned_probability_auc)

    def test_same_pattern_auc_is_invariant(self):
        rng = np.random.default_rng(42)
        rows = []
        for combo in get_all_modality_combinations(self.config["data"]["modalities"]):
            scores = rng.normal(size=20)
            labels = np.asarray([0, 1] * 10)
            for index, (score, label) in enumerate(zip(scores, labels)):
                rows.append(
                    {
                        "case_id": f"c{index}",
                        "split": "val",
                        "combo": "_".join(combo),
                        "mask": "".join(str(int(v)) for v in classifier_mask(self.model.modalities, self.model.mask_order, combo)),
                        "y_true": int(label),
                        "base_logit": float(score),
                        "base_prob": float(1 / (1 + np.exp(-score))),
                    }
                )
        _, _, _, auc_rows = evaluate_cache(self.model, pd.DataFrame(rows), torch.device("cpu"), self.config["data"]["modalities"])
        self.assertLessEqual(float(auc_rows["delta_auc"].abs().max()), 1e-12)

    def test_train_val_cache_provenance_and_equal_pattern_coverage(self):
        combos = get_all_modality_combinations(self.config["data"]["modalities"])
        rows = []
        for combo in combos:
            for case_id, label in [("train_a", 0), ("train_b", 1)]:
                rows.append(
                    {
                        "case_id": case_id,
                        "split": "train",
                        "combo": "_".join(combo),
                        "mask": "1111",
                        "y_true": label,
                        "base_logit": 0.0,
                        "base_prob": 0.5,
                    }
                )
        frame = pd.DataFrame(rows)
        validate_cache_frame(frame, "train", ["train_a", "train_b"], combos)
        with self.assertRaises(RuntimeError):
            validate_cache_frame(frame, "val", ["train_a", "train_b"], combos)

    def test_cache_reuse_requires_matching_sha_and_provenance(self):
        combos = get_all_modality_combinations(self.config["data"]["modalities"])
        splits = {
            "train": [SimpleNamespace(case_id="train_a", patient_id="train_a", label=0), SimpleNamespace(case_id="train_b", patient_id="train_b", label=1)],
            "val": [SimpleNamespace(case_id="val_a", patient_id="val_a", label=0), SimpleNamespace(case_id="val_b", patient_id="val_b", label=1)],
            "test": [SimpleNamespace(case_id="test_a", patient_id="test_a", label=0), SimpleNamespace(case_id="test_b", patient_id="test_b", label=1)],
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            cache_dir = root / "cache"
            cache_dir.mkdir()
            source_path = root / "best.pt"
            source_path.write_bytes(b"source checkpoint")
            split_metadata = {}
            for split_name in ["train", "val"]:
                rows = []
                for combo in combos:
                    combo_name = "_".join(combo)
                    mask = "".join(str(int(value)) for value in classifier_mask(self.model.modalities, self.model.mask_order, combo))
                    for record in splits[split_name]:
                        rows.append({"case_id": record.case_id, "split": split_name, "combo": combo_name, "mask": mask, "y_true": record.label, "base_logit": 0.0, "base_prob": 0.5})
                csv_path = cache_dir / f"{split_name}_base_logits.csv"
                pd.DataFrame(rows).to_csv(csv_path, index=False)
                split_metadata[split_name] = {"num_rows": len(rows), "num_subjects": 2, "num_patterns": 15, "sha256": sha256_file(csv_path)}
            canonical_manifest_sha256 = "manifest-sha"
            metadata = {
                "source_checkpoint_sha256": sha256_file(source_path),
                "canonical_manifest_sha256": canonical_manifest_sha256,
                "split_provenance_sha256": hashlib.sha256(json.dumps(split_provenance(splits), sort_keys=True).encode("utf-8")).hexdigest(),
                "mask_order": list(self.model.mask_order),
                "modalities": list(self.model.modalities),
                "patterns": ["_".join(combo) for combo in combos],
                "cached_before_checkpoint_selection": ["train", "val"],
                "test_cache_created_before_checkpoint_and_threshold_freeze": False,
                **split_metadata,
            }
            (cache_dir / "cache_provenance.json").write_text(json.dumps(metadata), encoding="utf-8")
            source_checkpoint = {"canonical_manifest_sha256": canonical_manifest_sha256}
            train_frame, val_frame, _ = load_reusable_train_val_cache(
                cache_dir, source_path, source_checkpoint, splits, combos, self.model
            )
            self.assertEqual(len(train_frame), 30)
            self.assertEqual(len(val_frame), 30)
            with (cache_dir / "val_base_logits.csv").open("a", encoding="utf-8") as handle:
                handle.write("\n")
            with self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"):
                load_reusable_train_val_cache(cache_dir, source_path, source_checkpoint, splits, combos, self.model)

    def test_smoke_selection_contains_both_classes(self):
        class Record:
            def __init__(self, case_id, label):
                self.case_id = case_id
                self.label = label

        selected = smoke_records([Record("zero_a", 0), Record("zero_b", 0), Record("one", 1)])
        self.assertEqual([record.label for record in selected], [0, 1])


if __name__ == "__main__":
    unittest.main()
