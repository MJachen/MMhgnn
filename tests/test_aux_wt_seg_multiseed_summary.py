from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.summarize_aux_wt_seg_multiseed import (
    FULL_MODALITY_COMBO,
    build_comparison,
    collect_arm,
    summarize_runs,
)


class AuxWTSegMultiSeedSummaryTests(unittest.TestCase):
    def _write_arm(self, root: Path, lambda_seg: float, offset: float) -> None:
        for seed_index, seed in enumerate([42, 43, 44]):
            run_dir = root / f"seed_{seed}"
            metrics_dir = run_dir / "metrics"
            metrics_dir.mkdir(parents=True)
            (run_dir / "resolved_config.json").write_text(
                json.dumps({"seed": seed, "train": {"lambda_seg": lambda_seg}}),
                encoding="utf-8",
            )
            value = 0.7 + offset + seed_index * 0.01
            pd.DataFrame(
                [
                    {
                        "combo": FULL_MODALITY_COMBO,
                        "acc": value,
                        "auc": value,
                        "f1": value,
                        "sen": value,
                        "spe": value,
                        "bal_acc": value,
                    }
                ]
            ).to_csv(metrics_dir / "test_metrics.csv", index=False)

    def test_collect_summarize_and_compare(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            aux_root = tmp_path / "lambda_005"
            base_root = tmp_path / "lambda_000"
            self._write_arm(aux_root, lambda_seg=0.05, offset=0.05)
            self._write_arm(base_root, lambda_seg=0.0, offset=0.0)

            aux = collect_arm(aux_root, "lambda_005", [42, 43, 44])
            base = collect_arm(base_root, "lambda_000", [42, 43, 44])
            summary = summarize_runs(pd.concat([aux, base], ignore_index=True))
            comparison = build_comparison(aux, base)

            self.assertEqual(len(aux), 3)
            self.assertEqual(set(summary["num_seeds"]), {3})
            self.assertTrue((comparison["bal_acc_delta"].round(8) == 0.05).all())
            self.assertTrue((comparison["auc_delta"].round(8) == 0.05).all())


if __name__ == "__main__":
    unittest.main()
