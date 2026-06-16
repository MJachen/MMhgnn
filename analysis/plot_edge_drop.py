from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from utils.visualization import save_metric_comparison


def parse_args():
    parser = argparse.ArgumentParser(description="Plot edge-type drop summaries")
    parser.add_argument("--input_dir", type=str, default="outputs/ablation")
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    edge_path = input_dir / "edge_type_drop_metrics.csv"
    if not edge_path.exists():
        raise FileNotFoundError(f"Missing {edge_path}. Run summarize_ablation.py first.")
    df = pd.read_csv(edge_path)
    full_df = df[df["experiment"] == "prior_modal_knn"].copy()
    if full_df.empty:
        full_df = df.groupby("setting", as_index=False)[["auc", "f1"]].mean()
        metric_df = pd.DataFrame({"experiment": full_df["setting"], "auc": full_df["auc"]})
    else:
        metric_df = pd.DataFrame({"experiment": full_df["setting"], "auc": full_df["auc"]})
    save_metric_comparison(metric_df, "auc", input_dir / "edge_type_drop.png", title="Edge-type drop AUC comparison")


if __name__ == "__main__":
    main()
