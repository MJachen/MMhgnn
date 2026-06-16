from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from utils.visualization import save_heatmap


def parse_args():
    parser = argparse.ArgumentParser(description="Plot ROI importance summaries")
    parser.add_argument("--input_dir", type=str, default="outputs/ablation")
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    source_path = input_dir / "roi_importance_heatmap_source.csv"
    if not source_path.exists():
        raise FileNotFoundError(f"Missing {source_path}. Run summarize_ablation.py first.")
    df = pd.read_csv(source_path)
    heatmap_df = df.pivot(index="experiment", columns="roi", values="mean_importance").reset_index()
    save_heatmap(heatmap_df, "experiment", input_dir / "roi_importance_heatmap.png", title="ROI importance heatmap across hyperedge settings")


if __name__ == "__main__":
    main()
