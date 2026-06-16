from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.visualization import save_dual_metric_comparison, save_metric_comparison


def parse_args():
    parser = argparse.ArgumentParser(description="Plot ablation metric comparisons")
    parser.add_argument("--input_dir", type=str, default="outputs/ablation")
    return parser.parse_args()


def save_fusion_barplot(df: pd.DataFrame, path: Path):
    x = np.arange(len(df))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x - width, df["alpha_mean"], width, label="alpha")
    ax.bar(x, df["beta_mean"], width, label="beta")
    ax.bar(x + width, df["gamma_mean"], width, label="gamma")
    ax.set_xticks(x)
    ax.set_xticklabels(df["experiment"], rotation=35)
    ax.set_title("Final fusion weights across hyperedge settings")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    summary_path = input_dir / "summary_mean_std.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing {summary_path}. Run summarize_ablation.py first.")

    df = pd.read_csv(summary_path)
    plot_df = pd.DataFrame({
        "experiment": df["experiment"],
        "auc": df["auc_mean"],
        "f1": df["f1_mean"],
        "sen": df["sen_mean"],
        "spe": df["spe_mean"],
    })
    save_metric_comparison(plot_df, "auc", input_dir / "auc_comparison.png", title="AUC comparison across hyperedge settings")
    save_metric_comparison(plot_df, "f1", input_dir / "f1_comparison.png", title="F1 comparison across hyperedge settings")
    save_dual_metric_comparison(plot_df, "sen", "spe", input_dir / "sen_spe_comparison.png", title="SEN/SPE comparison across hyperedge settings")

    if {"alpha_mean", "beta_mean", "gamma_mean"}.issubset(df.columns):
        save_fusion_barplot(df[["experiment", "alpha_mean", "beta_mean", "gamma_mean"]], input_dir / "fusion_weights_comparison.png")


if __name__ == "__main__":
    main()
