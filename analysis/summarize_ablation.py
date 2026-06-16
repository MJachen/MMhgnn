from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from utils.visualization import save_roi_barplot, save_roi_drop_plot

METRIC_COLS = ["acc", "auc", "f1", "sen", "spe"]
FUSION_COLS = ["alpha", "beta", "gamma"]


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize hyperedge ablation outputs")
    parser.add_argument("--input_dir", type=str, default="outputs/ablation")
    return parser.parse_args()


def aggregate_experiment_seed(seed_dir: Path):
    test_dir = seed_dir / "test"
    combo_metric_files = sorted(test_dir.glob("*/metrics.csv"))
    if not combo_metric_files:
        return None
    metric_frames = []
    roi_frames = []
    roi_drop_frames = []
    edge_drop_frames = []
    for metric_file in combo_metric_files:
        combo_name = metric_file.parent.name
        df = pd.read_csv(metric_file)
        df["combo"] = combo_name
        metric_frames.append(df)

        roi_path = metric_file.parent / "roi_importance_stats.csv"
        if roi_path.exists():
            roi_df = pd.read_csv(roi_path)
            roi_df["combo"] = combo_name
            roi_frames.append(roi_df)

        roi_drop_path = metric_file.parent / "roi_drop.csv"
        if roi_drop_path.exists():
            rd_df = pd.read_csv(roi_drop_path)
            rd_df["combo"] = combo_name
            roi_drop_frames.append(rd_df)

        edge_drop_path = metric_file.parent / "edge_type_drop_metrics.csv"
        if edge_drop_path.exists():
            ed_df = pd.read_csv(edge_drop_path)
            ed_df["combo"] = combo_name
            edge_drop_frames.append(ed_df)

    metrics_df = pd.concat(metric_frames, ignore_index=True)
    numeric_cols = [c for c in METRIC_COLS + FUSION_COLS + ["prior_norm", "modal_norm", "knn_norm"] if c in metrics_df.columns]
    metric_mean = metrics_df[numeric_cols].mean().to_dict()
    metric_mean["num_combos"] = len(metrics_df)
    metric_mean["seed"] = int(seed_dir.name.replace("seed_", "")) if seed_dir.name.startswith("seed_") else seed_dir.name

    outputs = {"metrics": metric_mean, "metrics_per_combo": metrics_df}
    if roi_frames:
        outputs["roi"] = pd.concat(roi_frames, ignore_index=True).groupby("roi", as_index=False)[["mean_importance", "std_importance"]].mean()
    if roi_drop_frames:
        outputs["roi_drop"] = pd.concat(roi_drop_frames, ignore_index=True).groupby("roi", as_index=False)[["mean_prob_drop", "std_prob_drop"]].mean()
    if edge_drop_frames:
        outputs["edge_drop"] = pd.concat(edge_drop_frames, ignore_index=True).groupby("setting", as_index=False)[METRIC_COLS].mean()
    return outputs


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    experiment_dirs = [p for p in input_dir.iterdir() if p.is_dir()]

    ablation_rows = []
    mean_std_rows = []
    roi_heatmap_rows = []
    edge_drop_rows = []

    for exp_dir in sorted(experiment_dirs):
        seed_dirs = sorted([p for p in exp_dir.iterdir() if p.is_dir() and p.name.startswith("seed_")])
        if not seed_dirs:
            continue
        per_seed_metrics = []
        exp_roi_frames = []
        exp_roi_drop_frames = []
        exp_edge_drop_frames = []

        for seed_dir in seed_dirs:
            aggregated = aggregate_experiment_seed(seed_dir)
            if aggregated is None:
                continue
            metrics = aggregated["metrics"]
            metrics["experiment"] = exp_dir.name
            per_seed_metrics.append(metrics)
            ablation_rows.append(metrics)

            if "roi" in aggregated:
                roi_df = aggregated["roi"].copy()
                roi_df["experiment"] = exp_dir.name
                exp_roi_frames.append(roi_df)
                roi_heatmap_rows.append(roi_df)
            if "roi_drop" in aggregated:
                roi_drop_df = aggregated["roi_drop"].copy()
                roi_drop_df["experiment"] = exp_dir.name
                exp_roi_drop_frames.append(roi_drop_df)
            if "edge_drop" in aggregated:
                edge_df = aggregated["edge_drop"].copy()
                edge_df["experiment"] = exp_dir.name
                exp_edge_drop_frames.append(edge_df)
                edge_drop_rows.append(edge_df)

        if not per_seed_metrics:
            continue
        seed_df = pd.DataFrame(per_seed_metrics)
        summary_row = {"experiment": exp_dir.name}
        for metric in METRIC_COLS:
            summary_row[f"{metric}_mean"] = seed_df[metric].mean()
            summary_row[f"{metric}_std"] = seed_df[metric].std(ddof=0)
        for metric in FUSION_COLS:
            if metric in seed_df.columns:
                summary_row[f"{metric}_mean"] = seed_df[metric].mean()
                summary_row[f"{metric}_std"] = seed_df[metric].std(ddof=0)
        mean_std_rows.append(summary_row)

        if exp_roi_frames:
            exp_roi = pd.concat(exp_roi_frames, ignore_index=True).groupby("roi", as_index=False)[["mean_importance", "std_importance"]].mean()
            exp_roi.to_csv(input_dir / f"roi_importance_{exp_dir.name}.csv", index=False)
            save_roi_barplot(exp_roi["roi"], exp_roi["mean_importance"], f"ROI importance: {exp_dir.name}", input_dir / f"roi_importance_{exp_dir.name}.png")

        if exp_roi_drop_frames:
            exp_roi_drop = pd.concat(exp_roi_drop_frames, ignore_index=True).groupby("roi", as_index=False)[["mean_prob_drop", "std_prob_drop"]].mean()
            exp_roi_drop.to_csv(input_dir / f"roi_drop_{exp_dir.name}.csv", index=False)
            save_roi_drop_plot(exp_roi_drop, input_dir / f"roi_drop_{exp_dir.name}.png", title=f"ROI drop: {exp_dir.name}")

    ablation_df = pd.DataFrame(ablation_rows)
    if not ablation_df.empty:
        ablation_df.to_csv(input_dir / "ablation_metrics_all.csv", index=False)

    mean_std_df = pd.DataFrame(mean_std_rows)
    if not mean_std_df.empty:
        mean_std_df.to_csv(input_dir / "summary_mean_std.csv", index=False)

    if roi_heatmap_rows:
        roi_df = pd.concat(roi_heatmap_rows, ignore_index=True)
        roi_summary = roi_df.groupby(["experiment", "roi"], as_index=False)["mean_importance"].mean()
        roi_summary.to_csv(input_dir / "roi_importance_heatmap_source.csv", index=False)

    if edge_drop_rows:
        edge_df = pd.concat(edge_drop_rows, ignore_index=True)
        edge_summary = edge_df.groupby(["experiment", "setting"], as_index=False)[METRIC_COLS].mean()
        edge_summary.to_csv(input_dir / "edge_type_drop_metrics.csv", index=False)


if __name__ == "__main__":
    main()
