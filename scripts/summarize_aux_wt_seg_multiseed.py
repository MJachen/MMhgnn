from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import pandas as pd


METRICS = ["acc", "auc", "f1", "sen", "spe", "bal_acc"]
FULL_MODALITY_COMBO = "t2_t1ce_t1_flair"


def parse_seed_list(value: str) -> list[int]:
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not seeds:
        raise ValueError("At least one expected seed is required.")
    if len(seeds) != len(set(seeds)):
        raise ValueError(f"Duplicate seeds are not allowed: {seeds}")
    return seeds


def collect_arm(run_root: Path, arm: str, expected_seeds: Iterable[int]) -> pd.DataFrame:
    rows = []
    expected_seeds = list(expected_seeds)
    for seed in expected_seeds:
        run_dir = run_root / f"seed_{seed}"
        config_path = run_dir / "resolved_config.json"
        metrics_path = run_dir / "metrics/test_metrics.csv"
        if not config_path.exists() or not metrics_path.exists():
            raise FileNotFoundError(
                f"Run seed={seed} is incomplete: expected {config_path} and {metrics_path}"
            )
        config = json.loads(config_path.read_text(encoding="utf-8"))
        resolved_seed = int(config["seed"])
        if resolved_seed != seed:
            raise ValueError(f"Seed mismatch in {config_path}: directory={seed}, config={resolved_seed}")
        frame = pd.read_csv(metrics_path)
        missing_columns = [column for column in ["combo", *METRICS] if column not in frame.columns]
        if missing_columns:
            raise ValueError(f"Missing columns in {metrics_path}: {missing_columns}")
        if frame["combo"].duplicated().any():
            raise ValueError(f"Duplicate combo rows in {metrics_path}")
        if FULL_MODALITY_COMBO not in set(frame["combo"]):
            raise ValueError(f"Full-modality combo missing from {metrics_path}")
        selected = frame[["combo", *METRICS]].copy()
        selected.insert(0, "seed", seed)
        selected.insert(0, "arm", arm)
        selected["lambda_seg"] = float(config["train"].get("lambda_seg", 0.0))
        selected["run_dir"] = str(run_dir)
        rows.append(selected)
    return pd.concat(rows, ignore_index=True)


def summarize_runs(run_metrics: pd.DataFrame) -> pd.DataFrame:
    grouped = run_metrics.groupby(["arm", "lambda_seg", "combo"], sort=True)[METRICS]
    summary = grouped.agg(["mean", "std", "min", "max"]).reset_index()
    summary.columns = [
        "_".join(part for part in column if part).rstrip("_")
        if isinstance(column, tuple)
        else column
        for column in summary.columns
    ]
    seed_counts = (
        run_metrics.groupby(["arm", "lambda_seg", "combo"], sort=True)["seed"]
        .nunique()
        .rename("num_seeds")
        .reset_index()
    )
    return summary.merge(seed_counts, on=["arm", "lambda_seg", "combo"], validate="one_to_one")


def build_comparison(auxiliary: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    aux = auxiliary[["seed", "combo", *METRICS]].copy()
    base = baseline[["seed", "combo", *METRICS]].copy()
    merged = aux.merge(base, on=["seed", "combo"], suffixes=("_aux", "_base"), validate="one_to_one")
    for metric in METRICS:
        merged[f"{metric}_delta"] = merged[f"{metric}_aux"] - merged[f"{metric}_base"]
    return merged


def write_report(
    output_path: Path,
    run_metrics: pd.DataFrame,
    summary: pd.DataFrame,
    comparison: pd.DataFrame | None,
) -> None:
    full = summary[summary["combo"] == FULL_MODALITY_COMBO].copy()
    lines = [
        "# Auxiliary WT segmentation multi-seed summary",
        "",
        "All rows use the tracked fixed subject split; seeds change initialization and stochastic training only.",
        "",
        "## Full-modality classification",
        "",
        "| arm | seeds | balanced accuracy mean ± SD | AUC mean ± SD | F1 mean ± SD |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, row in full.sort_values("arm").iterrows():
        lines.append(
            f"| {row['arm']} | {int(row['num_seeds'])} | "
            f"{row['bal_acc_mean']:.4f} ± {row['bal_acc_std']:.4f} | "
            f"{row['auc_mean']:.4f} ± {row['auc_std']:.4f} | "
            f"{row['f1_mean']:.4f} ± {row['f1_std']:.4f} |"
        )

    lines.extend(["", "## Paired auxiliary minus baseline deltas", ""])
    if comparison is None:
        lines.append(
            "No baseline root was supplied. This run estimates auxiliary-arm variability but cannot establish classification improvement."
        )
    else:
        full_comparison = comparison[comparison["combo"] == FULL_MODALITY_COMBO]
        lines.extend(
            [
                "| seed | Δ balanced accuracy | Δ AUC | Δ F1 |",
                "|---:|---:|---:|---:|",
            ]
        )
        for _, row in full_comparison.sort_values("seed").iterrows():
            lines.append(
                f"| {int(row['seed'])} | {row['bal_acc_delta']:+.4f} | "
                f"{row['auc_delta']:+.4f} | {row['f1_delta']:+.4f} |"
            )
        positive_bal = int((full_comparison["bal_acc_delta"] > 0).sum())
        positive_auc = int((full_comparison["auc_delta"] > 0).sum())
        lines.extend(
            [
                "",
                f"Positive direction count: balanced accuracy {positive_bal}/{len(full_comparison)}, "
                f"AUC {positive_auc}/{len(full_comparison)}.",
            ]
        )

    lines.extend(
        [
            "",
            "## Interpretation rule",
            "",
            "Keep the lightweight auxiliary head only if the paired classification improvement is larger than seed variability and directionally consistent. A better segmentation score alone is not evidence of a better classifier.",
            "",
            f"Raw run rows: {len(run_metrics)}; aggregated rows: {len(summary)}.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate three-seed auxiliary WT classification results.")
    parser.add_argument("--run-root", type=Path, required=True, help="Auxiliary arm root containing seed_<N> directories.")
    parser.add_argument("--baseline-root", type=Path, default=None, help="Optional matched lambda_000 root.")
    parser.add_argument("--expected-seeds", default="42,43,44")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    seeds = parse_seed_list(args.expected_seeds)
    output_dir = args.output_dir or args.run_root / "summary"
    output_dir.mkdir(parents=True, exist_ok=True)

    auxiliary = collect_arm(args.run_root, "lambda_005", seeds)
    frames = [auxiliary]
    baseline = None
    if args.baseline_root is not None:
        baseline = collect_arm(args.baseline_root, "lambda_000", seeds)
        frames.append(baseline)
    run_metrics = pd.concat(frames, ignore_index=True)
    summary = summarize_runs(run_metrics)
    comparison = build_comparison(auxiliary, baseline) if baseline is not None else None

    run_metrics.to_csv(output_dir / "multiseed_run_metrics.csv", index=False)
    summary.to_csv(output_dir / "multiseed_summary.csv", index=False)
    if comparison is not None:
        comparison.to_csv(output_dir / "multiseed_comparison.csv", index=False)
    write_report(output_dir / "multiseed_report.md", run_metrics, summary, comparison)
    print(f"Saved multi-seed summary to {output_dir}")


if __name__ == "__main__":
    main()
