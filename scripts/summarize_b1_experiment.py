from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.brats_dataset import get_all_modality_combinations
from utils.metrics import modality_group_from_combo


METRICS = ["bal_acc", "auc"]
MODALITIES = ["t2", "t1ce", "t1", "flair"]
FULL_COMBO = "_".join(MODALITIES)
EXPECTED_COMBOS = {"_".join(combo) for combo in get_all_modality_combinations(MODALITIES)}
CRITICAL_GROUP = "no_t1ce_with_t1"


def parse_seeds(value: str) -> list[int]:
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError(f"B1 requires exactly three distinct seeds, got {seeds}")
    return seeds


def collect_arm(root: Path, arm: str, seeds: list[int]) -> pd.DataFrame:
    rows = []
    for seed in seeds:
        run_dir = root / f"seed_{seed}"
        config_path = run_dir / "resolved_config.json"
        metrics_path = run_dir / "metrics" / "test_metrics.csv"
        if not config_path.exists() or not metrics_path.exists():
            raise FileNotFoundError(
                f"Incomplete seed={seed}: expected {config_path} and {metrics_path}"
            )
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if int(config["seed"]) != seed:
            raise ValueError(f"Seed mismatch in {config_path}")
        frame = pd.read_csv(metrics_path)
        missing = {"combo", *METRICS} - set(frame.columns)
        if missing:
            raise ValueError(f"Missing columns in {metrics_path}: {sorted(missing)}")
        if frame["combo"].duplicated().any():
            raise ValueError(f"Duplicate combo rows in {metrics_path}")
        combo_set = set(frame["combo"])
        if combo_set != EXPECTED_COMBOS:
            raise ValueError(
                f"{metrics_path} must contain exactly 15 combinations; "
                f"missing={sorted(EXPECTED_COMBOS - combo_set)}, "
                f"extra={sorted(combo_set - EXPECTED_COMBOS)}"
            )
        selected = frame[["combo", *METRICS]].copy()
        selected.insert(0, "seed", seed)
        selected.insert(0, "arm", arm)
        selected["group"] = selected["combo"].map(
            lambda name: modality_group_from_combo(name.split("_"))
        )
        rows.append(selected)
    return pd.concat(rows, ignore_index=True)


def summarize_seed_endpoints(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (arm, seed), seed_frame in frame.groupby(["arm", "seed"], sort=True):
        full = seed_frame.loc[seed_frame["combo"] == FULL_COMBO].iloc[0]
        critical = seed_frame.loc[seed_frame["group"] == CRITICAL_GROUP]
        rows.append(
            {
                "arm": arm,
                "seed": int(seed),
                "macro_bal_acc": float(seed_frame["bal_acc"].mean()),
                "macro_auc": float(seed_frame["auc"].mean()),
                "critical_bal_acc": float(critical["bal_acc"].mean()),
                "critical_auc": float(critical["auc"].mean()),
                "full_bal_acc": float(full["bal_acc"]),
                "full_auc": float(full["auc"]),
            }
        )
    return pd.DataFrame(rows)


def paired_deltas(b1: pd.DataFrame, b0: pd.DataFrame) -> pd.DataFrame:
    endpoints = [
        "macro_bal_acc",
        "macro_auc",
        "critical_bal_acc",
        "critical_auc",
        "full_bal_acc",
        "full_auc",
    ]
    merged = b1[["seed", *endpoints]].merge(
        b0[["seed", *endpoints]],
        on="seed",
        suffixes=("_b1", "_b0"),
        validate="one_to_one",
    )
    for endpoint in endpoints:
        merged[f"delta_{endpoint}"] = (
            merged[f"{endpoint}_b1"] - merged[f"{endpoint}_b0"]
        )
    return merged


def _same_direction(left: float, right: float) -> bool:
    if not math.isfinite(left) or not math.isfinite(right):
        return False
    return left == 0 or right == 0 or (left > 0) == (right > 0)


def evaluate_retention(
    deltas: pd.DataFrame,
    max_critical_drop: float = 0.01,
    min_full_gain: float = 0.03,
) -> dict:
    means = {
        column.removeprefix("delta_"): float(deltas[column].mean())
        for column in deltas.columns
        if column.startswith("delta_")
    }
    macro_ba_all_seeds = bool((deltas["delta_macro_bal_acc"] >= 0).all())
    critical_drop_ok = means["critical_bal_acc"] >= -float(max_critical_drop)
    direction_checks = {
        scope: _same_direction(
            means[f"{scope}_bal_acc"],
            means[f"{scope}_auc"],
        )
        for scope in ("macro", "critical", "full")
    }
    direction_agreement = all(direction_checks.values())
    full_gain_ok = means["full_bal_acc"] >= float(min_full_gain)
    criteria = {
        "macro_ba_noninferior_for_all_three_seeds": macro_ba_all_seeds,
        "critical_group_mean_ba_drop_at_most_0.01": critical_drop_ok,
        "mean_auc_and_ba_direction_agree": direction_agreement,
        "full_modality_mean_ba_gain_at_least_0.03": full_gain_ok,
    }
    return {
        "retain_b1": bool(all(criteria.values())),
        "criteria": criteria,
        "direction_checks": direction_checks,
        "paired_mean_deltas": means,
        "thresholds": {
            "max_critical_group_ba_drop": float(max_critical_drop),
            "min_full_modality_ba_gain": float(min_full_gain),
        },
    }


def write_report(path: Path, deltas: pd.DataFrame, decision: dict) -> None:
    lines = [
        "# B1 retention report",
        "",
        "B1 is retained only if all four pre-registered criteria pass.",
        "",
        "| seed | delta macro BA | delta macro AUC | delta critical BA | delta critical AUC | delta full BA | delta full AUC |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in deltas.sort_values("seed").iterrows():
        lines.append(
            f"| {int(row['seed'])} | {row['delta_macro_bal_acc']:+.4f} | "
            f"{row['delta_macro_auc']:+.4f} | {row['delta_critical_bal_acc']:+.4f} | "
            f"{row['delta_critical_auc']:+.4f} | {row['delta_full_bal_acc']:+.4f} | "
            f"{row['delta_full_auc']:+.4f} |"
        )
    lines.extend(["", "## Decision", ""])
    for name, passed in decision["criteria"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'}: `{name}`")
    lines.extend(
        [
            "",
            f"Final decision: **{'retain B1' if decision['retain_b1'] else 'do not retain B1'}**.",
            "",
            "The critical group is the mean across the four combinations that contain T1 but not T1ce. "
            "Macro values are unweighted means across all 15 non-empty modality combinations.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare three-seed B1 results with the matched B0 baseline."
    )
    parser.add_argument("--b1-root", type=Path, required=True)
    parser.add_argument("--b0-root", type=Path, required=True)
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    output_dir = args.output_dir or args.b1_root / "summary_vs_b0"
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.concat(
        [
            collect_arm(args.b1_root, "b1", seeds),
            collect_arm(args.b0_root, "b0", seeds),
        ],
        ignore_index=True,
    )
    endpoints = summarize_seed_endpoints(raw)
    deltas = paired_deltas(
        endpoints[endpoints["arm"] == "b1"],
        endpoints[endpoints["arm"] == "b0"],
    )
    decision = evaluate_retention(deltas)

    raw.to_csv(output_dir / "all_combo_metrics.csv", index=False)
    endpoints.to_csv(output_dir / "seed_endpoints.csv", index=False)
    deltas.to_csv(output_dir / "paired_deltas.csv", index=False)
    (output_dir / "retention_decision.json").write_text(
        json.dumps(decision, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_report(output_dir / "retention_report.md", deltas, decision)
    print(json.dumps(decision, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
