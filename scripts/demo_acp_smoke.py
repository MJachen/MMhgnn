from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import torch
from torch import nn

from models import HybridHypergraphClassifier
from utils.config import ensure_dir, load_config
from utils.metrics import build_drop_t1_ablation_report, calibrate_grouped_3way_t1ce_t1, modality_group_from_combo, threshold_dispatch_for_combo, compute_binary_metrics
from utils.visualization import save_roi_barplot

MASK_ORDER = ["t1", "t1ce", "t2", "flair"]


def make_batch(combo, roi_names, shape=(32, 40, 28)):
    images = torch.randn(1, 4, *shape)
    available = torch.tensor([[1.0 if m in combo else 0.0 for m in ["t2", "t1ce", "t1", "flair"]]], dtype=torch.float32)
    for idx, flag in enumerate(available[0]):
        if flag < 0.5:
            images[:, idx] = 0.0

    roi_masks = torch.zeros(1, len(roi_names), *shape)
    roi_masks[:, 0, 10:20, 12:22, 8:16] = 1.0
    roi_masks[:, 1, 8:22, 10:24, 6:18] = 1.0 - roi_masks[:, 0, 8:22, 10:24, 6:18]
    roi_masks[:, 2, 6:24, 8:26, 4:20] = 1.0
    roi_masks[:, 2] = torch.clamp(roi_masks[:, 2] - roi_masks[:, 0] - roi_masks[:, 1], min=0)
    roi_masks[:, 3, 2:28, 4:32, 2:24] = 1.0
    roi_masks[:, 3] = torch.clamp(roi_masks[:, 3] - roi_masks[:, 2] - roi_masks[:, 0] - roi_masks[:, 1], min=0)
    roi_masks[:, 4] = 1.0
    roi_masks[:, 4] = torch.clamp(roi_masks[:, 4] - roi_masks[:, 3] - roi_masks[:, 2] - roi_masks[:, 0] - roi_masks[:, 1], min=0)

    roi_valid = torch.ones(1, len(roi_names))
    seg = torch.zeros(1, *shape)
    seg[0, 8:22, 10:24, 6:18] = 1.0
    return {
        "case_id": ["synthetic_case"],
        "images": images,
        "label": torch.tensor([1], dtype=torch.long),
        "seg": seg,
        "roi_masks": roi_masks,
        "roi_valid": roi_valid,
        "available_modalities": available,
        "combo": [combo],
    }


def summarize_gates(gates_tensor):
    gates = gates_tensor.detach().cpu().numpy()
    global_mean = gates.mean(axis=0)
    return {
        "global": {mod: float(global_mean[idx]) for idx, mod in enumerate(MASK_ORDER)},
        "per_node": [
            {mod: float(gates[node_idx, idx]) for idx, mod in enumerate(MASK_ORDER)}
            for node_idx in range(gates.shape[0])
        ],
    }


def main():
    cfg = load_config("configs/default.yaml")
    model = HybridHypergraphClassifier(cfg)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    roi_names = cfg["model"]["roi_names"]
    full_combo = tuple(cfg["data"]["modalities"])
    batch = make_batch(full_combo, roi_names)

    out = model(batch)
    loss = criterion(out["logits"], batch["label"])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    fine_tune_batch = make_batch(("t2", "flair"), roi_names)
    fine_tune_out = model(fine_tune_batch)
    fine_tune_loss = criterion(fine_tune_out["logits"], fine_tune_batch["label"])
    optimizer.zero_grad()
    fine_tune_loss.backward()
    optimizer.step()

    combos = []
    mods = cfg["data"]["modalities"]
    for r in range(1, len(mods) + 1):
        combos.extend(itertools.combinations(mods, r))

    eval_results = []
    val_probs = []
    val_labels = []
    model.eval()
    with torch.no_grad():
        for combo in combos:
            combo_batch = make_batch(combo, roi_names)
            combo_out = model(combo_batch)
            prob = float(combo_out["prob"][0].item())
            gate_summary = summarize_gates(combo_out["modality_gates"][0])
            eval_results.append({
                "combo": list(combo),
                "prob_hgg": prob,
                "gate_summary": gate_summary,
            })
            val_probs.append(prob)
            group = modality_group_from_combo(combo)
            if group == "no_t1ce_with_t1":
                val_labels.append(1)
            else:
                val_labels.append(1 if len(combo) % 2 == 0 else 0)

    out_dir = ensure_dir("outputs/acp_demo")
    roi_scores = out["roi_attention"][0].detach().cpu().numpy()
    save_roi_barplot(roi_names, roi_scores, "ACP explicit node importance", out_dir / "node_importance.png")
    calibration = calibrate_grouped_3way_t1ce_t1(val_labels, val_probs, [item["combo"] for item in eval_results], metric="balanced_accuracy")
    val_metrics = compute_binary_metrics(val_labels, val_probs, threshold=calibration["threshold"])
    metrics_by_combo = {}
    for item in eval_results:
        combo_tuple = tuple(item["combo"])
        dispatch = threshold_dispatch_for_combo(calibration, combo_tuple, calibration["threshold"])
        metrics_by_combo["_".join(combo_tuple)] = {
            "acc": 0.5 + 0.01 * len(combo_tuple),
            "auc": item["prob_hgg"],
            "f1": 0.4 + 0.01 * len(combo_tuple),
            "sen": 0.45 + 0.01 * len(combo_tuple),
            "spe": 0.55 + 0.005 * len(combo_tuple),
            "bal_acc": 0.5 + 0.0075 * len(combo_tuple),
            "threshold_group": dispatch["threshold_group"],
            "applied_threshold": dispatch["applied_threshold"],
        }
    drop_t1_report = build_drop_t1_ablation_report(metrics_by_combo)

    gate_map = {"_".join(item["combo"]): item["gate_summary"] for item in eval_results}
    full_gate = gate_map["t2_t1ce_t1_flair"]["global"]
    t2_t1_gate = gate_map["t2_t1"]["global"]
    t1_flair_gate = gate_map["t1_flair"]["global"]
    t2_flair_gate = gate_map["t2_flair"]["global"]
    missing_modalities_zero = {
        "t2_t1ce_t1_flair": all(full_gate[mod] > 0.0 for mod in MASK_ORDER),
        "t2_only": gate_map["t2"]["global"]["t1"] == 0.0 and gate_map["t2"]["global"]["t1ce"] == 0.0 and gate_map["t2"]["global"]["flair"] == 0.0,
        "flair_only": gate_map["flair"]["global"]["t1"] == 0.0 and gate_map["flair"]["global"]["t1ce"] == 0.0 and gate_map["flair"]["global"]["t2"] == 0.0,
    }
    t1_downweight_checks = {
        "full_vs_t2_t1": {
            "full_t1": float(full_gate["t1"]),
            "t2_t1_t1": float(t2_t1_gate["t1"]),
            "t1_reduced": bool(t2_t1_gate["t1"] < full_gate["t1"]),
        },
        "t1_flair_vs_flair": {
            "t1_flair_t1": float(t1_flair_gate["t1"]),
            "t1_flair_flair": float(t1_flair_gate["flair"]),
            "t1_below_flair": bool(t1_flair_gate["t1"] < t1_flair_gate["flair"]),
        },
        "t2_t1_vs_t2": {
            "t2_t1_t1": float(t2_t1_gate["t1"]),
            "t2_t1_t2": float(t2_t1_gate["t2"]),
            "t1_below_t2": bool(t2_t1_gate["t1"] < t2_t1_gate["t2"]),
        },
        "t2_t1_flair_vs_t2_flair": {
            "t2_t1_flair_t1": float(gate_map["t2_t1_flair"]["global"]["t1"]),
            "t2_t1_flair_t2": float(gate_map["t2_t1_flair"]["global"]["t2"]),
            "t2_t1_flair_flair": float(gate_map["t2_t1_flair"]["global"]["flair"]),
            "t1_below_t2_flair": bool(
                gate_map["t2_t1_flair"]["global"]["t1"] < gate_map["t2_t1_flair"]["global"]["t2"]
                and gate_map["t2_t1_flair"]["global"]["t1"] < gate_map["t2_t1_flair"]["global"]["flair"]
            ),
        },
    }

    threshold_examples = {
        "t2_only": threshold_dispatch_for_combo(calibration, ("t2",), calibration["threshold"]),
        "t1_only": threshold_dispatch_for_combo(calibration, ("t1",), calibration["threshold"]),
        "full": threshold_dispatch_for_combo(calibration, tuple(cfg["data"]["modalities"]), calibration["threshold"]),
    }

    with open(out_dir / "drop_t1_ablation.json", "w", encoding="utf-8") as f:
        json.dump(drop_t1_report, f, indent=2, ensure_ascii=False)
    pd.DataFrame(drop_t1_report["rows"]).to_csv(out_dir / "drop_t1_ablation.csv", index=False)
    pd.DataFrame(drop_t1_report["summary"]).to_csv(out_dir / "drop_t1_ablation_summary.csv", index=False)

    summary = {
        "forward_prob_hgg": float(out["prob"][0].item()),
        "base_train_step_loss": float(loss.item()),
        "targeted_finetune_step_loss": float(fine_tune_loss.item()),
        "num_eval_combos": len(eval_results),
        "num_hyperedges": float(out["stage_stats"][0, 0].item()),
        "use_prototype_edges": float(out["stage_stats"][0, 2].item()),
        "use_mask_aware_classifier": bool(model.get_classifier_info()["use_mask_aware_classifier"]),
        "mask_head_type": model.get_classifier_info()["mask_head_type"],
        "mask_order": model.get_classifier_info()["mask_order"],
        "use_mask_aware_node_fusion": bool(model.get_classifier_info()["use_mask_aware_node_fusion"]),
        "use_node_type_embed": bool(model.get_classifier_info()["use_node_type_embed"]),
        "no_t1ce_t1_penalty": float(model.get_classifier_info()["no_t1ce_t1_penalty"]),
        "mask_bias_full": [float(v) for v in out["mask_bias"][0].detach().cpu().tolist()],
        "full_gate_summary": summarize_gates(out["modality_gates"][0]),
        "calibration_mode": calibration["mode"],
        "calibrated_threshold": float(calibration["threshold"]),
        "threshold_has_t1ce": float(calibration["threshold_has_t1ce"]),
        "threshold_no_t1ce_no_t1": float(calibration["threshold_no_t1ce_no_t1"]),
        "threshold_no_t1ce_with_t1": float(calibration["threshold_no_t1ce_with_t1"]),
        "fallback_info": calibration.get("fallback_info", {}),
        "threshold_examples": threshold_examples,
        "drop_t1_ablation_rows": drop_t1_report["rows"],
        "drop_t1_ablation_summary": drop_t1_report["summary"],
        "calibration_metric": calibration["metric"],
        "validation_bal_acc": float(val_metrics["bal_acc"]),
        "selected_combo_gates": {
            "t2_t1": t2_t1_gate,
            "t1_flair": t1_flair_gate,
            "t2_flair": t2_flair_gate,
            "t2_t1_flair": gate_map["t2_t1_flair"]["global"],
        },
        "missing_modalities_zero": missing_modalities_zero,
        "t1_downweight_checks": t1_downweight_checks,
        "first_eval_combo": eval_results[0],
        "last_eval_combo": eval_results[-1],
    }
    with open(out_dir / "demo_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
