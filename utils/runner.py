from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from utils.metrics import compute_binary_metrics
from utils.visualization import (
    save_average_roi_importance,
    save_case_overlay,
    save_confusion_matrix,
    save_roi_barplot,
    save_roi_drop_plot,
)

BRANCH_DROP_OPTIONS = {
    "full": {"prior": True, "modal": True, "knn": False},
    "drop_anatomy": {"prior": False, "modal": True, "knn": False},
    "drop_modal_aggregation": {"prior": True, "modal": False, "knn": False},
}


def move_batch_to_device(batch: Dict, device: torch.device) -> Dict:
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def choose_display_image(images: torch.Tensor, available_modalities: torch.Tensor) -> np.ndarray:
    available = available_modalities.detach().cpu().numpy()
    for idx, flag in enumerate(available.tolist()):
        if flag > 0.5:
            return images[idx].detach().cpu().numpy()
    return images[0].detach().cpu().numpy()


def save_metrics_files(metrics: Dict, output_dir: Path, filename_prefix: str = "metrics") -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / f"{filename_prefix}.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    pd.DataFrame([metrics]).to_csv(output_dir / f"{filename_prefix}.csv", index=False)


def _append_optional_stats(metrics: Dict, stage_stats: list) -> Dict:
    if stage_stats:
        stats = np.asarray(stage_stats, dtype=float)
        metrics["num_hyperedges"] = float(stats[:, 0].mean())
        metrics["use_anatomy_edges"] = float(stats[:, 1].mean())
        metrics["use_prototype_edges"] = float(stats[:, 2].mean())
    return metrics



def _append_classifier_info(metrics: Dict, model) -> Dict:
    if hasattr(model, "get_classifier_info"):
        info = model.get_classifier_info()
        metrics["use_mask_aware_classifier"] = bool(info.get("use_mask_aware_classifier", False))
        metrics["mask_head_type"] = str(info.get("mask_head_type", "none"))
        metrics["mask_order"] = ",".join(info.get("mask_order", []))
        metrics["use_mask_aware_node_fusion"] = bool(info.get("use_mask_aware_node_fusion", False))
        metrics["use_node_type_embed"] = bool(info.get("use_node_type_embed", False))
        metrics["no_t1ce_t1_penalty"] = float(info.get("no_t1ce_t1_penalty", 0.0))
    return metrics



def _append_gate_stats(metrics: Dict, modality_gates: np.ndarray, roi_names: Sequence[str], mask_order: Sequence[str]) -> Dict:
    if modality_gates.size == 0:
        return metrics
    gate_mean = modality_gates.mean(axis=(0, 1))
    for mod_idx, mod_name in enumerate(mask_order):
        metrics[f"avg_gate_{mod_name}"] = float(gate_mean[mod_idx])
    for roi_idx, roi_name in enumerate(roi_names):
        for mod_idx, mod_name in enumerate(mask_order):
            metrics[f"{roi_name}_avg_gate_{mod_name}"] = float(modality_gates[:, roi_idx, mod_idx].mean())
    return metrics


def run_epoch(model, loader, device, criterion=None, optimizer=None, grad_clip: float | None = None, threshold: float = 0.5, desc: str = "epoch"):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    y_true, y_prob = [], []
    stage_stats = []

    iterator = tqdm(loader, desc=desc, leave=False)
    for batch in iterator:
        batch = move_batch_to_device(batch, device)
        with torch.set_grad_enabled(training):
            output = model(batch)
            loss = criterion(output["logits"], batch["label"]) if criterion is not None else torch.tensor(0.0, device=device)
            if training:
                optimizer.zero_grad()
                loss.backward()
                if grad_clip is not None and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

        total_loss += float(loss.item()) * batch["label"].shape[0]
        y_true.extend(batch["label"].detach().cpu().tolist())
        y_prob.extend(output["prob"].detach().cpu().tolist())
        if "stage_stats" in output:
            stage_stats.extend(output["stage_stats"].detach().cpu().tolist())

    metrics = compute_binary_metrics(y_true, y_prob, threshold=threshold)
    metrics["loss"] = total_loss / max(len(loader.dataset), 1)
    metrics = _append_optional_stats(metrics, stage_stats)
    metrics = _append_classifier_info(metrics, model)
    return metrics


def compute_modality_auxiliary_loss(output, batch, criterion, modalities: Sequence[str]):
    """Average weighted CE over modality heads that have observed samples."""
    auxiliary_logits = output.get("aux_logits")
    if auxiliary_logits is None:
        raise RuntimeError("Auxiliary training requires model output['aux_logits'].")

    modality_losses = {}
    modality_counts = {}
    observed_losses = []
    availability = batch["available_modalities"]
    labels = batch["label"]
    for mod_idx, modality in enumerate(modalities):
        observed = availability[:, mod_idx] > 0.5
        count = int(observed.sum().item())
        modality_counts[modality] = count
        if count == 0:
            modality_losses[modality] = None
            continue
        modality_loss = criterion(auxiliary_logits[modality][observed], labels[observed])
        modality_losses[modality] = modality_loss
        observed_losses.append(modality_loss)

    if not observed_losses:
        raise RuntimeError("Auxiliary loss received an empty-modality training batch.")
    return torch.stack(observed_losses).mean(), modality_losses, modality_counts


def run_auxiliary_epoch(
    model,
    loader,
    device,
    criterion,
    optimizer,
    lambda_aux: float = 0.1,
    grad_clip: float | None = None,
    threshold: float = 0.5,
    desc: str = "auxiliary train",
):
    """Run E1 single-view training with observed-modality auxiliary classification."""
    model.train(True)
    total_main_loss = 0.0
    total_aux_loss = 0.0
    total_loss = 0.0
    y_true, y_prob = [], []
    stage_stats = []
    modalities = list(model.modalities)
    modality_loss_sums = {modality: 0.0 for modality in modalities}
    modality_counts = {modality: 0 for modality in modalities}

    for batch in tqdm(loader, desc=desc, leave=False):
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad()
        output = model(batch)
        main_loss = criterion(output["logits"], batch["label"])
        auxiliary_loss, per_modality_losses, per_modality_counts = compute_modality_auxiliary_loss(
            output, batch, criterion, modalities
        )
        loss = main_loss + float(lambda_aux) * auxiliary_loss
        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        batch_size = batch["label"].shape[0]
        total_main_loss += float(main_loss.item()) * batch_size
        total_aux_loss += float(auxiliary_loss.item()) * batch_size
        total_loss += float(loss.item()) * batch_size
        for modality in modalities:
            count = per_modality_counts[modality]
            modality_counts[modality] += count
            if count:
                modality_loss_sums[modality] += float(per_modality_losses[modality].item()) * count
        y_true.extend(batch["label"].detach().cpu().tolist())
        y_prob.extend(output["prob"].detach().cpu().tolist())
        if "stage_stats" in output:
            stage_stats.extend(output["stage_stats"].detach().cpu().tolist())

    num_samples = max(len(loader.dataset), 1)
    metrics = compute_binary_metrics(y_true, y_prob, threshold=threshold)
    metrics["loss"] = total_loss / num_samples
    metrics["main_loss"] = total_main_loss / num_samples
    metrics["aux_loss"] = total_aux_loss / num_samples
    metrics["total_loss"] = total_loss / num_samples
    metrics["lambda_aux"] = float(lambda_aux)
    for modality in modalities:
        count = modality_counts[modality]
        metrics[f"aux_{modality}_loss"] = modality_loss_sums[modality] / count if count else float("nan")
        metrics[f"aux_{modality}_count"] = count
    metrics = _append_optional_stats(metrics, stage_stats)
    metrics = _append_classifier_info(metrics, model)
    return metrics


def run_dual_view_epoch(
    model,
    loader,
    device,
    criterion,
    optimizer,
    lambda_missing: float = 1.0,
    grad_clip: float | None = None,
    threshold: float = 0.5,
    desc: str = "dual-view train",
):
    """Train on same-subject full/missing views while sharing the loaded image and ROI tensors."""
    model.train(True)
    total_loss = 0.0
    total_full_loss = 0.0
    total_missing_loss = 0.0
    y_true, full_prob, missing_prob = [], [], []
    stage_stats = []

    group_candidates = loader.dataset.targeted_missing_groups()
    subgroup_counts = Counter({name: 0 for name in group_candidates})
    pattern_counts = Counter({"_".join(combo): 0 for combos in group_candidates.values() for combo in combos})

    for batch in tqdm(loader, desc=desc, leave=False):
        if "missing_available_modalities" not in batch:
            raise RuntimeError("Dual-view training batch is missing the targeted availability mask.")
        batch = move_batch_to_device(batch, device)
        missing_mask = batch["missing_available_modalities"].float()
        broadcast_shape = [missing_mask.shape[0], missing_mask.shape[1]] + [1] * (batch["images"].ndim - 2)
        missing_batch = dict(batch)
        missing_batch["images"] = batch["images"] * missing_mask.view(*broadcast_shape)
        missing_batch["available_modalities"] = missing_mask
        missing_batch["combo"] = batch["missing_combo"]

        optimizer.zero_grad()
        full_output = model(batch)
        missing_output = model(missing_batch)
        full_loss = criterion(full_output["logits"], batch["label"])
        missing_loss = criterion(missing_output["logits"], batch["label"])
        loss = full_loss + float(lambda_missing) * missing_loss
        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        batch_size = batch["label"].shape[0]
        total_loss += float(loss.item()) * batch_size
        total_full_loss += float(full_loss.item()) * batch_size
        total_missing_loss += float(missing_loss.item()) * batch_size
        y_true.extend(batch["label"].detach().cpu().tolist())
        full_prob.extend(full_output["prob"].detach().cpu().tolist())
        missing_prob.extend(missing_output["prob"].detach().cpu().tolist())
        if "stage_stats" in missing_output:
            stage_stats.extend(missing_output["stage_stats"].detach().cpu().tolist())
        subgroup_counts.update(batch["targeted_subgroup"])
        pattern_counts.update("_".join(combo) for combo in batch["missing_combo"])

    num_samples = max(len(loader.dataset), 1)
    metrics = compute_binary_metrics(y_true, missing_prob, threshold=threshold)
    full_metrics = compute_binary_metrics(y_true, full_prob, threshold=threshold)
    metrics.update({f"full_{name}": value for name, value in full_metrics.items()})
    metrics["loss"] = total_loss / num_samples
    metrics["loss_full"] = total_full_loss / num_samples
    metrics["loss_missing"] = total_missing_loss / num_samples
    metrics["lambda_missing"] = float(lambda_missing)
    metrics = _append_optional_stats(metrics, stage_stats)
    metrics = _append_classifier_info(metrics, model)

    sampled_total = sum(subgroup_counts.values())
    cfg = getattr(loader.dataset, "curriculum_config", {})
    configured_ratios = {
        "full": float(cfg.get("targeted_ratio_full", 0.30)),
        "t1ce_absent_t2_present": float(cfg.get("targeted_ratio_t1ce_absent_t2_present", 0.25)),
        "t2_absent_t1ce_present": float(cfg.get("targeted_ratio_t2_absent_t1ce_present", 0.25)),
        "t1ce_t2_both_absent": float(cfg.get("targeted_ratio_t1ce_t2_both_absent", 0.20)),
    }
    sampling = {
        "num_samples": sampled_total,
        "configured_ratios": configured_ratios,
        "subgroup_counts": dict(subgroup_counts),
        "subgroup_frequencies": {name: count / max(sampled_total, 1) for name, count in subgroup_counts.items()},
        "pattern_counts": dict(pattern_counts),
        "pattern_frequencies": {name: count / max(sampled_total, 1) for name, count in pattern_counts.items()},
    }
    return {"metrics": metrics, "sampling": sampling}


@torch.no_grad()
def collect_predictions(model, loader, device, branch_override=None, explain_num_cases: int = 0):
    y_true, y_prob = [], []
    roi_scores_all, stage_stats_all = [], []
    modality_gates_all = []
    combos_all = []
    case_payloads = []
    exported = 0
    for batch in tqdm(loader, desc="evaluate", leave=False):
        batch = move_batch_to_device(batch, device)
        output = model(batch, branch_override=branch_override)
        probs = output["prob"].detach().cpu().numpy()
        attn = output["roi_attention"].detach().cpu().numpy()
        labels = batch["label"].detach().cpu().numpy()
        y_true.extend(labels.tolist())
        y_prob.extend(probs.tolist())
        roi_scores_all.extend(attn.tolist())
        if "stage_stats" in output:
            stage_stats_all.extend(output["stage_stats"].detach().cpu().tolist())
        if "modality_gates" in output:
            modality_gates_all.extend(output["modality_gates"].detach().cpu().tolist())
        combos_all.extend([tuple(c) for c in batch.get("combo", [])])
        if exported < explain_num_cases:
            batch_size = len(batch["case_id"])
            for i in range(batch_size):
                if exported >= explain_num_cases:
                    break
                case_payloads.append({
                    "case_id": batch["case_id"][i],
                    "images": batch["images"][i].detach().cpu(),
                    "available_modalities": batch["available_modalities"][i].detach().cpu(),
                    "roi_masks": batch["roi_masks"][i].detach().cpu(),
                    "attention": attn[i],
                })
                exported += 1
    return {
        "y_true": np.asarray(y_true, dtype=int),
        "y_prob": np.asarray(y_prob, dtype=float),
        "roi_scores": np.asarray(roi_scores_all, dtype=float),
        "stage_stats": np.asarray(stage_stats_all, dtype=float) if stage_stats_all else np.zeros((0, 3), dtype=float),
        "modality_gates": np.asarray(modality_gates_all, dtype=float) if modality_gates_all else np.zeros((0, 0, 0), dtype=float),
        "combos": combos_all,
        "case_payloads": case_payloads,
    }


def evaluate_with_explanations(
    model,
    loader,
    device,
    roi_names: Sequence[str],
    output_dir: str,
    threshold: float = 0.5,
    threshold_group: str = "global",
    explain_num_cases: int = 3,
    roi_drop_enabled: bool = True,
    edge_type_drop_enabled: bool = True,
    class_names=None,
):
    model.eval()
    output_dir = Path(output_dir)
    case_dir = output_dir / "case_explanations"
    case_dir.mkdir(parents=True, exist_ok=True)

    collected = collect_predictions(model, loader, device, explain_num_cases=explain_num_cases)
    y_true = collected["y_true"]
    y_prob = collected["y_prob"]
    roi_scores = collected["roi_scores"]
    stage_stats = collected["stage_stats"]
    modality_gates = collected["modality_gates"]

    metrics = compute_binary_metrics(y_true, y_prob, threshold=threshold)
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    display_names = list(class_names if class_names is not None else ["LGG(0)", "HGG(1)"])
    save_confusion_matrix(cm, display_names, output_dir / "confusion_matrix.png")
    metrics = _append_optional_stats(metrics, stage_stats.tolist() if len(stage_stats) > 0 else [])
    metrics = _append_classifier_info(metrics, model)
    metrics["threshold"] = float(threshold)
    metrics["threshold_group"] = threshold_group
    metrics["applied_threshold"] = float(threshold)
    mask_order = model.get_classifier_info().get("mask_order", []) if hasattr(model, "get_classifier_info") else []
    metrics = _append_gate_stats(metrics, modality_gates, roi_names, mask_order)
    save_metrics_files(metrics, output_dir)

    if modality_gates.size > 0 and hasattr(model, "get_classifier_info"):
        gate_rows = []
        mask_order = model.get_classifier_info().get("mask_order", [])
        global_gate = modality_gates.mean(axis=(0, 1))
        gate_rows.append({"node": "global", **{f"avg_gate_{mod}": float(global_gate[idx]) for idx, mod in enumerate(mask_order)}})
        for roi_idx, roi_name in enumerate(roi_names):
            gate_rows.append({"node": roi_name, **{f"avg_gate_{mod}": float(modality_gates[:, roi_idx, idx].mean()) for idx, mod in enumerate(mask_order)}})
        pd.DataFrame(gate_rows).to_csv(output_dir / "gating_stats.csv", index=False)

    if len(roi_scores) > 0:
        roi_mean = roi_scores.mean(axis=0)
        roi_std = roi_scores.std(axis=0)
    else:
        roi_mean = np.zeros(len(roi_names), dtype=float)
        roi_std = np.zeros(len(roi_names), dtype=float)
    roi_df = pd.DataFrame({"roi": list(roi_names), "mean_importance": roi_mean, "std_importance": roi_std})
    roi_df.to_csv(output_dir / "roi_importance_stats.csv", index=False)
    roi_df.to_csv(output_dir / "node_importance.csv", index=False)
    save_average_roi_importance(roi_names, roi_scores.tolist() if len(roi_scores) > 0 else [], output_dir / "avg_roi_importance.csv", output_dir / "avg_roi_importance.png")

    for payload in collected["case_payloads"]:
        case_id = payload["case_id"]
        attention = payload["attention"]
        score_map = {roi_names[j]: float(attention[j]) for j in range(len(roi_names))}
        save_roi_barplot(roi_names, attention, f"ROI importance: {case_id}", case_dir / f"{case_id}_roi_importance.png")
        image_3d = choose_display_image(payload["images"], payload["available_modalities"])
        roi_masks = {roi_names[j]: payload["roi_masks"][j].numpy() for j in range(len(roi_names))}
        save_case_overlay(image_3d, roi_masks, score_map, case_dir / f"{case_id}_overlay.png")

    if roi_drop_enabled:
        roi_drop_results = roi_drop_analysis(model, loader, device, roi_names)
        roi_drop_df = pd.DataFrame({
            "roi": list(roi_names),
            "mean_prob_drop": roi_drop_results.mean(axis=0) if len(roi_drop_results) > 0 else np.zeros(len(roi_names)),
            "std_prob_drop": roi_drop_results.std(axis=0) if len(roi_drop_results) > 0 else np.zeros(len(roi_names)),
        })
        roi_drop_df.to_csv(output_dir / "roi_drop.csv", index=False)
        save_roi_drop_plot(roi_drop_df, output_dir / "roi_drop.png", title="ROI dropping probability delta")

    if edge_type_drop_enabled:
        enabled = model.get_enabled_branches()
        if enabled["prior"] and enabled["modal"]:
            edge_drop_rows = []
            for name, override in BRANCH_DROP_OPTIONS.items():
                drop_collected = collect_predictions(model, loader, device, branch_override=override)
                drop_metrics = compute_binary_metrics(drop_collected["y_true"], drop_collected["y_prob"], threshold=threshold)
                drop_metrics = _append_optional_stats(drop_metrics, drop_collected["stage_stats"].tolist() if len(drop_collected["stage_stats"]) > 0 else [])
                drop_metrics = _append_classifier_info(drop_metrics, model)
                drop_metrics["threshold"] = float(threshold)
                drop_metrics["threshold_group"] = threshold_group
                drop_metrics["applied_threshold"] = float(threshold)
                drop_metrics["setting"] = name
                edge_drop_rows.append(drop_metrics)
            pd.DataFrame(edge_drop_rows).to_csv(output_dir / "edge_type_drop_metrics.csv", index=False)

    return metrics


@torch.no_grad()
def roi_drop_analysis(model, loader, device, roi_names: Sequence[str]) -> np.ndarray:
    deltas = []
    for batch in tqdm(loader, desc="roi_drop", leave=False):
        batch = move_batch_to_device(batch, device)
        base = model(batch)["prob"]
        batch_deltas = []
        for roi_idx in range(len(roi_names)):
            drop_mask = torch.ones_like(batch["roi_valid"])
            drop_mask[:, roi_idx] = 0.0
            dropped = model(batch, roi_drop_mask=drop_mask)["prob"]
            batch_deltas.append((base - dropped).detach().cpu().numpy())
        batch_deltas = np.stack(batch_deltas, axis=1)
        deltas.append(batch_deltas)
    if not deltas:
        return np.zeros((0, len(roi_names)), dtype=float)
    return np.concatenate(deltas, axis=0)
