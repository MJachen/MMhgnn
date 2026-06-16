from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def save_confusion_matrix(cm: np.ndarray, labels: List[str], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_roi_barplot(roi_names: Iterable[str], scores: Iterable[float], title: str, path: str | Path) -> None:
    roi_names = list(roi_names)
    scores = np.asarray(list(scores), dtype=float)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    order = np.argsort(scores)[::-1]
    ax.bar(np.array(roi_names)[order], scores[order], color="#1f77b4")
    ax.set_ylabel("Importance")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_average_roi_importance(roi_names: Iterable[str], all_scores: List[Iterable[float]], csv_path: str | Path, fig_path: str | Path) -> pd.DataFrame:
    roi_names = list(roi_names)
    scores = np.asarray([list(item) for item in all_scores], dtype=float)
    mean_scores = scores.mean(axis=0) if len(scores) > 0 else np.zeros(len(roi_names), dtype=float)
    df = pd.DataFrame({"roi": roi_names, "mean_importance": mean_scores})
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    save_roi_barplot(roi_names, mean_scores, "Average ROI Importance", fig_path)
    return df


def _get_slice(image_3d: np.ndarray, slice_axis: int = 2):
    index = image_3d.shape[slice_axis] // 2
    slicer = [slice(None), slice(None), slice(None)]
    slicer[slice_axis] = index
    return image_3d[tuple(slicer)], tuple(slicer)


def save_case_overlay(image_3d: np.ndarray, roi_masks: Dict[str, np.ndarray], roi_scores: Dict[str, float], path: str | Path, slice_axis: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image_slice, slicer = _get_slice(image_3d, slice_axis=slice_axis)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image_slice.T, cmap="gray", origin="lower")
    cmap = plt.get_cmap("tab10")
    for idx, (roi_name, mask_3d) in enumerate(roi_masks.items()):
        if mask_3d.sum() == 0:
            continue
        mask_slice = mask_3d[tuple(slicer)]
        alpha = float(np.clip(roi_scores.get(roi_name, 0.0), 0.0, 1.0))
        if alpha <= 0:
            continue
        overlay = np.zeros((*mask_slice.T.shape, 4), dtype=float)
        color = cmap(idx % 10)
        overlay[..., :3] = color[:3]
        overlay[..., 3] = mask_slice.T.astype(float) * (0.15 + 0.55 * alpha)
        ax.imshow(overlay, origin="lower")

    ax.set_title("ROI contribution overlay (middle slice)")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_case_partition_visualization(image_3d: np.ndarray, seg_3d: np.ndarray, roi_masks: Dict[str, np.ndarray], path: str | Path, slice_axis: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image_slice, slicer = _get_slice(image_3d, slice_axis=slice_axis)
    seg_slice = seg_3d[tuple(slicer)]

    original_regions = {
        "NCR (seg==1)": seg_slice == 1,
        "ED (seg==2)": seg_slice == 2,
        "ET (seg==4)": seg_slice == 4,
    }
    roi_slices = {name: mask[tuple(slicer)] > 0.5 for name, mask in roi_masks.items()}
    panels = [("Base image", None)] + list(original_regions.items()) + list(roi_slices.items())
    ncols = 4
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.atleast_1d(axes).reshape(nrows, ncols)

    for ax, (title, mask) in zip(axes.flat, panels):
        ax.imshow(image_slice.T, cmap="gray", origin="lower")
        if mask is not None and np.asarray(mask).sum() > 0:
            overlay = np.zeros((*mask.T.shape, 4), dtype=float)
            overlay[..., :3] = np.array([1.0, 0.4, 0.0])
            overlay[..., 3] = mask.T.astype(float) * 0.45
            ax.imshow(overlay, origin="lower")
        ax.set_title(title)
        ax.axis("off")

    for ax in axes.flat[len(panels):]:
        ax.axis("off")

    fig.suptitle("BraTS original regions and ROI node partition", fontsize=16)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_roi_drop_plot(df: pd.DataFrame, path: str | Path, title: str = "ROI drop") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(df["roi"], df["mean_prob_drop"], yerr=df.get("std_prob_drop"), color="#d62728", alpha=0.8)
    ax.set_ylabel("Mean probability drop")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_metric_comparison(df: pd.DataFrame, metric: str, path: str | Path, title: str | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(df["experiment"], df[metric], color="#1f77b4")
    ax.set_ylabel(metric.upper())
    ax.set_title(title or f"{metric.upper()} comparison")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_dual_metric_comparison(df: pd.DataFrame, metric_a: str, metric_b: str, path: str | Path, title: str | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(df))
    width = 0.38
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x - width / 2, df[metric_a], width, label=metric_a.upper())
    ax.bar(x + width / 2, df[metric_b], width, label=metric_b.upper())
    ax.set_xticks(x)
    ax.set_xticklabels(df["experiment"], rotation=35)
    ax.set_title(title or f"{metric_a.upper()}/{metric_b.upper()} comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_heatmap(df: pd.DataFrame, index_col: str, path: str | Path, title: str = "Heatmap") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pivot = df.set_index(index_col)
    fig, ax = plt.subplots(figsize=(max(8, pivot.shape[1]), max(4, pivot.shape[0] * 0.5)))
    sns.heatmap(pivot, cmap="viridis", annot=True, fmt=".3f", ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_fusion_weight_history(history_df: pd.DataFrame, path: str | Path, title: str = "Fusion weight history") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    for column, color in [("alpha", "#1f77b4"), ("beta", "#ff7f0e"), ("gamma", "#2ca02c")]:
        if column in history_df.columns:
            ax.plot(history_df["epoch"], history_df[column], label=column, color=color)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Weight")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
