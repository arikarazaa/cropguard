"""
utils.py — Helper utilities for CropGuard
Includes: metric computation, confusion matrix plot, per-class F1 chart,
          Grad-CAM grid visualisation, model size summary.
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(preds: np.ndarray, targets: np.ndarray, class_names: list[str]) -> dict:
    """Compute accuracy, macro F1, and per-class F1 from prediction arrays."""
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    acc = accuracy_score(targets, preds)
    f1_macro = f1_score(targets, preds, average="macro", zero_division=0)
    f1_weighted = f1_score(targets, preds, average="weighted", zero_division=0)
    precision = precision_score(targets, preds, average="macro", zero_division=0)
    recall = recall_score(targets, preds, average="macro", zero_division=0)

    per_class_f1 = f1_score(targets, preds, average=None, zero_division=0)
    per_class = {class_names[i]: float(per_class_f1[i]) for i in range(len(class_names))}

    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "precision_macro": precision,
        "recall_macro": recall,
        "per_class_f1": per_class,
    }


def print_classification_report(preds: np.ndarray, targets: np.ndarray, class_names: list[str]):
    print("\n" + "=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)
    print(classification_report(targets, preds, target_names=class_names, digits=3, zero_division=0))


# ── Visualisations ────────────────────────────────────────────────────────────
def plot_confusion_matrix(
    preds: np.ndarray,
    targets: np.ndarray,
    class_names: list[str],
    save_path: Optional[str] = None,
    top_n: int = 15,
) -> plt.Figure:
    """
    Plot normalised confusion matrix.
    Shows top_n most-confused classes for readability when many classes.
    """
    # Focus on top_n most frequent classes
    from collections import Counter
    counts = Counter(targets)
    top_classes = [i for i, _ in counts.most_common(top_n)]
    mask = np.isin(targets, top_classes)
    filtered_preds = preds[mask]
    filtered_targets = targets[mask]
    filtered_names = [class_names[i].split("___")[-1].replace("_", " ") for i in top_classes]

    cm = confusion_matrix(filtered_targets, filtered_preds, labels=top_classes, normalize="true")

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        cm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=filtered_names, yticklabels=filtered_names,
        ax=ax, linewidths=0.3, cbar_kws={"shrink": 0.8},
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(f"Confusion Matrix (top {top_n} classes, normalised)", fontsize=14)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved confusion matrix: {save_path}")

    return fig


def plot_per_class_f1(
    per_class_f1: dict[str, float],
    save_path: Optional[str] = None,
    top_n: int = 20,
) -> plt.Figure:
    """Horizontal bar chart of per-class F1 scores, sorted descending."""
    sorted_items = sorted(per_class_f1.items(), key=lambda x: x[1], reverse=True)
    if top_n:
        sorted_items = sorted_items[:top_n]

    labels = [k.split("___")[-1].replace("_", " ") for k, _ in sorted_items]
    values = [v for _, v in sorted_items]
    colors = ["#1D9E75" if v >= 0.95 else "#5DCAA5" if v >= 0.90 else "#FAC775" for v in values]

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.35)))
    bars = ax.barh(labels[::-1], values[::-1], color=colors[::-1], edgecolor="none", height=0.7)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("F1 Score", fontsize=11)
    ax.set_title("Per-class F1 Scores", fontsize=13)
    ax.axvline(0.95, color="#D85A30", linewidth=1, linestyle="--", alpha=0.7, label="0.95 threshold")
    ax.legend(fontsize=9)
    for bar, val in zip(bars, values[::-1]):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=8)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved F1 chart: {save_path}")

    return fig


def plot_gradcam_grid(
    image_paths: list[str],
    true_labels: list[str],
    pred_labels: list[str],
    heatmaps: list[np.ndarray],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot a grid of: original | heatmap | overlay for each sample.
    """
    n = len(image_paths)
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = [axes]

    col_titles = ["Original", "Grad-CAM Heatmap", "Overlay"]
    for col, title in enumerate(col_titles):
        axes[0][col].set_title(title, fontsize=12, fontweight="bold")

    for i, (img_path, true_lbl, pred_lbl, heatmap) in enumerate(
        zip(image_paths, true_labels, pred_labels, heatmaps)
    ):
        original = np.array(Image.open(img_path).convert("RGB"))
        original = cv2.resize(original, (224, 224))
        heatmap_resized = cv2.resize(heatmap, (224, 224))
        heatmap_colour = cv2.applyColorMap((heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap_colour = cv2.cvtColor(heatmap_colour, cv2.COLOR_BGR2RGB)
        overlay = (0.45 * heatmap_colour + 0.55 * original).astype(np.uint8)

        is_correct = true_lbl == pred_lbl
        colour = "green" if is_correct else "red"
        short_pred = pred_lbl.split("___")[-1].replace("_", " ")
        short_true = true_lbl.split("___")[-1].replace("_", " ")

        axes[i][0].imshow(original)
        axes[i][0].set_ylabel(f"True: {short_true}", fontsize=8)
        axes[i][0].axis("off")

        axes[i][1].imshow(heatmap_resized, cmap="jet")
        axes[i][1].axis("off")

        axes[i][2].imshow(overlay)
        axes[i][2].set_xlabel(f"Pred: {short_pred}", fontsize=8, color=colour)
        axes[i][2].axis("off")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_training_curves(history: dict, save_path: Optional[str] = None) -> plt.Figure:
    """Plot train/val loss, val accuracy, val F1."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs, history["train_loss"], label="Train", color="#D85A30", linewidth=2)
    axes[0].plot(epochs, history["val_loss"], label="Val", color="#1D9E75", linewidth=2)
    axes[0].set_title("Loss", fontsize=12); axes[0].legend(); axes[0].set_xlabel("Epoch")

    axes[1].plot(epochs, [a * 100 for a in history["val_acc"]], color="#378ADD", linewidth=2)
    axes[1].set_title("Val Accuracy (%)", fontsize=12); axes[1].set_xlabel("Epoch")

    axes[2].plot(epochs, history["val_f1"], color="#7F77DD", linewidth=2)
    axes[2].axhline(0.95, color="#D85A30", linestyle="--", alpha=0.6, label="0.95 target")
    axes[2].set_title("Val Macro F1", fontsize=12); axes[2].legend(); axes[2].set_xlabel("Epoch")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ── Model utils ───────────────────────────────────────────────────────────────
def model_summary(model: torch.nn.Module) -> dict:
    """Return param counts and estimated model size."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_mb = total * 4 / (1024 ** 2)  # float32

    print(f"Total parameters  : {total:,}")
    print(f"Trainable params  : {trainable:,}")
    print(f"Frozen params     : {total - trainable:,}")
    print(f"Model size (fp32) : {size_mb:.1f} MB")

    return {"total": total, "trainable": trainable, "size_mb": size_mb}


def export_onnx(model: torch.nn.Module, output_path: str, img_size: int = 224):
    """Export model to ONNX for edge deployment."""
    model.eval()
    dummy = torch.randn(1, 3, img_size, img_size)
    torch.onnx.export(
        model, dummy, output_path,
        export_params=True, opset_version=17,
        input_names=["image"], output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
    )
    size_mb = os.path.getsize(output_path) / (1024 ** 2)
    print(f"ONNX model saved: {output_path} ({size_mb:.1f} MB)")
