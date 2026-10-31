"""
Metrics and plots.

Headline metrics are BINARY with positive class = 'true' (cloudy), the
definition used by Chisari et al. (ResNet50: acc 89.57, P 0.90, R 0.96,
F1 0.93). Weighted averages are reported alongside for reference.
"""

import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

PAPER_RESNET50 = {"accuracy": 89.57, "f1": 0.93, "precision": 0.90, "recall": 0.96}


def compute_metrics(labels: np.ndarray, preds: np.ndarray) -> Dict[str, float]:
    """accuracy in percent (comparable with earlier W&B runs), the rest in [0, 1]."""
    labels, preds = np.asarray(labels), np.asarray(preds)
    return {
        "accuracy":           accuracy_score(labels, preds) * 100,
        "n_correct":          int((labels == preds).sum()),
        "n_total":            int(labels.size),
        "f1":                 f1_score(labels, preds, pos_label=1, zero_division=0),
        "precision":          precision_score(labels, preds, pos_label=1, zero_division=0),
        "recall":             recall_score(labels, preds, pos_label=1, zero_division=0),
        "f1_weighted":        f1_score(labels, preds, average="weighted", zero_division=0),
        "precision_weighted": precision_score(labels, preds, average="weighted", zero_division=0),
        "recall_weighted":    recall_score(labels, preds, average="weighted", zero_division=0),
    }


def plot_confusion_matrix(labels, preds, class_names: List[str], title: str, path: str = None) -> None:
    disp = ConfusionMatrixDisplay(confusion_matrix(labels, preds), display_labels=class_names)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(title)
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_training_history(history: Dict[str, List[float]], title: str, path: str = None) -> None:
    """`history` as returned by experiment.run_experiment (keys train_loss, val_loss, train_acc, val_acc)."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, metric in zip(axes, ("loss", "acc")):
        ax.plot(epochs, history[f"train_{metric}"], label="train")
        ax.plot(epochs, history[f"val_{metric}"], label="val")
        ax.set_xlabel("epoch")
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.4)
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()


def compare_models(results: Dict[str, Dict[str, float]], results_path: str) -> None:
    """Bar chart of accuracy and binary F1 (in %) across models."""
    names = list(results)
    x = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, [results[m]["accuracy"] for m in names], width, label="Accuracy (%)")
    ax.bar(x + width / 2, [results[m]["f1"] * 100 for m in names], width, label="F1 (%)")
    for bar in ax.patches:
        ax.annotate(f"{bar.get_height():.1f}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15)
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(axis="y", alpha=0.4)
    fig.tight_layout()
    fig.savefig(os.path.join(results_path, "model_comparison.png"), dpi=150, bbox_inches="tight")
    plt.show()
