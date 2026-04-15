"""
Evaluation module for ceilometer cloud detection models.
Computes metrics, generates plots and saves results for comparison.
"""

import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from typing import Dict, List, Tuple


def get_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device
) -> Tuple[List[int], List[int]]:
    """
    Runs inference on a DataLoader and returns predictions and ground truth.

    Args:
        model  (nn.Module)   : trained model to evaluate.
        loader (DataLoader)  : data loader (val or test).
        device (torch.device): 'cuda' or 'cpu'.

    Returns:
        Tuple[List[int], List[int]]:
            - all_preds  : predicted labels
            - all_labels : ground truth labels
    """

    model.eval()

    all_preds: List[int] = []
    all_labels: List[int] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs: torch.Tensor = model(images)
            predicted: torch.Tensor = outputs.argmax(dim=1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return all_preds, all_labels


def compute_metrics(
    all_preds: List[int],
    all_labels: List[int],
    model_name: str
) -> Dict[str, float]:
    """
    Computes classification metrics from predictions and ground truth.
    These are the same metrics used in the paper for comparison.

    Args:
        all_preds  (List[int]) : predicted labels from get_predictions().
        all_labels (List[int]) : ground truth labels from get_predictions().
        model_name (str)       : model name, used for printing results.

    Returns:
        Dict[str, float]: dictionary with keys:
            - 'accuracy'  : overall accuracy (0-100)
            - 'f1'        : F1 score (0-1)
            - 'precision' : precision (0-1)
            - 'recall'    : recall (0-1)
    """

    accuracy: float  = accuracy_score(all_labels, all_preds) * 100
    f1: float        = f1_score(all_labels, all_preds, average='weighted')
    precision: float = precision_score(all_labels, all_preds, average='weighted')
    recall: float    = recall_score(all_labels, all_preds, average='weighted')

    print(f"\nResults for {model_name}:")
    print(f"  Accuracy:  {accuracy:.2f}%")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")

    metrics: Dict[str, float] = {
        'accuracy':  accuracy,
        'f1':        f1,
        'precision': precision,
        'recall':    recall
    }

    return metrics


def plot_confusion_matrix(
    all_preds: List[int],
    all_labels: List[int],
    class_names: List[str],
    model_name: str,
    results_path: str
) -> None:
    """
    Plots and saves the confusion matrix to the results folder.

    Args:
        all_preds    (List[int])  : predicted labels.
        all_labels   (List[int])  : ground truth labels.
        class_names  (List[str])  : class names e.g. ['false', 'true'].
        model_name   (str)        : used for the plot title and filename.
        results_path (str)        : folder where to save the plot.
                                    e.g. '/content/project/results'
    """

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(f"Confusion Matrix — {model_name}")

    filepath: str = os.path.join(results_path, f"confusion_matrix_{model_name}.png")
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Confusion matrix saved: {filepath}")


def plot_training_history(
    history: dict,
    model_name: str,
    results_path: str
) -> None:
    """
    Plots and saves training and validation loss and accuracy curves.
    Useful to diagnose overfitting or underfitting.

    Args:
        history      (dict) : output of run_training() from train.py.
                              keys: train_loss, val_loss,
                                    train_accuracy, val_accuracy
        model_name   (str)  : used for the plot title and filename.
        results_path (str)  : folder where to save the plot.
    """

    epochs: List[int] = list(range(1, len(history['train_loss']) + 1))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # loss curves
    axes[0].plot(epochs, history['train_loss'], label='Train loss')
    axes[0].plot(epochs, history['val_loss'],   label='Val loss')
    axes[0].set_title(f"Loss — {model_name}")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    # accuracy curves
    axes[1].plot(epochs, history['train_accuracy'], label='Train accuracy')
    axes[1].plot(epochs, history['val_accuracy'],   label='Val accuracy')
    axes[1].set_title(f"Accuracy — {model_name}")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()
    axes[1].grid(True)

    plt.suptitle(model_name, fontsize=14)
    plt.tight_layout()

    filepath: str = os.path.join(results_path, f"training_history_{model_name}.png")
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Training history saved: {filepath}")


def compare_models(
    results: Dict[str, Dict[str, float]],
    results_path: str
) -> None:
    """
    Plots a bar chart comparing metrics across all tested models.
    This is the main comparison table for the final report.

    Args:
        results      (Dict[str, Dict[str, float]]) : dictionary where
                      keys are model names and values are metric dicts
                      from compute_metrics().
                      e.g. {
                          'resnet50':      {'accuracy': 89.57, ...},
                          'convnext_base': {'accuracy': 91.20, ...}
                      }
        results_path (str): folder where to save the plot.
    """

    model_names: List[str] = list(results.keys())
    accuracies: List[float] = [results[m]['accuracy'] for m in model_names]
    f1_scores: List[float]  = [results[m]['f1'] for m in model_names]

    x: np.ndarray = np.arange(len(model_names))
    width: float = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, accuracies, width, label='Accuracy (%)')
    ax.bar(x + width/2, [f * 100 for f in f1_scores], width, label='F1 Score (%)')

    ax.set_title("Model comparison — Ceilometer cloud detection")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15)
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(axis='y', alpha=0.5)

    # add value labels on top of each bar
    for bar in ax.patches:
        ax.annotate(
            f"{bar.get_height():.1f}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha='center', va='bottom', fontsize=9
        )

    plt.tight_layout()

    filepath: str = os.path.join(results_path, "model_comparison.png")
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Comparison chart saved: {filepath}")