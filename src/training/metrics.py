"""
metrics.py — Evaluation metrics and result visualisation.

Functions
---------
compute_metrics       : accuracy, per-class precision/recall/F1, confusion matrix
plot_training_history : 4-panel training curve figure
plot_confusion_matrix : seaborn heatmap (row-normalised for recall per class)
compare_models        : side-by-side F1 bar chart for hybrid vs. classical
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

from src.config import CFG


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def compute_metrics(
    model: nn.Module,
    loader: DataLoader,
    label_encoder: LabelEncoder,
    device: str = "cpu",
) -> dict:
    """
    Evaluate the model on a DataLoader split.

    Parameters
    ----------
    model         : trained nn.Module (hybrid or baseline)
    loader        : DataLoader for the split to evaluate
    label_encoder : fitted LabelEncoder (for human-readable class names in report)
    device        : "cpu" or "cuda"

    Returns
    -------
    dict with keys:
        "accuracy"         : float
        "report"           : str — sklearn classification_report
        "confusion_matrix" : np.ndarray, shape (n_classes, n_classes)
        "y_true"           : np.ndarray of true labels
        "y_pred"           : np.ndarray of predicted labels
    """
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            logits  = model(X_batch)
            preds   = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_batch.numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    accuracy = (y_true == y_pred).mean()
    cm       = confusion_matrix(y_true, y_pred)
    report   = classification_report(
        y_true, y_pred,
        target_names=label_encoder.classes_,
        zero_division=0,
    )

    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("\nClassification Report:\n")
    print(report)

    return {
        "accuracy":         accuracy,
        "report":           report,
        "confusion_matrix": cm,
        "y_true":           y_true,
        "y_pred":           y_pred,
    }


# ---------------------------------------------------------------------------
# Training history plot
# ---------------------------------------------------------------------------

def plot_training_history(
    history: dict,
    save_path: Path = None,
    title: str = "Training History",
) -> None:
    """
    4-panel figure: train/val loss (left) and train/val accuracy (right).

    Parameters
    ----------
    history   : dict returned by Trainer.fit()
    save_path : if provided, save the figure here (PNG)
    title     : figure super-title
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(title, fontsize=14)

    # Loss subplot
    axes[0].plot(epochs, history["train_loss"], label="Train Loss", marker="o", markersize=3)
    axes[0].plot(epochs, history["val_loss"],   label="Val Loss",   marker="o", markersize=3)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy subplot
    axes[1].plot(epochs, history["train_acc"], label="Train Acc", marker="o", markersize=3)
    axes[1].plot(epochs, history["val_acc"],   label="Val Acc",   marker="o", markersize=3)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Training history figure saved to '{save_path}'.")

    plt.show()


# ---------------------------------------------------------------------------
# Confusion matrix plot
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list,
    title: str = "Confusion Matrix",
    save_path: Path = None,
    normalise: bool = True,
) -> None:
    """
    Seaborn heatmap of a confusion matrix.

    Parameters
    ----------
    cm          : confusion matrix from sklearn.metrics.confusion_matrix
    class_names : list of genre name strings
    title       : plot title
    save_path   : if provided, save figure here (PNG)
    normalise   : if True, normalise by row (shows recall per class)
    """
    if normalise:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_plot  = cm.astype(float) / np.where(row_sums == 0, 1, row_sums)
        fmt      = ".2f"
        vmax     = 1.0
    else:
        cm_plot = cm
        fmt     = "d"
        vmax    = None

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm_plot,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        vmin=0,
        vmax=vmax,
        ax=ax,
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(title)
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Confusion matrix saved to '{save_path}'.")

    plt.show()


# ---------------------------------------------------------------------------
# Model comparison plot
# ---------------------------------------------------------------------------

def compare_models(
    hybrid_metrics: dict,
    baseline_metrics: dict,
    label_encoder: LabelEncoder,
    save_path: Path = None,
) -> None:
    """
    Side-by-side F1 bar chart comparing hybrid vs. classical baseline per genre.
    Also prints a summary accuracy table to stdout.

    Parameters
    ----------
    hybrid_metrics   : dict from compute_metrics() for the hybrid model
    baseline_metrics : dict from compute_metrics() for the classical baseline
    label_encoder    : to get genre class names
    save_path        : if provided, save figure here (PNG)
    """
    from sklearn.metrics import f1_score

    class_names = list(label_encoder.classes_)
    n_classes   = len(class_names)

    hybrid_f1   = f1_score(hybrid_metrics["y_true"],   hybrid_metrics["y_pred"],   average=None)
    baseline_f1 = f1_score(baseline_metrics["y_true"], baseline_metrics["y_pred"], average=None)

    x = np.arange(n_classes)
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width/2, hybrid_f1,   width, label="Hybrid QNN",        color="steelblue")
    ax.bar(x + width/2, baseline_f1, width, label="Classical Baseline", color="darkorange")

    ax.set_xlabel("Genre")
    ax.set_ylabel("F1 Score")
    ax.set_title("Per-Genre F1: Hybrid QNN vs Classical Baseline")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=15)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    # Print summary table
    print("\n=== Model Comparison Summary ===")
    print(f"{'Genre':<12}  {'Hybrid F1':>10}  {'Baseline F1':>12}")
    print("-" * 38)
    for genre, hf1, bf1 in zip(class_names, hybrid_f1, baseline_f1):
        print(f"{genre:<12}  {hf1:>10.4f}  {bf1:>12.4f}")
    print("-" * 38)
    print(f"{'Overall Acc':<12}  {hybrid_metrics['accuracy']:>10.4f}  {baseline_metrics['accuracy']:>12.4f}")

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\nComparison figure saved to '{save_path}'.")

    plt.show()
