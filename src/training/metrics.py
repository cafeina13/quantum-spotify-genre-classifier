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


# ---------------------------------------------------------------------------
# Training progression chart (all runs)
# ---------------------------------------------------------------------------

# Known results from all training runs — update after each new run.
# val_acc values taken from best validation epoch per run.
_RUN_HISTORY = [
    {"run": 1, "label": "Run 1\n(PCA + [0,π])",              "hybrid": 33.0, "baseline": 51.3},
    {"run": 2, "label": "Run 2\n(feat. selection + [-π,π])", "hybrid": 39.6, "baseline": 51.3},
    {"run": 3, "label": "Run 3\n(wide encoder + decoder)",   "hybrid": 48.7, "baseline": 54.2},
    {"run": 4, "label": "Run 4\n(+ReduceLR p=4 f=0.5)",      "hybrid": 51.3, "baseline": 54.7},
    {"run": 5, "label": "Run 5\n(+ReduceLR p=2 f=0.3)",      "hybrid": 48.6, "baseline": 54.2},
]


def plot_training_progression(
    new_run: dict = None,
    save_path: Path = None,
) -> None:
    """
    Bar chart showing hybrid vs classical val accuracy across all training runs.
    Visualises the narrowing gap as the architecture improved.

    Parameters
    ----------
    new_run   : optional dict {"run": int, "label": str, "hybrid": float, "baseline": float}
                Pass the latest run's results to append them automatically.
    save_path : if provided, save figure here (PNG)
    """
    runs = list(_RUN_HISTORY)
    if new_run is not None:
        runs.append(new_run)

    labels   = [r["label"]   for r in runs]
    hybrid   = [r["hybrid"]   for r in runs]
    baseline = [r["baseline"] for r in runs]
    gaps     = [b - h for h, b in zip(hybrid, baseline)]

    x     = np.arange(len(runs))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training Progression — All Runs", fontsize=14)

    # Left: grouped bar chart
    ax = axes[0]
    bars_h = ax.bar(x - width/2, hybrid,   width, label="Hybrid QNN",        color="steelblue")
    bars_b = ax.bar(x + width/2, baseline, width, label="Classical Baseline", color="darkorange")

    # Annotate bars
    for bar in bars_h:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=8)
    for bar in bars_b:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Training Run")
    ax.set_ylabel("Validation Accuracy (%)")
    ax.set_title("Val Accuracy per Run")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 70)
    ax.axhline(16.7, color="gray", linestyle="--", linewidth=0.8, label="Random (16.7%)")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    # Right: gap line chart
    ax2 = axes[1]
    ax2.plot(range(1, len(runs)+1), gaps, marker="o", color="crimson", linewidth=2, markersize=7)
    for i, (g, r) in enumerate(zip(gaps, runs)):
        ax2.text(i+1, g + 0.3, f"{g:.1f}%", ha="center", va="bottom", fontsize=9)

    ax2.set_xlabel("Training Run")
    ax2.set_ylabel("Gap: Baseline − Hybrid (%)")
    ax2.set_title("Narrowing Accuracy Gap")
    ax2.set_xticks(range(1, len(runs)+1))
    ax2.set_xticklabels([r["label"] for r in runs], fontsize=8)
    ax2.set_ylim(0, max(gaps) + 5)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Training progression figure saved to '{save_path}'.")

    plt.show()
