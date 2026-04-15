"""
run_evaluation.py — Step 4: Evaluate and compare models.

Loads the best checkpoints from Step 3 and evaluates both models on the
held-out test set. Generates:
  - Confusion matrices for both models
  - Training history curves
  - Side-by-side F1 comparison chart
  - Summary accuracy table printed to stdout

Run from the project root:
  python scripts/run_evaluation.py
"""

import sys
import json
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for scripted multi-figure saves
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import CFG
from src.data.preprocessor import load_processed
from src.quantum.device import get_device
from src.models.hybrid_model import HybridGenreClassifier
from src.models.classical_baseline import ClassicalBaseline
from src.training.metrics import (
    compute_metrics,
    plot_training_history,
    plot_confusion_matrix,
    compare_models,
    plot_training_progression,
)


def load_model_from_checkpoint(model: nn.Module, checkpoint_path: Path) -> nn.Module:
    """Load saved weights into a model instance."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def main():
    print("=" * 60)
    print("Step 4 — Model Evaluation")
    print("=" * 60)

    data = load_processed(CFG.processed_dir)
    encoder = data["encoder"]

    # --- Test loaders ---
    def to_loader(X_np, y_np):
        return DataLoader(
            TensorDataset(
                torch.tensor(X_np, dtype=torch.float32),
                torch.tensor(y_np, dtype=torch.long),
            ),
            batch_size=CFG.batch_size,
            shuffle=False,
        )

    hybrid_test_loader   = to_loader(data["X_test"], data["y_test"])
    baseline_test_loader = to_loader(data["X_test"], data["y_test"])

    # --- Load hybrid model ---
    hybrid_path = Path(CFG.models_dir) / "hybrid_qnn_best.pt"
    hybrid_model = HybridGenreClassifier(
        n_qubits=CFG.n_qubits,
        n_layers=CFG.n_layers,
        device=get_device(),
    )
    hybrid_model = load_model_from_checkpoint(hybrid_model, hybrid_path)
    print(f"\nLoaded Hybrid QNN from '{hybrid_path}'")

    # --- Load classical baseline ---
    baseline_path = Path(CFG.models_dir) / "classical_baseline_best.pt"
    baseline_model = ClassicalBaseline(input_dim=len(CFG.audio_features))
    baseline_model = load_model_from_checkpoint(baseline_model, baseline_path)
    print(f"Loaded Classical Baseline from '{baseline_path}'")

    # --- Evaluate ---
    print("\n--- Hybrid QNN Test Results ---")
    hybrid_metrics = compute_metrics(hybrid_model, hybrid_test_loader, encoder)

    print("\n--- Classical Baseline Test Results ---")
    baseline_metrics = compute_metrics(baseline_model, baseline_test_loader, encoder)

    # --- Detect current run number for figure subfolder ---
    results_dir = Path(CFG.results_dir)
    existing_runs = sorted([
        int(d.name.replace("run", ""))
        for d in Path(CFG.figures_dir).glob("run*") if d.is_dir()
        if d.name.replace("run", "").isdigit()
    ])
    current_run   = existing_runs[-1] + 1 if existing_runs else 4
    figures_dir   = Path(CFG.figures_dir) / f"run{current_run}"
    figures_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving figures for Run {current_run} to '{figures_dir}'")

    # Also save flat copies to the top-level figures dir (for quick access)
    figures_flat = Path(CFG.figures_dir)
    figures_flat.mkdir(parents=True, exist_ok=True)

    # Training histories (loaded from saved JSON)
    with open(results_dir / "hybrid_history.json") as f:
        hybrid_history = json.load(f)
    with open(results_dir / "baseline_history.json") as f:
        baseline_history = json.load(f)

    for title, history, stem in [
        ("Hybrid QNN Training History",        hybrid_history,   "hybrid_training_history"),
        ("Classical Baseline Training History", baseline_history, "baseline_training_history"),
    ]:
        plot_training_history(history, save_path=figures_dir / f"{stem}.png", title=title)
        plot_training_history(history, save_path=figures_flat / f"{stem}.png", title=title)

    # Confusion matrices
    for title, cm, stem in [
        ("Hybrid QNN — Confusion Matrix (Test Set)",        hybrid_metrics["confusion_matrix"],   "hybrid_confusion_matrix"),
        ("Classical Baseline — Confusion Matrix (Test Set)", baseline_metrics["confusion_matrix"], "baseline_confusion_matrix"),
    ]:
        plot_confusion_matrix(cm, class_names=list(encoder.classes_), title=title,
                              save_path=figures_dir / f"{stem}.png")
        plot_confusion_matrix(cm, class_names=list(encoder.classes_), title=title,
                              save_path=figures_flat / f"{stem}.png")

    # Model comparison
    compare_models(hybrid_metrics, baseline_metrics, encoder,
                   save_path=figures_dir / "model_comparison.png")
    compare_models(hybrid_metrics, baseline_metrics, encoder,
                   save_path=figures_flat / "model_comparison.png")

    # Training progression — all runs including this one
    new_run = {
        "run":      current_run,
        "label":    f"Run {current_run}\n(+ReduceLROnPlateau)",
        "hybrid":   round(max(hybrid_history["val_acc"]) * 100, 1),
        "baseline": round(max(baseline_history["val_acc"]) * 100, 1),
    }
    plot_training_progression(
        new_run=new_run,
        save_path=figures_flat / "training_progression.png",
    )

    print(f"\nStep 4 complete.")
    print(f"  Run-specific figures -> outputs/figures/run{current_run}/")
    print(f"  Latest flat copies   -> outputs/figures/")


if __name__ == "__main__":
    main()
