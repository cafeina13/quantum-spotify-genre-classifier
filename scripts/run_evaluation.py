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
from src.training.trainer import Trainer
from src.training.metrics import (
    compute_metrics,
    plot_training_history,
    plot_confusion_matrix,
    compare_models,
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

    hybrid_test_loader   = to_loader(data["Z_test"], data["y_test"])
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

    # --- Plots ---
    figures_dir = Path(CFG.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Training histories (loaded from saved JSON)
    results_dir = Path(CFG.results_dir)
    with open(results_dir / "hybrid_history.json") as f:
        hybrid_history = json.load(f)
    with open(results_dir / "baseline_history.json") as f:
        baseline_history = json.load(f)

    plot_training_history(
        hybrid_history,
        save_path=figures_dir / "hybrid_training_history.png",
        title="Hybrid QNN Training History",
    )
    plot_training_history(
        baseline_history,
        save_path=figures_dir / "baseline_training_history.png",
        title="Classical Baseline Training History",
    )

    # Confusion matrices
    plot_confusion_matrix(
        hybrid_metrics["confusion_matrix"],
        class_names=list(encoder.classes_),
        title="Hybrid QNN — Confusion Matrix (Test Set)",
        save_path=figures_dir / "hybrid_confusion_matrix.png",
    )
    plot_confusion_matrix(
        baseline_metrics["confusion_matrix"],
        class_names=list(encoder.classes_),
        title="Classical Baseline — Confusion Matrix (Test Set)",
        save_path=figures_dir / "baseline_confusion_matrix.png",
    )

    # Model comparison
    compare_models(
        hybrid_metrics,
        baseline_metrics,
        encoder,
        save_path=figures_dir / "model_comparison.png",
    )

    print("\nStep 4 complete. All figures saved to 'outputs/figures/'.")


if __name__ == "__main__":
    main()
