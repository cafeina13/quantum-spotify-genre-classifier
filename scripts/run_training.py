"""
run_training.py — Step 3: Train the hybrid and baseline models.

Loads the PCA-compressed arrays from Step 2 and trains:
  1. HybridGenreClassifier  (quantum + classical)
  2. ClassicalBaseline      (purely classical, same parameter budget)

Checkpoints are saved to outputs/models/.
Training history is saved to outputs/results/.

Run from the project root:
  python scripts/run_training.py

To switch to IBM hardware, set use_ibm_hardware=True in src/config.py
and ensure IBM_QUANTUM_TOKEN is set in your .env file.
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import CFG
from src.data.preprocessor import load_processed
from src.quantum.device import get_device
from src.models.hybrid_model import HybridGenreClassifier
from src.models.classical_baseline import ClassicalBaseline
from src.training.trainer import Trainer


def make_loaders(data: dict) -> tuple:
    """Create TensorDataset DataLoaders from processed numpy arrays."""
    def to_loader(X_np, y_np, shuffle: bool) -> DataLoader:
        X_t = torch.tensor(X_np, dtype=torch.float32)
        y_t = torch.tensor(y_np, dtype=torch.long)
        ds  = TensorDataset(X_t, y_t)
        return DataLoader(ds, batch_size=CFG.batch_size, shuffle=shuffle)

    train_loader = to_loader(data["Z_train"], data["y_train"], shuffle=True)
    val_loader   = to_loader(data["Z_val"],   data["y_val"],   shuffle=False)
    test_loader  = to_loader(data["Z_test"],  data["y_test"],  shuffle=False)
    return train_loader, val_loader, test_loader


def train_model(model, name: str, train_loader, val_loader) -> dict:
    """Train a model and return its history dict."""
    print(f"\n{'='*60}")
    print(f"Training: {name}  ({model.count_parameters()} parameters)")
    print(f"{'='*60}")

    optimizer = torch.optim.Adam(model.parameters(), lr=CFG.learning_rate)
    criterion = nn.CrossEntropyLoss()
    trainer   = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device="cpu",
        checkpoint_dir=CFG.models_dir,
    )

    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=CFG.n_epochs,
        patience=CFG.early_stop_patience,
        checkpoint_name=f"{name.lower().replace(' ', '_')}_best.pt",
    )
    return history


def main():
    print("=" * 60)
    print("Step 3 — Model Training")
    print("=" * 60)

    # 1. Load PCA-compressed data
    data = load_processed(CFG.processed_dir)
    train_loader, val_loader, _ = make_loaders(data)

    # 2. Create quantum device (local simulator by default)
    device = get_device()

    # 3. Train hybrid model
    hybrid_model = HybridGenreClassifier(
        n_qubits=CFG.n_qubits,
        n_layers=CFG.n_layers,
        device=device,
    )
    hybrid_history = train_model(hybrid_model, "Hybrid QNN", train_loader, val_loader)

    # 4. Train classical baseline (uses full 12 features, not PCA)
    # Rebuild loaders with the raw scaled features
    baseline_data = load_processed(CFG.processed_dir)

    def to_loader(X_np, y_np, shuffle):
        X_t = torch.tensor(X_np, dtype=torch.float32)
        y_t = torch.tensor(y_np, dtype=torch.long)
        from torch.utils.data import TensorDataset, DataLoader
        return DataLoader(TensorDataset(X_t, y_t), batch_size=CFG.batch_size, shuffle=shuffle)

    baseline_train = to_loader(baseline_data["X_train"], baseline_data["y_train"], shuffle=True)
    baseline_val   = to_loader(baseline_data["X_val"],   baseline_data["y_val"],   shuffle=False)

    baseline_model = ClassicalBaseline(input_dim=len(CFG.audio_features))
    baseline_history = train_model(baseline_model, "Classical Baseline", baseline_train, baseline_val)

    # 5. Save training histories
    results_dir = Path(CFG.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / "hybrid_history.json", "w") as f:
        json.dump(hybrid_history, f, indent=2)
    with open(results_dir / "baseline_history.json", "w") as f:
        json.dump(baseline_history, f, indent=2)

    print(f"\nTraining histories saved to '{results_dir}'.")
    print("\nStep 3 complete. Run scripts/run_evaluation.py to compare models.")


if __name__ == "__main__":
    main()
