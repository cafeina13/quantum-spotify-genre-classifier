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

    train_loader = to_loader(data["X_train"], data["y_train"], shuffle=True)
    val_loader   = to_loader(data["X_val"],   data["y_val"],   shuffle=False)
    test_loader  = to_loader(data["X_test"],  data["y_test"],  shuffle=False)
    return train_loader, val_loader, test_loader


def train_model(model, name: str, train_loader, val_loader, n_epochs: int = None) -> dict:
    """Train a model with ReduceLROnPlateau and return its history dict."""
    print(f"\n{'='*60}")
    print(f"Training: {name}  ({model.count_parameters()} parameters)")
    print(f"{'='*60}")

    optimizer = torch.optim.Adam(model.parameters(), lr=CFG.learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Halve LR after 4 epochs of no val_loss improvement.
    # Scheduler patience (4) < early-stop patience (10) so the model gets
    # at least one LR reduction before training gives up entirely.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.3,
        patience=2,
        min_lr=1e-5,
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device="cpu",
        checkpoint_dir=CFG.models_dir,
    )

    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=n_epochs or CFG.n_epochs,
        patience=10,
        checkpoint_name=f"{name.lower().replace(' ', '_')}_best.pt",
        scheduler=scheduler,
    )
    return history


def backup_run(run_number: int) -> None:
    """
    Copy current outputs (models, results, figures) into a numbered subfolder
    before a new training run overwrites them.
    Skips silently if there is nothing to back up.
    """
    import shutil
    dirs = {
        Path(CFG.models_dir):  Path(CFG.models_dir)  / f"run{run_number}",
        Path(CFG.results_dir): Path(CFG.results_dir) / f"run{run_number}",
        Path(CFG.figures_dir): Path(CFG.figures_dir) / f"run{run_number}",
    }
    backed_up = False
    for src, dst in dirs.items():
        if not src.exists():
            continue
        files = [f for f in src.glob("*") if f.is_file()]
        if not files:
            continue
        dst.mkdir(parents=True, exist_ok=True)
        for f in files:
            shutil.copy2(f, dst / f.name)
        backed_up = True

    if backed_up:
        print(f"Previous run backed up to run{run_number}/ subfolders.")


def detect_next_run_number() -> int:
    """
    Return the next run number by scanning run* subfolders across ALL output
    directories (models, results, figures) and taking the global maximum.
    Prevents the bug where only scanning one dir causes repeated overwriting.
    """
    scan_dirs = [Path(CFG.models_dir), Path(CFG.results_dir), Path(CFG.figures_dir)]
    numbers = []
    for d in scan_dirs:
        if not d.exists():
            continue
        for sub in d.glob("run*"):
            if sub.is_dir():
                try:
                    numbers.append(int(sub.name.replace("run", "")))
                except ValueError:
                    pass
    return max(numbers) + 1 if numbers else 4


def main():
    print("=" * 60)
    print("Step 3 — Model Training")
    print("=" * 60)

    # Auto-backup previous run before overwriting outputs
    run_number = detect_next_run_number()
    print(f"\nThis will be Run {run_number}. Backing up previous outputs ...")
    backup_run(run_number - 1)

    # 1. Load processed data
    data = load_processed(CFG.processed_dir)
    train_loader, val_loader, _ = make_loaders(data)

    # 2. Create quantum device (local simulator by default)
    device = get_device()

    # 3. Train hybrid model — 50 epochs with ReduceLROnPlateau
    hybrid_model = HybridGenreClassifier(
        n_qubits=CFG.n_qubits,
        n_layers=CFG.n_layers,
        device=device,
        input_dim=len(CFG.audio_features),
    )
    hybrid_history = train_model(hybrid_model, "Hybrid QNN", train_loader, val_loader, n_epochs=50)

    # 4. Train classical baseline — 50 epochs with ReduceLROnPlateau
    def to_loader(X_np, y_np, shuffle):
        X_t = torch.tensor(X_np, dtype=torch.float32)
        y_t = torch.tensor(y_np, dtype=torch.long)
        from torch.utils.data import TensorDataset, DataLoader
        return DataLoader(TensorDataset(X_t, y_t), batch_size=CFG.batch_size, shuffle=shuffle)

    baseline_data  = load_processed(CFG.processed_dir)
    baseline_train = to_loader(baseline_data["X_train"], baseline_data["y_train"], shuffle=True)
    baseline_val   = to_loader(baseline_data["X_val"],   baseline_data["y_val"],   shuffle=False)

    baseline_model   = ClassicalBaseline(input_dim=len(CFG.audio_features))
    baseline_history = train_model(baseline_model, "Classical Baseline", baseline_train, baseline_val, n_epochs=50)

    # 5. Save training histories
    results_dir = Path(CFG.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / "hybrid_history.json", "w") as f:
        json.dump(hybrid_history, f, indent=2)
    with open(results_dir / "baseline_history.json", "w") as f:
        json.dump(baseline_history, f, indent=2)

    print(f"\nTraining histories saved to '{results_dir}'.")
    print(f"\nRun {run_number} complete. Run scripts/run_evaluation.py to evaluate.")


if __name__ == "__main__":
    main()
