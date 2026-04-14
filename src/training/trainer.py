"""
trainer.py — Training loop for both HybridGenreClassifier and ClassicalBaseline.

Both models are nn.Module subclasses, so this Trainer works with either.
The main loop includes:
  - Per-epoch train and validation passes
  - Early stopping (patience on val_loss)
  - Best-model checkpointing
  - Loss and accuracy history logging
"""

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import CFG


class Trainer:
    """
    Manages the train/val loop, checkpointing, and history logging.

    Parameters
    ----------
    model          : nn.Module — HybridGenreClassifier or ClassicalBaseline
    optimizer      : torch.optim.Optimizer
    criterion      : loss function (nn.CrossEntropyLoss recommended)
    device         : "cpu" or "cuda"
    checkpoint_dir : directory to save best model weights
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: str = "cpu",
        checkpoint_dir: Path = None,
    ):
        self.model          = model.to(device)
        self.optimizer      = optimizer
        self.criterion      = criterion
        self.device         = device
        self.checkpoint_dir = Path(checkpoint_dir or CFG.models_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Single epoch methods
    # ------------------------------------------------------------------

    def train_epoch(self, loader: DataLoader) -> tuple:
        """
        One full training epoch.

        Returns
        -------
        (avg_loss, accuracy) : floats
        """
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(X_batch)
            loss   = self.criterion(logits, y_batch)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * len(y_batch)
            preds   = logits.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total   += len(y_batch)

        return total_loss / total, correct / total

    def validate(self, loader: DataLoader) -> tuple:
        """
        Validation pass (no gradient computation).

        Returns
        -------
        (avg_loss, accuracy) : floats
        """
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                logits = self.model(X_batch)
                loss   = self.criterion(logits, y_batch)

                total_loss += loss.item() * len(y_batch)
                preds   = logits.argmax(dim=1)
                correct += (preds == y_batch).sum().item()
                total   += len(y_batch)

        return total_loss / total, correct / total

    # ------------------------------------------------------------------
    # Full training loop
    # ------------------------------------------------------------------

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        n_epochs: int = None,
        patience: int = None,
        checkpoint_name: str = "best_model.pt",
    ) -> dict:
        """
        Full training loop with early stopping and best-model checkpointing.

        Parameters
        ----------
        train_loader    : DataLoader for training data
        val_loader      : DataLoader for validation data
        n_epochs        : maximum number of training epochs (default: CFG.n_epochs)
        patience        : early-stop patience on val_loss (default: CFG.early_stop_patience)
        checkpoint_name : filename for the best checkpoint

        Returns
        -------
        history : dict with keys
            "train_loss", "val_loss", "train_acc", "val_acc" (lists of per-epoch floats)
        """
        n_epochs = n_epochs or CFG.n_epochs
        patience = patience or CFG.early_stop_patience
        checkpoint_path = self.checkpoint_dir / checkpoint_name

        history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
        best_val_loss  = float("inf")
        epochs_no_improve = 0

        for epoch in range(1, n_epochs + 1):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc     = self.validate(val_loader)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)

            print(
                f"Epoch [{epoch:>3}/{n_epochs}]  "
                f"train_loss: {train_loss:.4f}  train_acc: {train_acc:.3f}  "
                f"val_loss: {val_loss:.4f}  val_acc: {val_acc:.3f}"
            )

            # Checkpoint if best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                self.save_checkpoint(checkpoint_path)
                print(f"  → New best model saved (val_loss: {best_val_loss:.4f})")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(
                        f"\nEarly stopping at epoch {epoch} "
                        f"(no improvement for {patience} consecutive epochs)."
                    )
                    break

        print(f"\nTraining complete. Best val_loss: {best_val_loss:.4f}")
        return history

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: Path) -> None:
        """Save model state_dict and optimizer state_dict to disk."""
        torch.save(
            {
                "model_state_dict":     self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )

    def load_checkpoint(self, path: Path) -> None:
        """Restore model and optimizer from a saved checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"Checkpoint loaded from '{path}'.")
