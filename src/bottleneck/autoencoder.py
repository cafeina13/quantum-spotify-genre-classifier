"""
autoencoder.py — Classical autoencoder bottleneck (optional upgrade to PCA).

This module provides an alternative to PCA for the dimensionality reduction
step. The autoencoder learns a non-linear compression of 12 audio features
down to n_qubits (default 6) latent dimensions.

When to use instead of PCA
---------------------------
If the quick SVC benchmark in the bottleneck notebook shows that PCA accuracy
is more than 5% below the full-feature accuracy, switch to this autoencoder.

Training the autoencoder
------------------------
1. Train with MSE reconstruction loss (input ≈ decoded output).
2. Freeze encoder weights.
3. Feed encoder output (scaled to [0, π]) into the quantum layer.

The autoencoder is trained BEFORE hybrid model training — it is NOT
backpropagated through the quantum circuit.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from src.config import CFG


class Autoencoder(nn.Module):
    """
    Symmetric bottleneck autoencoder: 12 → 8 → 6 → 8 → 12.

    The encoder output uses Sigmoid activation so values lie in (0, 1).
    Multiply encoder output by π before feeding into angle encoding.

    Parameters
    ----------
    input_dim  : number of input audio features (12)
    latent_dim : bottleneck size = number of qubits (default CFG.n_qubits = 6)
    """

    def __init__(self, input_dim: int = 12, latent_dim: int = None):
        super().__init__()
        latent_dim = latent_dim or CFG.n_qubits

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.ReLU(),
            nn.Linear(8, latent_dim),
            nn.Sigmoid(),   # output in (0, 1); scale by π before quantum layer
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 8),
            nn.ReLU(),
            nn.Linear(8, input_dim),
        )

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Full autoencoder pass.

        Parameters
        ----------
        x : torch.Tensor, shape (batch_size, input_dim)

        Returns
        -------
        (reconstructed_x, latent_z)
          reconstructed_x : shape (batch_size, input_dim)
          latent_z        : shape (batch_size, latent_dim), values in (0, 1)
        """
        latent_z       = self.encoder(x)
        reconstructed  = self.decoder(latent_z)
        return reconstructed, latent_z

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encoder-only pass. Used at inference time and during hybrid training.

        Returns values in (0, 1). Multiply by π before passing to quantum layer.
        """
        return self.encoder(x)


# ---------------------------------------------------------------------------
# Stand-alone training function
# ---------------------------------------------------------------------------

def train_autoencoder(
    model: Autoencoder,
    X_train: torch.Tensor,
    X_val: torch.Tensor,
    n_epochs: int = 50,
    learning_rate: float = 1e-3,
    batch_size: int = 64,
    checkpoint_path: Path = None,
) -> dict:
    """
    Train the autoencoder with MSE reconstruction loss.

    Parameters
    ----------
    model          : Autoencoder instance
    X_train        : torch.Tensor, shape (N_train, input_dim)
    X_val          : torch.Tensor, shape (N_val, input_dim)
    n_epochs       : number of training epochs
    learning_rate  : Adam learning rate
    batch_size     : mini-batch size
    checkpoint_path: if provided, save the best model weights here

    Returns
    -------
    dict with keys "train_loss" and "val_loss" (lists of per-epoch values)
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    train_dataset = torch.utils.data.TensorDataset(X_train)
    train_loader  = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")

    for epoch in range(1, n_epochs + 1):
        # --- Training ---
        model.train()
        epoch_loss = 0.0
        for (batch_x,) in train_loader:
            optimizer.zero_grad()
            reconstructed, _ = model(batch_x)
            loss = criterion(reconstructed, batch_x)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch_x)

        train_loss = epoch_loss / len(X_train)

        # --- Validation ---
        model.eval()
        with torch.no_grad():
            reconstructed_val, _ = model(X_val)
            val_loss = criterion(reconstructed_val, X_val).item()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch [{epoch:>3}/{n_epochs}]  train_loss: {train_loss:.5f}  val_loss: {val_loss:.5f}")

        # Save best checkpoint
        if val_loss < best_val_loss and checkpoint_path is not None:
            best_val_loss = val_loss
            checkpoint_path = Path(checkpoint_path)
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)

    if val_loss > 0.05:
        print(
            f"\n  WARNING: Final val_loss ({val_loss:.5f}) > 0.05. "
            "Consider increasing model capacity or training longer."
        )

    return history
