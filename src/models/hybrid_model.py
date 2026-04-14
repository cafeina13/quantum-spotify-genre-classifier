"""
hybrid_model.py — Hourglass Hybrid Genre Classifier.

Architecture overview
---------------------
Input (batch, 6)  ← top-6 selected features, scaled to [-π, π] BEFORE passing in
  │
  ├─ [Optional] Classical Encoder  (12 → 6)   if use_autoencoder_bottleneck=True
  │   The Autoencoder encoder replaces the external PCA step.
  │
  ▼
Quantum Layer     (6 → 6)    qml.qnn.TorchLayer wrapping the VQC
  ▼
BatchNorm1d(6)               stabilises near-zero quantum expectation values
  ▼
Linear(6, 6)                 raw logits (no Softmax — use CrossEntropyLoss)
  ▼
Output (batch, 6)

Why BatchNorm after the quantum layer?
  Quantum expectation values tend to cluster near zero due to the
  barren plateau phenomenon. BatchNorm re-centres and rescales them,
  giving the subsequent linear layer healthier gradient signal.

Why raw logits (no Softmax)?
  torch.nn.CrossEntropyLoss applies log_softmax internally.
  Adding Softmax in the model AND using CrossEntropyLoss would compute
  log(softmax(softmax(x))) — silently producing near-zero gradients.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml

from src.config import CFG
from src.quantum.circuit import build_vqc_circuit
from src.quantum.device import get_device


class HybridGenreClassifier(nn.Module):
    """
    Hourglass hybrid quantum-classical model.

    Parameters
    ----------
    n_qubits : int
        Number of qubits = latent dimension = number of output classes.
        Default: CFG.n_qubits (6).
    n_layers : int
        Ansatz depth (StronglyEntanglingLayers repetitions). Default: CFG.n_layers (2).
    device : qml.Device
        PennyLane device. If None, get_device() is called with CFG defaults.
    use_autoencoder_bottleneck : bool
        False (default): input is already PCA-reduced to shape (batch, n_qubits).
        True: input has full shape (batch, input_dim=12) and a small classical
              encoder is applied first to compress to (batch, n_qubits).
    input_dim : int
        Number of raw input features. Only used when use_autoencoder_bottleneck=True.
    """

    def __init__(
        self,
        n_qubits: int = None,
        n_layers: int = None,
        device = None,
        use_autoencoder_bottleneck: bool = False,
        input_dim: int = 12,
    ):
        super().__init__()
        self.n_qubits = n_qubits or CFG.n_qubits
        self.n_layers = n_layers or CFG.n_layers

        if device is None:
            device = get_device()

        # --- Optional classical encoder (replaces external PCA) ---
        if use_autoencoder_bottleneck:
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 8),
                nn.ReLU(),
                nn.Linear(8, self.n_qubits),
                nn.Sigmoid(),   # output in (0, 1); scaled to (0, π) in forward()
            )
        else:
            self.encoder = None  # PCA applied externally before calling model

        # --- Quantum layer ---
        circuit, weight_shapes = build_vqc_circuit(self.n_qubits, self.n_layers, device)
        self.quantum_layer = qml.qnn.TorchLayer(circuit, weight_shapes)

        # --- Post-quantum classical head ---
        # BatchNorm stabilises the small-variance quantum outputs
        self.bn = nn.BatchNorm1d(self.n_qubits)
        # Output raw logits — no Softmax (use CrossEntropyLoss)
        self.fc = nn.Linear(self.n_qubits, self.n_qubits)  # n_qubits == n_classes == 6

        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming initialisation for classical linear layers."""
        if self.encoder is not None:
            for layer in self.encoder:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
                    nn.init.zeros_(layer.bias)
        nn.init.kaiming_uniform_(self.fc.weight, nonlinearity="linear")
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            If encoder is None: shape (batch_size, n_qubits), values in [-π, π].
            If encoder is set:  shape (batch_size, input_dim=12).

        Returns
        -------
        torch.Tensor, shape (batch_size, n_qubits) — raw logits.
        """
        if self.encoder is not None:
            # Encoder output is in (0, 1); scale to (0, π) for RY gate angles
            x = self.encoder(x) * math.pi

        # Quantum layer: (batch, n_qubits) → (batch, n_qubits), values in [-1, 1]
        q_out = self.quantum_layer(x)

        # Stabilise + classify
        out = self.bn(q_out)
        out = self.fc(out)
        return out   # raw logits

    def count_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
