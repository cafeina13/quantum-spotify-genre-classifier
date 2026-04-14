"""
hybrid_model.py — Hourglass Hybrid Genre Classifier.

Architecture overview
---------------------
Input (batch, 12)  <- all 12 audio features, scaled to [-pi, pi]
  |
Classical Encoder  (12 -> 32 -> 64 -> 6)
  Learns the best 6-dim representation for the quantum layer end-to-end.
  Tanh output scaled by pi keeps values in (-pi, pi) for RY gate angles.
  Wide classical layers are safe here — no quantum noise yet.
  |
Quantum Layer  (6 -> 6)   qml.qnn.TorchLayer wrapping the VQC
  |
BatchNorm1d(6)
  Stabilises near-zero quantum expectation values (barren plateau mitigation).
  |
Linear(6 -> 16) -> ReLU -> Linear(16 -> 6)
  Modest expansion gives the decoder capacity to interpret quantum measurements
  without amplifying quantum noise into a high-dimensional space.
  |
Output (batch, 6)  <- raw logits (no Softmax, use CrossEntropyLoss)

Why Tanh x pi for the encoder output?
  RY gates accept angles in (-pi, pi). Tanh maps any real value to (-1, 1),
  scaling by pi gives (-pi, pi). The encoder learns which rotation angles
  best separate the 6 genres — better than hand-picked feature selection.

Why BatchNorm after the quantum layer?
  Quantum expectation values cluster near zero (barren plateau). BatchNorm
  re-centres and rescales them so the decoder gets a healthier gradient signal.

Why a 2-layer decoder (6->16->6) instead of a single Linear(6->6)?
  One linear layer has 42 weights to decode 6 quantum measurements into 6 class
  scores — too little capacity. The intermediate 16-dim layer lets the model
  learn non-linear combinations. We keep it at 16 (not wider) to avoid
  amplifying quantum noise in high-dimensional space.

Why raw logits (no Softmax)?
  CrossEntropyLoss applies log_softmax internally. Adding Softmax here too
  would compute log(softmax(softmax(x))) — silently near-zero gradients.
"""

import math
import torch
import torch.nn as nn
import pennylane as qml

from src.config import CFG
from src.quantum.circuit import build_vqc_circuit
from src.quantum.device import get_device


class HybridGenreClassifier(nn.Module):
    """
    Hybrid quantum-classical genre classifier.

    Parameters
    ----------
    n_qubits : int
        Number of qubits. Must equal the encoder output dim and n_classes.
        Default: CFG.n_qubits (6).
    n_layers : int
        Ansatz depth (StronglyEntanglingLayers repetitions). Default: CFG.n_layers (2).
    device : qml.Device
        PennyLane device. If None, get_device() is called.
    input_dim : int
        Number of raw input features (default 12).
    """

    def __init__(
        self,
        n_qubits: int = None,
        n_layers: int = None,
        device=None,
        input_dim: int = 12,
    ):
        super().__init__()
        self.n_qubits = n_qubits or CFG.n_qubits

        if device is None:
            device = get_device()

        # --- Classical pre-encoder ---
        # Wide classical layers compress 12 raw features to 6 quantum-ready angles.
        # Tanh x pi maps output to (-pi, pi) — the full RY rotation range.
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, self.n_qubits),
            nn.Tanh(),          # output in (-1, 1); scaled to (-pi, pi) in forward()
        )

        # --- Quantum layer ---
        n_layers = n_layers or CFG.n_layers
        circuit, weight_shapes = build_vqc_circuit(self.n_qubits, n_layers, device)
        self.quantum_layer = qml.qnn.TorchLayer(circuit, weight_shapes)

        # --- Classical post-decoder ---
        self.decoder = nn.Sequential(
            nn.BatchNorm1d(self.n_qubits),
            nn.Linear(self.n_qubits, 16),
            nn.ReLU(),
            nn.Linear(16, self.n_qubits),   # n_qubits == n_classes == 6
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for layer in list(self.encoder) + list(self.decoder):
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor, shape (batch_size, input_dim)
            Raw scaled audio features in [-pi, pi].

        Returns
        -------
        torch.Tensor, shape (batch_size, n_qubits) — raw logits.
        """
        # Encode: 12 -> 32 -> 64 -> 6, then scale Tanh output to (-pi, pi)
        angles = self.encoder(x) * math.pi

        # Quantum layer: (batch, n_qubits) -> (batch, n_qubits), values in [-1, 1]
        q_out = self.quantum_layer(angles)

        # Decode: BN -> 6->16->6, raw logits
        return self.decoder(q_out)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
