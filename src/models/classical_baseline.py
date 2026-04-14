"""
classical_baseline.py — Pure classical DNN for fair comparison.

Architecture: 12 -> 128 -> 64 -> 32 -> 6
  - Linear(12, 128) + ReLU + BatchNorm + Dropout(0.3)
  - Linear(128, 64) + ReLU + BatchNorm + Dropout(0.3)
  - Linear(64, 32)  + ReLU
  - Linear(32, 6)              <- raw logits, no Softmax

Dropout is added here (unlike the hybrid) because the classical layers
produce clean outputs — dropout acts as regularisation without the
noise-amplification risk that wider post-QNN layers would carry.
"""

import torch
import torch.nn as nn

from src.config import CFG


class ClassicalBaseline(nn.Module):
    """
    Classical DNN baseline for genre classification.

    Parameters
    ----------
    input_dim : int, number of raw audio features (12)
    n_classes  : int, number of output genres (6)
    """

    def __init__(self, input_dim: int = 12, n_classes: int = None):
        super().__init__()
        n_classes = n_classes or len(CFG.genre_classes)

        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, n_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
