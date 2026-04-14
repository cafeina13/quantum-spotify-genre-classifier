"""
classical_baseline.py — Pure classical DNN for fair comparison.

The baseline is intentionally constrained to a similar parameter budget as
the hybrid model (~80–100 total parameters) so that any performance difference
is attributable to the quantum layer, not raw model capacity.

Architecture: 12 → 16 → 8 → 6  (raw logits, no Softmax)
  - Linear(12, 16) + ReLU + BatchNorm1d(16)
  - Linear(16, 8)  + ReLU
  - Linear(8, 6)                     ← raw logits

Why the same parameter budget?
  Comparing a 36-parameter quantum model against a 1-million-parameter DNN
  is not a fair ablation. Keeping the parameter counts similar isolates the
  contribution of the quantum layer from the contribution of model capacity.
"""

import torch
import torch.nn as nn

from src.config import CFG


class ClassicalBaseline(nn.Module):
    """
    Classical DNN baseline for genre classification.

    Parameters
    ----------
    input_dim  : int, number of raw audio features (12)
    n_classes  : int, number of output genres (6)
    """

    def __init__(self, input_dim: int = 12, n_classes: int = None):
        super().__init__()
        n_classes = n_classes or len(CFG.genre_classes)

        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, n_classes),
            # No Softmax — CrossEntropyLoss expects raw logits
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor, shape (batch_size, input_dim)
            Raw audio features (any normalisation is fine; model handles it internally).

        Returns
        -------
        torch.Tensor, shape (batch_size, n_classes) — raw logits.
        """
        return self.net(x)

    def count_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
