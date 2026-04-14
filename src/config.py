"""
config.py — Central configuration for the Hybrid QNN project.
All hyperparameters and paths live here. Every other module imports from this file.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    raw_data_path: Path = Path("data/raw/spotify_songs.csv")
    processed_dir: Path = Path("data/processed")
    figures_dir:   Path = Path("outputs/figures")
    models_dir:    Path = Path("outputs/models")
    results_dir:   Path = Path("outputs/results")

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    audio_features: list = field(default_factory=lambda: [
        "danceability",
        "energy",
        "key",
        "loudness",
        "mode",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
        "tempo",
        "duration_ms",
    ])
    target_column: str = "playlist_genre"
    genre_classes: list = field(default_factory=lambda: [
        "edm", "latin", "pop", "r&b", "rap", "rock"
    ])

    # ------------------------------------------------------------------
    # Dimensionality reduction
    # ------------------------------------------------------------------
    # n_qubits == n_components: one qubit per compressed feature
    n_qubits:     int = 6
    n_components: int = 6   # PCA output dimensions

    # ------------------------------------------------------------------
    # Quantum circuit
    # ------------------------------------------------------------------
    n_layers:     int = 2          # StronglyEntanglingLayers repetitions
    entanglement: str = "linear"   # used for documentation; SEL handles this internally

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    batch_size:    int   = 64
    learning_rate: float = 0.01
    n_epochs:      int   = 30
    test_size:     float = 0.20
    val_size:      float = 0.10    # fraction of the training split
    random_seed:   int   = 42
    early_stop_patience: int = 5

    # ------------------------------------------------------------------
    # Hardware
    # ------------------------------------------------------------------
    use_ibm_hardware: bool = False
    # Set ibm_backend to whichever 7-qubit device is available on your account.
    # Check available backends at: https://quantum.ibm.com/
    ibm_backend: str = "ibm_brisbane"


# Singleton — import this object everywhere:
#   from src.config import CFG
CFG = Config()
