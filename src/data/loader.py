"""
loader.py — Dataset loading and train/val/test splitting.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import CFG


def load_dataset(path: Path = CFG.raw_data_path) -> pd.DataFrame:
    """
    Load spotify_songs.csv and validate that all expected audio feature columns exist.

    Parameters
    ----------
    path : Path
        Location of spotify_songs.csv. Defaults to CFG.raw_data_path.

    Returns
    -------
    pd.DataFrame
        Raw DataFrame — no modifications applied.

    Raises
    ------
    FileNotFoundError
        If the CSV does not exist at the given path.
    ValueError
        If one or more expected audio feature columns are missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{path}'.\n"
            "Place spotify_songs.csv in the data/raw/ directory."
        )

    df = pd.read_csv(path)
    print(f"Loaded dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")

    # Validate required columns
    missing = [col for col in CFG.audio_features + [CFG.target_column] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")

    print(f"Target column '{CFG.target_column}' — unique values: {sorted(df[CFG.target_column].dropna().unique())}")
    return df


def split_dataset(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = CFG.test_size,
    val_size: float = CFG.val_size,
    random_seed: int = CFG.random_seed,
) -> tuple:
    """
    Stratified train / val / test split.

    The split is done in two steps to avoid leakage:
      1. Split off the test set from the full data.
      2. Split the remaining data into train and val.

    Parameters
    ----------
    X : np.ndarray, shape (N, n_features)
    y : np.ndarray, shape (N,)  — integer-encoded class labels
    test_size  : fraction of total data reserved for testing
    val_size   : fraction of the *remaining* (non-test) data reserved for validation
    random_seed: reproducibility seed

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test : six np.ndarrays
    """
    # Step 1: hold out test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_seed,
    )

    # Step 2: split remaining into train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_size,
        stratify=y_temp,
        random_state=random_seed,
    )

    print(
        f"Split sizes — train: {len(X_train):,} | val: {len(X_val):,} | test: {len(X_test):,}"
    )
    return X_train, X_val, X_test, y_train, y_val, y_test
