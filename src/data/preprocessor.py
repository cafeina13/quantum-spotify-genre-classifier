"""
preprocessor.py — Data cleaning, feature scaling, and label encoding.

Critical design note on scaling:
  - Raw features are scaled to [-π, π] with MinMaxScaler BEFORE feature selection.
    Full [-π, π] range gives RY gates access to both rotation directions,
    producing richer quantum states and better class separability than [0, π].
  - Both scalers are fitted on training data only to prevent data leakage.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from src.config import CFG


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def clean_data(
    df: pd.DataFrame,
    audio_features: list = None,
    target_col: str = None,
) -> pd.DataFrame:
    """
    Prepare the raw DataFrame for modelling.

    Steps:
      1. Keep only audio_features + target_col columns.
      2. Drop rows where target_col is NaN.
      3. Drop duplicate rows (exact duplicates across all kept columns).
      4. Handle key == -1 (Spotify uses -1 when no key is detected).
         Strategy: replace with 6 (median of 0–11 pitch classes).
      5. Print a brief cleaning report.

    Parameters
    ----------
    df           : raw DataFrame from load_dataset()
    audio_features : list of feature column names (defaults to CFG.audio_features)
    target_col   : name of the target column (defaults to CFG.target_column)

    Returns
    -------
    pd.DataFrame — cleaned, with only audio_features + target_col columns.
    """
    if audio_features is None:
        audio_features = CFG.audio_features
    if target_col is None:
        target_col = CFG.target_column

    # Keep only relevant columns
    df = df[audio_features + [target_col]].copy()
    n_start = len(df)

    # Drop rows with missing target
    df = df.dropna(subset=[target_col])
    n_after_target = len(df)

    # Drop rows with any NaN in audio features
    df = df.dropna(subset=audio_features)
    n_after_nan = len(df)

    # Drop exact duplicates
    df = df.drop_duplicates()
    n_after_dedup = len(df)

    # Handle key == -1: replace with 6 (median pitch class)
    n_bad_key = (df["key"] == -1).sum()
    if n_bad_key > 0:
        df.loc[df["key"] == -1, "key"] = 6
        print(f"  Replaced {n_bad_key} rows where key == -1 with median value 6.")

    print(
        f"Cleaning report:\n"
        f"  Started with:          {n_start:>7,} rows\n"
        f"  After dropping NaN target: {n_after_target:>7,} rows\n"
        f"  After dropping NaN features: {n_after_nan:>7,} rows\n"
        f"  After deduplication:   {n_after_dedup:>7,} rows\n"
        f"  Rows removed total:    {n_start - n_after_dedup:>7,}"
    )

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Feature scaling
# ---------------------------------------------------------------------------

def scale_features(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
) -> tuple:
    """
    Scale all feature splits to [-π, π] using MinMaxScaler fitted on X_train only.

    [-π, π] gives RY gates access to both rotation directions on the Bloch sphere:
      - RY(-π)|0⟩ rotates fully one way
      - RY(0)|0⟩  = |0⟩ (no rotation — average features stay near ground state)
      - RY(π)|0⟩  rotates fully the other way
    This doubles the effective encoding range vs [0, π] and improves class separability.

    Parameters
    ----------
    X_train, X_val, X_test : np.ndarray, shape (N_split, n_features)

    Returns
    -------
    X_train_scaled, X_val_scaled, X_test_scaled : np.ndarray in [-π, π]
    scaler : fitted MinMaxScaler (save this to apply the same transform later)
    """
    scaler = MinMaxScaler(feature_range=(-float(np.pi), float(np.pi)))
    X_train_scaled = scaler.fit_transform(X_train)   # fit only on train
    X_val_scaled   = scaler.transform(X_val)
    X_test_scaled  = scaler.transform(X_test)

    print(
        f"Feature scaling (raw -> [-pi, pi]):\n"
        f"  X_train range: [{X_train_scaled.min():.4f}, {X_train_scaled.max():.4f}]\n"
        f"  X_val   range: [{X_val_scaled.min():.4f}, {X_val_scaled.max():.4f}]\n"
        f"  X_test  range: [{X_test_scaled.min():.4f}, {X_test_scaled.max():.4f}]"
    )

    # Sanity checks
    assert X_train_scaled.max() <= float(np.pi) + 1e-6, "Scaling error: values exceed pi"
    assert X_train_scaled.min() >= -float(np.pi) - 1e-6, "Scaling error: values below -pi"

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def scale_pca_output(
    Z_train: np.ndarray,
    Z_val: np.ndarray,
    Z_test: np.ndarray,
) -> tuple:
    """
    Re-scale PCA-projected features back to [-π, π].

    PCA output lives in an unbounded real space (positive and negative values).
    This second scaling step brings it back into the range expected by RY gates.
    The scaler is fitted on Z_train only.

    Parameters
    ----------
    Z_train, Z_val, Z_test : PCA-projected arrays, shape (N_split, n_components)

    Returns
    -------
    Z_train_scaled, Z_val_scaled, Z_test_scaled : np.ndarray in [-π, π]
    pca_scaler : fitted MinMaxScaler
    """
    pca_scaler = MinMaxScaler(feature_range=(-float(np.pi), float(np.pi)))
    Z_train_scaled = pca_scaler.fit_transform(Z_train)
    Z_val_scaled   = pca_scaler.transform(Z_val)
    Z_test_scaled  = pca_scaler.transform(Z_test)

    print(
        f"PCA output re-scaling (PCA space -> [-pi, pi]):\n"
        f"  Z_train range: [{Z_train_scaled.min():.4f}, {Z_train_scaled.max():.4f}]\n"
        f"  Z_val   range: [{Z_val_scaled.min():.4f}, {Z_val_scaled.max():.4f}]\n"
        f"  Z_test  range: [{Z_test_scaled.min():.4f}, {Z_test_scaled.max():.4f}]"
    )

    return Z_train_scaled, Z_val_scaled, Z_test_scaled, pca_scaler


# ---------------------------------------------------------------------------
# Label encoding
# ---------------------------------------------------------------------------

def encode_labels(
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    genre_classes: list = None,
) -> tuple:
    """
    Integer-encode genre string labels.

    The LabelEncoder is fitted on the canonical genre_classes list (not on
    y_train) to guarantee a consistent integer → genre mapping regardless
    of which genres appear in a given split.

    Parameters
    ----------
    y_train, y_val, y_test : np.ndarray of genre strings
    genre_classes : list of all possible genre strings (defaults to CFG.genre_classes)

    Returns
    -------
    y_train_enc, y_val_enc, y_test_enc : np.ndarray of integers
    encoder : fitted LabelEncoder
    """
    if genre_classes is None:
        genre_classes = CFG.genre_classes

    encoder = LabelEncoder()
    encoder.fit(genre_classes)   # fit on fixed list, not on data

    y_train_enc = encoder.transform(y_train)
    y_val_enc   = encoder.transform(y_val)
    y_test_enc  = encoder.transform(y_test)

    print(f"Label encoding: {dict(enumerate(encoder.classes_))}")
    assert len(np.unique(y_train_enc)) == len(genre_classes), \
        "Not all genre classes appear in the training set — check class balance."

    return y_train_enc, y_val_enc, y_test_enc, encoder


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def save_processed(arrays: dict, processed_dir: Path = CFG.processed_dir) -> None:
    """
    Save processed arrays and fitted objects to disk.

    Parameters
    ----------
    arrays : dict
        Keys are file stem names; values are np.ndarray or picklable objects.
        np.ndarray → saved as .npy
        Other objects (scalers, encoders) → saved as .pkl
    processed_dir : Path
    """
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    for name, obj in arrays.items():
        if isinstance(obj, np.ndarray):
            path = processed_dir / f"{name}.npy"
            np.save(path, obj)
            print(f"  Saved {path}  shape={obj.shape}")
        else:
            path = processed_dir / f"{name}.pkl"
            with open(path, "wb") as f:
                pickle.dump(obj, f)
            print(f"  Saved {path}")

    print(f"All processed files written to '{processed_dir}'.")


def load_processed(processed_dir: Path = CFG.processed_dir) -> dict:
    """
    Load all .npy and .pkl files from processed_dir into a dict.

    Returns
    -------
    dict — same structure as passed to save_processed()
    """
    processed_dir = Path(processed_dir)
    result = {}

    for path in sorted(processed_dir.iterdir()):
        if path.suffix == ".npy":
            result[path.stem] = np.load(path, allow_pickle=False)
        elif path.suffix == ".pkl":
            with open(path, "rb") as f:
                result[path.stem] = pickle.load(f)

    print(f"Loaded {len(result)} processed objects from '{processed_dir}'.")
    return result
