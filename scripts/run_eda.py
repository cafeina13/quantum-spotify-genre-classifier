"""
run_eda.py — Step 1: Data pipeline CLI entry point.

Loads spotify_songs.csv, cleans it, splits it, scales features,
encodes labels, and saves all processed arrays to data/processed/.

Run from the project root:
  python scripts/run_eda.py
"""

import sys
from pathlib import Path

# Allow src/ imports without pip install
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.config import CFG
from src.data.loader import load_dataset, split_dataset
from src.data.preprocessor import (
    clean_data,
    encode_labels,
    save_processed,
    scale_features,
)


def main():
    print("=" * 60)
    print("Step 1 — Data Pipeline")
    print("=" * 60)

    # 1. Load
    df = load_dataset(CFG.raw_data_path)

    # 2. Clean
    df = clean_data(df, CFG.audio_features, CFG.target_column)

    # 3. Separate features and target
    X = df[CFG.audio_features].values.astype(float)
    y = df[CFG.target_column].values

    # 4. Train/val/test split (stratified)
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y)

    # 5. Scale features to [-π, π]
    X_train_s, X_val_s, X_test_s, scaler = scale_features(X_train, X_val, X_test)

    # 6. Encode labels
    y_train_enc, y_val_enc, y_test_enc, encoder = encode_labels(y_train, y_val, y_test)

    # 7. Verify
    assert X_train_s.max() <= float(np.pi) + 1e-6, "Scale check failed: max > π"
    assert X_train_s.min() >= -float(np.pi) - 1e-6, "Scale check failed: min < -pi"
    assert np.isnan(X_train_s).sum() == 0, "NaN check failed"
    print("\nAll sanity checks passed.")

    # 8. Save
    print("\nSaving processed arrays...")
    save_processed(
        {
            "X_train": X_train_s,
            "X_val":   X_val_s,
            "X_test":  X_test_s,
            "y_train": y_train_enc,
            "y_val":   y_val_enc,
            "y_test":  y_test_enc,
            "scaler":  scaler,
            "encoder": encoder,
        },
        CFG.processed_dir,
    )

    print("\nStep 1 complete. Open notebooks/01_eda.ipynb for EDA visualisations.")


if __name__ == "__main__":
    main()
