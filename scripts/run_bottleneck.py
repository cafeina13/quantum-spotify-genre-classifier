"""
run_bottleneck.py — Step 2: Feature selection bottleneck.

Selects the 6 most genre-discriminative features from the 12 scaled features
(determined by F-score + mutual information analysis) and saves them as Z arrays
for the quantum layer.

This replaces PCA. PCA maximises variance, not class separation — analysis showed
a 13.8% accuracy drop when compressing 12->6 via PCA. Direct feature selection
retains the original scaled values of the most informative features.

Selected features (CFG.selected_features):
  speechiness, danceability, energy, instrumentalness, tempo, acousticness

Run from the project root:
  python scripts/run_bottleneck.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from src.config import CFG
from src.data.preprocessor import load_processed, save_processed


def svc_benchmark(X_train, y_train, X_test, y_test, label: str) -> float:
    """Train a quick SVC and return test accuracy."""
    clf = SVC(kernel="rbf", C=1.0, random_state=CFG.random_seed)
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    print(f"  SVC accuracy ({label}): {acc:.4f} ({acc*100:.2f}%)")
    return acc


def main():
    print("=" * 60)
    print("Step 2 -- Feature Selection Bottleneck")
    print("=" * 60)

    # 1. Load processed arrays from Step 1
    data = load_processed(CFG.processed_dir)
    X_train = data["X_train"]
    X_val   = data["X_val"]
    X_test  = data["X_test"]
    y_train = data["y_train"]
    y_test  = data["y_test"]

    # 2. Benchmark: SVC on full 12-feature representation
    print("\nSVC benchmark on full 12 features:")
    full_acc = svc_benchmark(X_train, y_train, X_test, y_test, "12 features")

    # 3. Get indices of selected features in the full audio_features list
    selected_indices = [
        CFG.audio_features.index(f) for f in CFG.selected_features
    ]
    print(f"\nSelected features: {CFG.selected_features}")
    print(f"Column indices:    {selected_indices}")

    # 4. Extract selected columns — no second scaler needed,
    #    features are already scaled to [-pi, pi] from Step 1
    Z_train = X_train[:, selected_indices]
    Z_val   = X_val[:,   selected_indices]
    Z_test  = X_test[:,  selected_indices]

    # 5. Benchmark: SVC on selected 6 features
    print(f"\nSVC benchmark on {len(CFG.selected_features)} selected features:")
    sel_acc = svc_benchmark(Z_train, y_train, Z_test, y_test,
                            f"{len(CFG.selected_features)} selected features")

    # 6. Report accuracy drop
    acc_drop = full_acc - sel_acc
    if acc_drop > 0.05:
        print(
            f"\n  WARNING: accuracy drop is {acc_drop:.3f} (>5%).\n"
            f"  Consider revising CFG.selected_features."
        )
    else:
        print(f"\n  Accuracy drop: {acc_drop:.3f} -- within acceptable range (<5%).")

    # 7. Sanity checks
    import math
    assert Z_train.max() <= math.pi + 1e-6, "Range check failed: max > pi"
    assert Z_train.min() >= -math.pi - 1e-6, "Range check failed: min < -pi"
    print("Range check passed: Z arrays in [-pi, pi].")

    # 8. Save selected-feature arrays
    print("\nSaving feature-selected arrays...")
    save_processed(
        {
            "Z_train": Z_train,
            "Z_val":   Z_val,
            "Z_test":  Z_test,
        },
        CFG.processed_dir,
    )

    print("\nStep 2 complete. Run scripts/run_training.py to train.")


if __name__ == "__main__":
    main()
