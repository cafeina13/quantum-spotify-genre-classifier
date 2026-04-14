"""
run_bottleneck.py — Step 2: Classical bottleneck (PCA) CLI entry point.

Loads the processed arrays from Step 1, applies PCA to compress 12 features
to n_qubits (default 6), re-scales the output to [0, π], and saves the
compressed arrays.

Also runs a quick SVC benchmark to verify the PCA does not lose too much
discriminative information.

Run from the project root:
  python scripts/run_bottleneck.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from src.config import CFG
from src.data.preprocessor import load_processed, save_processed
from src.bottleneck.pca_reducer import PCAReducer


def svc_benchmark(X_train, y_train, X_test, y_test, label: str) -> float:
    """Train a quick SVC and return test accuracy."""
    clf = SVC(kernel="rbf", C=1.0, random_state=CFG.random_seed)
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    print(f"  SVC accuracy ({label}): {acc:.4f} ({acc*100:.2f}%)")
    return acc


def main():
    print("=" * 60)
    print("Step 2 — Classical Bottleneck (PCA)")
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

    # 3. Fit PCA and compress to n_qubits dimensions
    print(f"\nApplying PCA: 12 → {CFG.n_components} components...")
    reducer = PCAReducer(n_components=CFG.n_components)
    Z_train = reducer.fit_transform(X_train)
    Z_val   = reducer.transform(X_val)
    Z_test  = reducer.transform(X_test)

    # 4. Benchmark: SVC on compressed representation
    print(f"\nSVC benchmark on PCA-compressed {CFG.n_components} features:")
    pca_acc = svc_benchmark(Z_train, y_train, Z_test, y_test, f"{CFG.n_components} PCA features")

    # 5. Warn if information loss is too high
    acc_drop = full_acc - pca_acc
    if acc_drop > 0.05:
        print(
            f"\n  WARNING: PCA accuracy drop is {acc_drop:.3f} (>{5}%).\n"
            f"  Consider increasing CFG.n_components or using the Autoencoder bottleneck.\n"
            f"  See src/bottleneck/autoencoder.py."
        )
    else:
        print(f"\n  PCA accuracy drop: {acc_drop:.3f} — within acceptable range (<5%).")

    # 6. Sanity checks
    assert Z_train.max() <= float(np.pi) + 1e-6, "PCA output re-scaling failed: max > π"
    assert Z_train.min() >= -1e-6, "PCA output re-scaling failed: min < 0"
    print("\nPCA output range check passed.")

    # 7. Save PCA arrays and fitted reducer
    print("\nSaving PCA-compressed arrays and reducer...")
    save_processed(
        {
            "Z_train":      Z_train,
            "Z_val":        Z_val,
            "Z_test":       Z_test,
            "pca_reducer":  reducer,
        },
        CFG.processed_dir,
    )

    print("\nStep 2 complete. Open notebooks/02_bottleneck.ipynb for variance plots.")


if __name__ == "__main__":
    main()
