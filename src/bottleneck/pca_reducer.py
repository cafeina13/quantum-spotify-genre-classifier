"""
pca_reducer.py — Classical dimensionality reduction via PCA.

Compresses 12 audio features down to n_components (default 6) principal
components, fitting the PCA on training data only to prevent leakage.

After projection, the output is re-scaled to [0, π] via a second
MinMaxScaler (also fitted on train only) because PCA output is unbounded.
"""

import pickle
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

from src.config import CFG


class PCAReducer:
    """
    Wraps sklearn PCA with the required [0, π] re-scaling step.

    Usage
    -----
    reducer = PCAReducer(n_components=6)
    Z_train = reducer.fit_transform(X_train_scaled)
    Z_val   = reducer.transform(X_val_scaled)
    Z_test  = reducer.transform(X_test_scaled)
    reducer.save(CFG.processed_dir / "pca_reducer.pkl")
    """

    def __init__(
        self,
        n_components: int = None,
        random_state: int = None,
    ):
        self.n_components  = n_components  or CFG.n_components
        self.random_state  = random_state  or CFG.random_seed
        self.pca           = PCA(n_components=self.n_components, random_state=self.random_state)
        self.post_scaler   = MinMaxScaler(feature_range=(-float(np.pi), float(np.pi)))
        self.is_fitted     = False

    # ------------------------------------------------------------------

    def fit(self, X_train: np.ndarray) -> "PCAReducer":
        """
        Fit PCA and the post-projection scaler on training data.

        Parameters
        ----------
        X_train : np.ndarray, shape (N_train, n_features)
            Should already be scaled to [0, π] by scale_features().

        Returns
        -------
        self
        """
        self.pca.fit(X_train)
        Z_train = self.pca.transform(X_train)
        self.post_scaler.fit(Z_train)   # fit post-scaler on PCA output of train
        self.is_fitted = True
        self._print_variance_report()
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Project X into PCA space and re-scale to [0, π].

        Parameters
        ----------
        X : np.ndarray, shape (N, n_features)

        Returns
        -------
        np.ndarray, shape (N, n_components), values in [0, π]
        """
        self._check_fitted()
        Z = self.pca.transform(X)
        return np.clip(self.post_scaler.transform(Z), -float(np.pi), float(np.pi))

    def fit_transform(self, X_train: np.ndarray) -> np.ndarray:
        """Fit on X_train and return its transformed version."""
        self.fit(X_train)
        Z_train = self.pca.transform(X_train)
        return self.post_scaler.transform(Z_train)

    # ------------------------------------------------------------------

    def explained_variance_report(self) -> dict:
        """
        Return a dict with per-component and cumulative explained variance.

        Example return value
        --------------------
        {
            "per_component":  [0.31, 0.22, 0.15, 0.10, 0.08, 0.06],
            "cumulative":     [0.31, 0.53, 0.68, 0.78, 0.86, 0.92],
            "total_explained": 0.92,
        }
        """
        self._check_fitted()
        evr        = self.pca.explained_variance_ratio_.tolist()
        cumulative = np.cumsum(evr).tolist()
        return {
            "per_component":   evr,
            "cumulative":      cumulative,
            "total_explained": cumulative[-1],
        }

    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """Pickle the entire PCAReducer (PCA + post_scaler) to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"PCAReducer saved to '{path}'.")

    @classmethod
    def load(cls, path: Path) -> "PCAReducer":
        """Load a previously saved PCAReducer from disk."""
        path = Path(path)
        with open(path, "rb") as f:
            obj = pickle.load(f)
        print(f"PCAReducer loaded from '{path}'.")
        return obj

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("PCAReducer has not been fitted yet. Call fit() or fit_transform() first.")

    def _print_variance_report(self) -> None:
        report = self.explained_variance_report()
        print(f"\nPCA Explained Variance Report ({self.n_components} components):")
        for i, (ev, cum) in enumerate(zip(report["per_component"], report["cumulative"])):
            print(f"  PC{i+1}: {ev:.3f}  (cumulative: {cum:.3f})")
        print(f"  Total explained variance: {report['total_explained']:.3f}")
        if report["total_explained"] < 0.75:
            print(
                "  WARNING: Less than 75% of variance explained. "
                "Consider increasing n_components."
            )
