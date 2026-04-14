"""
smoke_test.py — Quick sanity check before committing to full training.

Runs 2 batches through both models and verifies:
  - Imports work (pennylane, torch, qiskit, etc.)
  - Processed data loads correctly
  - Forward pass produces the right output shape
  - Output values are finite (no NaN / Inf)
  - Loss computes and backward pass flows gradients through the quantum layer
  - Classical baseline works the same way

Takes ~30 seconds on CPU. If this passes, run_training.py should complete
without surprises.

Run from the project root:
  python scripts/smoke_test.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

PASS = "[PASS]"
FAIL = "[FAIL]"


def check(label: str, condition: bool, detail: str = "") -> bool:
    status = PASS if condition else FAIL
    print(f"  {status}  {label}" + (f" — {detail}" if detail else ""))
    return condition


def main():
    all_ok = True

    # ------------------------------------------------------------------
    print("\n-- 1. Imports ----------------------------------------------")
    # ------------------------------------------------------------------
    try:
        import pennylane as qml
        check("pennylane", True, qml.__version__)
    except ImportError as e:
        all_ok = check("pennylane", False, str(e))

    try:
        import torch
        check("torch", True, torch.__version__)
    except ImportError as e:
        all_ok = check("torch", False, str(e))

    try:
        import sklearn
        check("scikit-learn", True, sklearn.__version__)
    except ImportError as e:
        all_ok = check("scikit-learn", False, str(e))

    try:
        import qiskit
        check("qiskit", True, qiskit.__version__)
    except ImportError as e:
        all_ok = check("qiskit", False, str(e))

    # ------------------------------------------------------------------
    print("\n-- 2. Config & processed data ------------------------------")
    # ------------------------------------------------------------------
    try:
        from src.config import CFG
        check("config loaded", True, f"n_qubits={CFG.n_qubits}, n_classes={len(CFG.genre_classes)}")
    except Exception as e:
        print(f"  {FAIL}  config — {e}")
        sys.exit(1)   # nothing else can run without config

    try:
        from src.data.preprocessor import load_processed
        data = load_processed(CFG.processed_dir)
        has_keys = all(k in data for k in ["Z_train", "Z_val", "y_train", "y_val", "encoder"])
        all_ok &= check("processed data keys", has_keys, str(list(data.keys())))
        Z_train, y_train = data["Z_train"], data["y_train"]
        all_ok &= check(
            "Z_train shape",
            Z_train.shape[1] == CFG.n_qubits,
            f"{Z_train.shape} — expected (N, {CFG.n_qubits})"
        )
        import numpy as np
        all_ok &= check(
            "Z_train range [-pi, pi]",
            Z_train.min() >= -float(np.pi) - 1e-6 and Z_train.max() <= float(np.pi) + 1e-6,
            f"[{Z_train.min():.4f}, {Z_train.max():.4f}]"
        )
        all_ok &= check(
            "label classes",
            len(data["encoder"].classes_) == len(CFG.genre_classes),
            str(list(data["encoder"].classes_))
        )
    except Exception as e:
        print(f"  {FAIL}  loading processed data — {e}")
        print("         Did you run scripts/run_bottleneck.py first?")
        sys.exit(1)

    # ------------------------------------------------------------------
    print("\n-- 3. Quantum device ---------------------------------------")
    # ------------------------------------------------------------------
    try:
        from src.quantum.device import get_device
        device = get_device(use_ibm_hardware=False)
        all_ok &= check("default.qubit device", device is not None)
    except Exception as e:
        all_ok &= check("default.qubit device", False, str(e))

    # ------------------------------------------------------------------
    print("\n-- 4. Circuit draw -----------------------------------------")
    # ------------------------------------------------------------------
    try:
        from src.quantum.circuit import draw_circuit
        draw_circuit()
        all_ok &= check("circuit diagram", True)

    except Exception as e:
        all_ok &= check("circuit diagram", False, str(e))

    # ------------------------------------------------------------------
    print("\n-- 5. Hybrid model — forward pass --------------------------")
    # ------------------------------------------------------------------
    try:
        from src.models.hybrid_model import HybridGenreClassifier

        model = HybridGenreClassifier(
            device=get_device(use_ibm_hardware=False),
            input_dim=len(CFG.audio_features),
        )
        n_params = model.count_parameters()
        all_ok &= check("model instantiated", True, f"{n_params} trainable parameters")

        # 2-sample mini-batch — hybrid now takes full 12 features (X_train)
        X_batch = torch.tensor(data["X_train"][:2], dtype=torch.float32)
        y_batch = torch.tensor(y_train[:2], dtype=torch.long)

        logits = model(X_batch)
        all_ok &= check(
            "output shape",
            logits.shape == (2, len(CFG.genre_classes)),
            f"{tuple(logits.shape)}"
        )
        all_ok &= check(
            "output finite",
            torch.isfinite(logits).all().item(),
            f"min={logits.min().item():.4f}  max={logits.max().item():.4f}"
        )
    except Exception as e:
        all_ok &= check("hybrid forward pass", False, str(e))
        model = None

    # ------------------------------------------------------------------
    print("\n-- 6. Hybrid model — backward pass -------------------------")
    # ------------------------------------------------------------------
    if model is not None:
        try:
            criterion = nn.CrossEntropyLoss()
            loss = criterion(logits, y_batch)
            loss.backward()

            q_grad = model.quantum_layer.weights.grad
            all_ok &= check(
                "quantum layer gradients",
                q_grad is not None and torch.isfinite(q_grad).all().item(),
                f"grad norm={q_grad.norm().item():.6f}" if q_grad is not None else "None"
            )
            all_ok &= check("loss finite", torch.isfinite(loss).item(), f"loss={loss.item():.6f}")
        except Exception as e:
            all_ok &= check("backward pass", False, str(e))

    # ------------------------------------------------------------------
    print("\n-- 7. Classical baseline -----------------------------------")
    # ------------------------------------------------------------------
    try:
        from src.models.classical_baseline import ClassicalBaseline
        from src.data.preprocessor import load_processed

        baseline = ClassicalBaseline(input_dim=len(CFG.audio_features))
        X_full = torch.tensor(data["X_train"][:2], dtype=torch.float32)
        y_b    = torch.tensor(data["y_train"][:2], dtype=torch.long)

        out  = baseline(X_full)
        loss = nn.CrossEntropyLoss()(out, y_b)
        loss.backward()

        all_ok &= check(
            "baseline forward+backward",
            torch.isfinite(out).all().item() and torch.isfinite(loss).item(),
            f"output shape {tuple(out.shape)}"
        )
    except Exception as e:
        all_ok &= check("classical baseline", False, str(e))

    # ------------------------------------------------------------------
    print("\n" + "-" * 60)
    if all_ok:
        print("  All checks passed. Safe to run scripts/run_training.py.")
    else:
        print("  One or more checks FAILED. Fix the issues above before training.")
    print("-" * 60 + "\n")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
