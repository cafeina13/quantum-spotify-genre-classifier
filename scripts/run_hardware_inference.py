"""
run_hardware_inference.py — Optimized IBM hardware inference via Qiskit EstimatorV2.

Architecture of this script
----------------------------
Instead of PennyLane's qiskit.remote (1 IBM job per sample = quota disaster),
this script uses Qiskit's native EstimatorV2 primitive to submit ALL samples
as a single batched job.

How it works
------------
The hybrid model splits cleanly into three parts:

  [Classical Encoder]  12 -> 32 -> 64 -> 6  (PyTorch, runs on CPU, free)
          |
  [Quantum VQC]        6 -> 6               (Qiskit EstimatorV2, 1 job total)
          |
  [Classical Decoder]  BN -> 6->16->6       (PyTorch, runs on CPU, free)

Step 1: Run the encoder on all test samples locally → angles (n_samples, 6)
Step 2: Build a Qiskit ParameterizedCircuit for the VQC with FIXED trained
        quantum weights baked in, only the 6 input angles are parameters.
Step 3: Transpile at optimization_level=3 → Qiskit picks best physical qubits,
        minimises SWAP overhead, decomposes to native gate set (ECR/RZ/SX/X).
Step 4: Submit one EstimatorV2 job: all n_samples angle vectors + 6 PauliZ
        observables → get back (n_samples, 6) expectation values in one call.
Step 5: Run the decoder on the expectation values locally → logits → predictions.

Result: 1 IBM job instead of n_samples jobs. Quota-efficient.

Circuit reconstruction
----------------------
StronglyEntanglingLayers (n_qubits=6, n_layers=2, ranges=[1,2]):
  - AngleEmbedding: RY(x_i) on qubit i
  - Layer 0: Rot(w[0,i]) on each qubit, then CNOT(i, (i+1)%6)
  - Layer 1: Rot(w[1,i]) on each qubit, then CNOT(i, (i+2)%6)
  PennyLane Rot(phi,theta,omega) = RZ(phi) -> RY(theta) -> RZ(omega) in circuit order.

Run from project root:
  py -3.12 -u scripts/run_hardware_inference.py
  (-u forces unbuffered stdout so progress prints immediately)
"""

import sys
import json
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from src.config import CFG
from src.data.preprocessor import load_processed
from src.models.hybrid_model import HybridGenreClassifier

RESULTS_PATH = Path(CFG.results_dir) / "hardware_inference_results.json"
SAMPLE_SIZE  = 128


# ---------------------------------------------------------------------------
# Step 1 — Build the Qiskit VQC circuit from trained weights
# ---------------------------------------------------------------------------

def build_qiskit_vqc(quantum_weights: np.ndarray, n_qubits: int = 6) -> "QuantumCircuit":
    """
    Reconstruct the PennyLane VQC as a Qiskit ParameterizedCircuit.

    quantum_weights : np.ndarray, shape (n_layers, n_qubits, 3)
        Trained Rot gate angles — FIXED, baked directly into the circuit.
        Only the 6 AngleEmbedding inputs remain as free Parameters.

    Returns a QuantumCircuit with ParameterVector("x", n_qubits).
    """
    from qiskit.circuit import QuantumCircuit, ParameterVector

    n_layers = quantum_weights.shape[0]
    inputs   = ParameterVector("x", n_qubits)

    qc = QuantumCircuit(n_qubits)

    # --- AngleEmbedding: RY(x_i) on each qubit ---
    for i in range(n_qubits):
        qc.ry(inputs[i], i)

    # --- StronglyEntanglingLayers ---
    # PennyLane Rot(phi, theta, omega) applies: RZ(phi) then RY(theta) then RZ(omega)
    # Default ranges for n_layers=2: [1, 2]
    ranges = list(range(1, n_layers + 1))

    for l in range(n_layers):
        # Rot gates with trained (fixed) weights
        for i in range(n_qubits):
            phi   = float(quantum_weights[l, i, 0])
            theta = float(quantum_weights[l, i, 1])
            omega = float(quantum_weights[l, i, 2])
            qc.rz(phi, i)
            qc.ry(theta, i)
            qc.rz(omega, i)
        # CNOT entanglement with range r
        r = ranges[l]
        for i in range(n_qubits):
            qc.cx(i, (i + r) % n_qubits)

    return qc


# ---------------------------------------------------------------------------
# Step 2 — Transpile for target backend
# ---------------------------------------------------------------------------

def transpile_circuit(qc, backend, optimization_level: int = 3):
    """
    Transpile the logical circuit to the physical backend.

    optimization_level=3:
    - Picks the 6 physical qubits with lowest gate error rates from calibration data
    - Finds shortest SWAP paths for non-adjacent 2-qubit gates
    - Cancels redundant gates (e.g. RZ(0), back-to-back CNOTs)
    - Decomposes to native gate set: ECR, RZ, SX, X (for ibm_fez)
    """
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    print(f"  Original circuit: depth={qc.depth()}, gates={dict(qc.count_ops())}", flush=True)

    pm  = generate_preset_pass_manager(backend=backend, optimization_level=optimization_level)
    isa = pm.run(qc)

    print(f"  Transpiled circuit: depth={isa.depth()}, gates={dict(isa.count_ops())}", flush=True)
    print(f"  Physical qubits selected: {sorted(isa.layout.final_layout.get_physical_bits().keys())}", flush=True)

    return isa


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60, flush=True)
    print("IBM Hardware Inference — Optimized (EstimatorV2 batched)", flush=True)
    print("=" * 60, flush=True)

    # --- Load data ---
    data        = load_processed(CFG.processed_dir)
    class_names = list(data["encoder"].classes_)
    X_test      = data["X_test"]
    y_test      = data["y_test"]

    rng = np.random.default_rng(CFG.random_seed)
    idx = rng.choice(len(X_test), size=min(SAMPLE_SIZE, len(X_test)), replace=False)
    X_hw = torch.tensor(X_test[idx], dtype=torch.float32)
    y_hw = y_test[idx]

    print(f"\nSamples to evaluate : {len(X_hw)}", flush=True)
    print(f"Results will save to : {RESULTS_PATH}", flush=True)

    # --- Load checkpoint and reconstruct model ---
    print("\nLoading checkpoint ...", flush=True)
    checkpoint_path = Path(CFG.models_dir) / "hybrid_qnn_best.pt"
    checkpoint      = torch.load(checkpoint_path, map_location="cpu")

    model = HybridGenreClassifier(n_qubits=CFG.n_qubits, n_layers=CFG.n_layers)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # --- Step 1: Run encoder locally (free, no IBM quota) ---
    print("\nStep 1 — Running encoder on CPU ...", flush=True)
    with torch.no_grad():
        angles = (model.encoder(X_hw) * math.pi).numpy()  # (n_samples, 6)
    print(f"  Angles shape: {angles.shape}  range: [{angles.min():.3f}, {angles.max():.3f}]", flush=True)

    # --- Simulator baseline (also free) ---
    print("\nRunning simulator baseline on same subset ...", flush=True)
    with torch.no_grad():
        sim_preds = model(X_hw).argmax(dim=1).numpy()
    sim_acc = (sim_preds == y_hw).mean()
    print(f"  Simulator accuracy: {sim_acc*100:.2f}%", flush=True)

    # --- Step 2: Build Qiskit VQC circuit ---
    print("\nStep 2 — Building Qiskit VQC circuit ...", flush=True)
    quantum_weights = model.quantum_layer.weights.detach().numpy()  # (2, 6, 3)
    qc = build_qiskit_vqc(quantum_weights, n_qubits=CFG.n_qubits)
    print(f"  Logical circuit built: {qc.num_qubits} qubits, {qc.num_parameters} input parameters", flush=True)

    # --- Connect to IBM ---
    print(f"\nConnecting to IBM backend: {CFG.ibm_backend} ...", flush=True)
    from dotenv import load_dotenv
    import os
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    token = os.getenv("IBM_QUANTUM_TOKEN")

    from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2
    service = QiskitRuntimeService(channel="ibm_quantum_platform", token=token)
    backend = service.backend(CFG.ibm_backend)
    status  = backend.status()
    print(f"  Backend: {backend.name}  qubits={backend.num_qubits}  pending_jobs={status.pending_jobs}", flush=True)

    # --- Step 3: Transpile ---
    print("\nStep 3 — Transpiling circuit (optimization_level=3) ...", flush=True)
    isa_circuit = transpile_circuit(qc, backend, optimization_level=3)

    # --- Define PauliZ observables, apply transpiled layout ---
    from qiskit.quantum_info import SparsePauliOp
    n_qubits = CFG.n_qubits
    z_ops = []
    for i in range(n_qubits):
        pauli = "I" * (n_qubits - 1 - i) + "Z" + "I" * i
        z_ops.append(SparsePauliOp(pauli).apply_layout(isa_circuit.layout))
    obs_array = np.array(z_ops)  # shape (6,) — one Z observable per qubit

    # --- Step 4: Submit one batched EstimatorV2 job ---
    # param_values shape (n_samples, n_input_params) broadcasts against obs_array (6,)
    # Result evs shape: (n_samples, 6)
    print(f"\nStep 4 — Submitting EstimatorV2 job ({len(X_hw)} samples, 6 observables) ...", flush=True)
    print("  This is ONE IBM job — not one per sample.", flush=True)

    estimator = EstimatorV2(mode=backend)
    pub       = (isa_circuit, obs_array, angles)   # angles shape: (n_samples, 6)
    job       = estimator.run([pub])

    print(f"  Job submitted. Job ID: {job.job_id()}", flush=True)
    print("  Waiting for results ...", flush=True)

    result = job.result()
    evs    = result[0].data.evs  # (n_samples, 6) expectation values in [-1, 1]
    print(f"  Results received. EVs shape: {evs.shape}", flush=True)

    # --- Step 5: Run decoder locally ---
    print("\nStep 5 — Running decoder on CPU ...", flush=True)
    q_out = torch.tensor(evs, dtype=torch.float32)  # (n_samples, 6)
    with torch.no_grad():
        logits = model.decoder(q_out)               # BN uses trained running stats
    hw_preds = logits.argmax(dim=1).numpy()
    hw_acc   = (hw_preds == y_hw).mean()

    # --- Save results ---
    results = {
        "backend":          CFG.ibm_backend,
        "n_samples":        len(y_hw),
        "simulator_acc":    float(sim_acc),
        "hardware_acc":     float(hw_acc),
        "nisq_penalty":     float(sim_acc - hw_acc),
        "y_true":           y_hw.tolist(),
        "y_pred_hw":        hw_preds.tolist(),
        "y_pred_sim":       sim_preds.tolist(),
        "evs":              evs.tolist(),
        "job_id":           job.job_id(),
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    # --- Report ---
    print("\n" + "=" * 60, flush=True)
    print("Hardware Inference Results", flush=True)
    print("=" * 60, flush=True)
    print(f"Backend           : {CFG.ibm_backend}", flush=True)
    print(f"Samples evaluated : {len(y_hw)}", flush=True)
    print(f"Simulator accuracy: {sim_acc*100:.2f}%", flush=True)
    print(f"Hardware accuracy : {hw_acc*100:.2f}%", flush=True)
    print(f"NISQ noise penalty: {(sim_acc - hw_acc)*100:.2f}%", flush=True)

    print("\nPer-class accuracy (hardware):", flush=True)
    for i, name in enumerate(class_names):
        mask = y_hw == i
        if mask.sum() == 0:
            continue
        cls_acc = (hw_preds[mask] == y_hw[mask]).mean()
        print(f"  {name:<8}: {cls_acc*100:.1f}%  (n={mask.sum()})", flush=True)

    print(f"\nResults saved to: {RESULTS_PATH}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
