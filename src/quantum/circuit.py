"""
circuit.py — Variational Quantum Circuit (VQC) definition.

Architecture
------------
1. AngleEmbedding (RY gates)
   Each of the 6 classical features is encoded as a rotation angle on one qubit:
     RY(x_i) |0⟩  for i in 0..n_qubits-1
   This is O(n) in circuit depth and preserves the continuous structure of
   the audio features. Because inputs are pre-scaled to [0, π], the rotation
   spans the full |0⟩ → |1⟩ range.

2. StronglyEntanglingLayers (Ansatz)
   PennyLane's built-in template applies L layers of:
     - Single-qubit Rot gates (Rz·Ry·Rz) on every qubit — parameterized.
     - A structured CNOT entanglement pattern that creates long-range correlations.
   For n=6 qubits and L=2 layers: 2 × 6 × 3 = 36 trainable parameters.

3. Pauli-Z measurements
   Expectation value <Z_i> on every qubit, returning a 6-dim vector in [-1, 1].
   The 6-dim output feeds directly into the 6-class classification head.

Gradient method: parameter-shift rule
--------------------------------------
diff_method="parameter-shift" computes exact gradients by evaluating the
circuit at θ ± π/2. It works identically on simulators AND real IBM hardware,
so no code change is needed when switching to hardware.
(backprop would be faster on a simulator but fails on hardware.)
"""

import numpy as np
import pennylane as qml
import torch

from src.config import CFG
from src.quantum.device import get_device


def build_vqc_circuit(
    n_qubits: int = None,
    n_layers: int = None,
    device: qml.devices.Device = None,
) -> tuple:
    """
    Build the Variational Quantum Circuit (QNode) for use with TorchLayer.

    Parameters
    ----------
    n_qubits : int
        Number of qubits (= latent dimension from bottleneck). Default: CFG.n_qubits.
    n_layers : int
        Number of StronglyEntanglingLayers repetitions. Default: CFG.n_layers.
    device : qml.Device
        PennyLane device. If None, get_device() is called with CFG defaults.

    Returns
    -------
    qnode : callable
        QNode with signature qnode(inputs, weights) → list of n_qubits tensors.
        Wrapped by qml.qnn.TorchLayer in hybrid_model.py.
    weight_shapes : dict
        {"weights": (n_layers, n_qubits, 3)} — required by TorchLayer to
        know how many trainable parameters to allocate.
    """
    n_qubits = n_qubits or CFG.n_qubits
    n_layers = n_layers or CFG.n_layers
    if device is None:
        device = get_device()

    @qml.qnode(device, interface="torch", diff_method="parameter-shift")
    def circuit(inputs: torch.Tensor, weights: torch.Tensor):
        """
        inputs  : shape (n_qubits,)  — one feature per qubit, in [0, π]
        weights : shape (n_layers, n_qubits, 3) — ansatz parameters

        Returns a list of n_qubits scalar expectation values.
        """
        # Step 1: Encode classical features as RY rotation angles
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")

        # Step 2: Apply parameterized entangling ansatz
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))

        # Step 3: Measure Pauli-Z expectation on every qubit → values in [-1, 1]
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    # weight_shapes tells TorchLayer the shape of each trainable tensor
    weight_shapes = {"weights": (n_layers, n_qubits, 3)}

    return circuit, weight_shapes


def draw_circuit(n_qubits: int = None, n_layers: int = None) -> None:
    """
    Print an ASCII diagram of the VQC for visual inspection.

    Call this after build_vqc_circuit() to verify the encoding and ansatz
    layers appear correctly before training.

    Example usage
    -------------
    from src.quantum.circuit import draw_circuit
    draw_circuit()
    """
    import numpy as np

    n_qubits = n_qubits or CFG.n_qubits
    n_layers = n_layers or CFG.n_layers

    # Build circuit on a local device just for drawing
    dev     = qml.device("default.qubit", wires=n_qubits)
    circuit, weight_shapes = build_vqc_circuit(n_qubits, n_layers, dev)

    dummy_inputs  = torch.zeros(n_qubits)
    dummy_weights = torch.zeros(*weight_shapes["weights"])

    print("\nVQC Circuit Diagram:")
    print(qml.draw(circuit)(dummy_inputs, dummy_weights))
