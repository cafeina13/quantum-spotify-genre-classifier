"""
device.py — PennyLane device factory.

Returns the appropriate PennyLane device based on configuration:
  - Local simulator : "default.qubit"   (fast, no installation required)
  - IBM real hardware: "qiskit.remote"  (pennylane-qiskit 0.38+, qiskit-ibm-runtime 0.43+)

The old "qiskit.ibmq" device name was removed in pennylane-qiskit 0.38.
IBM hardware is now accessed via QiskitRuntimeService + qiskit.remote device.

IBM token is loaded from the .env file via python-dotenv. It is NEVER hardcoded.
"""

import os
from pathlib import Path

import pennylane as qml
from dotenv import load_dotenv

from src.config import CFG

# Load .env file from the project root (one level above src/)
load_dotenv(Path(__file__).resolve().parents[2] / ".env")


def get_device(
    use_ibm_hardware: bool = None,
    ibm_backend: str = None,
    n_qubits: int = None,
):
    """
    Create and return a PennyLane device.

    Parameters
    ----------
    use_ibm_hardware : bool
        False → local default.qubit simulator (default from CFG).
        True  → IBM hardware via qiskit.remote + QiskitRuntimeService.
    ibm_backend : str
        IBM backend name, e.g. "ibm_brisbane" (only used when use_ibm_hardware=True).
    n_qubits : int
        Number of qubits / wires (default from CFG.n_qubits).

    Returns
    -------
    PennyLane device instance.

    Raises
    ------
    EnvironmentError
        If use_ibm_hardware=True but IBM_QUANTUM_TOKEN is not set in .env.
    """
    use_ibm_hardware = use_ibm_hardware if use_ibm_hardware is not None else CFG.use_ibm_hardware
    ibm_backend      = ibm_backend      or CFG.ibm_backend
    n_qubits         = n_qubits         or CFG.n_qubits

    if not use_ibm_hardware:
        device = qml.device("default.qubit", wires=n_qubits)
        print(f"Using local simulator: default.qubit ({n_qubits} qubits)")
        return device

    # --- IBM hardware path (pennylane-qiskit 0.38+ / qiskit-ibm-runtime 0.43+) ---
    token = os.getenv("IBM_QUANTUM_TOKEN")
    if not token:
        raise EnvironmentError(
            "IBM_QUANTUM_TOKEN not found in environment.\n"
            "Add it to your .env file:\n"
            "  IBM_QUANTUM_TOKEN=your_token_here\n"
            "Get your token at: https://quantum.ibm.com/"
        )

    from qiskit_ibm_runtime import QiskitRuntimeService

    service = QiskitRuntimeService(channel="ibm_quantum_platform", token=token)
    backend = service.backend(ibm_backend)

    # "qiskit.remote" replaces the removed "qiskit.ibmq" device name
    device = qml.device("qiskit.remote", wires=n_qubits, backend=backend)
    print(f"Using IBM hardware backend: {ibm_backend} ({n_qubits} qubits)")
    return device
