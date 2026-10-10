"""
C++ binding adapter for StateVector.

Wraps the pybind11-compiled _zuntenium_cpp.StateVectorCPP in a thin
Python class that mirrors the StateVector interface.

Falls back gracefully to NumPy if the C++ module is unavailable.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from zuntenium.quantum_core.exceptions import ZunteniumError


class StatevectorCPPAdapter:
    """
    Thin Python adapter around StateVectorCPP.

    Keeps the same interface as statevector_numpy functions so
    StateVector can dispatch to either backend transparently.
    """

    def __init__(self, num_qubits: int) -> None:
        self._num_qubits = num_qubits
        self._handle: Any = self._load(num_qubits)

    @staticmethod
    def _load(num_qubits: int) -> Any:
        try:
            from zuntenium.bindings.binding_loader import load_cpp_module
            cpp = load_cpp_module()
            return cpp.StateVectorCPP(num_qubits)
        except Exception as exc:
            raise ZunteniumError(
                "C++ statevector module is not available. "
                "Run 'bash scripts/build_bindings.sh' to compile it, "
                "or use backend='numpy'."
            ) from exc

    def apply_single_qubit_gate(
        self, matrix: np.ndarray, qubit: int
    ) -> None:
        """Apply 2×2 gate in-place on the C++ state."""
        flat = matrix.flatten().tolist()
        self._handle.apply_single_qubit_gate(flat, qubit)

    def apply_two_qubit_gate(
        self, matrix: np.ndarray, qubit0: int, qubit1: int
    ) -> None:
        """Apply 4×4 gate in-place on the C++ state."""
        flat = matrix.flatten().tolist()
        self._handle.apply_two_qubit_gate(flat, qubit0, qubit1)

    def get_probabilities(self) -> np.ndarray:
        probs = self._handle.get_probabilities()
        return np.array(probs, dtype=np.float64)

    def get_amplitudes(self) -> np.ndarray:
        amps = self._handle.get_amplitudes()
        return np.array(amps, dtype=np.complex128)

    def measure(self, shots: int, seed: int = 0) -> dict[str, int]:
        return self._handle.measure(shots, seed)

