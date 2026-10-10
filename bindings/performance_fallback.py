"""
performance_fallback.py — Pure NumPy fallback for C++ extension functions.

Provides the same API as the compiled _zuntenium_core module so that
code calling bindings/loader.py works identically regardless of whether
the C++ extension has been compiled.
"""
from __future__ import annotations

import numpy as np

from zuntenium.quantum_core.statevector.statevector_numpy import (
    apply_n_qubit_gate,
    apply_single_qubit_gate,
    apply_two_qubit_gate,
)


class StateVectorNumPy:
    """
    Pure NumPy StateVector with the same interface as the C++ StateVectorCPP.

    This allows the Python layer to call the same API regardless of backend.
    """

    def __init__(self, num_qubits: int) -> None:
        self._num_qubits = num_qubits
        self._dim = 2 ** num_qubits
        self._amplitudes = np.zeros(self._dim, dtype=np.complex128)
        self._amplitudes[0] = 1.0

    def apply_single_qubit_gate(
        self, matrix_flat: list[complex], qubit: int
    ) -> None:
        """Apply a 2×2 gate given as a flat list of 4 complex values."""
        matrix = np.array(matrix_flat, dtype=np.complex128).reshape(2, 2)
        self._amplitudes = apply_single_qubit_gate(
            self._amplitudes, matrix, qubit, self._num_qubits
        )

    def apply_two_qubit_gate(
        self, matrix_flat: list[complex], qubit0: int, qubit1: int
    ) -> None:
        """Apply a 4×4 gate given as a flat list of 16 complex values."""
        matrix = np.array(matrix_flat, dtype=np.complex128).reshape(4, 4)
        self._amplitudes = apply_two_qubit_gate(
            self._amplitudes, matrix, qubit0, qubit1, self._num_qubits
        )

    def apply_n_qubit_gate(
        self, matrix_flat: list[complex], qubits: list[int]
    ) -> None:
        """Apply a general k-qubit gate."""
        k = len(qubits)
        dim = 2 ** k
        matrix = np.array(matrix_flat, dtype=np.complex128).reshape(dim, dim)
        self._amplitudes = apply_n_qubit_gate(
            self._amplitudes, matrix, qubits, self._num_qubits
        )

    def get_amplitudes(self) -> list[complex]:
        """Return amplitudes as a Python list (mirrors C++ API)."""
        return self._amplitudes.tolist()

    def get_probabilities(self) -> list[float]:
        """Return measurement probabilities as a Python list."""
        return (np.abs(self._amplitudes) ** 2).tolist()

    def sample(self, shots: int, seed: int | None = None) -> list[int]:
        """
        Sample ``shots`` computational basis states.

        Returns a list of integer state indices.
        """
        probs = np.abs(self._amplitudes) ** 2
        probs /= probs.sum()
        rng = np.random.default_rng(seed)
        return rng.choice(self._dim, size=shots, p=probs).tolist()

    @property
    def num_qubits(self) -> int:
        return self._num_qubits

    def __repr__(self) -> str:
        return f"StateVectorNumPy(num_qubits={self._num_qubits})"


def is_cpp_available() -> bool:
    """Always False for the NumPy fallback module."""
    return False


# Mirror the _zuntenium_core module API
def create_statevector(num_qubits: int) -> StateVectorNumPy:
    """Factory: create a NumPy-backed StateVector."""
    return StateVectorNumPy(num_qubits)
