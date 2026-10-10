"""
fallback.py — Pure-Python fallback when native C++/Rust extensions are unavailable.

Re-exports the NumPy-based statevector API so that callers do not need to
know whether the compiled extension is present.
"""
from __future__ import annotations

import numpy as np


class StateVectorFallback:
    """
    Pure-Python (NumPy) statevector that matches the C++ StateVector API.

    This is used automatically when _zuntenium_core is not compiled.
    """

    def __init__(self, num_qubits: int) -> None:
        if num_qubits < 1 or num_qubits > 30:
            raise ValueError(f"num_qubits must be in [1, 30], got {num_qubits}")
        self._num_qubits = num_qubits
        self._dim = 2 ** num_qubits
        self._amplitudes = np.zeros(self._dim, dtype=np.complex128)
        self._amplitudes[0] = 1.0

    def apply_single_qubit_gate(self, matrix: list[complex], qubit: int) -> None:
        """Apply a 2×2 unitary (4 elements, row-major) to the given qubit."""
        from zuntenium.quantum_core.statevector.statevector_numpy import (
            apply_single_qubit_gate,
        )
        mat = np.array(matrix, dtype=np.complex128).reshape(2, 2)
        self._amplitudes = apply_single_qubit_gate(
            self._amplitudes, mat, qubit, self._num_qubits
        )

    def apply_two_qubit_gate(self, matrix: list[complex], q0: int, q1: int) -> None:
        """Apply a 4×4 unitary (16 elements, row-major) to qubits (q0, q1)."""
        from zuntenium.quantum_core.statevector.statevector_numpy import (
            apply_two_qubit_gate,
        )
        mat = np.array(matrix, dtype=np.complex128).reshape(4, 4)
        self._amplitudes = apply_two_qubit_gate(
            self._amplitudes, mat, q0, q1, self._num_qubits
        )

    def get_amplitudes(self) -> list[complex]:
        return self._amplitudes.tolist()

    def get_probabilities(self) -> list[float]:
        return (np.abs(self._amplitudes) ** 2).tolist()

    def sample_measurements(self, shots: int, seed: int = 42) -> list[int]:
        rng = np.random.default_rng(seed)
        probs = np.abs(self._amplitudes) ** 2
        probs /= probs.sum()
        return rng.choice(self._dim, size=shots, p=probs).tolist()

    def num_qubits(self) -> int:
        return self._num_qubits

    def dim(self) -> int:
        return self._dim

    def reset(self) -> None:
        self._amplitudes[:] = 0.0
        self._amplitudes[0] = 1.0

    def __repr__(self) -> str:
        return f"<StateVector(numpy) num_qubits={self._num_qubits}>"


def is_cpp_available() -> bool:
    """Always False — this is the fallback module."""
    return False
