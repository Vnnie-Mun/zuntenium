"""
Type stubs for the _zuntenium_core C++ extension module.

Provides IDE autocomplete, hover documentation, and static type checking
for the compiled pybind11 native engine.
"""
from __future__ import annotations

from typing import Sequence

__version__: str
__backend__: str

def is_cpp_available() -> bool:
    """Returns True if the compiled C++ extension is loaded."""
    ...

def create_statevector(num_qubits: int) -> StateVector:
    """Factory function to allocate a new C++ StateVector simulator."""
    ...

class StateVector:
    """
    High-performance C++ statevector simulator.

    Memory layout: amplitudes[i] = coefficient of |i⟩.
    Qubit 0 is the most significant bit.

    Args:
        num_qubits: Number of qubits (1–30).
    """

    def __init__(self, num_qubits: int) -> None: ...
    def apply_single_qubit_gate(self, matrix: Sequence[complex], qubit: int) -> None:
        """Apply a 2×2 unitary (4 complex values, row-major) to qubit."""
        ...
    def apply_two_qubit_gate(self, matrix: Sequence[complex], q0: int, q1: int) -> None:
        """Apply a 4×4 unitary (16 complex values, row-major) to qubits (q0, q1)."""
        ...
    def get_amplitudes(self) -> list[complex]:
        """Return the complete statevector as complex amplitudes."""
        ...
    def get_probabilities(self) -> list[float]:
        """Return measurement probabilities for all 2^N basis states."""
        ...
    def sample_measurements(self, shots: int, seed: int = 42) -> list[int]:
        """Sample discrete measurement outcomes in [0, 2^N - 1]."""
        ...
    def num_qubits(self) -> int:
        """Return number of qubits simulated."""
        ...
    def dim(self) -> int:
        """Return Hilbert space dimension (2^N)."""
        ...
    def reset(self) -> None:
        """Reset statevector to |0...0⟩."""
        ...

class gates:
    """Standard quantum gate matrices as flat lists of complex numbers."""

    @staticmethod
    def i_matrix() -> list[complex]: ...
    @staticmethod
    def x_matrix() -> list[complex]: ...
    @staticmethod
    def y_matrix() -> list[complex]: ...
    @staticmethod
    def z_matrix() -> list[complex]: ...
    @staticmethod
    def h_matrix() -> list[complex]: ...
    @staticmethod
    def s_matrix() -> list[complex]: ...
    @staticmethod
    def t_matrix() -> list[complex]: ...
    @staticmethod
    def rx_matrix(theta: float) -> list[complex]: ...
    @staticmethod
    def ry_matrix(theta: float) -> list[complex]: ...
    @staticmethod
    def rz_matrix(theta: float) -> list[complex]: ...
    @staticmethod
    def u3_matrix(theta: float, phi: float, lam: float) -> list[complex]: ...
    @staticmethod
    def cx_matrix() -> list[complex]: ...
    @staticmethod
    def cz_matrix() -> list[complex]: ...
    @staticmethod
    def swap_matrix() -> list[complex]: ...
