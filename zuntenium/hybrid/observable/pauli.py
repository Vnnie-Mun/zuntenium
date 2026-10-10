"""
PauliString — Tensor products of single-qubit Pauli operators.

A PauliString is a multi-qubit operator of the form σ₀ ⊗ σ₁ ⊗ ... ⊗ σₙ₋₁
where each σᵢ ∈ {I, X, Y, Z}.

Examples:
    PauliString("ZZ")     → Z⊗Z on qubits [0,1]
    PauliString("XIZI")   → X on q0, I on q1, Z on q2, I on q3
"""
from __future__ import annotations

import numpy as np

_PAULI_MATRICES: dict[str, np.ndarray] = {
    "I": np.eye(2, dtype=np.complex128),
    "X": np.array([[0, 1], [1, 0]], dtype=np.complex128),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
    "Z": np.array([[1, 0], [0, -1]], dtype=np.complex128),
}


class PauliString:
    """
    A tensor product of single-qubit Pauli operators.

    Args:
        string: Pauli string, e.g. "ZZ", "XIZ", "IXYZ".
                Must contain only characters from {I, X, Y, Z}.

    Attributes:
        string: The Pauli string.
        num_qubits: Number of qubits this operator acts on.
    """

    def __init__(self, string: str) -> None:
        string = string.upper()
        invalid = set(string) - {"I", "X", "Y", "Z"}
        if invalid:
            raise ValueError(
                f"PauliString contains invalid characters: {invalid}. "
                "Only I, X, Y, Z are allowed."
            )
        self.string = string
        self.num_qubits = len(string)

    def to_matrix(self) -> np.ndarray:
        """Return the 2^n × 2^n matrix representation via tensor product."""
        result = _PAULI_MATRICES[self.string[0]]
        for char in self.string[1:]:
            result = np.kron(result, _PAULI_MATRICES[char])
        return result

    def expectation_value(self, statevector: np.ndarray) -> float:
        """Compute ⟨ψ|P|ψ⟩ for this Pauli string."""
        psi = statevector.flatten()
        mat = self.to_matrix()
        return float(np.real(psi.conj() @ mat @ psi))

    def __mul__(self, scalar: complex) -> tuple[complex, PauliString]:
        return (scalar, self)

    def __rmul__(self, scalar: complex) -> tuple[complex, PauliString]:
        return (scalar, self)

    def __repr__(self) -> str:
        return f"PauliString('{self.string}')"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PauliString):
            return NotImplemented
        return self.string == other.string

    def __hash__(self) -> int:
        return hash(self.string)
