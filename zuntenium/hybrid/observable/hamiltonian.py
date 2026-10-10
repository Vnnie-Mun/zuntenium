"""
Hamiltonian — Linear combination of Pauli operators.

H = Σᵢ cᵢ · Pᵢ   where cᵢ ∈ ℂ and Pᵢ are PauliStrings.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from zuntenium.hybrid.observable.pauli import PauliString

if TYPE_CHECKING:
    from zuntenium.quantum_core.executor.result import ExecutionResult


class Hamiltonian:
    """
    Hamiltonian as a linear combination of Pauli operators.

    H = Σᵢ coeffᵢ · PauliStringᵢ

    Args:
        terms: List of (coefficient, PauliString) tuples.

    Example::

        H = Hamiltonian([
            (0.5,  PauliString("ZZ")),
            (-0.3, PauliString("XX")),
        ])
        energy = H.expectation_value(result)
    """

    def __init__(self, terms: list[tuple[complex, PauliString]]) -> None:
        if not terms:
            raise ValueError("Hamiltonian must have at least one term.")
        self.terms = terms
        self.num_qubits = terms[0][1].num_qubits
        # Validate all terms have the same qubit count
        for coeff, ps in terms:
            if ps.num_qubits != self.num_qubits:
                raise ValueError(
                    f"All PauliStrings must have the same length. "
                    f"Expected {self.num_qubits}, got {ps.num_qubits} for '{ps.string}'."
                )

    def expectation_value(self, result: ExecutionResult) -> float:
        """
        Compute ⟨ψ|H|ψ⟩ from an ExecutionResult.

        Uses the statevector if available (simulator), otherwise falls back
        to estimating from measurement counts using the diagonal terms.

        Args:
            result: ExecutionResult from backend.run().

        Returns:
            Real-valued expectation value.
        """
        if result.statevector is not None:
            return self._sv_expectation(result.statevector)
        return self._counts_expectation(result.counts, result.shots)

    def _sv_expectation(self, statevector: np.ndarray) -> float:
        """⟨ψ|H|ψ⟩ via exact statevector computation."""
        total = 0.0
        for coeff, ps in self.terms:
            total += float(np.real(coeff)) * ps.expectation_value(statevector)
        return total

    def _counts_expectation(self, counts: dict[str, int], shots: int) -> float:
        """Estimate ⟨H⟩ from measurement counts (diagonal Hamiltonian only)."""
        total = 0.0
        for coeff, ps in self.terms:
            # Only diagonal Pauli strings (I and Z only) can be estimated from counts
            if all(c in ("I", "Z") for c in ps.string):
                ev = 0.0
                for bitstring, count in counts.items():
                    eigenvalue = 1.0
                    for i, pauli in enumerate(reversed(ps.string)):
                        if pauli == "Z":
                            bit = int(bitstring[-(i + 1)]) if i < len(bitstring) else 0
                            eigenvalue *= (1 - 2 * bit)
                    ev += eigenvalue * count / shots
                total += float(np.real(coeff)) * ev
        return total

    def ground_state_energy(self) -> float:
        """Exact ground state energy via full diagonalization (small H only)."""
        mat = self.to_matrix()
        eigenvalues = np.linalg.eigvalsh(mat)
        return float(np.min(eigenvalues))

    def to_matrix(self) -> np.ndarray:
        """Return the full 2^n × 2^n Hamiltonian matrix."""
        dim = 2 ** self.num_qubits
        result = np.zeros((dim, dim), dtype=np.complex128)
        for coeff, ps in self.terms:
            result += complex(coeff) * ps.to_matrix()
        return result

    def __add__(self, other: Hamiltonian) -> Hamiltonian:
        return Hamiltonian(self.terms + other.terms)

    def __repr__(self) -> str:
        terms_str = " + ".join(
            f"{c:.3f}·{ps.string}" for c, ps in self.terms
        )
        return f"Hamiltonian({terms_str})"
