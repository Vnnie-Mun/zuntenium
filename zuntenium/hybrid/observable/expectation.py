"""
Expectation value computation utilities.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from zuntenium.hybrid.observable.pauli import PauliString


def compute_expectation(
    statevector: np.ndarray,
    observable: np.ndarray,
) -> float:
    """
    Compute ⟨ψ|O|ψ⟩ for a Hermitian observable matrix O.

    Args:
        statevector: Complex128 array of shape (2^N,).
        observable: Hermitian 2^N × 2^N matrix.

    Returns:
        Real-valued expectation value.
    """
    psi = statevector.flatten()
    return float(np.real(psi.conj() @ observable @ psi))


def pauli_expectation(
    statevector: np.ndarray,
    pauli_string: PauliString,
) -> float:
    """
    Compute ⟨ψ|P|ψ⟩ for a PauliString P.

    Args:
        statevector: Complex128 array of shape (2^N,).
        pauli_string: PauliString to evaluate.

    Returns:
        Real-valued expectation value.
    """
    return compute_expectation(statevector, pauli_string.to_matrix())
