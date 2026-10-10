"""
Quantum state distance measures: fidelity, trace distance, Bures distance.
"""
from __future__ import annotations

import numpy as np


def state_fidelity(psi: np.ndarray, phi: np.ndarray) -> float:
    """
    Fidelity between two pure states: F = |⟨ψ|φ⟩|².

    Args:
        psi: First statevector (shape 2^N,).
        phi: Second statevector (shape 2^N,).

    Returns:
        Fidelity in [0, 1].
    """
    return float(abs(np.dot(psi.conj(), phi)) ** 2)


def density_matrix_fidelity(rho: np.ndarray, sigma: np.ndarray) -> float:
    """
    Uhlmann fidelity between two density matrices: F = Tr(√(√ρ σ √ρ))².

    Args:
        rho: First density matrix.
        sigma: Second density matrix.

    Returns:
        Fidelity in [0, 1].
    """
    sqrt_rho = _matrix_sqrt(rho)
    inner = sqrt_rho @ sigma @ sqrt_rho
    eigenvalues = np.linalg.eigvalsh(inner)
    eigenvalues = np.maximum(eigenvalues, 0)   # clip numerical negatives
    return float(np.sum(np.sqrt(eigenvalues)) ** 2)


def trace_distance(rho: np.ndarray, sigma: np.ndarray) -> float:
    """
    Trace distance: D(ρ,σ) = (1/2) Tr|ρ - σ|.

    Args:
        rho: First density matrix.
        sigma: Second density matrix.

    Returns:
        Trace distance in [0, 1].
    """
    diff = rho - sigma
    eigenvalues = np.linalg.eigvalsh(diff)
    return float(0.5 * np.sum(np.abs(eigenvalues)))


def bures_distance(rho: np.ndarray, sigma: np.ndarray) -> float:
    """
    Bures distance: B(ρ,σ) = √(2(1 - √F(ρ,σ))).
    """
    f = density_matrix_fidelity(rho, sigma)
    return float(np.sqrt(2 * (1 - np.sqrt(max(0.0, f)))))


def _matrix_sqrt(A: np.ndarray) -> np.ndarray:
    """Compute the matrix square root of a positive semi-definite matrix."""
    evals, evecs = np.linalg.eigh(A)
    evals = np.maximum(evals, 0)
    return evecs @ np.diag(np.sqrt(evals)) @ evecs.conj().T
