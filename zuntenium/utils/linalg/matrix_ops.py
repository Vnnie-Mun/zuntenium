"""
matrix_ops — Core linear algebra operations for quantum computing.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm as _scipy_expm


def tensor_product(*matrices: np.ndarray) -> np.ndarray:
    """
    Compute the Kronecker (tensor) product of N matrices.

    tensor_product(A, B, C) = A ⊗ B ⊗ C

    Args:
        *matrices: One or more 2D complex numpy arrays.

    Returns:
        Tensor product matrix.
    """
    result = matrices[0]
    for m in matrices[1:]:
        result = np.kron(result, m)
    return result


def partial_trace(
    rho: np.ndarray,
    keep: list[int],
    dims: list[int],
) -> np.ndarray:
    """
    Compute the partial trace over all subsystems not in ``keep``.

    Args:
        rho: Density matrix of shape (D, D) where D = prod(dims).
        keep: List of subsystem indices to keep (0-indexed).
        dims: List of subsystem dimensions.

    Returns:
        Reduced density matrix.
    """
    n = len(dims)
    d_keep = [dims[i] for i in keep]

    rho_tensor = rho.reshape(dims + dims)
    trace_indices = [i for i in range(n) if i not in keep]

    for ax in sorted(trace_indices, reverse=True):
        rho_tensor = np.trace(rho_tensor, axis1=ax, axis2=ax + n)
        n -= 1

    d = int(np.prod(d_keep))
    return rho_tensor.reshape(d, d)


def matrix_exponential(A: np.ndarray) -> np.ndarray:
    """Compute the matrix exponential exp(A) using scipy."""
    return _scipy_expm(A)


def commutator(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Compute the commutator [A, B] = AB - BA."""
    return A @ B - B @ A


def anticommutator(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Compute the anticommutator {A, B} = AB + BA."""
    return A @ B + B @ A


def is_unitary(U: np.ndarray, tol: float = 1e-10) -> bool:
    """Return True if U is unitary within tolerance."""
    n = U.shape[0]
    diff = np.linalg.norm(U.conj().T @ U - np.eye(n, dtype=np.complex128))
    return bool(diff < tol)


def is_hermitian(H: np.ndarray, tol: float = 1e-10) -> bool:
    """Return True if H is Hermitian within tolerance."""
    return bool(np.linalg.norm(H - H.conj().T) < tol)
