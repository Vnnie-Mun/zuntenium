"""
Pure NumPy statevector gate kernels.

All functions are pure (no mutation) and return the updated amplitude array.
These are the reference implementations — correctness over performance.
For num_qubits ≤ 20 these are fast enough; C++ takes over above that.
"""
from __future__ import annotations

import numpy as np


def apply_single_qubit_gate(
    amplitudes: np.ndarray,
    matrix: np.ndarray,
    qubit: int,
    num_qubits: int,
) -> np.ndarray:
    """
    Apply a 2×2 unitary to a single qubit using tensor reshape.

    The algorithm reshapes the statevector into a rank-N tensor, applies the
    matrix along the target qubit axis, then reshapes back.

    Args:
        amplitudes: Complex128 array of shape (2^N,).
        matrix: 2×2 complex128 unitary.
        qubit: Target qubit index (0 = least significant).
        num_qubits: Total number of qubits.

    Returns:
        Updated amplitudes array of shape (2^N,).
    """
    state = amplitudes.reshape([2] * num_qubits)
    # tensordot contracts the inner qubit axis with the matrix column index
    result = np.tensordot(matrix, state, axes=([1], [qubit]))
    # tensordot moves the contracted axis to position 0; move it back
    result = np.moveaxis(result, 0, qubit)
    return result.reshape(amplitudes.shape)


def apply_two_qubit_gate(
    amplitudes: np.ndarray,
    matrix: np.ndarray,
    qubit0: int,
    qubit1: int,
    num_qubits: int,
) -> np.ndarray:
    """
    Apply a 4×4 unitary to two qubits.

    Args:
        amplitudes: Complex128 array of shape (2^N,).
        matrix: 4×4 complex128 unitary (row-major, qubit0 is most-significant).
        qubit0: First qubit index (control for CNOT etc.).
        qubit1: Second qubit index (target for CNOT etc.).
        num_qubits: Total number of qubits.

    Returns:
        Updated amplitudes array of shape (2^N,).
    """
    state = amplitudes.reshape([2] * num_qubits)
    gate_tensor = matrix.reshape(2, 2, 2, 2)
    # Contract over both input qubit axes
    result = np.tensordot(gate_tensor, state, axes=([2, 3], [qubit0, qubit1]))
    # Restore qubit axes to original positions
    result = np.moveaxis(result, [0, 1], [qubit0, qubit1])
    return result.reshape(amplitudes.shape)


def apply_n_qubit_gate(
    amplitudes: np.ndarray,
    matrix: np.ndarray,
    qubits: list[int],
    num_qubits: int,
) -> np.ndarray:
    """
    Apply an N-qubit unitary via general tensor contraction.

    Args:
        amplitudes: Complex128 array of shape (2^N,).
        matrix: 2^k × 2^k unitary for k qubits in ``qubits``.
        qubits: List of target qubit indices.
        num_qubits: Total number of qubits.

    Returns:
        Updated amplitudes array of shape (2^N,).
    """
    k = len(qubits)
    state = amplitudes.reshape([2] * num_qubits)
    gate_tensor = matrix.reshape([2] * (2 * k))
    input_axes = list(range(k, 2 * k))
    result = np.tensordot(gate_tensor, state, axes=(input_axes, qubits))
    output_axes = list(range(k))
    result = np.moveaxis(result, output_axes, qubits)
    return result.reshape(amplitudes.shape)
