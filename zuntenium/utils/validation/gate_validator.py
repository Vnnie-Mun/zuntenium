"""Gate validation utilities."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from zuntenium.quantum_core.gates.base_gate import BaseGate


def check_unitarity(gate: BaseGate, tol: float = 1e-10) -> bool:
    """Return True if the gate matrix is unitary within tolerance."""
    m = gate.matrix()
    eye = np.eye(m.shape[0], dtype=np.complex128)
    error = float(np.linalg.norm(m.conj().T @ m - eye))
    return error < tol


def check_dimension(gate: BaseGate) -> bool:
    """Return True if gate matrix dimensions match num_qubits."""
    expected = 2 ** gate.num_qubits
    m = gate.matrix()
    return m.shape == (expected, expected)
