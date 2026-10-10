"""
Multi-qubit gate implementations.

Includes: Toffoli (CCX), Fredkin (CSWAP), and multi-controlled gate builder.
"""
from __future__ import annotations

import numpy as np

from zuntenium.quantum_core.gates.base_gate import BaseGate


class ToffoliGate(BaseGate):
    """
    Toffoli gate (CCX): flips target when both controls are |1⟩.

    Qubit ordering: [control0, control1, target].
    Standard 8×8 matrix in the computational basis.
    """
    name = "CCX"
    num_qubits = 3

    def matrix(self) -> np.ndarray:
        m = np.eye(8, dtype=np.complex128)
        # Swap |110⟩ ↔ |111⟩ rows (indices 6 and 7)
        m[6, 6] = 0
        m[7, 7] = 0
        m[6, 7] = 1
        m[7, 6] = 1
        return m


class FredkinGate(BaseGate):
    """
    Fredkin gate (CSWAP): swaps target qubits when control is |1⟩.

    Qubit ordering: [control, target0, target1].
    """
    name = "CSWAP"
    num_qubits = 3

    def matrix(self) -> np.ndarray:
        m = np.eye(8, dtype=np.complex128)
        # Swap |101⟩ ↔ |110⟩ rows (indices 5 and 6)
        m[5, 5] = 0
        m[6, 6] = 0
        m[5, 6] = 1
        m[6, 5] = 1
        return m


class MultiControlledGate(BaseGate):
    """
    Programmatic multi-controlled gate.

    Builds a 2^(num_controls + base.num_qubits) unitary that applies
    ``base`` only when all control qubits are |1⟩.

    Args:
        base: The target gate to control.
        num_controls: Number of control qubits.
    """

    def __init__(self, base: BaseGate, num_controls: int) -> None:
        self._base = base
        self._num_controls = num_controls
        self.name = "C" * num_controls + base.name
        self.num_qubits = base.num_qubits + num_controls
        self.is_parametric = base.is_parametric

    def matrix(self) -> np.ndarray:
        base_dim = 2 ** self._base.num_qubits
        ctrl_dim = 2 ** self._num_controls
        total_dim = ctrl_dim * base_dim
        result = np.eye(total_dim, dtype=np.complex128)
        result[-base_dim:, -base_dim:] = self._base.matrix()
        return result

    @property
    def params(self) -> list[float]:
        return self._base.params
