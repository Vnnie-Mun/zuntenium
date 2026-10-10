"""
Two-qubit gate implementations.

Includes: CX (CNOT), CZ, CY, SWAP, iSWAP, CRx, CRy, CRz.
Qubit ordering convention: qubit0 = control (MSB), qubit1 = target (LSB)
in the 4×4 matrix basis {|00⟩, |01⟩, |10⟩, |11⟩}.
"""
from __future__ import annotations

import numpy as np

from zuntenium.quantum_core.gates.base_gate import BaseGate

_I = complex(0, 1)


# ── Entangling Clifford gates ─────────────────────────────────────────────

class CXGate(BaseGate):
    """Controlled-NOT (CNOT): flips target when control is |1⟩."""
    name = "CX"
    num_qubits = 2

    def matrix(self) -> np.ndarray:
        return np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0],
             [0, 0, 0, 1],
             [0, 0, 1, 0]],
            dtype=np.complex128,
        )


class CZGate(BaseGate):
    """Controlled-Z: applies Z to target when control is |1⟩."""
    name = "CZ"
    num_qubits = 2

    def matrix(self) -> np.ndarray:
        return np.diag([1, 1, 1, -1]).astype(np.complex128)


class CYGate(BaseGate):
    """Controlled-Y."""
    name = "CY"
    num_qubits = 2

    def matrix(self) -> np.ndarray:
        return np.array(
            [[1, 0, 0,   0],
             [0, 1, 0,   0],
             [0, 0, 0, -_I],
             [0, 0, _I,  0]],
            dtype=np.complex128,
        )


class SWAPGate(BaseGate):
    """SWAP gate: exchanges two qubits."""
    name = "SWAP"
    num_qubits = 2

    def matrix(self) -> np.ndarray:
        return np.array(
            [[1, 0, 0, 0],
             [0, 0, 1, 0],
             [0, 1, 0, 0],
             [0, 0, 0, 1]],
            dtype=np.complex128,
        )


class ISWAPGate(BaseGate):
    """iSWAP gate: SWAP with an imaginary phase on swapped terms."""
    name = "ISWAP"
    num_qubits = 2

    def matrix(self) -> np.ndarray:
        return np.array(
            [[1,  0,  0, 0],
             [0,  0, _I, 0],
             [0, _I,  0, 0],
             [0,  0,  0, 1]],
            dtype=np.complex128,
        )


# ── Controlled rotation gates ─────────────────────────────────────────────

class CRxGate(BaseGate):
    """Controlled-Rx(θ): applies Rx(θ) to target when control is |1⟩."""
    name = "CRx"
    num_qubits = 2
    is_parametric = True

    def __init__(self, theta: float) -> None:
        self._theta = theta

    def matrix(self) -> np.ndarray:
        c = np.cos(self._theta / 2)
        s = np.sin(self._theta / 2)
        return np.array(
            [[1, 0,       0,       0],
             [0, 1,       0,       0],
             [0, 0,       c, -_I * s],
             [0, 0, -_I * s,       c]],
            dtype=np.complex128,
        )

    @property
    def params(self) -> list[float]:
        return [self._theta]


class CRyGate(BaseGate):
    """Controlled-Ry(θ)."""
    name = "CRy"
    num_qubits = 2
    is_parametric = True

    def __init__(self, theta: float) -> None:
        self._theta = theta

    def matrix(self) -> np.ndarray:
        c = np.cos(self._theta / 2)
        s = np.sin(self._theta / 2)
        return np.array(
            [[1, 0,  0, 0],
             [0, 1,  0, 0],
             [0, 0,  c, -s],
             [0, 0,  s,  c]],
            dtype=np.complex128,
        )

    @property
    def params(self) -> list[float]:
        return [self._theta]


class CRzGate(BaseGate):
    """Controlled-Rz(θ)."""
    name = "CRz"
    num_qubits = 2
    is_parametric = True

    def __init__(self, theta: float) -> None:
        self._theta = theta

    def matrix(self) -> np.ndarray:
        return np.array(
            [[1, 0,                          0,                         0],
             [0, 1,                          0,                         0],
             [0, 0, np.exp(-_I * self._theta / 2),                      0],
             [0, 0,                          0, np.exp(_I * self._theta / 2)]],
            dtype=np.complex128,
        )

    @property
    def params(self) -> list[float]:
        return [self._theta]
