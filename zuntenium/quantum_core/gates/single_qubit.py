"""
Single-qubit gate implementations.

Includes: H, X, Y, Z, S, T, Rx, Ry, Rz, U1, U2, U3
All matrices are in the standard computational basis {|0⟩, |1⟩}.
"""
from __future__ import annotations

import numpy as np

from zuntenium.quantum_core.gates.base_gate import BaseGate

_I = complex(0, 1)  # imaginary unit
_SQRT2_INV = 1.0 / np.sqrt(2)


# ── Pauli & Clifford gates ─────────────────────────────────────────────────

class HGate(BaseGate):
    """Hadamard gate: H = (X + Z) / √2."""
    name = "H"
    num_qubits = 1

    def matrix(self) -> np.ndarray:
        return _SQRT2_INV * np.array([[1, 1], [1, -1]], dtype=np.complex128)


class XGate(BaseGate):
    """Pauli-X (NOT) gate."""
    name = "X"
    num_qubits = 1

    def matrix(self) -> np.ndarray:
        return np.array([[0, 1], [1, 0]], dtype=np.complex128)


class YGate(BaseGate):
    """Pauli-Y gate."""
    name = "Y"
    num_qubits = 1

    def matrix(self) -> np.ndarray:
        return np.array([[0, -_I], [_I, 0]], dtype=np.complex128)


class ZGate(BaseGate):
    """Pauli-Z gate."""
    name = "Z"
    num_qubits = 1

    def matrix(self) -> np.ndarray:
        return np.array([[1, 0], [0, -1]], dtype=np.complex128)


class SGate(BaseGate):
    """S gate (phase gate): S = diag(1, i)."""
    name = "S"
    num_qubits = 1

    def matrix(self) -> np.ndarray:
        return np.array([[1, 0], [0, _I]], dtype=np.complex128)


class TGate(BaseGate):
    """T gate: T = diag(1, e^{iπ/4})."""
    name = "T"
    num_qubits = 1

    def matrix(self) -> np.ndarray:
        return np.array([[1, 0], [0, np.exp(_I * np.pi / 4)]], dtype=np.complex128)


# ── Rotation gates ──────────────────────────────────────────────────────────

class RxGate(BaseGate):
    """Rotation around X-axis: Rx(θ) = exp(-i θ/2 · X)."""
    name = "Rx"
    num_qubits = 1
    is_parametric = True

    def __init__(self, theta: float) -> None:
        self._theta = theta

    def matrix(self) -> np.ndarray:
        c = np.cos(self._theta / 2)
        s = np.sin(self._theta / 2)
        return np.array([[c, -_I * s], [-_I * s, c]], dtype=np.complex128)

    @property
    def params(self) -> list[float]:
        return [self._theta]

    def __repr__(self) -> str:
        return f"Rx({self._theta:.4f})"


class RyGate(BaseGate):
    """Rotation around Y-axis: Ry(θ) = exp(-i θ/2 · Y)."""
    name = "Ry"
    num_qubits = 1
    is_parametric = True

    def __init__(self, theta: float) -> None:
        self._theta = theta

    def matrix(self) -> np.ndarray:
        c = np.cos(self._theta / 2)
        s = np.sin(self._theta / 2)
        return np.array([[c, -s], [s, c]], dtype=np.complex128)

    @property
    def params(self) -> list[float]:
        return [self._theta]


class RzGate(BaseGate):
    """Rotation around Z-axis: Rz(θ) = exp(-i θ/2 · Z) = diag(e^{-iθ/2}, e^{iθ/2})."""
    name = "Rz"
    num_qubits = 1
    is_parametric = True

    def __init__(self, theta: float) -> None:
        self._theta = theta

    def matrix(self) -> np.ndarray:
        return np.array(
            [[np.exp(-_I * self._theta / 2), 0],
             [0, np.exp(_I * self._theta / 2)]],
            dtype=np.complex128,
        )

    @property
    def params(self) -> list[float]:
        return [self._theta]


# ── Universal gates ──────────────────────────────────────────────────────────

class U1Gate(BaseGate):
    """U1(λ) = Rz(λ) up to global phase: diag(1, e^{iλ})."""
    name = "U1"
    num_qubits = 1
    is_parametric = True

    def __init__(self, lam: float) -> None:
        self._lam = lam

    def matrix(self) -> np.ndarray:
        return np.array(
            [[1, 0], [0, np.exp(_I * self._lam)]], dtype=np.complex128
        )

    @property
    def params(self) -> list[float]:
        return [self._lam]


class U2Gate(BaseGate):
    """U2(φ, λ) = (1/√2) [[1, -e^{iλ}], [e^{iφ}, e^{i(φ+λ)}]]."""
    name = "U2"
    num_qubits = 1
    is_parametric = True

    def __init__(self, phi: float, lam: float) -> None:
        self._phi = phi
        self._lam = lam

    def matrix(self) -> np.ndarray:
        return _SQRT2_INV * np.array(
            [[1, -np.exp(_I * self._lam)],
             [np.exp(_I * self._phi), np.exp(_I * (self._phi + self._lam))]],
            dtype=np.complex128,
        )

    @property
    def params(self) -> list[float]:
        return [self._phi, self._lam]


class U3Gate(BaseGate):
    """
    General single-qubit unitary U3(θ, φ, λ).

    U3(θ, φ, λ) = [[cos(θ/2), -e^{iλ}sin(θ/2)],
                    [e^{iφ}sin(θ/2), e^{i(φ+λ)}cos(θ/2)]]

    This parameterizes ALL single-qubit unitaries (up to global phase).
    """
    name = "U3"
    num_qubits = 1
    is_parametric = True

    def __init__(self, theta: float, phi: float, lam: float) -> None:
        self._theta = theta
        self._phi = phi
        self._lam = lam

    def matrix(self) -> np.ndarray:
        c = np.cos(self._theta / 2)
        s = np.sin(self._theta / 2)
        return np.array(
            [[c, -np.exp(_I * self._lam) * s],
             [np.exp(_I * self._phi) * s,
              np.exp(_I * (self._phi + self._lam)) * c]],
            dtype=np.complex128,
        )

    @property
    def params(self) -> list[float]:
        return [self._theta, self._phi, self._lam]
