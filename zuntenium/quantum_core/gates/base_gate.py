"""
BaseGate — Abstract protocol all quantum gates must satisfy.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

from zuntenium.quantum_core.exceptions import GateDimensionError, GateNotUnitaryError


@dataclass(frozen=True, slots=True)
class GateIR:
    """
    Serializable gate record used in QuantumIR instructions.

    Immutable value object — safe for caching and hashing.

    Attributes:
        name: Gate identifier string (e.g. "H", "CX", "Rx").
        num_qubits: Number of qubits the gate acts on.
        params: Ordered tuple of numeric parameters (rotations, etc.).
        matrix_data: Flattened complex128 matrix as a tuple (for JSON).
    """
    name: str
    num_qubits: int
    params: tuple[float, ...] = ()
    matrix_data: tuple[complex, ...] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "num_qubits": self.num_qubits,
            "params": list(self.params),
            "matrix_data": (
                [[v.real, v.imag] for v in self.matrix_data]
                if self.matrix_data is not None else None
            ),
        }


class BaseGate(ABC):
    """
    Abstract base class for all ZUNTENIUM gates.

    Subclasses must implement:
      - ``name`` class attribute
      - ``num_qubits`` class attribute
      - ``matrix()`` method returning a 2^n × 2^n complex128 unitary

    Optional overrides:
      - ``is_parametric`` (default False)
      - ``params`` property
      - ``dagger()``
      - ``controlled()``
    """

    name: str = ""
    num_qubits: int = 1
    is_parametric: bool = False

    # ── Core interface ─────────────────────────────────────────────────────

    @abstractmethod
    def matrix(self) -> np.ndarray:
        """Return the gate's unitary matrix as a complex128 ndarray."""
        ...

    @property
    def params(self) -> list[float]:
        """Return ordered list of gate parameters (empty for fixed gates)."""
        return []

    # ── Derived operations ─────────────────────────────────────────────────

    def dagger(self) -> BaseGate:
        """Return the Hermitian conjugate (†) of this gate."""
        return _ConjugatedGate(self)

    def controlled(self, num_controls: int = 1) -> BaseGate:
        """Return a controlled version of this gate."""
        return _ControlledGate(self, num_controls)

    def to_ir(self) -> GateIR:
        """Convert gate to serializable IR record."""
        mat = self.matrix()
        return GateIR(
            name=self.name,
            num_qubits=self.num_qubits,
            params=tuple(self.params),
            matrix_data=tuple(mat.flatten().tolist()),
        )

    # ── Validation helpers ─────────────────────────────────────────────────

    def validate_unitary(self, tol: float = 1e-10) -> None:
        """Raise GateNotUnitaryError if the matrix deviates from unitarity."""
        m = self.matrix()
        eye = np.eye(m.shape[0], dtype=np.complex128)
        error = float(np.linalg.norm(m.conj().T @ m - eye))
        if error > tol:
            raise GateNotUnitaryError(self.name, error, tol)

    def validate_dimension(self) -> None:
        """Raise GateDimensionError if matrix shape doesn't match num_qubits."""
        m = self.matrix()
        expected = 2 ** self.num_qubits
        if m.shape != (expected, expected):
            raise GateDimensionError(self.name, expected, m.shape[0])

    # ── Dunder ─────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        if self.params:
            param_str = ", ".join(f"{p:.4f}" for p in self.params)
            return f"{self.name}({param_str})"
        return self.name

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BaseGate):
            return NotImplemented
        return (
            self.name == other.name
            and self.num_qubits == other.num_qubits
            and np.allclose(self.matrix(), other.matrix())
        )

    def __hash__(self) -> int:
        """Hash based on gate identity (name + params). Consistent with __eq__."""
        return hash((self.name, self.num_qubits, tuple(self.params)))


# ── Derived gate wrappers ─────────────────────────────────────────────────

class _ConjugatedGate(BaseGate):
    """Internal wrapper: applies Hermitian conjugate of a gate."""

    def __init__(self, base: BaseGate) -> None:
        self._base = base
        self.name = f"{base.name}†"
        self.num_qubits = base.num_qubits
        self.is_parametric = base.is_parametric

    def matrix(self) -> np.ndarray:
        return self._base.matrix().conj().T

    def dagger(self) -> BaseGate:
        return self._base  # (A†)† = A


class _ControlledGate(BaseGate):
    """Internal wrapper: controlled-U with k control qubits."""

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
        # Apply base gate only when all control qubits are |1⟩
        result[-base_dim:, -base_dim:] = self._base.matrix()
        return result
