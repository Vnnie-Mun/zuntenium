"""
ZUNTENIUM exception hierarchy.

All exceptions inherit from ZunteniumError so callers can catch the
entire family with a single except clause.
"""
from __future__ import annotations


class ZunteniumError(Exception):
    """Base exception for all ZUNTENIUM errors."""


# ── Circuit / Qubit errors ────────────────────────────────────────────────

class QubitIndexError(ZunteniumError, IndexError):
    """Raised when a qubit index is out of range for a given circuit."""

    def __init__(self, index: int, num_qubits: int) -> None:
        super().__init__(
            f"Qubit index {index} is out of range for a {num_qubits}-qubit circuit. "
            f"Valid indices: 0..{num_qubits - 1}."
        )
        self.index = index
        self.num_qubits = num_qubits


class GateDimensionError(ZunteniumError, ValueError):
    """Raised when a gate matrix has incorrect dimensions."""

    def __init__(self, gate_name: str, expected: int, got: int) -> None:
        super().__init__(
            f"Gate '{gate_name}' expects a {expected}×{expected} unitary matrix, "
            f"got {got}×{got}."
        )
        self.gate_name = gate_name
        self.expected = expected
        self.got = got


class GateNotUnitaryError(ZunteniumError, ValueError):
    """Raised when a provided matrix is not unitary within tolerance."""

    def __init__(self, gate_name: str, error: float, tol: float) -> None:
        super().__init__(
            f"Gate '{gate_name}' matrix is not unitary (||U†U - I||={error:.2e}, tol={tol:.2e})."
        )


# ── Compilation errors ────────────────────────────────────────────────────

class CircuitCompilationError(ZunteniumError):
    """Raised when circuit compilation or IR generation fails."""


class CircuitValidationError(ZunteniumError, ValueError):
    """Raised when circuit topology or gate support validation fails."""


# ── Backend errors ────────────────────────────────────────────────────────

class BackendNotFoundError(ZunteniumError, KeyError):
    """Raised when a requested backend name is not registered."""

    def __init__(self, name: str) -> None:
        super().__init__(f"Backend '{name}' is not registered. "
                         "Use BackendProvider.available_backends() to list options.")
        self.name = name


class BackendExecutionError(ZunteniumError):
    """Raised when a backend fails to execute a circuit."""


class HardwareConnectionError(ZunteniumError, ConnectionError):
    """Raised when a hardware backend cannot connect to the remote service."""


# ── Transpiler errors ─────────────────────────────────────────────────────

class TranspilerError(ZunteniumError):
    """Raised when the transpiler pipeline fails."""


class RoutingError(TranspilerError):
    """Raised when SWAP-based routing cannot satisfy connectivity constraints."""


# ── Hybrid / Optimizer errors ─────────────────────────────────────────────

class OptimizerConvergenceError(ZunteniumError):
    """Raised when a classical optimizer fails to converge."""


class HamiltonianError(ZunteniumError, ValueError):
    """Raised for invalid Hamiltonian specifications."""


# ── IO errors ─────────────────────────────────────────────────────────────

class QASMParseError(ZunteniumError, SyntaxError):
    """Raised when QASM parsing fails."""


class CircuitSerializationError(ZunteniumError):
    """Raised when circuit serialization or deserialization fails."""
