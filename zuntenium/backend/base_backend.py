"""
AbstractBackend — Universal backend interface.

Simulators and hardware backends look identical to user code.
All backends are registered via BackendProvider and discovered by name.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from zuntenium.circuit.circuit import QuantumCircuit
    from zuntenium.circuit.ir.quantum_ir import QuantumIR
    from zuntenium.quantum_core.executor.result import ExecutionResult
    from zuntenium.transpiler.target import Target


@dataclass
class BackendStatus:
    """Operational status of a backend."""
    name: str
    operational: bool
    pending_jobs: int = 0
    message: str = ""


@dataclass
class ValidationResult:
    """Result of circuit validation against a backend."""
    valid: bool
    errors: list[str]
    warnings: list[str]

    def __bool__(self) -> bool:
        return self.valid


@dataclass
class BackendProperties:
    """Hardware properties: fidelities, timing, etc."""
    name: str
    num_qubits: int
    t1_us: dict[int, float]
    t2_us: dict[int, float]
    gate_errors: dict[str, float]
    readout_errors: dict[int, float]
    metadata: dict[str, Any]


class AbstractBackend(ABC):
    """
    Universal backend interface — simulators and hardware look identical.

    All backends must implement ``run()`` and ``validate_circuit()``.
    The ``name``, ``max_qubits``, ``supported_gates``, and ``is_simulator``
    class attributes must be set by each subclass.
    """

    name: str = ""
    max_qubits: int = 0
    supported_gates: list[str] = []
    is_simulator: bool = True

    @abstractmethod
    def run(
        self,
        circuit: QuantumCircuit | QuantumIR,
        shots: int = 1024,
        **options: Any,
    ) -> ExecutionResult:
        """
        Execute a circuit and return results.

        Args:
            circuit: A QuantumCircuit or pre-compiled QuantumIR.
            shots: Number of measurement samples.
            **options: Backend-specific options.

        Returns:
            ExecutionResult with counts and optional statevector.
        """
        ...

    @abstractmethod
    def validate_circuit(
        self,
        circuit: QuantumCircuit,
    ) -> ValidationResult:
        """
        Validate that a circuit is compatible with this backend.

        Args:
            circuit: The circuit to validate.

        Returns:
            ValidationResult with errors/warnings list.
        """
        ...

    def status(self) -> BackendStatus:
        """Return the current operational status of this backend."""
        return BackendStatus(name=self.name, operational=True)

    def properties(self) -> BackendProperties | None:
        """Return hardware properties (T1, T2, gate fidelities). None for simulators."""
        return None

    @property
    def target(self) -> Target:
        """Return a Target object describing this backend's capabilities."""
        from zuntenium.transpiler.target import Target
        return Target.simulator(self.max_qubits)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name='{self.name}', max_qubits={self.max_qubits})"
        )
