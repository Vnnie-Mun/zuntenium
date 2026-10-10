"""
AbstractHardwareBackend — Base for all real hardware backends.

Extends AbstractBackend with job submission, polling, and credential management.
"""
from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any

from zuntenium.backend.base_backend import AbstractBackend, ValidationResult

if TYPE_CHECKING:
    from zuntenium.backend.job.job import QuantumJob
    from zuntenium.circuit.circuit import QuantumCircuit


class AbstractHardwareBackend(AbstractBackend):
    """
    Base class for all physical quantum hardware backends.

    Extends AbstractBackend with async job lifecycle methods.
    Hardware backends do not return statevectors; only counts are available.
    """

    is_simulator = False

    @abstractmethod
    def submit_job(
        self,
        circuit: QuantumCircuit,
        shots: int = 1024,
        **options: Any,
    ) -> QuantumJob:
        """Submit a circuit to the hardware queue and return a job handle."""
        ...

    @abstractmethod
    def fetch_result(self, job_id: str) -> Any:
        """Fetch the result of a completed job by job ID."""
        ...

    @abstractmethod
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a pending or running job. Returns True if successful."""
        ...

    def validate_circuit(self, circuit: QuantumCircuit) -> ValidationResult:
        errors: list[str] = []
        warnings: list[str] = []
        if circuit.num_qubits > self.max_qubits:
            errors.append(
                f"Circuit requires {circuit.num_qubits} qubits; "
                f"backend supports {self.max_qubits}."
            )
        for instr in circuit.instructions:
            if (
                instr.operation == "gate"
                and instr.gate is not None
                and instr.gate.name not in self.supported_gates
            ):
                warnings.append(
                    f"Gate '{instr.gate.name}' is not in the backend's native gate set. "
                    "Transpiler will decompose it."
                )
        return ValidationResult(valid=not errors, errors=errors, warnings=warnings)
