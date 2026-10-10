"""
IBM Quantum backend adapter (v0.2 stub).

Will connect to IBM Quantum via qiskit-ibm-runtime in v0.2.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from zuntenium.backend.hardware.hardware_backend import AbstractHardwareBackend
from zuntenium.backend.base_backend import BackendProperties, ValidationResult
from zuntenium.backends.ibm_backend import IBMBackend as ProviderIBMBackend

if TYPE_CHECKING:
    from zuntenium.backend.job.job import QuantumJob
    from zuntenium.circuit.circuit import QuantumCircuit
    from zuntenium.quantum_core.executor.result import ExecutionResult


class IBMBackend(AbstractHardwareBackend):
    """
    IBM Quantum hardware backend.

    v0.1: Stub — raises NotImplementedError on run().
    v0.2: Full implementation using qiskit-ibm-runtime.

    Args:
        backend_name: IBM backend name, e.g. "ibm_nairobi".
        credentials: IBMCredentials object.
    """

    name = "ibm_quantum"
    max_qubits = 127       # IBM Eagle processor
    supported_gates = ["H", "X", "CX", "Rz", "SX"]
    is_simulator = False

    def __init__(
        self,
        backend_name: str = "ibm_nairobi",
        credentials: object | None = None,
    ) -> None:
        self._backend_name = backend_name
        self._credentials = credentials
        token = getattr(credentials, "token", None) if credentials is not None else None
        self._provider = ProviderIBMBackend(
            token=token,
            ibm_backend_name=backend_name,
            fallback_to_simulator=True,
        )

    def run(
        self,
        circuit: QuantumCircuit | Any,
        shots: int = 1024,
        **options: Any,
    ) -> ExecutionResult:
        return self._provider.execute(circuit, shots=shots, **options)

    def validate_circuit(self, circuit: QuantumCircuit) -> ValidationResult:
        try:
            from zuntenium.transpiler.v2 import Circuit as V2Circuit
            if isinstance(circuit, V2Circuit):
                from zuntenium.execution.quantum_executor import validate_circuit_compatibility
                validate_circuit_compatibility(circuit, max_qubits=self.max_qubits)
                return ValidationResult(True, [], [])
        except Exception as exc:
            return ValidationResult(False, [str(exc)], [])
        return ValidationResult(
            False,
            ["IBM backend adapter accepts zuntenium.transpiler.v2.Circuit instances."],
            [],
        )

    def submit_job(self, circuit: QuantumCircuit, shots: int = 1024, **options: Any) -> QuantumJob:
        raise NotImplementedError("Use zuntenium.backends.ibm_backend.IBMBackend for asynchronous provider jobs.")

    def fetch_result(self, job_id: str) -> Any:
        return self._provider.job_status(job_id)

    def cancel_job(self, job_id: str) -> bool:
        return self._provider.cancel_job(job_id)
