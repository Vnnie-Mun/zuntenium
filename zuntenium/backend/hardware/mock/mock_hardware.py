"""
MockHardwareBackend — Deterministic fake hardware for testing.

Behaves like a hardware backend (async jobs, limited gate set, finite
qubit count) but executes locally using the statevector simulator.
Used in unit/integration tests that test hardware-specific code paths.
"""
from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

from zuntenium.backend.base_backend import BackendStatus
from zuntenium.backend.hardware.hardware_backend import AbstractHardwareBackend

if TYPE_CHECKING:
    from zuntenium.backend.job.job import QuantumJob
    from zuntenium.circuit.circuit import QuantumCircuit
    from zuntenium.quantum_core.executor.result import ExecutionResult


class MockHardwareBackend(AbstractHardwareBackend):
    """
    Fake hardware backend for testing.

    Runs circuits locally via StatevectorBackend but exposes the hardware
    backend interface (async job handle, limited gate set, job queuing).
    """

    name = "mock_hardware"
    max_qubits = 5
    supported_gates = ["H", "X", "CX", "Rz"]
    is_simulator = False

    def __init__(self, seed: int | None = 42) -> None:
        self._seed = seed
        self._jobs: dict[str, Any] = {}

    def run(
        self,
        circuit: QuantumCircuit | Any,
        shots: int = 1024,
        **options: Any,
    ) -> ExecutionResult:
        """Run synchronously using the statevector backend."""
        from zuntenium.backend.simulator.statevector_backend import StatevectorBackend
        sim = StatevectorBackend(seed=self._seed)
        result = sim.run(circuit, shots=shots)
        result.metadata["backend"] = self.name
        return result

    def submit_job(
        self,
        circuit: QuantumCircuit,
        shots: int = 1024,
        **options: Any,
    ) -> QuantumJob:
        from zuntenium.backend.job.job import QuantumJob
        job_id = str(uuid.uuid4())
        result = self.run(circuit, shots=shots)
        self._jobs[job_id] = result
        return QuantumJob(job_id=job_id, backend_name=self.name, status="DONE")

    def fetch_result(self, job_id: str) -> ExecutionResult:
        if job_id not in self._jobs:
            raise KeyError(f"Job '{job_id}' not found.")
        return self._jobs[job_id]

    def cancel_job(self, job_id: str) -> bool:
        if job_id in self._jobs:
            del self._jobs[job_id]
            return True
        return False

    def status(self) -> BackendStatus:
        return BackendStatus(
            name=self.name,
            operational=True,
            pending_jobs=0,
            message="Mock hardware: always online.",
        )
