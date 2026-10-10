"""
SnapshotBackend — Captures statevector snapshots at specific circuit points.

Useful for debugging variational circuits by observing the state
mid-circuit without collapsing it.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from zuntenium.backend.base_backend import AbstractBackend, ValidationResult
from zuntenium.quantum_core.statevector.statevector import StateVector

if TYPE_CHECKING:
    from zuntenium.circuit.circuit import QuantumCircuit
    from zuntenium.circuit.ir.quantum_ir import QuantumIR
    from zuntenium.quantum_core.executor.result import ExecutionResult


class SnapshotBackend(AbstractBackend):
    """
    Executes a circuit and captures statevector snapshots at each layer.

    After ``run()``, access ``.snapshots`` to retrieve the state at each
    circuit layer boundary.
    """

    name = "zuntenium_snapshot_simulator"
    is_simulator = True
    max_qubits = 25

    def __init__(self) -> None:
        self._snapshots: list[np.ndarray] = []

    def run(
        self,
        circuit: QuantumCircuit | QuantumIR,
        shots: int = 1024,
        **options: Any,
    ) -> ExecutionResult:
        from zuntenium.circuit.circuit import QuantumCircuit
        from zuntenium.quantum_core.executor.measurement import measure_all
        from zuntenium.quantum_core.executor.result import ExecutionResult

        if isinstance(circuit, QuantumCircuit):
            ir = circuit.to_ir()
        else:
            ir = circuit

        sv = StateVector(ir.num_qubits)
        self._snapshots = [sv.get_amplitudes().copy()]

        layers = ir.to_layers()
        for layer in layers:
            for node in layer:
                if node.is_gate() and node.gate is not None:
                    sv.apply_gate(node.gate, node.qubits)
            self._snapshots.append(sv.get_amplitudes().copy())

        measurement = measure_all(sv.get_amplitudes(), ir.num_qubits, shots)
        return ExecutionResult(
            counts=measurement.counts,
            num_qubits=ir.num_qubits,
            shots=shots,
            statevector=sv.get_amplitudes(),
            metadata={"backend": self.name, "num_snapshots": len(self._snapshots)},
        )

    def validate_circuit(self, circuit: QuantumCircuit) -> ValidationResult:
        errors: list[str] = []
        warnings: list[str] = []
        if circuit.num_qubits > self.max_qubits:
            errors.append(f"SnapshotBackend limit is {self.max_qubits} qubits.")
        return ValidationResult(valid=not errors, errors=errors, warnings=warnings)

    @property
    def snapshots(self) -> list[np.ndarray]:
        """Return list of statevector snapshots (one per layer + initial)."""
        return list(self._snapshots)
