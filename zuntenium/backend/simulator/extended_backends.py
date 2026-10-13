"""
extended_backends.py — Universal AbstractBackend adapters for extended simulators.

Bridges DensityMatrixSimulator, StabilizerSimulator, TensorNetworkBackend,
and GPUStatevectorSimulator into the canonical AbstractBackend interface so they
can be discovered by name, retrieved via get_backend(), and executed transparently
via execute(circuit, backend=...) from any external script or IDE.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from zuntenium.backend.base_backend import AbstractBackend, ValidationResult
from zuntenium.backends.density_matrix_simulator import DensityMatrixSimulator
from zuntenium.backends.gpu_backend import GPUStatevectorSimulator
from zuntenium.backends.stabilizer import StabilizerSimulator
from zuntenium.backends.tensor_network_backend import TensorNetworkBackend
from zuntenium.quantum_core.executor.result import ExecutionResult
from zuntenium.transpiler.v2.core import Circuit, Gate

if TYPE_CHECKING:
    from zuntenium.circuit.circuit import QuantumCircuit
    from zuntenium.circuit.ir.quantum_ir import QuantumIR


def to_v2_circuit(circuit: Any) -> Circuit:
    """Convert QuantumCircuit, QuantumIR, or v2 Circuit to canonical v2 Circuit."""
    if hasattr(circuit, "gates") and isinstance(circuit.gates, list):
        return circuit

    if hasattr(circuit, "instructions"):
        v2_circ = Circuit(circuit.num_qubits, name=getattr(circuit, "name", ""))
        for inst in circuit.instructions:
            if getattr(inst, "gate", None) is None:
                op = getattr(inst, "operation", "MEASURE").upper()
                v2_circ.add_gate(Gate(op, targets=tuple(inst.qubits)))
                continue

            name = inst.gate.name.upper()
            qubits = tuple(inst.qubits)
            params = tuple(getattr(inst.gate, "params", ()))
            if len(qubits) == 1:
                v2_circ.add_gate(Gate(name, targets=qubits, params=params))
            elif len(qubits) == 2 and name in ("CX", "CNOT", "CZ", "CY", "CRX", "CRY", "CRZ"):
                v2_circ.add_gate(
                    Gate(name, targets=(qubits[1],), controls=(qubits[0],), params=params)
                )
            else:
                v2_circ.add_gate(Gate(name, targets=qubits, params=params))
        return v2_circ

    raise TypeError(f"Cannot convert circuit of type {type(circuit)} to Circuit")


class DensityMatrixBackend(AbstractBackend):
    """Density matrix mixed-state simulation backend."""

    name = "zuntenium_density_matrix_simulator"
    max_qubits = 15
    is_simulator = True
    supported_gates = ["H", "X", "Y", "Z", "S", "T", "CX", "CZ", "RX", "RY", "RZ", "U", "U3"]

    def run(
        self,
        circuit: QuantumCircuit | QuantumIR | Circuit,
        shots: int = 1024,
        **options: Any,
    ) -> ExecutionResult:
        v2 = to_v2_circuit(circuit)
        sim = DensityMatrixSimulator(v2.num_qubits)
        res = sim.execute(v2, shots=shots, **options)
        return ExecutionResult(
            counts=res.counts,
            num_qubits=v2.num_qubits,
            shots=shots,
            metadata=getattr(res, "metadata", {"backend": self.name}),
        )

    def validate_circuit(self, circuit: Any) -> ValidationResult:
        errors = []
        if getattr(circuit, "num_qubits", 0) > self.max_qubits:
            errors.append(
                f"Circuit has {circuit.num_qubits} qubits, exceeding "
                f"density matrix limit of {self.max_qubits}."
            )
        return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=[])


class StabilizerBackend(AbstractBackend):
    """Clifford binary tableau simulation backend."""

    name = "zuntenium_stabilizer_simulator"
    max_qubits = 10000
    is_simulator = True
    supported_gates = ["H", "S", "X", "Y", "Z", "CX", "CZ", "SWAP", "MEASURE"]

    def run(
        self,
        circuit: QuantumCircuit | QuantumIR | Circuit,
        shots: int = 1024,
        **options: Any,
    ) -> ExecutionResult:
        v2 = to_v2_circuit(circuit)
        sim = StabilizerSimulator(v2.num_qubits)
        res = sim.execute(v2, shots=shots, **options)
        return ExecutionResult(
            counts=res.counts,
            num_qubits=v2.num_qubits,
            shots=shots,
            metadata=getattr(res, "metadata", {"backend": self.name}),
        )

    def validate_circuit(self, circuit: Any) -> ValidationResult:
        errors = []
        if getattr(circuit, "num_qubits", 0) > self.max_qubits:
            errors.append(
                f"Circuit has {circuit.num_qubits} qubits, exceeding "
                f"stabilizer limit of {self.max_qubits}."
            )
        return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=[])


class TensorNetworkBackendAdapter(AbstractBackend):
    """Matrix Product State (MPS) tensor network backend."""

    name = "zuntenium_tensor_network_simulator"
    max_qubits = 100
    is_simulator = True

    def run(
        self,
        circuit: QuantumCircuit | QuantumIR | Circuit,
        shots: int = 1024,
        **options: Any,
    ) -> ExecutionResult:
        v2 = to_v2_circuit(circuit)
        tn = TensorNetworkBackend()
        res = tn.execute(v2, shots=shots, **options)
        return ExecutionResult(
            counts=res.counts,
            num_qubits=v2.num_qubits,
            shots=shots,
            metadata=getattr(res, "metadata", {"backend": self.name}),
        )

    def validate_circuit(self, circuit: Any) -> ValidationResult:
        errors = []
        if getattr(circuit, "num_qubits", 0) > self.max_qubits:
            errors.append(
                f"Circuit has {circuit.num_qubits} qubits, exceeding "
                f"tensor network limit of {self.max_qubits}."
            )
        return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=[])


class GPUStatevectorBackend(AbstractBackend):
    """GPU-accelerated statevector backend with CPU fallback."""

    name = "zuntenium_gpu_statevector_simulator"
    max_qubits = 30
    is_simulator = True

    def run(
        self,
        circuit: QuantumCircuit | QuantumIR | Circuit,
        shots: int = 1024,
        **options: Any,
    ) -> ExecutionResult:
        v2 = to_v2_circuit(circuit)
        gpu = GPUStatevectorSimulator()
        res = gpu.execute(v2, shots=shots, **options)
        return ExecutionResult(
            counts=res.counts,
            num_qubits=v2.num_qubits,
            shots=shots,
            statevector=getattr(res, "statevector", None),
            metadata=getattr(res, "metadata", {"backend": self.name}),
        )

    def validate_circuit(self, circuit: Any) -> ValidationResult:
        errors = []
        if getattr(circuit, "num_qubits", 0) > self.max_qubits:
            errors.append(
                f"Circuit has {circuit.num_qubits} qubits, exceeding "
                f"GPU statevector limit of {self.max_qubits}."
            )
        return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=[])
