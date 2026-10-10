"""
UnitaryBackend — Full unitary matrix simulation backend.

Instead of sampling, computes the full 2^N × 2^N unitary matrix of the circuit.
Useful for small circuits (≤12 qubits) where you want the exact gate matrix.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from zuntenium.backend.base_backend import AbstractBackend, ValidationResult

if TYPE_CHECKING:
    from zuntenium.circuit.circuit import QuantumCircuit
    from zuntenium.circuit.ir.quantum_ir import QuantumIR
    from zuntenium.quantum_core.executor.result import ExecutionResult


class UnitaryBackend(AbstractBackend):
    """
    Computes the full unitary matrix of a circuit.

    WARNING: Memory scales as O(4^N). Practical limit ~12 qubits.

    Args:
        max_qubits: Override safety limit (default 14).
    """

    name = "zuntenium_unitary_simulator"
    is_simulator = True

    def __init__(self, max_qubits: int = 14) -> None:
        self.max_qubits = max_qubits

    def run(
        self,
        circuit: QuantumCircuit | QuantumIR,
        shots: int = 1024,
        **options: Any,
    ) -> ExecutionResult:
        from zuntenium.circuit.circuit import QuantumCircuit
        from zuntenium.quantum_core.executor.result import ExecutionResult

        if isinstance(circuit, QuantumCircuit):
            ir = circuit.to_ir()
        else:
            ir = circuit

        n = ir.num_qubits
        if n > self.max_qubits:
            raise ValueError(
                f"UnitaryBackend: {n} qubits exceeds limit {self.max_qubits}. "
                "Use StatevectorBackend for large circuits."
            )

        dim = 2 ** n
        unitary = np.eye(dim, dtype=np.complex128)

        for node in ir.instructions:
            if node.is_gate() and node.gate is not None:
                gate_matrix = node.gate.matrix()
                full = self._embed_gate(gate_matrix, node.qubits, n)
                unitary = full @ unitary

        # Apply to |0...0⟩ to get statevector
        state = unitary[:, 0]

        from zuntenium.quantum_core.executor.measurement import measure_all
        measurement = measure_all(state, n, shots)

        return ExecutionResult(
            counts=measurement.counts,
            num_qubits=n,
            shots=shots,
            statevector=state,
            metadata={"backend": self.name, "unitary_shape": (dim, dim)},
        )

    def validate_circuit(self, circuit: QuantumCircuit) -> ValidationResult:
        errors: list[str] = []
        warnings: list[str] = []
        if circuit.num_qubits > self.max_qubits:
            errors.append(
                f"UnitaryBackend: circuit has {circuit.num_qubits} qubits, "
                f"limit is {self.max_qubits}."
            )
        return ValidationResult(valid=not errors, errors=errors, warnings=warnings)

    @staticmethod
    def _embed_gate(gate_matrix: np.ndarray, qubits: list[int], n: int) -> np.ndarray:
        """Embed a k-qubit gate matrix into the full N-qubit space."""
        from zuntenium.quantum_core.statevector.statevector_numpy import apply_n_qubit_gate
        # Build the full unitary by applying gate to each basis vector
        dim = 2 ** n
        result = np.zeros((dim, dim), dtype=np.complex128)
        basis = np.eye(dim, dtype=np.complex128)
        for col in range(dim):
            result[:, col] = apply_n_qubit_gate(basis[:, col], gate_matrix, qubits, n)
        return result
