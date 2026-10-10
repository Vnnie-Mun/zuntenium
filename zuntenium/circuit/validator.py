"""
CircuitValidator — validates circuit topology and gate support before compilation.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from zuntenium.quantum_core.exceptions import CircuitValidationError, QubitIndexError

if TYPE_CHECKING:
    from zuntenium.circuit.circuit import QuantumCircuit


class CircuitValidator:
    """
    Validates a QuantumCircuit before IR compilation.

    Checks:
    - All qubit/clbit indices are in range
    - No duplicate qubits within a single instruction
    - Measure targets have corresponding classical bits
    - Gate qubit count matches gate.num_qubits

    Args:
        circuit: The circuit to validate.
    """

    def __init__(self, circuit: QuantumCircuit) -> None:
        self._circuit = circuit

    def validate(self) -> None:
        """Run all validation checks. Raises on first failure."""
        for i, instr in enumerate(self._circuit.instructions):
            self._check_qubit_indices(instr, i)
            self._check_clbit_indices(instr, i)
            self._check_no_duplicate_qubits(instr, i)
            if instr.operation == "gate":
                self._check_gate_qubit_count(instr, i)

    def _check_qubit_indices(self, instr: object, idx: int) -> None:
        from zuntenium.circuit.instruction import Instruction
        assert isinstance(instr, Instruction)
        n = self._circuit.num_qubits
        for q in instr.qubits:
            if not (0 <= q < n):
                raise QubitIndexError(q, n)

    def _check_clbit_indices(self, instr: object, idx: int) -> None:
        from zuntenium.circuit.instruction import Instruction
        assert isinstance(instr, Instruction)
        n = self._circuit.num_clbits
        for c in instr.clbits:
            if not (0 <= c < n):
                raise CircuitValidationError(
                    f"Instruction {idx}: classical bit index {c} out of range "
                    f"[0, {n}) (circuit has {n} classical bits)."
                )

    def _check_no_duplicate_qubits(self, instr: object, idx: int) -> None:
        from zuntenium.circuit.instruction import Instruction
        assert isinstance(instr, Instruction)
        if len(instr.qubits) != len(set(instr.qubits)):
            raise CircuitValidationError(
                f"Instruction {idx}: duplicate qubit indices {instr.qubits}. "
                "A gate cannot act on the same qubit twice."
            )

    def _check_gate_qubit_count(self, instr: object, idx: int) -> None:
        from zuntenium.circuit.instruction import Instruction
        assert isinstance(instr, Instruction)
        if instr.gate is None:
            return
        expected = instr.gate.num_qubits
        got = len(instr.qubits)
        if expected != got:
            raise CircuitValidationError(
                f"Instruction {idx}: gate '{instr.gate.name}' expects "
                f"{expected} qubit(s), got {got}."
            )
