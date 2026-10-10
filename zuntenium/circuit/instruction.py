"""
Instruction — a single gate application in a circuit.

An Instruction is the internal representation within the QuantumCircuit
instruction list, prior to IR compilation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from zuntenium.quantum_core.gates.base_gate import BaseGate


@dataclass
class Instruction:
    """
    A single circuit instruction: gate applied to qubits, optionally with clbits.

    Attributes:
        operation: Instruction type string: "gate", "measure", "reset", "barrier".
        gate: The gate object (None for non-gate operations like measure/barrier).
        qubits: List of qubit indices this instruction acts on.
        clbits: List of classical bit indices (for measure operations).
        params: Gate parameters (mirrors gate.params for convenience).
        label: Optional human-readable label for visualization.
        metadata: Arbitrary key-value data for pass annotations.
    """
    operation: str                          # "gate" | "measure" | "reset" | "barrier"
    gate: BaseGate | None = None
    qubits: list[int] = field(default_factory=list)
    clbits: list[int] = field(default_factory=list)
    params: list[float] = field(default_factory=list)
    label: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        valid_ops = {"gate", "measure", "reset", "barrier"}
        if self.operation not in valid_ops:
            raise ValueError(
                f"Unknown operation '{self.operation}'. Must be one of {valid_ops}."
            )
        if self.operation == "gate" and self.gate is None:
            raise ValueError("Instruction with operation='gate' must have a gate object.")

    def __repr__(self) -> str:
        if self.operation == "gate":
            return f"Instruction(gate={self.gate}, qubits={self.qubits})"
        return f"Instruction(op='{self.operation}', qubits={self.qubits}, clbits={self.clbits})"
