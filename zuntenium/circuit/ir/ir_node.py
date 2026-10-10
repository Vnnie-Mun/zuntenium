"""
IRNode — A single typed instruction node in the QuantumIR.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from zuntenium.quantum_core.gates.base_gate import BaseGate


@dataclass
class IRNode:
    """
    A single instruction in the QuantumIR instruction list.

    Attributes:
        operation: "gate" | "measure" | "reset" | "barrier"
        gate: Gate object (None for non-gate operations).
        qubits: Qubit indices this instruction acts on.
        clbits: Classical bit indices (for measure).
        params: Gate parameter values (mirrors gate.params).
        annotations: Key-value metadata added by transpiler passes.
    """
    operation: str
    gate: BaseGate | None = None
    qubits: list[int] = field(default_factory=list)
    clbits: list[int] = field(default_factory=list)
    params: list[float] = field(default_factory=list)
    annotations: dict[str, Any] = field(default_factory=dict)

    def is_gate(self) -> bool:
        return self.operation == "gate" and self.gate is not None

    def gate_name(self) -> str:
        if self.gate is not None:
            return self.gate.name
        return self.operation.upper()

    def __repr__(self) -> str:
        if self.is_gate():
            return f"IRNode({self.gate}, qubits={self.qubits})"
        return f"IRNode(op='{self.operation}', qubits={self.qubits})"
