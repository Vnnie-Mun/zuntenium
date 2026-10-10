"""
GateCountPass — Count gates by type and annotate IR metadata.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from zuntenium.transpiler.passes.base_pass import TranspilerPass

if TYPE_CHECKING:
    from zuntenium.circuit.ir.quantum_ir import QuantumIR


class GateCountPass(TranspilerPass):
    """
    Analysis pass: counts gates by type; annotates IR with gate_counts.
    """

    name = "GateCount"
    description = "Counts gates by type and annotates IR metadata."

    def run(self, ir: QuantumIR, **kwargs: object) -> QuantumIR:
        counts = ir.gate_counts()
        ir.metadata["gate_counts"] = counts
        ir.metadata["total_gates"] = sum(counts.values())
        ir.metadata["num_qubits"] = ir.num_qubits
        return ir

    def is_idempotent(self) -> bool:
        return True
