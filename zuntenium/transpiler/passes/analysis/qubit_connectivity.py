"""
QubitConnectivityPass — Analyse which qubit pairs interact in the circuit.

Output is used by routing passes to determine SWAP insertion needs.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from zuntenium.transpiler.passes.base_pass import TranspilerPass

if TYPE_CHECKING:
    from zuntenium.circuit.ir.quantum_ir import QuantumIR


class QubitConnectivityPass(TranspilerPass):
    """
    Builds the circuit's logical qubit interaction graph and annotates IR.

    For each two-qubit gate, records the (control, target) pair.
    The result is stored in ir.metadata['interaction_graph'].
    """

    name = "QubitConnectivity"
    description = "Analyses qubit pair interactions to build the circuit connectivity graph."

    def run(self, ir: QuantumIR, **kwargs: object) -> QuantumIR:
        interactions: dict[tuple[int, int], int] = {}

        for node in ir.instructions:
            if node.is_gate() and len(node.qubits) == 2:
                pair = (node.qubits[0], node.qubits[1])
                interactions[pair] = interactions.get(pair, 0) + 1

        ir.metadata["interaction_graph"] = {
            f"{q0}-{q1}": count for (q0, q1), count in interactions.items()
        }
        ir.metadata["num_two_qubit_gates"] = sum(
            1 for n in ir.instructions if n.is_gate() and len(n.qubits) >= 2
        )
        return ir

    def is_idempotent(self) -> bool:
        return True
