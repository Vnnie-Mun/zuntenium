"""
GateCancellationPass — Remove consecutive self-inverse gate pairs.

Examples of cancelled pairs:
  H · H = I         (Hadamard is self-inverse)
  X · X = I         (NOT is self-inverse)
  CX · CX = I       (CNOT applied twice = identity)

Algorithm: single forward scan; when the next gate is the inverse of the
last gate on the same qubits, both are removed.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from zuntenium.circuit.ir.ir_node import IRNode
from zuntenium.transpiler.passes.base_pass import TranspilerPass

if TYPE_CHECKING:
    from zuntenium.circuit.ir.quantum_ir import QuantumIR

# Gates that are their own inverse (U² = I)
_SELF_INVERSE = {"H", "X", "Y", "Z", "CX", "CZ", "CY", "SWAP"}


class GateCancellationPass(TranspilerPass):
    """
    Remove consecutive pairs of self-inverse gates acting on the same qubits.

    Conservative: only cancels adjacent gates. For non-adjacent cancellation
    see the full commutation-analysis pass (v0.2).
    """

    name = "GateCancellation"
    description = "Removes consecutive self-inverse gate pairs (H·H=I, X·X=I, CX·CX=I)."

    def run(self, ir: QuantumIR, **kwargs: object) -> QuantumIR:
        from zuntenium.circuit.ir.quantum_ir import QuantumIR

        changed = True
        instructions = list(ir.instructions)

        while changed:
            changed = False
            new_instructions: list[IRNode] = []
            skip_next = False

            for i, node in enumerate(instructions):
                if skip_next:
                    skip_next = False
                    continue

                if i + 1 < len(instructions):
                    next_node = instructions[i + 1]
                    next_gate = next_node.gate
                    if (
                        node.is_gate()
                        and next_node.is_gate()
                        and node.gate is not None
                        and next_gate is not None
                        and node.gate.name in _SELF_INVERSE
                        and node.gate.name == next_gate.name
                        and node.qubits == next_node.qubits
                    ):
                        # Cancel this pair
                        skip_next = True
                        changed = True
                        continue

                new_instructions.append(node)

            instructions = new_instructions

        return QuantumIR(
            num_qubits=ir.num_qubits,
            num_clbits=ir.num_clbits,
            instructions=instructions,
            metadata={**ir.metadata, "gate_cancellation": True},
        )

    def is_idempotent(self) -> bool:
        return True
