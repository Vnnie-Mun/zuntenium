"""
RoutingPass — SWAP insertion for hardware connectivity.

v0.1: Basic greedy SWAP routing.
v0.2: SABRE heuristic routing (see sabre_router.py).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from zuntenium.circuit.ir.ir_node import IRNode
from zuntenium.transpiler.passes.base_pass import TranspilerPass

if TYPE_CHECKING:
    from zuntenium.circuit.ir.quantum_ir import QuantumIR


class RoutingPass(TranspilerPass):
    """
    Inserts SWAP gates to satisfy hardware connectivity constraints.

    v0.1: Simple greedy routing — inserts SWAPs along shortest path.
    Requires LayoutPass to have run first (needs ir.metadata['layout']).
    """

    name = "Routing"
    description = "Inserts SWAP gates to route two-qubit gates to connected qubit pairs."
    requires = ["Layout", "QubitConnectivity"]

    def run(self, ir: QuantumIR, **kwargs: object) -> QuantumIR:
        from zuntenium.circuit.ir.quantum_ir import QuantumIR
        from zuntenium.quantum_core.gates import SWAPGate
        from zuntenium.transpiler.target import Target

        target_raw = kwargs.get("target")
        if target_raw is None or not isinstance(target_raw, Target) or target_raw.coupling_map is None:
            # No connectivity constraint — pass through
            return ir

        target: Target = target_raw
        assert target.coupling_map is not None  # Guarded above
        cm = target.coupling_map
        layout: dict[int, int] = dict(ir.metadata.get("layout", {q: q for q in range(ir.num_qubits)}))
        routed: list[IRNode] = []

        for node in ir.instructions:
            if node.is_gate() and len(node.qubits) == 2:
                p0 = layout[node.qubits[0]]
                p1 = layout[node.qubits[1]]

                if not cm.is_connected(p0, p1):
                    # Insert SWAPs along shortest path
                    path = cm.shortest_path(p0, p1)
                    for k in range(len(path) - 2):
                        a, b = path[k], path[k + 1]
                        swap_node = IRNode(
                            operation="gate",
                            gate=SWAPGate(),
                            qubits=[a, b],
                            annotations={"inserted_by": "RoutingPass"},
                        )
                        routed.append(swap_node)
                        # Update layout
                        inv = {v: k_ for k_, v in layout.items()}
                        if a in inv and b in inv:
                            la, lb = inv[a], inv[b]
                            layout[la], layout[lb] = b, a

                # Apply gate with updated layout
                routed.append(IRNode(
                    operation="gate",
                    gate=node.gate,
                    qubits=[layout[node.qubits[0]], layout[node.qubits[1]]],
                    params=node.params,
                ))
            else:
                # Single-qubit or non-gate: remap qubits through layout
                new_qubits = [layout.get(q, q) for q in node.qubits]
                routed.append(IRNode(
                    operation=node.operation,
                    gate=node.gate,
                    qubits=new_qubits,
                    clbits=node.clbits,
                    params=node.params,
                ))

        return QuantumIR(
            num_qubits=ir.num_qubits,
            num_clbits=ir.num_clbits,
            instructions=routed,
            metadata={**ir.metadata, "routed": True, "final_layout": layout},
        )
