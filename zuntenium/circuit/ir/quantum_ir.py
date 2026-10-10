"""
QuantumIR — Canonical Intermediate Representation.

This is the contract between the circuit builder, transpiler, and backends.
It is independent of any specific gate library.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from zuntenium.circuit.ir.ir_node import IRNode


@dataclass
class QuantumIR:
    """
    Canonical Intermediate Representation for quantum circuits.

    Independent of the gate library; used by the transpiler and all backends.
    Created by QuantumCircuit.to_ir() and consumed by TranspilerPipeline.run()
    and AbstractBackend.run().

    Attributes:
        num_qubits: Number of qubits.
        num_clbits: Number of classical bits.
        instructions: Ordered list of IRNode instructions.
        metadata: Arbitrary metadata dict (circuit name, source, etc.).
    """
    num_qubits: int
    num_clbits: int
    instructions: list[IRNode]
    metadata: dict[str, Any] = field(default_factory=dict)

    # ── Analysis ───────────────────────────────────────────────────────────

    def depth(self) -> int:
        """Compute circuit depth (critical path over qubit layers)."""
        latest: dict[int, int] = {}
        for node in self.instructions:
            if node.operation not in ("gate", "measure", "reset"):
                continue
            layer = max((latest.get(q, 0) for q in node.qubits), default=0) + 1
            for q in node.qubits:
                latest[q] = layer
        return max(latest.values()) if latest else 0

    def num_gates(self) -> int:
        """Return count of gate instructions."""
        return sum(1 for n in self.instructions if n.is_gate())

    def gate_counts(self) -> dict[str, int]:
        """Return {gate_name: count} for all gate instructions."""
        counts: dict[str, int] = {}
        for node in self.instructions:
            if node.is_gate():
                name = node.gate_name()
                counts[name] = counts.get(name, 0) + 1
        return counts

    def to_layers(self) -> list[list[IRNode]]:
        """
        Partition instructions into parallel layers.

        Each layer contains gates that act on disjoint qubit sets
        and can be executed simultaneously.
        """
        layers: list[list[IRNode]] = []
        qubit_layer: dict[int, int] = {}   # qubit → last layer index it was used

        for node in self.instructions:
            if node.operation == "barrier":
                continue
            layer_idx = max((qubit_layer.get(q, -1) for q in node.qubits), default=-1) + 1
            while layer_idx >= len(layers):
                layers.append([])
            layers[layer_idx].append(node)
            for q in node.qubits:
                qubit_layer[q] = layer_idx

        return layers

    def to_dag(self) -> DAGCircuit:
        """Convert to a DAGCircuit for topological analysis."""
        from zuntenium.circuit.ir.dag import DAGCircuit
        return DAGCircuit.from_ir(self)

    def copy(self) -> QuantumIR:
        """Return a deep copy of this IR."""
        import copy
        return copy.deepcopy(self)

    def __len__(self) -> int:
        return len(self.instructions)

    def __repr__(self) -> str:
        return (
            f"QuantumIR(num_qubits={self.num_qubits}, "
            f"num_gates={self.num_gates()}, "
            f"depth={self.depth()})"
        )
