"""
DAGCircuit — Directed Acyclic Graph representation of a quantum circuit.

Nodes represent quantum operations (gates, measurements, resets).
Edges represent data-flow dependencies through qubit wires.

Used for topological analysis, critical-path scheduling,
and dependency-aware pass ordering.

Reference: Qiskit Terra DAG architecture (simplified, no external deps).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterator

if TYPE_CHECKING:
    from zuntenium.circuit.ir.ir_node import IRNode
    from zuntenium.circuit.ir.quantum_ir import QuantumIR


# ── DAG Node ──────────────────────────────────────────────────────────────

@dataclass
class DAGNode:
    """
    A node in the DAGCircuit.

    Attributes:
        node_id:    Unique integer identifier within the DAG.
        ir_node:    The underlying IRNode instruction.
        qubits:     Qubit indices this operation touches.
        clbits:     Classical bit indices (for measure).
        layer:      Computed layer index (set during layering).
        predecessors: Set of predecessor node IDs (data dependencies).
        successors:   Set of successor node IDs.
    """
    node_id: int
    ir_node: IRNode
    qubits: list[int] = field(default_factory=list)
    clbits: list[int] = field(default_factory=list)
    layer: int = -1
    predecessors: set[int] = field(default_factory=set)
    successors: set[int] = field(default_factory=set)

    @property
    def name(self) -> str:
        """Gate or operation name."""
        if self.ir_node.is_gate() and self.ir_node.gate is not None:
            return self.ir_node.gate.name
        return self.ir_node.operation.upper()

    @property
    def is_gate(self) -> bool:
        return self.ir_node.is_gate()

    def __repr__(self) -> str:
        return (
            f"DAGNode(id={self.node_id}, {self.name}, "
            f"qubits={self.qubits}, layer={self.layer})"
        )


# ── Special sentinel nodes ────────────────────────────────────────────────

_INPUT_OP = "input"
_OUTPUT_OP = "output"


# ── DAGCircuit ────────────────────────────────────────────────────────────

class DAGCircuit:
    """
    Directed Acyclic Graph representation of a quantum circuit.

    Provides topological ordering, critical-path analysis,
    dependency-aware layering, and subgraph extraction.

    Example::

        from zuntenium.circuit.ir.dag import DAGCircuit

        dag = DAGCircuit.from_ir(my_ir)
        for node in dag.topological_sort():
            print(node.name, node.qubits)

        print(f"Critical path length: {dag.critical_path_length()}")
        layers = dag.layers()
        for i, layer in enumerate(layers):
            print(f"Layer {i}: {[n.name for n in layer]}")
    """

    def __init__(self, num_qubits: int, num_clbits: int = 0) -> None:
        self.num_qubits = num_qubits
        self.num_clbits = num_clbits
        self._nodes: dict[int, DAGNode] = {}
        self._next_id: int = 0

        # Track the latest node on each qubit wire for edge construction
        self._qubit_last: dict[int, int] = {}  # qubit → last node_id
        self._clbit_last: dict[int, int] = {}  # clbit → last node_id

    # ── Node management ───────────────────────────────────────────────

    def _alloc_id(self) -> int:
        nid = self._next_id
        self._next_id += 1
        return nid

    def add_node(self, ir_node: IRNode) -> int:
        """
        Add an instruction node and wire edges from its qubit predecessors.

        Returns the assigned node ID.
        """
        nid = self._alloc_id()
        dag_node = DAGNode(
            node_id=nid,
            ir_node=ir_node,
            qubits=list(ir_node.qubits),
            clbits=list(ir_node.clbits),
        )

        # Wire edges: this node depends on the previous op on each qubit
        for q in ir_node.qubits:
            prev = self._qubit_last.get(q)
            if prev is not None:
                dag_node.predecessors.add(prev)
                self._nodes[prev].successors.add(nid)
            self._qubit_last[q] = nid

        for c in ir_node.clbits:
            prev = self._clbit_last.get(c)
            if prev is not None:
                dag_node.predecessors.add(prev)
                self._nodes[prev].successors.add(nid)
            self._clbit_last[c] = nid

        self._nodes[nid] = dag_node
        return nid

    def get_node(self, node_id: int) -> DAGNode:
        """Retrieve a node by ID."""
        return self._nodes[node_id]

    @property
    def nodes(self) -> list[DAGNode]:
        """All DAG nodes in insertion order."""
        return list(self._nodes.values())

    @property
    def num_nodes(self) -> int:
        return len(self._nodes)

    # ── Traversal ─────────────────────────────────────────────────────

    def predecessors(self, node_id: int) -> list[DAGNode]:
        """Direct predecessor nodes of the given node."""
        return [self._nodes[pid] for pid in self._nodes[node_id].predecessors]

    def successors(self, node_id: int) -> list[DAGNode]:
        """Direct successor nodes of the given node."""
        return [self._nodes[sid] for sid in self._nodes[node_id].successors]

    def ancestors(self, node_id: int) -> set[int]:
        """All transitive predecessors (BFS)."""
        visited: set[int] = set()
        queue = deque(self._nodes[node_id].predecessors)
        while queue:
            nid = queue.popleft()
            if nid in visited:
                continue
            visited.add(nid)
            queue.extend(self._nodes[nid].predecessors)
        return visited

    def descendants(self, node_id: int) -> set[int]:
        """All transitive successors (BFS)."""
        visited: set[int] = set()
        queue = deque(self._nodes[node_id].successors)
        while queue:
            nid = queue.popleft()
            if nid in visited:
                continue
            visited.add(nid)
            queue.extend(self._nodes[nid].successors)
        return visited

    # ── Topological sort (Kahn's algorithm) ───────────────────────────

    def topological_sort(self) -> list[DAGNode]:
        """
        Return nodes in topological order (Kahn's algorithm).

        Raises ValueError if the graph contains a cycle (shouldn't happen
        for well-formed circuits, but safety check).
        """
        in_degree: dict[int, int] = {
            nid: len(node.predecessors) for nid, node in self._nodes.items()
        }
        queue: deque[int] = deque(
            nid for nid, deg in in_degree.items() if deg == 0
        )
        result: list[DAGNode] = []

        while queue:
            nid = queue.popleft()
            result.append(self._nodes[nid])
            for sid in self._nodes[nid].successors:
                in_degree[sid] -= 1
                if in_degree[sid] == 0:
                    queue.append(sid)

        if len(result) != len(self._nodes):
            raise ValueError(
                "Cycle detected in DAGCircuit — this should not happen "
                "for a well-formed quantum circuit."
            )
        return result

    # ── Layering ──────────────────────────────────────────────────────

    def layers(self) -> list[list[DAGNode]]:
        """
        Partition nodes into parallel execution layers.

        Each layer contains nodes whose predecessors are all in earlier
        layers — i.e., they can execute simultaneously.
        """
        if not self._nodes:
            return []

        # Compute layer index for each node
        layer_of: dict[int, int] = {}
        for node in self.topological_sort():
            if not node.predecessors:
                layer_of[node.node_id] = 0
            else:
                layer_of[node.node_id] = (
                    max(layer_of[pid] for pid in node.predecessors) + 1
                )
            node.layer = layer_of[node.node_id]

        # Group by layer
        num_layers = max(layer_of.values()) + 1 if layer_of else 0
        result: list[list[DAGNode]] = [[] for _ in range(num_layers)]
        for nid, layer_idx in layer_of.items():
            result[layer_idx].append(self._nodes[nid])
        return result

    # ── Critical path ─────────────────────────────────────────────────

    def critical_path(self) -> list[DAGNode]:
        """
        Compute the longest path through the DAG (critical path).

        This determines the minimum circuit depth under maximum
        parallelism. Uses dynamic programming on topological order.
        """
        if not self._nodes:
            return []

        topo = self.topological_sort()

        # dist[nid] = length of longest path ending at nid
        dist: dict[int, int] = {}
        parent: dict[int, int | None] = {}

        for node in topo:
            if not node.predecessors:
                dist[node.node_id] = 1
                parent[node.node_id] = None
            else:
                best_pred = max(
                    node.predecessors, key=lambda pid: dist[pid]
                )
                dist[node.node_id] = dist[best_pred] + 1
                parent[node.node_id] = best_pred

        # Trace back from the node with the longest path
        end_node = max(dist, key=lambda nid: dist[nid])
        path: list[DAGNode] = []
        current: int | None = end_node
        while current is not None:
            path.append(self._nodes[current])
            current = parent[current]
        path.reverse()
        return path

    def critical_path_length(self) -> int:
        """Return the length of the critical path (= circuit depth)."""
        path = self.critical_path()
        return len(path)

    # ── Subgraph extraction ───────────────────────────────────────────

    def nodes_on_qubits(self, qubits: set[int]) -> list[DAGNode]:
        """Return all nodes that touch any of the given qubits."""
        return [
            node for node in self._nodes.values()
            if set(node.qubits) & qubits
        ]

    def gate_nodes(self) -> list[DAGNode]:
        """Return only gate nodes (exclude barriers, measures, etc.)."""
        return [node for node in self._nodes.values() if node.is_gate]

    def two_qubit_gates(self) -> list[DAGNode]:
        """Return nodes for 2-qubit gates."""
        return [
            node for node in self._nodes.values()
            if node.is_gate and len(node.qubits) >= 2
        ]

    # ── Back to IR ────────────────────────────────────────────────────

    def to_ir(self) -> QuantumIR:
        """Convert the DAG back to a QuantumIR in topological order."""
        from zuntenium.circuit.ir.quantum_ir import QuantumIR

        topo = self.topological_sort()
        instructions = [node.ir_node for node in topo]
        return QuantumIR(
            num_qubits=self.num_qubits,
            num_clbits=self.num_clbits,
            instructions=instructions,
        )

    # ── Construction from IR ──────────────────────────────────────────

    @classmethod
    def from_ir(cls, ir: QuantumIR) -> DAGCircuit:
        """
        Build a DAGCircuit from a QuantumIR.

        Iterates through the IR instructions and wires edges based on
        qubit-wire dependencies.
        """
        dag = cls(num_qubits=ir.num_qubits, num_clbits=ir.num_clbits)
        for node in ir.instructions:
            if node.operation == "barrier":
                continue
            dag.add_node(node)
        return dag

    # ── Analysis utilities ────────────────────────────────────────────

    def depth(self) -> int:
        """Circuit depth (number of layers)."""
        layers = self.layers()
        return len(layers)

    def width(self) -> int:
        """Maximum parallelism (widest layer)."""
        layers = self.layers()
        return max(len(layer) for layer in layers) if layers else 0

    def count_ops(self) -> dict[str, int]:
        """Count of each gate type."""
        counts: dict[str, int] = {}
        for node in self._nodes.values():
            if node.is_gate:
                name = node.name
                counts[name] = counts.get(name, 0) + 1
        return counts

    def __repr__(self) -> str:
        return (
            f"DAGCircuit(num_qubits={self.num_qubits}, "
            f"nodes={self.num_nodes}, depth={self.depth()})"
        )

    def __len__(self) -> int:
        return self.num_nodes
