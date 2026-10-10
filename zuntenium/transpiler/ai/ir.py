"""
Circuit Intermediate Representation — Gate, DAG, HardwareConfig, TranspilerResult.

Provides the core data structures for the AI Transpiler's internal circuit
representation, independent of the higher-level ``QuantumIR`` / ``IRNode``
used by the main Zuntenium transpiler pipeline.

Gate dataclass is frozen-style (custom __hash__/__eq__) for O(1) set
membership and safe use as DAG node keys.

CircuitDAG wraps ``networkx.DiGraph`` where:
  - Nodes store a ``Gate``
  - Directed edges (u→v) encode "v depends on the output of u on qubit q"
  - Edge attribute ``qubit`` records which qubit creates the dependency

Author:  ZUNTENIUM Core Team
License: Apache 2.0
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════
# Gate dataclass
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Gate:
    """Immutable record for a single gate in the intermediate representation.

    Attributes:
        name:       Canonical gate name (e.g. ``'H'``, ``'RZ'``, ``'CNOT'``).
        qubits:     Target qubit indices as a tuple (little-endian).
        params:     Continuous parameters, e.g. ``(theta,)`` for ``RX(theta)``.
        duration:   Gate duration in nanoseconds (default 0 = unknown).
        error_rate: Depolarizing error probability (default 0).

    Notes:
        ``params`` is stored as a ``tuple[float, ...]`` (not a NumPy array)
        to keep hashing and equality comparison exact and fast.
    """

    name: str
    qubits: Tuple[int, ...]
    params: Tuple[float, ...] = field(default_factory=tuple)
    duration: float = 0.0
    error_rate: float = 0.0

    def __hash__(self) -> int:
        return hash((self.name, self.qubits, self.params))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Gate):
            return NotImplemented
        return (
            self.name == other.name
            and self.qubits == other.qubits
            and self.params == other.params
        )

    @property
    def num_qubits(self) -> int:
        """Number of qubits this gate acts on."""
        return len(self.qubits)

    def is_single_qubit(self) -> bool:
        """Return ``True`` if gate acts on exactly one qubit."""
        return len(self.qubits) == 1

    def is_parametric(self) -> bool:
        """Return ``True`` if gate has continuous parameters."""
        return len(self.params) > 0

    def __repr__(self) -> str:
        p = (
            f"({', '.join(f'{x:.4f}' for x in self.params)})"
            if self.params
            else ""
        )
        return f"{self.name}{p}@{self.qubits}"


# ═══════════════════════════════════════════════════════════════════════════
# HardwareConfig
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class HardwareConfig:
    """Target hardware specification for SWAP routing and noise-aware passes.

    Attributes:
        num_qubits:     Total physical qubits.
        coupling_map:   Adjacency list ``{qubit: [neighbor, ...]}``.
        native_gates:   Gate names natively supported by the device.
        gate_errors:    ``{gate_name: {qubit_tuple: error_rate}}``.
        t1:             Amplitude-damping times per qubit (μs).
        t2:             Dephasing times per qubit (μs).
        gate_durations: ``{gate_name: duration_ns}``.
    """

    num_qubits: int
    coupling_map: Dict[int, List[int]]
    native_gates: Set[str] = field(
        default_factory=lambda: {
            "H", "X", "Y", "Z", "S", "T",
            "RX", "RY", "RZ", "CNOT", "CZ", "SWAP",
        }
    )
    gate_errors: Dict[str, Dict[Tuple[int, ...], float]] = field(
        default_factory=dict
    )
    t1: Dict[int, float] = field(default_factory=dict)
    t2: Dict[int, float] = field(default_factory=dict)
    gate_durations: Dict[str, float] = field(
        default_factory=lambda: {
            "X": 50, "Y": 50, "Z": 50, "H": 50, "S": 50, "T": 50,
            "RX": 80, "RY": 80, "RZ": 80,
            "CNOT": 200, "CZ": 200, "SWAP": 600,
        }
    )

    @classmethod
    def fully_connected(cls, n: int) -> HardwareConfig:
        """All-to-all connectivity (simulator default)."""
        coupling = {i: [j for j in range(n) if j != i] for i in range(n)}
        return cls(num_qubits=n, coupling_map=coupling)

    @classmethod
    def linear_chain(cls, n: int) -> HardwareConfig:
        """Linear chain topology: 0–1–2–…–(n-1)."""
        coupling: Dict[int, List[int]] = {i: [] for i in range(n)}
        for i in range(n - 1):
            coupling[i].append(i + 1)
            coupling[i + 1].append(i)
        return cls(num_qubits=n, coupling_map=coupling)

    def are_connected(self, q0: int, q1: int) -> bool:
        """Return ``True`` if ``q1`` is in ``q0``'s adjacency list."""
        return q1 in self.coupling_map.get(q0, [])

    def shortest_swap_path(self, q0: int, q1: int) -> List[Tuple[int, int]]:
        """BFS shortest path expressed as SWAP pairs (excluding final hop).

        Returns an empty list if ``q0 == q1`` or ``q0, q1`` are already
        connected.  Each returned ``(a, b)`` means "SWAP qubits a and b".
        """
        if q0 == q1 or self.are_connected(q0, q1):
            return []
        G = nx.Graph(self.coupling_map)
        try:
            path = nx.shortest_path(G, q0, q1)
        except nx.NetworkXNoPath:
            return []
        return [(path[i], path[i + 1]) for i in range(len(path) - 2)]


# ═══════════════════════════════════════════════════════════════════════════
# TranspilerResult
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TranspilerResult:
    """Output of ``AITranspiler.optimize()`` with rich metadata."""

    circuit: List[Gate]
    original_depth: int
    optimized_depth: int
    original_gate_count: int
    optimized_gate_count: int
    estimated_fidelity: float
    optimization_time_ms: float
    phase: str
    pass_log: List[str] = field(default_factory=list)

    @property
    def depth_reduction_pct(self) -> float:
        """Percentage depth reduction."""
        if self.original_depth == 0:
            return 0.0
        return 100.0 * (self.original_depth - self.optimized_depth) / self.original_depth

    @property
    def gate_reduction_pct(self) -> float:
        """Percentage gate-count reduction."""
        if self.original_gate_count == 0:
            return 0.0
        return 100.0 * (
            self.original_gate_count - self.optimized_gate_count
        ) / self.original_gate_count

    def summary(self) -> str:
        """Human-readable one-line summary."""
        return (
            f"[{self.phase}] "
            f"Gates: {self.original_gate_count}→{self.optimized_gate_count} "
            f"({self.gate_reduction_pct:.1f}%↓) | "
            f"Depth: {self.original_depth}→{self.optimized_depth} "
            f"({self.depth_reduction_pct:.1f}%↓) | "
            f"Fidelity: {self.estimated_fidelity:.4f} | "
            f"{self.optimization_time_ms:.1f}ms"
        )


# ═══════════════════════════════════════════════════════════════════════════
# CircuitDAG
# ═══════════════════════════════════════════════════════════════════════════

class CircuitDAG:
    """DAG representation of a quantum circuit.

    Nodes = gates (stored as node attribute ``'gate'``).
    Edges = qubit data-dependencies (edge attribute ``'qubit'``).
    An edge ``(u → v)`` exists iff ``v`` depends on the output of ``u``
    on some qubit.

    Depth is computed via a single topological-order traversal tracking
    the longest chain — ``O(V + E)``.
    """

    def __init__(self) -> None:
        self._g: nx.DiGraph = nx.DiGraph()
        self._node_counter: int = 0
        self._qubit_frontier: Dict[int, int] = {}

    def add_gate(self, gate: Gate) -> int:
        """Append *gate* to the DAG, returning its node id."""
        nid = self._node_counter
        self._node_counter += 1
        self._g.add_node(nid, gate=gate)
        for q in gate.qubits:
            if q in self._qubit_frontier:
                self._g.add_edge(self._qubit_frontier[q], nid, qubit=q)
            self._qubit_frontier[q] = nid
        return nid

    def get_gate(self, node_id: int) -> Gate:
        """Return the ``Gate`` stored at *node_id*."""
        return self._g.nodes[node_id]["gate"]

    def set_gate(self, node_id: int, gate: Gate) -> None:
        """Replace the gate stored at *node_id*."""
        self._g.nodes[node_id]["gate"] = gate

    def remove_node(self, node_id: int) -> None:
        """Remove *node_id* and re-wire predecessor/successor edges."""
        preds = list(self._g.predecessors(node_id))
        succs = list(self._g.successors(node_id))
        gate = self.get_gate(node_id)
        for q in gate.qubits:
            pred_q = [
                p for p in preds
                if self._g.edges[p, node_id].get("qubit") == q
            ]
            succ_q = [
                s for s in succs
                if self._g.edges[node_id, s].get("qubit") == q
            ]
            for p, s in itertools.product(pred_q, succ_q):
                if not self._g.has_edge(p, s):
                    self._g.add_edge(p, s, qubit=q)
        self._g.remove_node(node_id)

    def topological_order(self) -> List[int]:
        """Node ids in topological (execution) order."""
        return list(nx.topological_sort(self._g))

    def to_gate_list(self) -> List[Gate]:
        """Flatten the DAG back into a gate list (topological order)."""
        return [self.get_gate(nid) for nid in self.topological_order()]

    def depth(self) -> int:
        """Critical-path length via longest-path DP on DAG — ``O(V+E)``."""
        if len(self._g) == 0:
            return 0
        layers: Dict[int, int] = {}
        for nid in self.topological_order():
            preds = list(self._g.predecessors(nid))
            layers[nid] = 1 + max((layers[p] for p in preds), default=0)
        return max(layers.values(), default=0)

    def gate_count(self) -> int:
        """Total number of gates in the DAG."""
        return self._g.number_of_nodes()

    def copy(self) -> CircuitDAG:
        """Return a shallow copy (gates are shared but graph is independent)."""
        new_dag = CircuitDAG()
        new_dag._g = self._g.copy()
        new_dag._node_counter = self._node_counter
        new_dag._qubit_frontier = self._qubit_frontier.copy()
        return new_dag


# ═══════════════════════════════════════════════════════════════════════════
# Conversion functions
# ═══════════════════════════════════════════════════════════════════════════

def circuit_to_dag(gates: List[Gate]) -> CircuitDAG:
    """Build a ``CircuitDAG`` from a flat gate list."""
    dag = CircuitDAG()
    for gate in gates:
        dag.add_gate(gate)
    return dag


def dag_to_circuit(dag: CircuitDAG) -> List[Gate]:
    """Flatten a ``CircuitDAG`` into a gate list (topological order)."""
    return dag.to_gate_list()


def circuit_to_zuntenium_format(gates: List[Gate]) -> Any:
    """Convert a gate list to a ``StatevectorSimulator``-compatible circuit.

    Uses the simulator-local ``QuantumCircuit`` class (different from the
    builder-level ``zuntenium.circuit.circuit.QuantumCircuit``).

    Raises:
        ImportError: If the statevector sub-package cannot be imported.
    """
    from zuntenium.quantum_core.statevector import (
        SimulatorQuantumCircuit,
    )

    n = max((max(g.qubits) for g in gates), default=0) + 1
    qc = SimulatorQuantumCircuit(n)
    for gate in gates:
        nm = gate.name.upper()
        q = gate.qubits
        p = gate.params
        if nm == "X":
            qc.x(q[0])
        elif nm == "Y":
            qc.y(q[0])
        elif nm == "Z":
            qc.z(q[0])
        elif nm == "H":
            qc.h(q[0])
        elif nm == "S":
            qc.s(q[0])
        elif nm == "T":
            qc.t(q[0])
        elif nm == "SDG":
            qc.sdg(q[0])
        elif nm == "TDG":
            qc.tdg(q[0])
        elif nm == "RX":
            qc.rx(p[0], q[0])
        elif nm == "RY":
            qc.ry(p[0], q[0])
        elif nm == "RZ":
            qc.rz(p[0], q[0])
        elif nm == "U3":
            qc.u3(p[0], p[1], p[2], q[0])
        elif nm in ("CNOT", "CX"):
            qc.cx(q[0], q[1])
        elif nm == "CZ":
            qc.cz(q[0], q[1])
        elif nm == "SWAP":
            qc.swap(q[0], q[1])
        elif nm in ("CCX", "TOFFOLI"):
            qc.ccx(q[0], q[1], q[2])
    return qc
