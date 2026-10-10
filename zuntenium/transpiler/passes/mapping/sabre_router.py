"""
SABRERouter — SABRE heuristic routing algorithm.

Reference: Li et al., "Tackling the Qubit Mapping Problem for NISQ-Era
Quantum Devices", ASPLOS 2019.

Algorithm:
    1. Forward pass: iterate over the front layer of the DAG, score
       candidate SWAPs by a heuristic combining nearest-neighbor distance
       and an extended-set lookahead, insert the best SWAP, and repeat.
    2. Backward pass: reverse the circuit and re-route to refine the
       initial layout.
    3. The final routing uses the better of the two passes.
"""
from __future__ import annotations

import copy
import math
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from zuntenium.transpiler.passes.base_pass import TranspilerPass

if TYPE_CHECKING:
    from zuntenium.circuit.ir.quantum_ir import QuantumIR


# ── Heuristic parameters ─────────────────────────────────────────────

_DECAY_DELTA = 0.001    # Per-SWAP decay increment
_DECAY_RESET = 1.0      # Initial decay value
_EXTENDED_SET_WEIGHT = 0.5  # Weight for the extended-set lookahead term
_SWAP_COST = 3          # A SWAP = 3 CX gates


@dataclass
class _Gate:
    """Lightweight gate for routing."""
    name: str
    qubits: list[int]
    params: list[float] = field(default_factory=list)
    original_index: int = 0

    @property
    def is_two_qubit(self) -> bool:
        return len(self.qubits) >= 2


class SABRERouter(TranspilerPass):
    """
    SABRE heuristic qubit routing pass.

    Inserts SWAP gates to satisfy hardware connectivity constraints
    defined by a coupling map. Uses forward + backward heuristic
    scoring with decay parameters and extended set lookahead.

    Args:
        coupling_edges: List of (q0, q1) edges in the coupling map.
                        If None, extracted from kwargs["coupling_map"]
                        or defaults to a linear chain.
        num_physical: Number of physical qubits. If None, inferred
                      from coupling_edges or the circuit.
        num_trials: Number of forward+backward iterations (default 1).
    """

    name = "SABRERouting"
    description = "SABRE heuristic qubit routing (v0.2)."
    requires = ["Layout"]

    def __init__(
        self,
        coupling_edges: list[tuple[int, int]] | None = None,
        num_physical: int | None = None,
        num_trials: int = 1,
    ) -> None:
        self._coupling_edges = coupling_edges
        self._num_physical = num_physical
        self._num_trials = max(1, num_trials)

    # ── Coupling map helpers ──────────────────────────────────────────

    def _build_adjacency(
        self, edges: list[tuple[int, int]], n: int
    ) -> dict[int, set[int]]:
        adj: dict[int, set[int]] = {i: set() for i in range(n)}
        for q0, q1 in edges:
            adj[q0].add(q1)
            adj[q1].add(q0)
        return adj

    def _bfs_distance(
        self, adj: dict[int, set[int]], src: int, tgt: int
    ) -> int:
        if src == tgt:
            return 0
        visited = {src}
        queue: deque[tuple[int, int]] = deque([(src, 0)])
        while queue:
            node, dist = queue.popleft()
            for nbr in adj[node]:
                if nbr == tgt:
                    return dist + 1
                if nbr not in visited:
                    visited.add(nbr)
                    queue.append((nbr, dist + 1))
        return 999  # disconnected

    def _all_pairs_distance(
        self, adj: dict[int, set[int]], n: int
    ) -> list[list[int]]:
        """BFS from every node — O(V(V+E))."""
        dist = [[999] * n for _ in range(n)]
        for src in range(n):
            dist[src][src] = 0
            visited = {src}
            queue: deque[tuple[int, int]] = deque([(src, 0)])
            while queue:
                node, d = queue.popleft()
                for nbr in adj[node]:
                    if nbr not in visited:
                        visited.add(nbr)
                        dist[src][nbr] = d + 1
                        queue.append((nbr, d + 1))
        return dist

    # ── Gate extraction from IR ───────────────────────────────────────

    def _ir_to_gates(self, ir: QuantumIR) -> list[_Gate]:
        gates: list[_Gate] = []
        for idx, node in enumerate(ir.instructions):
            if node.operation == "barrier":
                continue
            gates.append(_Gate(
                name=node.gate.name if node.gate else node.operation,
                qubits=list(node.qubits),
                params=list(node.params),
                original_index=idx,
            ))
        return gates

    # ── Front layer computation ───────────────────────────────────────

    def _compute_front_layer(
        self,
        gates: list[_Gate],
        executed: set[int],
        n_qubits: int,
    ) -> list[int]:
        """
        Find the front layer: gates whose qubit dependencies are satisfied.

        A gate is in the front layer if no unexecuted gate before it
        acts on the same qubits.
        """
        qubit_blocked: dict[int, int] = {}  # qubit → earliest blocking gate idx
        front: list[int] = []

        for i, gate in enumerate(gates):
            if i in executed:
                continue
            blocked = False
            for q in gate.qubits:
                if q in qubit_blocked and qubit_blocked[q] not in executed:
                    blocked = True
                    break
            if not blocked:
                front.append(i)
                for q in gate.qubits:
                    if q not in qubit_blocked:
                        qubit_blocked[q] = i
            else:
                for q in gate.qubits:
                    if q not in qubit_blocked:
                        qubit_blocked[q] = i

        return front

    def _extended_set(
        self,
        gates: list[_Gate],
        front: list[int],
        executed: set[int],
        depth: int = 20,
    ) -> list[int]:
        """
        Extended set: next `depth` unexecuted 2-qubit gates beyond the front.
        """
        front_set = set(front)
        extended: list[int] = []
        count = 0
        for i, gate in enumerate(gates):
            if i in executed or i in front_set:
                continue
            if gate.is_two_qubit:
                extended.append(i)
                count += 1
                if count >= depth:
                    break
        return extended

    # ── SABRE heuristic scoring ───────────────────────────────────────

    def _heuristic_score(
        self,
        swap: tuple[int, int],
        front: list[int],
        extended: list[int],
        gates: list[_Gate],
        layout: list[int],
        dist_matrix: list[list[int]],
        decay: list[float],
    ) -> float:
        """
        Score a candidate SWAP(p0, p1).

        H = decay_max · (
            Σ_{g ∈ F} D(g) +
            W · Σ_{g ∈ E} D(g)
        )

        where D(g) is the physical distance between g's qubits after
        applying the SWAP, and decay_max = max(decay[p0], decay[p1]).
        """
        # Apply the SWAP to a trial layout
        trial = layout[:]
        p0, p1 = swap
        trial[p0], trial[p1] = trial[p1], trial[p0]

        # Build reverse mapping: logical → physical
        log_to_phys: dict[int, int] = {}
        for phys, log_q in enumerate(trial):
            log_to_phys[log_q] = phys

        # Front layer distance
        front_cost = 0.0
        for gi in front:
            g = gates[gi]
            if g.is_two_qubit:
                pq0 = log_to_phys.get(g.qubits[0], 0)
                pq1 = log_to_phys.get(g.qubits[1], 0)
                front_cost += dist_matrix[pq0][pq1]

        # Extended set distance
        ext_cost = 0.0
        for gi in extended:
            g = gates[gi]
            if g.is_two_qubit:
                pq0 = log_to_phys.get(g.qubits[0], 0)
                pq1 = log_to_phys.get(g.qubits[1], 0)
                ext_cost += dist_matrix[pq0][pq1]

        decay_factor = max(decay[p0], decay[p1])
        return decay_factor * (front_cost + _EXTENDED_SET_WEIGHT * ext_cost)

    # ── Single routing pass ───────────────────────────────────────────

    def _route_pass(
        self,
        gates: list[_Gate],
        n_logical: int,
        n_physical: int,
        adj: dict[int, set[int]],
        dist_matrix: list[list[int]],
        initial_layout: list[int],
    ) -> tuple[list[_Gate], list[int]]:
        """
        One forward pass of the SABRE algorithm.

        Returns (routed_gates, final_layout).
        """
        layout = initial_layout[:]  # physical[i] = logical qubit i
        log_to_phys: dict[int, int] = {}
        for phys, log_q in enumerate(layout):
            if log_q >= 0:
                log_to_phys[log_q] = phys

        routed: list[_Gate] = []
        executed: set[int] = set()
        decay = [_DECAY_RESET] * n_physical
        max_iters = len(gates) * n_physical * 2  # safety bound

        iters = 0
        while len(executed) < len(gates) and iters < max_iters:
            iters += 1
            front = self._compute_front_layer(gates, executed, n_logical)
            if not front:
                break

            # Execute gates that are already nearest-neighbor
            progress = False
            for gi in front:
                g = gates[gi]
                if not g.is_two_qubit:
                    # Single-qubit gate: remap qubit and execute
                    mapped_q = log_to_phys.get(g.qubits[0], g.qubits[0])
                    routed.append(_Gate(
                        name=g.name,
                        qubits=[mapped_q],
                        params=g.params,
                    ))
                    executed.add(gi)
                    progress = True
                else:
                    pq0 = log_to_phys.get(g.qubits[0], g.qubits[0])
                    pq1 = log_to_phys.get(g.qubits[1], g.qubits[1])
                    if dist_matrix[pq0][pq1] == 1:
                        routed.append(_Gate(
                            name=g.name,
                            qubits=[pq0, pq1],
                            params=g.params,
                        ))
                        executed.add(gi)
                        progress = True
                        # Reset decay
                        decay[pq0] = _DECAY_RESET
                        decay[pq1] = _DECAY_RESET

            if progress:
                continue

            # No gate executable — find the best SWAP
            extended = self._extended_set(gates, front, executed)

            # Candidate SWAPs: all edges in the coupling map that involve
            # a qubit used by a front-layer gate
            front_physical: set[int] = set()
            for gi in front:
                g = gates[gi]
                for q in g.qubits:
                    front_physical.add(log_to_phys.get(q, q))

            candidates: list[tuple[int, int]] = []
            for p in front_physical:
                for nbr in adj.get(p, set()):
                    swap = (min(p, nbr), max(p, nbr))
                    if swap not in candidates:
                        candidates.append(swap)

            if not candidates:
                # Fallback: try all coupling edges
                candidates = [(min(a, b), max(a, b)) for a, b in adj.items()
                              for b in adj.get(a, set()) if a < b]
                if not candidates:
                    break

            # Score each candidate
            best_score = float("inf")
            best_swap = candidates[0]
            for swap in candidates:
                score = self._heuristic_score(
                    swap, front, extended, gates, layout, dist_matrix, decay
                )
                if score < best_score:
                    best_score = score
                    best_swap = swap

            # Insert the SWAP
            p0, p1 = best_swap
            routed.append(_Gate(name="SWAP", qubits=[p0, p1]))

            # Update layout
            layout[p0], layout[p1] = layout[p1], layout[p0]
            log_to_phys = {}
            for phys, log_q in enumerate(layout):
                if log_q >= 0:
                    log_to_phys[log_q] = phys

            # Update decay
            decay[p0] += _DECAY_DELTA
            decay[p1] += _DECAY_DELTA

        return routed, layout

    # ── TranspilerPass.run() ──────────────────────────────────────────

    def run(self, ir: QuantumIR, **kwargs: object) -> QuantumIR:
        """
        Apply SABRE routing to the IR.

        Accepts optional kwargs:
            coupling_map: CouplingMap instance or list of edges.
            num_physical: Number of physical qubits.
            initial_layout: List mapping physical → logical qubit.
        """
        from zuntenium.circuit.ir.ir_node import IRNode
        from zuntenium.circuit.ir.quantum_ir import QuantumIR as QIR

        n_logical = ir.num_qubits

        # Resolve coupling map
        coupling_edges = self._coupling_edges
        if coupling_edges is None:
            cm = kwargs.get("coupling_map")
            if cm is not None:
                if hasattr(cm, "edges"):
                    coupling_edges = cm.edges
                elif isinstance(cm, list):
                    coupling_edges = cm
            if coupling_edges is None:
                # Default: linear chain
                coupling_edges = [(i, i + 1) for i in range(n_logical - 1)]

        n_physical = self._num_physical
        if n_physical is None:
            n_physical = int(kwargs.get("num_physical", n_logical))  # type: ignore[arg-type]
        n_physical = max(n_physical, n_logical)

        adj = self._build_adjacency(coupling_edges, n_physical)
        dist_matrix = self._all_pairs_distance(adj, n_physical)

        gates = self._ir_to_gates(ir)

        # Initial layout: identity mapping
        initial_layout = list(range(n_physical))

        best_routed: list[_Gate] | None = None
        best_cost = float("inf")
        best_layout = initial_layout

        for trial in range(self._num_trials):
            # Forward pass
            fwd_routed, fwd_layout = self._route_pass(
                gates, n_logical, n_physical, adj, dist_matrix,
                initial_layout[:],
            )

            # Backward pass: reverse the circuit and re-route
            reversed_gates = list(reversed(gates))
            bwd_routed, bwd_layout = self._route_pass(
                reversed_gates, n_logical, n_physical, adj, dist_matrix,
                fwd_layout[:],
            )

            # Choose the pass with fewer SWAPs
            fwd_swaps = sum(1 for g in fwd_routed if g.name == "SWAP")
            bwd_swaps = sum(1 for g in bwd_routed if g.name == "SWAP")
            fwd_cost = fwd_swaps * _SWAP_COST + len(fwd_routed)
            bwd_cost = bwd_swaps * _SWAP_COST + len(bwd_routed)

            if fwd_cost <= bwd_cost:
                if fwd_cost < best_cost:
                    best_cost = fwd_cost
                    best_routed = fwd_routed
                    best_layout = fwd_layout
            else:
                if bwd_cost < best_cost:
                    best_cost = bwd_cost
                    best_routed = list(reversed(bwd_routed))
                    best_layout = bwd_layout

        if best_routed is None:
            return ir

        # Convert routed gates back to QuantumIR
        from zuntenium.quantum_core.gates import (
            HGate, XGate, YGate, ZGate, SGate, TGate,
            CXGate, CZGate, SWAPGate,
            RxGate, RyGate, RzGate, U3Gate,
        )

        _GATE_MAP: dict[str, type] = {
            "H": HGate, "X": XGate, "Y": YGate, "Z": ZGate,
            "S": SGate, "T": TGate,
            "CX": CXGate, "CNOT": CXGate, "CZ": CZGate, "SWAP": SWAPGate,
        }
        _PARAM_MAP: dict[str, type] = {
            "Rx": RxGate, "Ry": RyGate, "Rz": RzGate, "U3": U3Gate,
        }

        instructions: list[IRNode] = []
        for g in best_routed:
            gate_cls = _GATE_MAP.get(g.name) or _GATE_MAP.get(g.name.upper())
            if gate_cls:
                gate_obj = gate_cls()
            else:
                param_cls = _PARAM_MAP.get(g.name)
                if param_cls and g.params:
                    gate_obj = param_cls(*g.params)
                else:
                    gate_obj = None

            if gate_obj is not None:
                instructions.append(IRNode(
                    operation="gate",
                    gate=gate_obj,
                    qubits=g.qubits,
                    params=g.params,
                ))
            else:
                # Passthrough non-gate operations
                instructions.append(IRNode(
                    operation=g.name.lower(),
                    qubits=g.qubits,
                ))

        return QIR(
            num_qubits=n_physical,
            num_clbits=ir.num_clbits,
            instructions=instructions,
            metadata={
                **ir.metadata,
                "sabre_routing": True,
                "swaps_inserted": sum(
                    1 for g in best_routed if g.name == "SWAP"
                ),
                "final_layout": best_layout,
            },
        )
