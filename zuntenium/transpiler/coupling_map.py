"""
CouplingMap — hardware qubit connectivity graph.

Represents which pairs of physical qubits can directly interact via
two-qubit gates. Used by routing passes to determine SWAP insertion.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CouplingMap:
    """
    Directed or undirected graph of qubit-qubit connections.

    Args:
        num_qubits: Total number of physical qubits.
        edges: List of (control, target) qubit pairs that can interact.
        directed: If False (default), edges are treated as bidirectional.

    Example::

        # Linear 5-qubit chain: 0-1-2-3-4
        cm = CouplingMap(5, [(0,1),(1,2),(2,3),(3,4)])
        cm.is_connected(0, 1)  # True
        cm.is_connected(0, 2)  # False (no direct edge)
        cm.shortest_path(0, 3) # [0, 1, 2, 3]
    """
    num_qubits: int
    edges: list[tuple[int, int]] = field(default_factory=list)
    directed: bool = False

    def __post_init__(self) -> None:
        self._adj: dict[int, set[int]] = {i: set() for i in range(self.num_qubits)}
        for q0, q1 in self.edges:
            self._adj[q0].add(q1)
            if not self.directed:
                self._adj[q1].add(q0)

    def is_connected(self, q0: int, q1: int) -> bool:
        """Return True if there is a direct edge between q0 and q1."""
        return q1 in self._adj.get(q0, set())

    def neighbors(self, qubit: int) -> set[int]:
        """Return the set of qubits directly connected to ``qubit``."""
        return set(self._adj.get(qubit, set()))

    def shortest_path(self, source: int, target: int) -> list[int]:
        """BFS shortest path between source and target qubits."""
        if source == target:
            return [source]
        visited = {source}
        queue: list[list[int]] = [[source]]
        while queue:
            path = queue.pop(0)
            node = path[-1]
            for nbr in self._adj.get(node, set()):
                if nbr == target:
                    return path + [nbr]
                if nbr not in visited:
                    visited.add(nbr)
                    queue.append(path + [nbr])
        return []   # disconnected

    def distance(self, q0: int, q1: int) -> int:
        """Return shortest path length between q0 and q1, or -1 if unreachable."""
        path = self.shortest_path(q0, q1)
        return len(path) - 1 if path else -1

    @classmethod
    def linear(cls, num_qubits: int) -> CouplingMap:
        """Create a linear chain: 0-1-2-..-(n-1)."""
        edges = [(i, i + 1) for i in range(num_qubits - 1)]
        return cls(num_qubits=num_qubits, edges=edges)

    @classmethod
    def fully_connected(cls, num_qubits: int) -> CouplingMap:
        """Create a fully connected (all-to-all) coupling map."""
        edges = [(i, j) for i in range(num_qubits) for j in range(num_qubits) if i != j]
        return cls(num_qubits=num_qubits, edges=edges, directed=True)

    def __repr__(self) -> str:
        return f"CouplingMap(num_qubits={self.num_qubits}, edges={self.edges})"
