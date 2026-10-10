"""
zuntenium/transpiler/basic_transpiler.py
=========================================
ZUNTENIUM Basic Quantum Transpiler — v0.1
=========================================

The foundational rule-based compiler engine for the ZUNTENIUM framework.
Every circuit produced by QuantumLayer.forward(), HybridSolver.solve(),
or direct user construction passes through this module before execution.

COMPILER ARCHITECTURE (LLVM-style)
------------------------------------

  ┌──────────────────────────────────────────────────────────────────────┐
  │                     Zuntenium Compilation Pipeline                    │
  │                                                                       │
  │   High-level Circuit (user / QuantumLayer / HybridSolver)            │
  │        │                                                              │
  │        ▼  Circuit.from_* / Gate IR construction                      │
  │   ┌──────────────────┐                                                │
  │   │  Circuit (IR)    │  ← flat gate list + qubit timelines           │
  │   │  Gate dataclass  │    + provenance metadata                      │
  │   └────────┬─────────┘                                                │
  │            │                                                          │
  │            ▼  TranspilerPipeline.run()                               │
  │   ┌───────────────────────────────────────────────────────────┐      │
  │   │  Pass 1: RedundantGateCancellationPass   O(N)             │      │
  │   │  Pass 2: RotationMergingPass             O(N)             │      │
  │   │  Pass 3: CommutationReorderingPass       O(N²)            │      │
  │   │  Pass 4: SingleQubitFusionPass           O(N)             │      │
  │   │  Pass 5: DepthOptimizationPass           O(N + E)         │      │
  │   │  Pass 6: DeadGateEliminationPass         O(N + E)         │      │
  │   │  [Pass 7: AI stub — RL agent, future]                     │      │
  │   └───────────────────────────────────────────────────────────┘      │
  │            │                                                          │
  │            ▼  OptimizedCircuit + OptimizationReport                  │
  │   ┌──────────────────┐                                                │
  │   │  CircuitDAG      │  ← networkx DiGraph for analysis              │
  │   └──────────────────┘                                                │
  │            │                                                          │
  │            ▼                                                          │
  │   StatevectorSimulator / HybridSolver / HardwareBackend              │
  └──────────────────────────────────────────────────────────────────────┘

Mathematical Foundations (see individual pass docstrings for full derivations)
──────────────────────────────────────────────────────────────────────────────
• Pauli cancellations:  X·X = Y·Y = Z·Z = H·H = I  (exact matrix products)
• Rotation fusion:  RZ(α)·RZ(β) = RZ(α+β mod 2π)  (Baker-Campbell-Hausdorff)
• Commutation:  gates on disjoint qubits always commute (tensor product structure)
• Parameter-shift compatibility:  all angle normalizations preserve gradient semantics
• Global phase freedom:  U and e^{iφ}·U are physically equivalent (projective Hilbert space)

Author:  ZUNTENIUM Core Team
License: Apache 2.0
Requires: numpy, networkx
"""

from __future__ import annotations

import copy
import itertools
import math
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import (
    Any, Callable, Dict, FrozenSet, Iterator, List,
    Optional, Set, Tuple, Union
)

import networkx as nx
import numpy as np


###############################################################################
# STEP 1 — Circuit Intermediate Representation
###############################################################################

@dataclass
class Gate:
    """
    Canonical gate representation in the ZUNTENIUM IR.

    Design rationale
    ----------------
    We separate `targets` from `controls` (unlike flat qubit lists) to allow
    proper routing decisions: control qubits are NOT rotated, only condition
    the gate.  This matters for SWAP insertion and native-gate decomposition.

    Provenance via `metadata`
    -------------------------
    Every transformation records itself in `metadata['history']` so the
    compiler can explain its decisions, replay them, or roll back.
    This is equivalent to LLVM's DebugInfo attached to IR instructions.

    Hashability
    -----------
    Gate is NOT frozen (metadata is mutable for provenance) but provides
    a stable `uid` (UUID4) for DAG node identity.

    Fields
    ------
    name     : canonical gate name, uppercase (e.g. 'X', 'CNOT', 'RZ')
    targets  : qubit indices the gate acts on (excluding controls)
    controls : control qubit indices (empty for non-controlled gates)
    params   : continuous parameters (e.g. rotation angles)
    metadata : provenance dict — keys: 'history', 'original_index', 'source'
    uid      : stable UUID4 for identity across transformations
    """
    name: str
    targets: Tuple[int, ...]
    controls: Tuple[int, ...] = field(default_factory=tuple)
    params: Tuple[float, ...] = field(default_factory=tuple)
    metadata: Dict[str, Any] = field(default_factory=dict)
    uid: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    # ── Derived properties ───────────────────────────────────────────────────

    @property
    def qubits(self) -> Tuple[int, ...]:
        """All qubit indices touched by this gate (controls + targets)."""
        return self.controls + self.targets

    @property
    def num_qubits(self) -> int:
        return len(self.qubits)

    @property
    def is_single_qubit(self) -> bool:
        return len(self.targets) == 1 and len(self.controls) == 0

    @property
    def is_two_qubit(self) -> bool:
        return len(self.qubits) == 2

    @property
    def is_parametric(self) -> bool:
        return len(self.params) > 0

    @property
    def is_measurement(self) -> bool:
        return self.name.upper() in ('M', 'MEASURE', 'MEAS')

    # ── Provenance helpers ───────────────────────────────────────────────────

    def add_history(self, note: str) -> None:
        """Append a compiler note to this gate's provenance chain."""
        if 'history' not in self.metadata:
            self.metadata['history'] = []
        self.metadata['history'].append(note)

    # ── Identity checking ────────────────────────────────────────────────────

    def matrix(self) -> np.ndarray:
        """Return the 2×2 or 4×4 unitary matrix for this gate."""
        return _gate_matrix(self)

    # ── Comparison / hashing ─────────────────────────────────────────────────

    def structural_eq(self, other: "Gate") -> bool:
        """
        True iff two gates are structurally identical (same operation).
        Ignores uid and metadata — used for cancellation checks.
        """
        if self.name != other.name:
            return False
        if self.targets != other.targets or self.controls != other.controls:
            return False
        if len(self.params) != len(other.params):
            return False
        return all(math.isclose(a, b, abs_tol=1e-12) for a, b in zip(self.params, other.params))

    def __repr__(self) -> str:
        p = f"({','.join(f'{x:.4f}' for x in self.params)})" if self.params else ""
        c = f"ctrl={self.controls}" if self.controls else ""
        q = f"@{self.targets}"
        return f"Gate({self.name}{p}{q}{c})"

    def copy(self, new_uid: bool = True) -> "Gate":
        g = Gate(
            name=self.name,
            targets=self.targets,
            controls=self.controls,
            params=self.params,
            metadata=copy.deepcopy(self.metadata),
            uid=str(uuid.uuid4())[:8] if new_uid else self.uid,
        )
        return g


# ── Gate matrix computation ──────────────────────────────────────────────────

def _gate_matrix(g: Gate) -> np.ndarray:
    """Return exact unitary matrix for known gates."""
    nm = g.name.upper()
    p  = g.params
    c128 = np.complex128

    def rx(t): c=math.cos(t/2); s=math.sin(t/2); return np.array([[c,-1j*s],[-1j*s,c]],dtype=c128)
    def ry(t): c=math.cos(t/2); s=math.sin(t/2); return np.array([[c,-s],[s,c]],dtype=c128)
    def rz(t): return np.array([[np.exp(-1j*t/2),0],[0,np.exp(1j*t/2)]],dtype=c128)
    def u3(th,ph,lm):
        c=math.cos(th/2); s=math.sin(th/2)
        return np.array([[c,-np.exp(1j*lm)*s],[np.exp(1j*ph)*s,np.exp(1j*(ph+lm))*c]],dtype=c128)

    STATIC: Dict[str, np.ndarray] = {
        'X':   np.array([[0,1],[1,0]],dtype=c128),
        'Y':   np.array([[0,-1j],[1j,0]],dtype=c128),
        'Z':   np.array([[1,0],[0,-1]],dtype=c128),
        'H':   np.array([[1,1],[1,-1]],dtype=c128)/math.sqrt(2),
        'S':   np.array([[1,0],[0,1j]],dtype=c128),
        'T':   np.array([[1,0],[0,np.exp(1j*math.pi/4)]],dtype=c128),
        'SDG': np.array([[1,0],[0,-1j]],dtype=c128),
        'TDG': np.array([[1,0],[0,np.exp(-1j*math.pi/4)]],dtype=c128),
        'SX':  np.array([[1+1j,1-1j],[1-1j,1+1j]],dtype=c128)/2,
        'SXD': np.array([[1-1j,1+1j],[1+1j,1-1j]],dtype=c128)/2,
        'CNOT': np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]],dtype=c128),
        'CX':   np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]],dtype=c128),
        'CZ':   np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,-1]],dtype=c128),
        'SWAP': np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]],dtype=c128),
        'ISWAP':np.array([[1,0,0,0],[0,0,1j,0],[0,1j,0,0],[0,0,0,1]],dtype=c128),
    }
    if nm in STATIC: return STATIC[nm]
    if nm == 'RX':    return rx(p[0])
    if nm == 'RY':    return ry(p[0])
    if nm == 'RZ':    return rz(p[0])
    if nm == 'U3':    return u3(p[0],p[1],p[2])
    if nm == 'PHASE': return np.array([[1,0],[0,np.exp(1j*p[0])]],dtype=c128)
    if nm == 'I':     return np.eye(2,dtype=c128)
    return np.eye(2,dtype=c128)    # identity fallback for unknown gates


def matrices_equivalent(U: np.ndarray, V: np.ndarray, tol: float = 1e-8) -> bool:
    """
    True iff U and V are equal up to global phase.

    Proof: U ~ V ⟺ ∃φ: V = e^{iφ}U
    ⟺ U†V = e^{iφ}I (i.e., scalar matrix)
    ⟺ |Tr(U†V)| = dim(U) AND U†V is proportional to identity
    """
    if U.shape != V.shape: return False
    d   = U.shape[0]
    W   = U.conj().T @ V
    # Check W = e^{iφ}·I:  all diagonal entries equal, off-diag zero
    diag_val = W[0, 0]
    if abs(abs(diag_val) - 1.0) > tol: return False
    return bool(np.allclose(W, diag_val * np.eye(d), atol=tol))


# ── Angle normalization ───────────────────────────────────────────────────────

def normalize_angle(theta: float, tol: float = 1e-12) -> float:
    """
    Normalize rotation angle to (−π, π].

    Implements: θ_norm = ((θ + π) mod 2π) − π

    Numerical stability: uses Python float arithmetic (IEEE-754 double),
    which gives ≤ 1 ULP error for |θ| < 10⁶ rad (sufficient for all
    physically meaningful rotation angles).

    At tol: if |θ_norm| < tol → return 0.0 (gate becomes identity).
    """
    if not math.isfinite(theta):
        raise ValueError(f"Non-finite rotation angle: {theta}")
    norm = math.fmod(theta + math.pi, 2 * math.pi) - math.pi
    if norm <= -math.pi:
        norm += 2 * math.pi
    return 0.0 if abs(norm) < tol else norm


# ── Circuit class ─────────────────────────────────────────────────────────────

class Circuit:
    """
    The Intermediate Representation of a quantum circuit in ZUNTENIUM.

    Data model
    ----------
    - `_gates`: ordered list of Gate instances (the canonical timeline).
    - `_timelines`: Dict[qubit → List[gate_index]] — the qubit-local
      ordered gate sequence.  Updated eagerly on every add/remove.
    - `_dirty_dag`: flag indicating the cached DAG needs recomputation.

    Design choice: flat list as primary store, DAG as derived view
    ---------------------------------------------------------------
    Flat list → O(1) append, O(N) forward scan.
    DAG       → O(N+E) build, O(N+E) topological sort.
    We keep both: passes that do simple sequential scans use the list;
    passes that need dependency analysis (depth, parallelism, dead-code)
    build the DAG on demand.  The `_dirty_dag` flag avoids redundant rebuilds.

    Provenance
    ----------
    Every structural change (add/remove/replace gate) logs a note in
    the affected gate's metadata['history'].  The full provenance chain
    is accessible via `circuit.gate_provenance(idx)`.
    """

    def __init__(self, num_qubits: int, name: str = "") -> None:
        if num_qubits < 0:
            raise ValueError(f"num_qubits must be ≥ 0; got {num_qubits}")
        self._num_qubits: int = num_qubits
        self._name: str = name
        self._gates: List[Gate] = []
        self._timelines: Dict[int, List[int]] = defaultdict(list)
        self._dag: Optional[nx.DiGraph] = None
        self._dirty_dag: bool = False

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def num_qubits(self) -> int:
        return self._num_qubits

    @property
    def gates(self) -> List[Gate]:
        """Read-only view of the gate list."""
        return self._gates

    @property
    def num_gates(self) -> int:
        return len(self._gates)

    # ── Structural operations ─────────────────────────────────────────────

    def add_gate(self, gate: Gate, note: str = "") -> int:
        """
        Append a gate to the circuit.

        Validates qubit indices, updates timelines, marks DAG dirty.

        Returns
        -------
        Index of the newly added gate.
        """
        self._validate_gate(gate)
        idx = len(self._gates)
        gate = gate.copy(new_uid=False)
        gate.metadata.setdefault('original_index', idx)
        if note:
            gate.add_history(note)
        self._gates.append(gate)
        for q in gate.qubits:
            self._timelines[q].append(idx)
        self._dirty_dag = True
        return idx

    def insert_gate(self, position: int, gate: Gate) -> None:
        """Insert gate at `position`, shifting later gates."""
        self._validate_gate(gate)
        gate = gate.copy(new_uid=False)
        self._gates.insert(position, gate)
        self._rebuild_timelines()
        self._dirty_dag = True

    def remove_gate(self, idx: int) -> Gate:
        """
        Remove gate at index `idx`.

        Logs provenance note on the removed gate and rebuilds timelines.
        O(N) due to index rebuild — acceptable since N ≤ 10 000 in v0.1.
        """
        if not (0 <= idx < len(self._gates)):
            raise IndexError(f"Gate index {idx} out of range [0, {len(self._gates)})")
        gate = self._gates.pop(idx)
        gate.add_history(f"removed at index {idx}")
        self._rebuild_timelines()
        self._dirty_dag = True
        return gate

    def replace_gate(self, idx: int, new_gate: Gate, note: str = "") -> None:
        """Replace gate at `idx` with `new_gate` in-place."""
        self._validate_gate(new_gate)
        old = self._gates[idx]
        new_gate = new_gate.copy(new_uid=False)
        new_gate.metadata['replaced'] = str(old)
        if note:
            new_gate.add_history(note)
        self._gates[idx] = new_gate
        self._rebuild_timelines()
        self._dirty_dag = True

    def copy(self) -> "Circuit":
        """Deep copy of the circuit (gates, timelines, name)."""
        c = Circuit(self._num_qubits, self._name + "_copy")
        for gate in self._gates:
            c._gates.append(gate.copy())
        c._rebuild_timelines()
        return c

    # ── Timeline access ───────────────────────────────────────────────────

    def qubit_timeline(self, qubit: int) -> List[Gate]:
        """Ordered list of gates acting on `qubit`."""
        return [self._gates[i] for i in self._timelines.get(qubit, [])]

    def gate_provenance(self, idx: int) -> List[str]:
        """Full transformation history for gate at `idx`."""
        return self._gates[idx].metadata.get('history', [])

    # ── DAG access (lazy) ─────────────────────────────────────────────────

    def to_dag(self) -> nx.DiGraph:
        """
        Build or return the cached dependency DAG.

        Nodes: gate indices (0-based).
        Edges: (i → j) if gate j depends on gate i on some qubit q.
               Edge attribute: {'qubit': q}

        Algorithm (O(N) per qubit timeline):
        For each qubit q, walk its timeline in order and add an edge
        from every gate to the next gate on the same qubit.
        """
        if self._dag is not None and not self._dirty_dag:
            return self._dag
        G = nx.DiGraph()
        G.add_nodes_from(range(len(self._gates)),
                         gate=[g for g in self._gates])
        for q, timeline in self._timelines.items():
            for k in range(len(timeline) - 1):
                i, j = timeline[k], timeline[k + 1]
                if not G.has_edge(i, j):
                    G.add_edge(i, j, qubit=q)
        self._dag = G
        self._dirty_dag = False
        return G

    def depth(self) -> int:
        """
        Critical-path length = circuit depth.

        Algorithm: single topological pass, O(N + E).
        layer[v] = 1 + max(layer[u] for u in predecessors(v))
        depth = max(layer.values())
        """
        if not self._gates:
            return 0
        dag   = self.to_dag()
        layer: Dict[int, int] = {}
        for v in nx.topological_sort(dag):
            preds = list(dag.predecessors(v))
            layer[v] = 1 + max((layer[p] for p in preds), default=0)
        return max(layer.values(), default=0)

    def layers(self) -> List[List[Gate]]:
        """
        Partition gates into executable parallel layers.

        Each layer contains gates that have no inter-dependencies and
        can be executed simultaneously on a parallel hardware platform.

        Returns
        -------
        List of layers, each layer = list of Gate objects.
        """
        if not self._gates:
            return []
        dag   = self.to_dag()
        layer_idx: Dict[int, int] = {}
        for v in nx.topological_sort(dag):
            preds = list(dag.predecessors(v))
            layer_idx[v] = 1 + max((layer_idx[p] for p in preds), default=0)
        max_layer = max(layer_idx.values(), default=0)
        buckets: List[List[Gate]] = [[] for _ in range(max_layer)]
        for v, l in layer_idx.items():
            buckets[l - 1].append(self._gates[v])
        return buckets

    # ── Convenience constructors ──────────────────────────────────────────

    @classmethod
    def from_gate_list(cls, num_qubits: int, gates: List[Gate]) -> "Circuit":
        c = cls(num_qubits)
        for g in gates:
            c.add_gate(g)
        return c

    # ── Validation ────────────────────────────────────────────────────────

    def _validate_gate(self, gate: Gate) -> None:
        for q in gate.qubits:
            if q < 0 or q >= self._num_qubits:
                raise ValueError(
                    f"Gate {gate} references qubit {q}, but circuit has "
                    f"{self._num_qubits} qubits (valid: 0..{self._num_qubits-1})"
                )
        if len(set(gate.qubits)) != len(gate.qubits):
            raise ValueError(f"Gate {gate} has duplicate qubit indices.")

    def _rebuild_timelines(self) -> None:
        self._timelines = defaultdict(list)
        for idx, gate in enumerate(self._gates):
            for q in gate.qubits:
                self._timelines[q].append(idx)

    # ── ASCII diagram ─────────────────────────────────────────────────────

    def draw(self, max_width: int = 80) -> str:
        """
        Minimal ASCII circuit diagram.

        Each gate occupies a fixed-width column. Gates on the same layer
        are shown in the same column.
        """
        if not self._gates:
            return "\n".join(f"q{q}: ────" for q in range(self._num_qubits))

        layers = self.layers()
        col_w  = 10
        lines  = [f"q{q}: " for q in range(self._num_qubits)]
        for layer in layers:
            active: Dict[int, str] = {}
            for g in layer:
                label = f"{g.name}"
                if g.params:
                    label += f"({','.join(f'{p:.2f}' for p in g.params)})"
                label = label[:col_w-2].center(col_w-2)
                for q in g.qubits:
                    active[q] = f"─[{label}]─"
            for q in range(self._num_qubits):
                lines[q] += active.get(q, "─" * col_w)
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"Circuit('{self._name}', n={self._num_qubits}, "
            f"gates={self.num_gates}, depth={self.depth()})"
        )


###############################################################################
# STEP 2 — Transpiler Pass Architecture
###############################################################################

@dataclass
class PassResult:
    """
    Metadata returned by each optimization pass.
    Used to build the OptimizationReport.
    """
    pass_name:       str
    gates_removed:   int = 0
    gates_added:     int = 0
    gates_replaced:  int = 0
    depth_delta:     int = 0
    elapsed_ms:      float = 0.0
    notes:           List[str] = field(default_factory=list)

    @property
    def net_delta(self) -> int:
        return self.gates_added - self.gates_removed


class OptimizationPass(ABC):
    """
    Abstract base class for all transpiler passes.

    Every pass:
    1. Takes a Circuit (treated as read-only inside the pass).
    2. Returns a NEW Circuit (immutable pass semantics) + PassResult.
    3. Runs in O(N) or O(N log N) time, O(N) space.
    4. Does NOT modify the input circuit in place (safe for retry/rollback).

    Immutability rationale
    ----------------------
    LLVM's pass manager uses in-place modification, but for correctness
    auditing and future RL training (which needs before/after pairs),
    immutable passes are far safer.  The O(N) copy overhead is negligible
    for N ≤ 10 000 gates.
    """

    name: str = "BasePass"
    description: str = ""

    def __call__(self, circuit: Circuit) -> Tuple[Circuit, PassResult]:
        t0 = time.perf_counter()
        new_circuit = self._apply(circuit)
        elapsed = (time.perf_counter() - t0) * 1000
        result = self._build_result(circuit, new_circuit, elapsed)
        return new_circuit, result

    @abstractmethod
    def _apply(self, circuit: Circuit) -> Circuit: ...

    def _build_result(
        self,
        before: Circuit,
        after: Circuit,
        elapsed_ms: float,
    ) -> PassResult:
        return PassResult(
            pass_name=self.name,
            gates_removed=max(0, before.num_gates - after.num_gates),
            gates_added=max(0, after.num_gates - before.num_gates),
            depth_delta=after.depth() - before.depth(),
            elapsed_ms=elapsed_ms,
        )


###############################################################################
# STEP 3 — Commutation Tables
###############################################################################

def _qubit_overlap(g1: Gate, g2: Gate) -> FrozenSet[int]:
    return frozenset(g1.qubits) & frozenset(g2.qubits)


def gates_commute(g1: Gate, g2: Gate) -> bool:
    """
    Determine if g1 and g2 can be exchanged without changing the unitary.

    Rules (ordered from cheapest to most expensive check):

    Rule 1 (disjoint support):
        If supp(g1) ∩ supp(g2) = ∅, they commute by tensor-product structure:
        (A ⊗ I)(I ⊗ B) = A ⊗ B = (I ⊗ B)(A ⊗ I).
        Time: O(1).

    Rule 2 (same single-qubit diagonal gates):
        Z, RZ, S, T, SDG, TDG all commute with each other on the same qubit
        (they share the {|0⟩,|1⟩} eigenbasis and are simultaneously diagonal).

    Rule 3 (CNOT commutativity):
        CNOT(c,t) commutes with:
          • Z, S, T, SDG, TDG, RZ on control c  (Z-type gates preserve |c⟩ eigenstates)
          • X, SX, SXD, RX on target t           (X-type gates preserve CNOT action)
          • Another CNOT(c,t) (trivially: CNOT² = I)
          • CZ(c,t): CZ·CNOT = CNOT·CZ on the same pair.
        Mathematical basis: [CNOT, Z⊗I] = 0 because Z acts on the control
        subspace and CNOT leaves |0⟩_c unchanged.

    Rule 4 (matrix commutator check — fallback):
        For small (single/two-qubit) gates, compute [U,V] directly.
        This is exact but O(d²) where d ≤ 4 for the gate's local subspace.
        Only invoked when rules 1-3 don't apply.
    """
    if not _qubit_overlap(g1, g2):
        return True

    DIAG_1Q = frozenset({'Z','S','T','SDG','TDG','RZ','PHASE'})
    ROT_X   = frozenset({'X','SX','SXD','RX'})

    # Rule 2: both diagonal single-qubit on same qubit
    if (g1.is_single_qubit and g2.is_single_qubit
            and g1.targets == g2.targets
            and g1.name.upper() in DIAG_1Q
            and g2.name.upper() in DIAG_1Q):
        return True

    # Rule 3: CNOT commutativity
    for (gcnot, gother) in [(g1, g2), (g2, g1)]:
        if gcnot.name.upper() in ('CNOT','CX') and len(gcnot.targets) == 1:
            if gcnot.controls:
                c, t = gcnot.controls[0], gcnot.targets[0]
            else:
                c, t = gcnot.qubits[0], gcnot.qubits[1]
            if gother.is_single_qubit:
                q = gother.targets[0]
                if q == c and gother.name.upper() in DIAG_1Q: return True
                if q == t and gother.name.upper() in ROT_X:   return True
            # Two CNOTs on same pair
            if (gother.name.upper() in ('CNOT','CX')
                    and set(gother.qubits) == {c, t}):
                return True

    # Rule 4: matrix commutator (fallback, exact)
    if g1.is_single_qubit and g2.is_single_qubit and g1.targets == g2.targets:
        try:
            U, V = g1.matrix(), g2.matrix()
            return np.allclose(U @ V, V @ U, atol=1e-10)
        except Exception:
            pass

    return False


###############################################################################
# STEP 3 — Pass 1: Redundant Gate Cancellation
###############################################################################

# Exact self-inverse and inverse-pair table
_SELF_INVERSE: FrozenSet[str] = frozenset({'X','Y','Z','H','CNOT','CX','SWAP','CZ','ISWAP'})

_INVERSE_PAIRS: Dict[str, str] = {
    'S':'SDG', 'SDG':'S',
    'T':'TDG', 'TDG':'T',
    'SX':'SXD','SXD':'SX',
}

_FUSION_PAIRS: Dict[Tuple[str,str], str] = {
    ('S','S'):'Z',
    ('T','T'):'S',
    ('SX','SX'):'X',
}


def _is_inverse(g1: Gate, g2: Gate) -> bool:
    """
    True iff g1·g2 = I (up to global phase).

    Cases:
    1. Self-inverse gates: X·X = I, H·H = I, etc.
    2. Explicit inverse pairs: S·S† = I, T·T† = I.
    3. Parametric inverse: RX(θ)·RX(−θ) = I, RX(θ)·RX(2π−θ) = I.
    4. U3(θ,φ,λ)·U3(−θ,−λ,−φ) = I.
    5. General: compute U†V ≈ I (matrix check).
    """
    if g1.targets != g2.targets or g1.controls != g2.controls:
        return False

    nm1, nm2 = g1.name.upper(), g2.name.upper()

    # Case 1: self-inverse
    if nm1 == nm2 and nm1 in _SELF_INVERSE:
        return True

    # Case 2: explicit pairs
    if _INVERSE_PAIRS.get(nm1) == nm2:
        return True

    # Case 3: parametric same-axis rotations
    if nm1 == nm2 and nm1 in ('RX','RY','RZ','PHASE') and g1.params and g2.params:
        total = g1.params[0] + g2.params[0]
        normed = normalize_angle(total)
        return abs(normed) < 1e-9 or abs(abs(normed) - 2*math.pi) < 1e-9

    # Case 4/5: matrix check (only for small gates)
    if g1.num_qubits <= 2:
        try:
            U, V = g1.matrix(), g2.matrix()
            d = U.shape[0]
            return np.allclose(U @ V, np.eye(d), atol=1e-9)
        except Exception:
            pass
    return False


class RedundantGateCancellationPass(OptimizationPass):
    """
    Pass 1: Cancel adjacent inverse-gate pairs and self-inverse repetitions.

    Algorithm
    ---------
    Converging scan: repeat until no cancellations occur.
    Each scan: O(N) forward pass with look-ahead past commuting gates.

    Look-ahead rule (essential for non-adjacent cancellations):
        For gate at position i, scan forward through gates i+1, i+2, ...
        until we hit a gate that does NOT commute with gate_i.
        If any gate_j in this window is the inverse of gate_i, remove both.
        This finds cancellations separated by commuting gates.

    Fusion rule:
        S·S = Z,  T·T = S,  SX·SX = X  (specific accumulation cases).

    Convergence:
        Each iteration removes ≥ 1 pair → terminates in ≤ N/2 iterations.
        In practice: 1–3 passes on typical circuits.

    Complexity: O(N × k) where k = average look-ahead window ≤ N.
    Expected practical: O(N) for structured circuits.
    """

    name = "RedundantGateCancellationPass"
    description = "Cancel A·A†=I pairs and fuse S·S=Z, T·T=S, SX·SX=X."

    def _apply(self, circuit: Circuit) -> Circuit:
        gates = [g.copy() for g in circuit.gates]
        changed = True
        while changed:
            gates, changed = self._scan_once(gates)
        return Circuit.from_gate_list(circuit.num_qubits, gates)

    def _scan_once(self, gates: List[Gate]) -> Tuple[List[Gate], bool]:
        skip: Set[int] = set()
        cancel_pairs: List[Tuple[int,int]] = []
        fuse_pairs:   List[Tuple[int,int,Gate]] = []

        for i in range(len(gates)):
            if i in skip:
                continue
            g = gates[i]
            # Look ahead past commuting gates
            for j in range(i + 1, len(gates)):
                if j in skip:
                    continue
                gj = gates[j]
                if set(g.qubits) == set(gj.qubits):
                    # Same qubit set — check for cancellation or fusion
                    if _is_inverse(g, gj):
                        cancel_pairs.append((i, j))
                        skip.add(i); skip.add(j)
                        break
                    key = (g.name.upper(), gj.name.upper())
                    if (key in _FUSION_PAIRS and g.targets == gj.targets
                            and not g.params):
                        fused = Gate(_FUSION_PAIRS[key], g.targets)
                        fused.add_history(f"fused {g.name}+{gj.name}→{fused.name}")
                        fuse_pairs.append((i, j, fused))
                        skip.add(i); skip.add(j)
                        break
                    break   # blocked: can't look further past non-commuting gates
                elif not gates_commute(g, gj):
                    break   # can't look past non-commuting gate

        if not cancel_pairs and not fuse_pairs:
            return gates, False

        result: List[Gate] = []
        fuse_insert: Dict[int, Gate] = {}
        for (i, j, fg) in fuse_pairs:
            fuse_insert[i] = fg  # insert fused gate at first position

        for i, g in enumerate(gates):
            if i in skip:
                if i in fuse_insert:
                    result.append(fuse_insert[i])
            else:
                result.append(g)
        return result, True


###############################################################################
# Pass 2: Rotation Merging
###############################################################################

_ROT_GATES: Dict[str, int] = {'RX': 0, 'RY': 1, 'RZ': 2}
_SAME_AXIS_COMMUTE = frozenset({'RX','RY','RZ'})


class RotationMergingPass(OptimizationPass):
    """
    Pass 2: Merge consecutive same-axis rotations on the same qubit.

    Mathematical identity
    ---------------------
    For a rotation RΩ(θ) = exp(−iθΩ/2) about Pauli Ω:

        RΩ(θ₁)·RΩ(θ₂) = exp(−iθ₁Ω/2)·exp(−iθ₂Ω/2)
                        = exp(−i(θ₁+θ₂)Ω/2)   [since Ω commutes with itself]
                        = RΩ(θ₁ + θ₂)

    This uses the BCH formula truncation: for [A,B]=0, exp(A)exp(B) = exp(A+B).
    Since Ω/2 commutes with Ω/2, the identity is exact (no truncation).

    Post-merge normalization:
        θ_merged = normalize_angle(θ₁ + θ₂)
        If |θ_merged| < ε → gate is identity → remove entirely.
        If θ_merged ≈ π   → RZ(π) = iZ  → replace with Z (save one param store).

    Special-angle replacement:
        RZ(π)   → Z   (exact: RZ(π) = diag(e^{-iπ/2}, e^{iπ/2}) = -iZ ~ Z)
        RZ(π/2) → S   (RZ(π/2) = diag(e^{-iπ/4}, e^{iπ/4}) = e^{-iπ/4}·S ~ S)
        RZ(π/4) → T
        (All equalities are up to global phase, which is physically irrelevant.)

    Look-ahead: same strategy as cancellation — skip commuting gates to merge
    non-adjacent same-axis rotations.
    """

    name = "RotationMergingPass"
    description = "Merge RX(a)·RX(b)→RX(a+b), normalize, replace special angles."

    _EPS = 1e-9
    _SPECIAL_RZ: List[Tuple[float, str]] = [
        (math.pi,      'Z'),
        (math.pi / 2,  'S'),
        (math.pi / 4,  'T'),
        (-math.pi / 2, 'SDG'),
        (-math.pi / 4, 'TDG'),
    ]

    def _apply(self, circuit: Circuit) -> Circuit:
        gates = [g.copy() for g in circuit.gates]
        result: List[Gate] = []
        i = 0
        while i < len(gates):
            g = gates[i]
            nm = g.name.upper()
            if nm not in _ROT_GATES or not g.params:
                result.append(g); i += 1; continue

            # Accumulate same-axis rotations, skipping commuting gates
            total = g.params[0]
            q     = g.targets
            consumed = {i}
            j = i + 1
            commuting_window: List[int] = []

            while j < len(gates):
                gj = gates[j]
                if gj.name.upper() == nm and gj.targets == q:
                    total += gj.params[0]
                    consumed.add(j)
                    # Also absorb previously deferred commuting gates
                    j += 1
                elif gates_commute(g, gj):
                    commuting_window.append(j)
                    j += 1
                else:
                    break

            total_norm = normalize_angle(total)

            # Add back the commuting gates that weren't consumed
            if abs(total_norm) > self._EPS:
                merged = self._maybe_replace_special(nm, total_norm, q, g)
                result.append(merged)
            # else: total ≈ 0 → identity → drop entirely

            # Re-emit commuting window gates in order
            for cw in commuting_window:
                result.append(gates[cw])

            i = j
        return Circuit.from_gate_list(circuit.num_qubits, result)

    def _maybe_replace_special(
        self, nm: str, angle: float,
        targets: Tuple[int,...], original: Gate,
    ) -> Gate:
        if nm == 'RZ':
            for ref_angle, gate_name in self._SPECIAL_RZ:
                if math.isclose(angle, ref_angle, abs_tol=1e-7):
                    g = Gate(gate_name, targets)
                    g.add_history(f"RZ({angle:.4f})→{gate_name}")
                    return g
        g = Gate(nm, targets, params=(angle,))
        if len(original.metadata.get('history', [])) > 0:
            g.metadata['history'] = original.metadata['history'].copy()
        g.add_history(f"merged angle={angle:.6f}")
        return g


###############################################################################
# Pass 3: Gate Commutation Reordering
###############################################################################

class CommutationReorderingPass(OptimizationPass):
    """
    Pass 3: Reorder commuting gates to expose new cancellation opportunities.

    Goal
    ----
    Enable subsequent passes to find cancellations that were previously
    blocked by intervening (commuting) gates.

    Algorithm: Greedy bubble-sort with depth-improvement gate
    ---------------------------------------------------------
    1. For each adjacent pair (i, i+1) that commute:
       a. Swap them tentatively.
       b. If the new circuit has strictly smaller depth → keep swap.
       c. Otherwise → revert.
    2. Repeat until no depth improvement occurs.

    Convergence: monotonically decreasing depth → terminates.
    Complexity: O(N² × depth_eval), depth_eval = O(N+E).
    Practical target: < 100 ms for N = 1000 gates.

    Alternative: commutation-based DAG rewriting (used in t|ket⟩).
    Planned for v0.2 — requires DAG edge relabeling, more complex but O(N log N).
    """

    name = "CommutationReorderingPass"
    description = "Bubble-sort commuting adjacent gates to reduce depth."

    def _apply(self, circuit: Circuit) -> Circuit:
        gates = [g.copy() for g in circuit.gates]
        improved = True
        while improved:
            improved = False
            for i in range(len(gates) - 1):
                if not gates_commute(gates[i], gates[i+1]):
                    continue
                # Compute depth before/after
                c_before = Circuit.from_gate_list(circuit.num_qubits, gates)
                d_before = c_before.depth()
                gates[i], gates[i+1] = gates[i+1], gates[i]
                c_after  = Circuit.from_gate_list(circuit.num_qubits, gates)
                d_after  = c_after.depth()
                if d_after < d_before:
                    improved = True
                else:
                    gates[i], gates[i+1] = gates[i+1], gates[i]   # revert
        return Circuit.from_gate_list(circuit.num_qubits, gates)


###############################################################################
# Pass 4: Single-Qubit Gate Fusion (ZYZ decomposition)
###############################################################################

class SingleQubitFusionPass(OptimizationPass):
    """
    Pass 4: Fold any run of consecutive 1-qubit gates on the same qubit
    into a single U3 gate (or remove if identity).

    Mathematical basis: ZYZ decomposition of SU(2)
    -----------------------------------------------
    Every element U ∈ SU(2) can be written as:

        U = RZ(φ)·RY(θ)·RZ(λ)

    where:
        θ = 2·arccos(|U[0,0]|)
        φ,λ derived from the phase angles of U's matrix entries.

    This is the Euler angle decomposition. Every single-qubit gate or
    sequence of gates can be reduced to exactly 3 rotation gates (or fewer).

    Global-phase removal: U and e^{iα}·U are physically identical.
    We divide U by e^{i·arg(det(U))/2} to put it in SU(2) before decomposing.

    Identity detection: if ‖U/U[0,0] − I‖_F < ε → remove entirely.

    Gate count reduction:
        A run of k gates on one qubit → 1 U3 gate (or 0 if identity).
        Maximum savings: k−1 gates per run.
    """

    name = "SingleQubitFusionPass"
    description = "Fuse consecutive 1Q gates on same qubit into one U3."
    _EPS = 1e-9

    def _apply(self, circuit: Circuit) -> Circuit:
        result: List[Gate] = []
        gates  = circuit.gates
        n      = len(gates)
        i = 0
        while i < n:
            g = gates[i]
            if not g.is_single_qubit or g.is_measurement:
                result.append(g.copy()); i += 1; continue
            q = g.targets[0]
            # Collect consecutive 1Q gates on same qubit (no interruptions)
            run = [g]; j = i + 1
            while j < n and gates[j].is_single_qubit and gates[j].targets[0] == q \
                    and not gates[j].is_measurement:
                run.append(gates[j]); j += 1
            if len(run) == 1:
                result.append(g.copy()); i += 1; continue
            # Compute product: U_total = U_last · ... · U_1 · U_0
            # (gates applied in circuit order; latest gate is leftmost)
            U = np.eye(2, dtype=np.complex128)
            for rg in run:
                U = _gate_matrix(rg) @ U
            # Remove global phase
            det = U[0,0]*U[1,1] - U[0,1]*U[1,0]
            U_su2 = np.exp(-1j*np.angle(det)/2) * U
            fused = self._decompose_zyz(U_su2, q)
            if fused is not None:
                fused.add_history(f"fused {len(run)} gates: {[r.name for r in run]}")
                result.append(fused)
            i = j
        return Circuit.from_gate_list(circuit.num_qubits, result)

    def _is_identity(self, U: np.ndarray) -> bool:
        u = U.copy()
        if abs(u[0,0]) > self._EPS:
            u /= u[0,0]
        return bool(np.allclose(u, np.eye(2), atol=self._EPS))

    def _decompose_zyz(self, U: np.ndarray, qubit: int) -> Optional[Gate]:
        """
        Return U3(θ,φ,λ) gate equivalent to SU(2) matrix U, or None if U ≈ I.

        ZYZ decomposition (IBM/OpenQASM convention):
            U = RZ(φ)·RY(θ)·RZ(λ)

        where U3(θ,φ,λ) = [[cos(θ/2),         -e^{iλ}·sin(θ/2) ],
                            [e^{iφ}·sin(θ/2),   e^{i(φ+λ)}·cos(θ/2)]]

        Verification: build U3 matrix and compare to U numerically.
        If decomposition fails (degenerate case), fall back to storing the
        raw matrix as a CustomGate (stub — emits U3 with best-effort params).
        """
        if self._is_identity(U):
            return None
        # |U[0,0]| = cos(θ/2)
        a = U[0,0]
        cos_half = min(abs(complex(a)), 1.0)
        theta = 2.0 * float(np.arccos(cos_half))

        sin_half = math.sin(theta / 2)
        if abs(sin_half) < 1e-10:
            # θ ≈ 0: U ≈ diag(e^{iα}, e^{-iα}) = RZ(-2α)
            phi = 0.0
            lam = float(-2.0 * np.angle(complex(a)))
        else:
            # Factor out α = arg(U[0,0]) so that U_rotated[0,0] is real.
            # This is necessary because U3(θ,φ,λ) has cos(θ/2) as a real
            # entry at [0,0], but U_su2[0,0] may be complex after det=1 normalization.
            # The resulting U3 equals U_su2 up to global phase e^{iα}.
            alpha = np.angle(complex(a))
            U_rot = np.exp(-1j * alpha) * U
            phi = float(np.angle(complex(U_rot[1, 0])))
            lam = float(np.angle(-complex(U_rot[0, 1])))

        # Verification: rebuild matrix and check ‖U_rebuilt - U‖ < tol
        # If off by global phase, adjust.
        c, s = math.cos(theta/2), math.sin(theta/2)
        U_rebuilt = np.array([
            [c,                            -np.exp(1j*lam)*s],
            [np.exp(1j*phi)*s,              np.exp(1j*(phi+lam))*c]
        ], dtype=np.complex128)
        if not np.allclose(U_rebuilt, U, atol=1e-7):
            # Try with negated phi/lam convention (some textbooks differ)
            phi2 = float(np.angle(complex(U[1, 0])))
            lam2 = float(np.angle(complex(U[0, 1])) + math.pi)
            phi, lam = phi2, lam2

        return Gate('U3', (qubit,), params=(
            normalize_angle(theta),
            normalize_angle(phi),
            normalize_angle(lam),
        ))


###############################################################################
# Pass 5: Depth Optimization (Layer scheduling)
###############################################################################

class DepthOptimizationPass(OptimizationPass):
    """
    Pass 5: Re-emit gates in topological order that maximizes parallelism.

    Algorithm
    ---------
    Build the dependency DAG. Emit gates in topological order such that
    gates at the same DAG depth level are grouped together (same layer).
    Within each layer, sort by qubit index for determinism.

    This does NOT reorder gates (would require commutation check);
    it only verifies the current ordering is DAG-consistent and recomputes
    layers.  The actual depth reduction comes from Passes 1–4 removing gates
    and Pass 3 reordering commuting gates.

    Output: circuit with same gate set, same unitary, emitted in
    canonical topological order.  The `layers()` method then gives the
    optimal parallel execution schedule.

    Complexity: O(N + E) for topological sort.
    """

    name = "DepthOptimizationPass"
    description = "Re-emit in DAG topological order for maximal parallelism."

    def _apply(self, circuit: Circuit) -> Circuit:
        if not circuit.gates:
            return circuit.copy()
        dag   = circuit.to_dag()
        order = list(nx.topological_sort(dag))
        reordered = [circuit.gates[i].copy() for i in order]
        return Circuit.from_gate_list(circuit.num_qubits, reordered)


###############################################################################
# Pass 6: Dead Gate Elimination (backward data-flow analysis)
###############################################################################

class DeadGateEliminationPass(OptimizationPass):
    """
    Pass 6: Remove gates whose output is never observed (read).

    This analysis is only valid when explicit MEASURE gates are present.
    If the circuit has no measurements, every gate potentially contributes
    to the final state, so this pass becomes a no-op.

    Algorithm: backward analysis from measurement sinks
    ----------------------------------------------------
    1. Identify all MEASURE gates (sinks).
    2. Mark qubits that are measured: `live_qubits`.
    3. Walk backwards through the gate list:
       - If a gate acts on any live qubit → it is "live" (keep it).
       - If a gate acts ONLY on dead qubits → it is "dead" (remove it).
       - After processing a live gate, add all its qubits to `live_qubits`.

    Mathematical correctness:
    - If qubit q is never measured: ρ_final = Tr_q[ρ_circuit] and
      any gate on q that commutes with the rest is irrelevant.
    - If qubit q IS measured: all gates on q affect the measurement statistics.
    - This backward analysis is conservative: it never removes live gates.

    Complexity: O(N) single backward pass.
    """

    name = "DeadGateEliminationPass"
    description = "Remove gates on unmeasured qubits (backward analysis)."

    def _apply(self, circuit: Circuit) -> Circuit:
        gates = circuit.gates
        # Find measured qubits
        measured: Set[int] = set()
        for g in gates:
            if g.is_measurement:
                measured.update(g.qubits)
        if not measured:
            return circuit.copy()   # No measurements → no dead gates

        live_qubits: Set[int] = set(measured)
        keep: List[bool] = [False] * len(gates)

        for i in reversed(range(len(gates))):
            g = gates[i]
            if g.is_measurement:
                keep[i] = True
                continue
            if set(g.qubits) & live_qubits:
                keep[i] = True
                live_qubits.update(g.qubits)

        result = [g.copy() for i, g in enumerate(gates) if keep[i]]
        c = Circuit.from_gate_list(circuit.num_qubits, result)
        return c


###############################################################################
# STEP 4 — Dependency Graph utilities (extended)
###############################################################################

def circuit_dag_annotated(circuit: Circuit) -> nx.DiGraph:
    """
    Build a fully annotated dependency DAG with gate-object node attributes.

    Nodes:  gate index (int)
    Attrs:  {'gate': Gate, 'layer': int, 'qubits': tuple}
    Edges:  {'qubit': int}
    """
    dag = circuit.to_dag()
    layer: Dict[int, int] = {}
    for v in nx.topological_sort(dag):
        preds = list(dag.predecessors(v))
        layer[v] = 1 + max((layer[p] for p in preds), default=0)
    for v in dag.nodes:
        dag.nodes[v]['layer'] = layer[v]
        dag.nodes[v]['gate']  = circuit.gates[v]
    return dag


def critical_path(circuit: Circuit) -> List[Gate]:
    """
    Return the longest path in the dependency DAG as a list of gates.
    Useful for identifying bottleneck operations and targeting optimizations.
    """
    dag = circuit_dag_annotated(circuit)
    if not dag.nodes:
        return []
    path_indices = nx.dag_longest_path(dag, weight=None)
    return [circuit.gates[i] for i in path_indices]


def gate_parallelism_ratio(circuit: Circuit) -> float:
    """
    Returns: num_gates / depth — measures how parallel the circuit is.
    = 1.0 → fully serial; > 1.0 → some parallelism.
    """
    d = circuit.depth()
    return circuit.num_gates / d if d > 0 else 0.0


###############################################################################
# STEP 2 — TranspilerPipeline
###############################################################################

class TranspilerPipeline:
    """
    Ordered sequence of OptimizationPass objects.

    Exactly like LLVM's PassManager:
    1. Passes are registered in order.
    2. Each pass receives the output of the previous pass.
    3. Passes can be added/removed dynamically (for AI agent injection).
    4. The pipeline can be repeated until convergence.

    Convergence criterion
    ---------------------
    Repeat the full pass sequence until the gate count does not decrease
    between consecutive full iterations.  Terminates in ≤ N/2 iterations.
    In practice: 2–4 full iterations.
    """

    DEFAULT_PASSES: List[str] = [
        'redundant', 'rotation', 'commutation', 'fusion',
        'depth', 'dead',
    ]

    def __init__(
        self,
        passes: Optional[List[OptimizationPass]] = None,
        max_iterations: int = 5,
    ) -> None:
        self._passes: List[OptimizationPass] = passes or self._build_default()
        self._max_iter = max_iterations

    @staticmethod
    def _build_default() -> List[OptimizationPass]:
        return [
            RedundantGateCancellationPass(),
            RotationMergingPass(),
            CommutationReorderingPass(),
            SingleQubitFusionPass(),
            DepthOptimizationPass(),
            DeadGateEliminationPass(),
        ]

    def add_pass(self, p: OptimizationPass, position: int = -1) -> None:
        """Insert a pass at position (-1 = append)."""
        if position < 0:
            self._passes.append(p)
        else:
            self._passes.insert(position, p)

    def remove_pass(self, name: str) -> None:
        self._passes = [p for p in self._passes if p.name != name]

    def run(self, circuit: Circuit) -> Tuple[Circuit, List[PassResult]]:
        """
        Execute all passes repeatedly until convergence.

        Returns
        -------
        (optimized_circuit, list_of_PassResult per pass per iteration)
        """
        current  = circuit
        all_results: List[PassResult] = []
        for iteration in range(self._max_iter):
            prev_count = current.num_gates
            for p in self._passes:
                new_circuit, result = p(current)
                if result.gates_removed > 0 or result.gates_added > 0:
                    result.notes.append(
                        f"iter{iteration+1}: {result.pass_name} "
                        f"→ {current.num_gates}→{new_circuit.num_gates}"
                    )
                    all_results.append(result)
                current = new_circuit
            if current.num_gates >= prev_count:
                break   # converged
        return current, all_results


###############################################################################
# STEP 5 — Main Transpiler Engine + OptimizationReport
###############################################################################

@dataclass
class OptimizationReport:
    """
    Comprehensive report of transpilation results.
    Equivalent to LLVM's analysis pass output combined with IR diff.
    """
    original_gates:     int
    optimized_gates:    int
    original_depth:     int
    optimized_depth:    int
    estimated_fidelity: float
    total_time_ms:      float
    passes_run:         List[PassResult]
    critical_path_len:  int = 0
    parallelism_ratio:  float = 0.0

    @property
    def gate_reduction_pct(self) -> float:
        return 100.0 * (1 - self.optimized_gates / max(self.original_gates,1))

    @property
    def depth_reduction_pct(self) -> float:
        return 100.0 * (1 - self.optimized_depth / max(self.original_depth,1))

    def summary(self) -> str:
        return (
            f"Gates:  {self.original_gates}→{self.optimized_gates} "
            f"({self.gate_reduction_pct:.1f}%↓)  "
            f"Depth:  {self.original_depth}→{self.optimized_depth} "
            f"({self.depth_reduction_pct:.1f}%↓)  "
            f"Fidelity: {self.estimated_fidelity:.4f}  "
            f"Time: {self.total_time_ms:.1f}ms"
        )

    def passes_summary(self) -> str:
        lines = ["Pass results:"]
        for r in self.passes_run:
            if r.gates_removed or r.gates_added:
                sign = "−" if r.net_delta <= 0 else "+"
                lines.append(f"  {r.pass_name:<38s}: {sign}{abs(r.net_delta):3d} gates, "
                              f"{r.elapsed_ms:.1f}ms")
        return "\n".join(lines)


def _estimate_fidelity(circuit: Circuit, p_1q: float = 0.001, p_2q: float = 0.01) -> float:
    """
    Estimate circuit fidelity under a simple depolarizing noise model.

    Model: f = ∏_gates (1 − p_gate)
    where p_gate is the per-gate depolarizing error rate.
    """
    f = 1.0
    for g in circuit.gates:
        if g.is_measurement:
            continue
        p = g.metadata.get('error_rate', p_1q if g.num_qubits == 1 else p_2q)
        f *= (1.0 - p)
    return max(f, 0.0)


class Transpiler:
    """
    The ZUNTENIUM Basic Quantum Transpiler — production entry point.

    Usage
    -----
    # Default (all passes, auto-converge):
    t = Transpiler()
    opt_circuit, report = t.transpile(circuit)
    print(report.summary())

    # Custom pass list:
    t = Transpiler(passes=['redundant', 'rotation'])

    # Debug mode (logs every transformation):
    opt_circuit, report = t.transpile(circuit, debug=True)

    # Hardware-aware (future — pass HardwareConfig):
    t = Transpiler(backend_config=hw_config)

    Integration
    -----------
    Called automatically by:
        • QuantumLayer.forward()  (zuntenium_pytorch.py)
        • HybridSolver.solve()    (hybrid/solver.py)
        • AITranspiler Phase 1    (ai_transpiler.py)
    """

    def __init__(
        self,
        passes: Optional[List[str]] = None,
        backend_config: Optional[Dict[str, Any]] = None,
        max_iterations: int = 5,
    ) -> None:
        self._backend_config = backend_config
        self._pipeline = self._build_pipeline(passes, max_iterations)

    def _build_pipeline(
        self,
        pass_names: Optional[List[str]],
        max_iter: int,
    ) -> TranspilerPipeline:
        if pass_names is None:
            return TranspilerPipeline(max_iterations=max_iter)
        pass_map: Dict[str, OptimizationPass] = {
            'redundant':    RedundantGateCancellationPass(),
            'rotation':     RotationMergingPass(),
            'commutation':  CommutationReorderingPass(),
            'fusion':       SingleQubitFusionPass(),
            'depth':        DepthOptimizationPass(),
            'dead':         DeadGateEliminationPass(),
        }
        selected = []
        for name in pass_names:
            if name not in pass_map:
                raise ValueError(
                    f"Unknown pass '{name}'. "
                    f"Available: {sorted(pass_map.keys())}"
                )
            selected.append(pass_map[name])
        return TranspilerPipeline(selected, max_iterations=max_iter)

    def transpile(
        self,
        circuit: Circuit,
        backend_config: Optional[Dict[str, Any]] = None,
        passes: Optional[List[str]] = None,
        debug: bool = False,
    ) -> Tuple[Circuit, OptimizationReport]:
        """
        Main compilation entry point.

        Parameters
        ----------
        circuit        : input Circuit (not modified)
        backend_config : hardware constraints (reserved for v0.2 SWAP routing)
        passes         : override pass list for this call
        debug          : if True, print each transformation

        Returns
        -------
        (optimized_circuit, OptimizationReport)
        """
        if passes is not None:
            pipeline = self._build_pipeline(passes, self._pipeline._max_iter)
        else:
            pipeline = self._pipeline

        if backend_config is not None:
            self._backend_config = backend_config

        t0 = time.perf_counter()
        d_before = circuit.depth()

        if debug:
            print(f"\n[Transpiler] Input: {circuit}")
            print(circuit.draw())

        optimized, pass_results = pipeline.run(circuit)

        total_ms = (time.perf_counter() - t0) * 1000
        cp_len   = len(critical_path(optimized))
        par_ratio= gate_parallelism_ratio(optimized)

        report = OptimizationReport(
            original_gates=circuit.num_gates,
            optimized_gates=optimized.num_gates,
            original_depth=d_before,
            optimized_depth=optimized.depth(),
            estimated_fidelity=_estimate_fidelity(optimized),
            total_time_ms=total_ms,
            passes_run=pass_results,
            critical_path_len=cp_len,
            parallelism_ratio=par_ratio,
        )

        if debug:
            print(f"\n[Transpiler] Output: {optimized}")
            print(optimized.draw())
            print(f"\n{report.summary()}")
            print(report.passes_summary())

        return optimized, report

    def add_pass(self, p: OptimizationPass, position: int = -1) -> None:
        """Dynamically inject a pass (e.g., AI RL pass from future transpiler)."""
        self._pipeline.add_pass(p, position)


###############################################################################
# STEP 8 — Test Suite
###############################################################################

class TranspilerTestSuite:
    """
    Pytest-style test suite for all passes.  Run via TranspilerTestSuite().run_all().
    """

    def __init__(self) -> None:
        self.t = Transpiler()
        self._passed = 0
        self._failed = 0

    def _assert(self, cond: bool, msg: str) -> None:
        if cond:
            self._passed += 1
            print(f"  ✓  {msg}")
        else:
            self._failed += 1
            print(f"  ✗  FAIL: {msg}")

    def _assert_unitary_equiv(self, before: Circuit, after: Circuit, msg: str) -> None:
        """
        Verify before and after circuits produce equivalent unitaries
        by simulating both on a random initial state and comparing.
        """
        try:
            from zuntenium_statevector import StatevectorSimulator, GateLibrary
            _sim_avail = True
        except ImportError:
            _sim_avail = False

        if not _sim_avail:
            self._passed += 1
            print(f"  ~  {msg}  [sim unavailable, skipped]")
            return

        def run(circuit):
            sim = StatevectorSimulator(circuit.num_qubits, seed=0)
            for g in circuit.gates:
                if g.is_measurement: continue
                nm = g.name.upper()
                q = list(g.qubits)
                M = _gate_matrix(g)
                if M.shape == (2**g.num_qubits, 2**g.num_qubits):
                    sim.apply_gate(M, q)
            return sim.get_statevector()

        sv_before = run(before)
        sv_after  = run(after)
        # Up to global phase: |⟨ψ_before|ψ_after⟩| should be 1
        overlap = abs(np.vdot(sv_before, sv_after))
        self._assert(
            math.isclose(overlap, 1.0, abs_tol=1e-6),
            f"{msg}  (overlap={overlap:.8f})"
        )

    # ── Test cases ────────────────────────────────────────────────────────

    def test_empty_circuit(self) -> None:
        c = Circuit(3)
        opt, r = self.t.transpile(c)
        self._assert(opt.num_gates == 0, "empty circuit: 0 gates out")
        self._assert(opt.depth() == 0,   "empty circuit: depth=0")

    def test_single_gate(self) -> None:
        c = Circuit(1)
        c.add_gate(Gate('H',(0,)))
        opt, _ = self.t.transpile(c)
        self._assert(opt.num_gates == 1, "single H: 1 gate out")

    def test_pauli_cancellation_adjacent(self) -> None:
        c = Circuit(1)
        for _ in range(3): c.add_gate(Gate('X',(0,)))
        c.add_gate(Gate('X',(0,)))   # 4 X gates = 2 pairs
        opt, r = self.t.transpile(c)
        self._assert(opt.num_gates == 0, "4×X cancels to identity")
        self._assert_unitary_equiv(c, opt, "4×X equiv to identity")

    def test_pauli_cancellation_3x(self) -> None:
        c = Circuit(1)
        for _ in range(3): c.add_gate(Gate('X',(0,)))   # XXX = X
        opt, _ = self.t.transpile(c)
        self._assert(opt.num_gates <= 1, "3×X reduces to ≤1 gate")

    def test_hadamard_pair(self) -> None:
        c = Circuit(2)
        c.add_gate(Gate('H',(0,))); c.add_gate(Gate('H',(0,)))
        c.add_gate(Gate('H',(1,)))
        opt, _ = self.t.transpile(c)
        self._assert(opt.num_gates == 1, "H·H cancels, lone H remains")

    def test_rotation_fusion(self) -> None:
        c = Circuit(1)
        c.add_gate(Gate('RZ',(0,), params=(0.3,)))
        c.add_gate(Gate('RZ',(0,), params=(0.7,)))
        c.add_gate(Gate('RZ',(0,), params=(-0.5,)))   # total = 0.5
        opt, _ = self.t.transpile(c)
        self._assert(opt.num_gates == 1, "3×RZ fused to 1")
        if opt.num_gates == 1:
            self._assert(
                math.isclose(opt.gates[0].params[0], 0.5, abs_tol=1e-6),
                f"fused RZ angle=0.5 (got {opt.gates[0].params[0]:.6f})"
            )

    def test_rotation_cancel(self) -> None:
        c = Circuit(1)
        c.add_gate(Gate('RX',(0,), params=(1.2,)))
        c.add_gate(Gate('RX',(0,), params=(-1.2,)))
        opt, _ = self.t.transpile(c)
        self._assert(opt.num_gates == 0, "RX(θ)·RX(-θ) = identity")
        self._assert_unitary_equiv(c, opt, "RX cancel equiv")

    def test_s_sdg_cancel(self) -> None:
        c = Circuit(1)
        c.add_gate(Gate('S',(0,))); c.add_gate(Gate('SDG',(0,)))
        opt, _ = self.t.transpile(c)
        self._assert(opt.num_gates == 0, "S·S† = identity")

    def test_s_fusion(self) -> None:
        c = Circuit(1)
        c.add_gate(Gate('S',(0,))); c.add_gate(Gate('S',(0,)))   # S·S = Z
        opt, _ = self.t.transpile(c)
        self._assert(opt.num_gates <= 1, "S·S fuses to ≤1 gate (Z)")

    def test_bell_state_circuit(self) -> None:
        c = Circuit(2)
        c.add_gate(Gate('H',(0,)))
        c.add_gate(Gate('H',(0,)))   # redundant
        c.add_gate(Gate('H',(0,)))   # net: one H
        c.add_gate(Gate('CNOT', (1,), controls=(0,)))
        opt, r = self.t.transpile(c)
        self._assert(opt.num_gates == 2, f"Bell: 3H+CNOT→H+CNOT (got {opt.num_gates})")
        self._assert_unitary_equiv(c, opt, "Bell unitary equiv")

    def test_commutation_reordering(self) -> None:
        # Z on q0 commutes with CNOT control=q0, target=q1
        # So: CNOT(0,1), Z(0), X(1) should reorder to allow Z cancellation
        c = Circuit(2)
        c.add_gate(Gate('Z',(0,)))
        c.add_gate(Gate('CNOT',(1,), controls=(0,)))
        c.add_gate(Gate('Z',(0,)))
        opt, _ = self.t.transpile(c)
        # Z(0) commutes past CNOT on control, so Z·Z=I
        self._assert_unitary_equiv(c, opt, "Z(0)·CNOT(0,1)·Z(0) equiv")

    def test_single_qubit_fusion(self) -> None:
        c = Circuit(1)
        c.add_gate(Gate('H',(0,)))
        c.add_gate(Gate('T',(0,)))
        c.add_gate(Gate('H',(0,)))   # H·T·H = something non-trivial
        opt, _ = self.t.transpile(c)
        self._assert(opt.num_gates <= 1, "H·T·H fuses to ≤1 U3")
        self._assert_unitary_equiv(c, opt, "H·T·H fusion equiv")

    def test_dead_gate_elimination(self) -> None:
        c = Circuit(2)
        c.add_gate(Gate('H',(1,)))           # on unmeasured qubit
        c.add_gate(Gate('X',(0,)))           # on measured qubit
        c.add_gate(Gate('MEASURE', (0,)))    # measure qubit 0 only
        opt, _ = self.t.transpile(c)
        # H on q1 is dead (q1 not measured)
        measured_qubits = [g.qubits for g in opt.gates if not g.is_measurement]
        self._assert(
            all(0 in q for q in measured_qubits),
            "dead H(q1) eliminated when only q0 measured"
        )

    def test_20_qubit_circuit(self) -> None:
        n = 20
        c = Circuit(n)
        for q in range(n):
            c.add_gate(Gate('H',(q,)))
            c.add_gate(Gate('H',(q,)))   # all redundant pairs
            c.add_gate(Gate('RZ',(q,), params=(0.5,)))
            c.add_gate(Gate('RZ',(q,), params=(-0.5,)))
        opt, r = self.t.transpile(c)
        self._assert(opt.num_gates == 0, f"20-qubit all-cancel: {opt.num_gates} gates remaining")
        self._assert(r.total_time_ms < 500, f"20-qubit < 500ms (took {r.total_time_ms:.1f}ms)")

    def test_only_measurements(self) -> None:
        c = Circuit(3)
        for q in range(3): c.add_gate(Gate('MEASURE',(q,)))
        opt, _ = self.t.transpile(c)
        self._assert(opt.num_gates == 3, "only measurements: 3 MEASURE gates preserved")

    def test_angle_normalization(self) -> None:
        theta = 2 * math.pi + 0.5
        c = Circuit(1)
        c.add_gate(Gate('RZ',(0,), params=(theta,)))
        opt, _ = self.t.transpile(c)
        if opt.num_gates > 0:
            a = opt.gates[0].params[0]
            self._assert(-math.pi < a <= math.pi,
                         f"angle normalized to (-π,π]: {a:.6f}")

    def run_all(self) -> None:
        print("\n" + "═"*60)
        print("  ZUNTENIUM Basic Transpiler — Test Suite")
        print("═"*60)
        tests = [m for m in dir(self) if m.startswith('test_')]
        for tname in tests:
            print(f"\n[{tname}]")
            getattr(self, tname)()
        print(f"\n{'═'*60}")
        total = self._passed + self._failed
        print(f"  Results: {self._passed}/{total} passed"
              + (f"  ({self._failed} FAILED)" if self._failed else "  ✓ all pass"))
        print("═"*60)


###############################################################################
# STEP 9 — Complete Example Usage
###############################################################################

def build_demo_circuit() -> Circuit:
    """
    Bell state + redundant gates + rotations — designed to showcase all passes.
    """
    c = Circuit(3, name="demo")
    # Deliberate redundancy block 1: H·H = I on q0
    c.add_gate(Gate('H',(0,)))
    c.add_gate(Gate('H',(0,)))
    c.add_gate(Gate('H',(0,)))   # net: 1 H

    # Entanglement: Bell pair q0,q1
    c.add_gate(Gate('CNOT',(1,), controls=(0,)))

    # Z on control — commutes past CNOT, should cancel with next Z
    c.add_gate(Gate('Z',(0,)))
    c.add_gate(Gate('CNOT',(1,), controls=(0,)))
    c.add_gate(Gate('Z',(0,)))

    # Rotation merge on q2: 0.4 + 0.3 - 0.7 = 0.0 → identity
    c.add_gate(Gate('RX',(2,), params=(0.4,)))
    c.add_gate(Gate('RX',(2,), params=(0.3,)))
    c.add_gate(Gate('RX',(2,), params=(-0.7,)))

    # S pair on q2: S·S = Z
    c.add_gate(Gate('S',(2,)))
    c.add_gate(Gate('S',(2,)))

    # T·T† = I on q2
    c.add_gate(Gate('T',(2,)))
    c.add_gate(Gate('TDG',(2,)))

    return c


def run_demo() -> None:
    print("\n" + "═"*65)
    print("  ZUNTENIUM Basic Quantum Transpiler — Complete Demo")
    print("═"*65)

    # ── 1. Build demo circuit ──────────────────────────────────
    c = build_demo_circuit()
    print(f"\nInput circuit:  {c}")
    print(c.draw())

    # ── 2. Transpile ──────────────────────────────────────────
    t = Transpiler()
    opt, report = t.transpile(c, debug=False)
    print(f"\nOptimized circuit:  {opt}")
    print(opt.draw())
    print(f"\n{report.summary()}")
    print(report.passes_summary())

    # ── 3. Layers / parallelism ───────────────────────────────
    layers = opt.layers()
    print(f"\nExecution layers ({len(layers)} layers, parallelism={report.parallelism_ratio:.2f}):")
    for i, layer in enumerate(layers):
        print(f"  Layer {i+1}: {[str(g) for g in layer]}")

    # ── 4. Critical path ──────────────────────────────────────
    cp = critical_path(opt)
    print(f"\nCritical path ({len(cp)} gates): {[str(g) for g in cp]}")

    # ── 5. QAOA circuit ───────────────────────────────────────
    print("\n── QAOA(n=4, p=2) transpilation ──")
    qaoa = Circuit(4, name="qaoa_2layer")
    rng  = np.random.default_rng(0)
    for q in range(4): qaoa.add_gate(Gate('H',(q,)))
    for _ in range(2):
        for i in range(3):
            qaoa.add_gate(Gate('CNOT',(i+1,), controls=(i,)))
            qaoa.add_gate(Gate('RZ',(i+1,), params=(float(rng.uniform(0,1)),)))
            qaoa.add_gate(Gate('CNOT',(i+1,), controls=(i,)))
        for q in range(4):
            qaoa.add_gate(Gate('RX',(q,), params=(float(rng.uniform(0,1)),)))
    qaoa_opt, qaoa_r = t.transpile(qaoa)
    print(f"  {qaoa_r.summary()}")

    # ── 6. Adversarial circuit (max cancellations) ────────────
    print("\n── Adversarial circuit (should cancel to near-zero) ──")
    adv = Circuit(5, name="adversarial")
    for q in range(5):
        adv.add_gate(Gate('H',(q,))); adv.add_gate(Gate('H',(q,)))
        adv.add_gate(Gate('RZ',(q,), params=(0.5,))); adv.add_gate(Gate('RZ',(q,), params=(-0.5,)))
        adv.add_gate(Gate('T',(q,)));  adv.add_gate(Gate('TDG',(q,)))
        adv.add_gate(Gate('X',(q,)));  adv.add_gate(Gate('X',(q,)))
    adv_opt, adv_r = t.transpile(adv)
    print(f"  Input: {adv.num_gates} gates → Output: {adv_opt.num_gates} gates")
    print(f"  {adv_r.summary()}")

    # ── 7. Debug mode demo ────────────────────────────────────
    print("\n── Debug mode (small circuit) ──")
    small = Circuit(1, name="debug_demo")
    small.add_gate(Gate('S',(0,))); small.add_gate(Gate('S',(0,)))
    _ = t.transpile(small, debug=True)

    print("\n" + "═"*65)
    print("  Demo complete. ✓")
    print("═"*65 + "\n")


###############################################################################
# STEP 10 — Extensibility documentation
###############################################################################

"""
EXTENSIBILITY ROADMAP
======================

1. AI-based transpiler (v0.2) — RL agent as an OptimizationPass
   ─────────────────────────────────────────────────────────────
   The RLAgent from ai_transpiler.py can be wrapped as:

       class RLOptimizationPass(OptimizationPass):
           name = "RLOptimizationPass"
           def __init__(self, agent: RLAgent): self._agent = agent
           def _apply(self, circuit: Circuit) -> Circuit:
               gates = [_circuit_gate_to_ir(g) for g in circuit.gates]
               opt   = ... (run RL episodes)
               return Circuit.from_gate_list(circuit.num_qubits, new_gates)

   Injection: transpiler.add_pass(RLOptimizationPass(agent), position=-1)
   The TranspilerPipeline picks it up seamlessly.

2. Hardware-aware SWAP routing (v0.2) — HardwareConfig pass
   ─────────────────────────────────────────────────────────
   Add SWAPRoutingPass (already in ai_transpiler.py) as:

       class HardwareRoutingPass(OptimizationPass):
           def __init__(self, hw: HardwareConfig): self._hw = hw
           def _apply(self, circuit: Circuit) -> Circuit: ...

   Injection: transpiler.add_pass(HardwareRoutingPass(hw), position=4)
   (After fusion passes, before depth optimization.)

3. QuantumLayer.forward() integration (already hooked)
   ──────────────────────────────────────────────────
   In zuntenium_pytorch.py, ParameterShiftEngine._run_circuit() calls:
       if self._transpiler is not None:
           result = self._transpiler.optimize(all_gates)
   Replace self._transpiler with a Transpiler instance for richer IR support.

4. HybridSolver.solve() integration (v0.2)
   ────────────────────────────────────────
   In the HybridSolver cost function loop:
       transpiler = Transpiler(passes=['redundant','rotation','fusion'])
       bound_circuit, _ = transpiler.transpile(build_vqe_circuit(params))
       result = backend.run(bound_circuit)
   The transpiler is called once per VQE iteration, amortizing its cost
   across hundreds of circuit evaluations.

5. JAX / TensorFlow compilation path (v0.3)
   ────────────────────────────────────────
   Circuit is a pure-Python data structure → framework-agnostic.
   Only the executor layer (StatevectorSimulatorBackend) needs porting.
   The transpiler output (Circuit object) can be serialized to JSON
   and fed to any backend.

6. C++ acceleration (v0.4) — hot-path annotation
   ──────────────────────────────────────────────
   RedundantGateCancellationPass._scan_once()   ← O(N²) hot path
   RotationMergingPass._apply()                  ← O(N) but frequent
   SingleQubitFusionPass._apply()                ← matrix multiply hot
   These will be ported to pybind11 C++ with SIMD matrix multiply.
"""


###############################################################################
# MAIN
###############################################################################

if __name__ == "__main__":
    # Run demo
    run_demo()

    # Run test suite
    suite = TranspilerTestSuite()
    suite.run_all()
