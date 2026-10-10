"""
Transpiler v2 Core — Gate IR, Circuit, commutation, angle normalization.

Audit fixes applied:
  [HIGH-2]  NaN/Inf params rejected at Gate construction
  [LOW-2]   batch_remove for O(N) bulk removal
"""
from __future__ import annotations
import copy, math, uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple
import networkx as nx
import numpy as np

_ANGLE_TOL = 1e-12

def normalize_angle(theta: float, tol: float = _ANGLE_TOL) -> float:
    """Normalize to (-pi, pi]. Raises ValueError on NaN/Inf (fix HIGH-2)."""
    if not math.isfinite(theta):
        raise ValueError(f"Non-finite rotation angle: {theta!r}.")
    norm = math.fmod(theta + math.pi, 2.0 * math.pi) - math.pi
    if norm <= -math.pi + tol:
        norm += 2.0 * math.pi
    return 0.0 if abs(norm) < tol else norm

@dataclass
class Gate:
    """Canonical ZUNTENIUM IR gate node. Validates params at construction (HIGH-2)."""
    name: str
    targets: Tuple[int, ...]
    controls: Tuple[int, ...] = field(default_factory=tuple)
    params: Tuple[float, ...] = field(default_factory=tuple)
    metadata: Dict[str, Any] = field(default_factory=dict)
    uid: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    def __post_init__(self) -> None:
        for i, p in enumerate(self.params):
            if not math.isfinite(p):
                raise ValueError(f"Gate '{self.name}' param[{i}] = {p!r} is not finite.")
        object.__setattr__(self, 'name', self.name.upper())

    @property
    def qubits(self) -> Tuple[int, ...]: return self.controls + self.targets
    @property
    def num_qubits(self) -> int: return len(self.qubits)
    @property
    def is_single_qubit(self) -> bool: return len(self.targets) == 1 and len(self.controls) == 0
    @property
    def is_parametric(self) -> bool: return len(self.params) > 0
    @property
    def is_measurement(self) -> bool: return self.name in ('M', 'MEASURE', 'MEAS')

    def add_history(self, note: str) -> None:
        self.metadata.setdefault('history', []).append(note)

    def matrix(self) -> np.ndarray: return _gate_matrix(self)

    def structural_eq(self, other: "Gate") -> bool:
        if self.name != other.name: return False
        if self.targets != other.targets or self.controls != other.controls: return False
        if len(self.params) != len(other.params): return False
        return all(math.isclose(a, b, abs_tol=1e-12) for a, b in zip(self.params, other.params))

    def copy(self, new_uid: bool = True) -> "Gate":
        return Gate(name=self.name, targets=self.targets, controls=self.controls,
                    params=self.params, metadata=copy.deepcopy(self.metadata),
                    uid=str(uuid.uuid4())[:8] if new_uid else self.uid)

    def __repr__(self) -> str:
        p = f"({','.join(f'{x:.4f}' for x in self.params)})" if self.params else ""
        return f"Gate({self.name}{p}@{self.targets})"


def _gate_matrix(g: Gate) -> np.ndarray:
    nm, p, c128 = g.name, g.params, np.complex128
    def rx(t): c=math.cos(t/2); s=math.sin(t/2); return np.array([[c,-1j*s],[-1j*s,c]],dtype=c128)
    def ry(t): c=math.cos(t/2); s=math.sin(t/2); return np.array([[c,-s],[s,c]],dtype=c128)
    def rz(t): return np.array([[np.exp(-1j*t/2),0],[0,np.exp(1j*t/2)]],dtype=c128)
    STATIC = {
        'X':np.array([[0,1],[1,0]],dtype=c128), 'Y':np.array([[0,-1j],[1j,0]],dtype=c128),
        'Z':np.array([[1,0],[0,-1]],dtype=c128), 'H':np.array([[1,1],[1,-1]],dtype=c128)/math.sqrt(2),
        'S':np.array([[1,0],[0,1j]],dtype=c128), 'T':np.array([[1,0],[0,np.exp(1j*math.pi/4)]],dtype=c128),
        'SDG':np.array([[1,0],[0,-1j]],dtype=c128), 'TDG':np.array([[1,0],[0,np.exp(-1j*math.pi/4)]],dtype=c128),
        'SX':np.array([[1+1j,1-1j],[1-1j,1+1j]],dtype=c128)/2,
        'SXD':np.array([[1-1j,1+1j],[1+1j,1-1j]],dtype=c128)/2,
        'CNOT':np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]],dtype=c128),
        'CX':np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]],dtype=c128),
        'CZ':np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,-1]],dtype=c128),
        'SWAP':np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]],dtype=c128),
        'ISWAP':np.array([[1,0,0,0],[0,0,1j,0],[0,1j,0,0],[0,0,0,1]],dtype=c128),
        'ECR':np.array([[0, 1, 0, 1j], [1, 0, -1j, 0], [0, 1j, 0, 1], [-1j, 0, 1, 0]], dtype=c128)/math.sqrt(2),
        'I':np.eye(2,dtype=c128),
    }
    if nm in STATIC: return STATIC[nm]
    if nm == 'RX': return rx(p[0])
    if nm == 'RY': return ry(p[0])
    if nm == 'RZ': return rz(p[0])
    if nm == 'XX':
        th = p[0] if p else math.pi/4
        c, s = math.cos(th/2), math.sin(th/2)
        return np.array([[c, 0, 0, -1j*s], [0, c, -1j*s, 0], [0, -1j*s, c, 0], [-1j*s, 0, 0, c]], dtype=c128)
    if nm == 'PHASE': return np.array([[1,0],[0,np.exp(1j*p[0])]],dtype=c128)
    if nm == 'U3':
        th,ph,lm = p[0],p[1],p[2]; c,s=math.cos(th/2),math.sin(th/2)
        return np.array([[c,-np.exp(1j*lm)*s],[np.exp(1j*ph)*s,np.exp(1j*(ph+lm))*c]],dtype=c128)
    return np.eye(2,dtype=c128)


def matrices_equivalent(U: np.ndarray, V: np.ndarray, tol: float = 1e-7) -> bool:
    if U.shape != V.shape: return False
    W = U.conj().T @ V; d = U.shape[0]; dv = W[0,0]
    if abs(abs(dv) - 1.0) > tol: return False
    return bool(np.allclose(W, dv * np.eye(d), atol=tol))


# ── Commutation ──
_DIAG_1Q = frozenset({'Z','S','T','SDG','TDG','RZ','PHASE'})
_ROT_X = frozenset({'X','SX','SXD','RX'})

def gates_commute(g1: Gate, g2: Gate) -> bool:
    if not (frozenset(g1.qubits) & frozenset(g2.qubits)): return True
    if g1.is_single_qubit and g2.is_single_qubit and g1.targets == g2.targets:
        if g1.name in _DIAG_1Q and g2.name in _DIAG_1Q: return True
    for gcnot, gother in [(g1,g2),(g2,g1)]:
        if gcnot.name in ('CNOT','CX'):
            c = gcnot.controls[0] if gcnot.controls else gcnot.qubits[0]
            t = gcnot.targets[0] if gcnot.controls else gcnot.qubits[1]
            if gother.is_single_qubit:
                q = gother.targets[0]
                if q == c and gother.name in _DIAG_1Q: return True
                if q == t and gother.name in _ROT_X: return True
            if gother.name in ('CNOT','CX') and set(gother.qubits) == {c,t}: return True
    if g1.is_single_qubit and g2.is_single_qubit and g1.targets == g2.targets:
        try: return bool(np.allclose(g1.matrix()@g2.matrix(), g2.matrix()@g1.matrix(), atol=1e-10))
        except Exception: pass
    return False


# ── Circuit ──
class Circuit:
    """Circuit IR with batch_remove (fix LOW-2)."""
    def __init__(self, num_qubits: int, name: str = "") -> None:
        if num_qubits < 0: raise ValueError(f"num_qubits must be >= 0; got {num_qubits}")
        self._num_qubits = num_qubits; self._name = name
        self._gates: List[Gate] = []; self._timelines: Dict[int, List[int]] = defaultdict(list)
        self._dag: Optional[nx.DiGraph] = None; self._dirty_dag = False

    @property
    def num_qubits(self) -> int: return self._num_qubits
    @property
    def gates(self) -> List[Gate]: return self._gates
    @property
    def num_gates(self) -> int: return len(self._gates)

    def add_gate(self, gate: Gate, note: str = "") -> int:
        self._validate_gate(gate); idx = len(self._gates)
        gate = gate.copy(new_uid=False); gate.metadata.setdefault('original_index', idx)
        if note: gate.add_history(note)
        self._gates.append(gate)
        for q in gate.qubits: self._timelines[q].append(idx)
        self._dirty_dag = True; return idx

    def insert_gate(self, position: int, gate: Gate) -> None:
        self._validate_gate(gate); gate = gate.copy(new_uid=False)
        self._gates.insert(position, gate); self._rebuild_timelines(); self._dirty_dag = True

    def remove_gate(self, idx: int) -> Gate:
        if not (0 <= idx < len(self._gates)): raise IndexError(f"Index {idx} out of range")
        gate = self._gates.pop(idx); gate.add_history(f"removed at {idx}")
        self._rebuild_timelines(); self._dirty_dag = True; return gate

    def batch_remove(self, indices: Set[int]) -> List[Gate]:
        """Remove multiple gates in O(N) with single rebuild (fix LOW-2)."""
        removed, new = [], []
        for i, g in enumerate(self._gates):
            if i in indices: g.add_history(f"batch_removed at {i}"); removed.append(g)
            else: new.append(g)
        self._gates = new; self._rebuild_timelines(); self._dirty_dag = True; return removed

    def replace_gate(self, idx: int, new_gate: Gate, note: str = "") -> None:
        self._validate_gate(new_gate); old = self._gates[idx]
        new_gate = new_gate.copy(new_uid=False); new_gate.metadata['replaced'] = str(old)
        if note: new_gate.add_history(note)
        self._gates[idx] = new_gate; self._rebuild_timelines(); self._dirty_dag = True

    def to_dag(self) -> nx.DiGraph:
        if self._dag is not None and not self._dirty_dag: return self._dag
        G = nx.DiGraph(); G.add_nodes_from(range(len(self._gates)))
        for q, tl in self._timelines.items():
            for k in range(len(tl)-1):
                i, j = tl[k], tl[k+1]
                if not G.has_edge(i, j): G.add_edge(i, j, qubit=q)
        self._dag = G; self._dirty_dag = False; return G

    def depth(self) -> int:
        if not self._gates: return 0
        dag = self.to_dag(); layer: Dict[int,int] = {}
        for v in nx.topological_sort(dag):
            layer[v] = 1 + max((layer[p] for p in dag.predecessors(v)), default=0)
        return max(layer.values(), default=0)

    def layers(self) -> List[List[Gate]]:
        if not self._gates: return []
        dag = self.to_dag(); layer: Dict[int,int] = {}
        for v in nx.topological_sort(dag):
            layer[v] = 1 + max((layer[p] for p in dag.predecessors(v)), default=0)
        mx = max(layer.values(), default=0)
        buckets: List[List[Gate]] = [[] for _ in range(mx)]
        for v, l in layer.items(): buckets[l-1].append(self._gates[v])
        return buckets

    def qubit_timeline(self, qubit: int) -> List[Gate]:
        return [self._gates[i] for i in self._timelines.get(qubit, [])]

    def gate_provenance(self, idx: int) -> List[str]:
        return self._gates[idx].metadata.get('history', [])

    @classmethod
    def from_gate_list(cls, num_qubits: int, gates: List[Gate]) -> "Circuit":
        c = cls(num_qubits)
        for g in gates: c.add_gate(g)
        return c

    def copy(self) -> "Circuit":
        c = Circuit(self._num_qubits, self._name + "_copy")
        for g in self._gates: c._gates.append(g.copy())
        c._rebuild_timelines(); return c

    def draw(self) -> str:
        if not self._gates: return "\n".join(f"q{q}: ────" for q in range(self._num_qubits))
        layers = self.layers(); col_w = 10
        lines = [f"q{q}: " for q in range(self._num_qubits)]
        for layer in layers:
            active: Dict[int,str] = {}
            for g in layer:
                lbl = g.name
                if g.params: lbl += f"({','.join(f'{p:.2f}' for p in g.params)})"
                lbl = lbl[:col_w-2].center(col_w-2)
                for q in g.qubits: active[q] = f"─[{lbl}]─"
            for q in range(self._num_qubits): lines[q] += active.get(q, "─"*col_w)
        return "\n".join(lines)

    def _validate_gate(self, gate: Gate) -> None:
        for q in gate.qubits:
            if q < 0 or q >= self._num_qubits:
                raise ValueError(f"Gate {gate} references qubit {q}, circuit has {self._num_qubits}.")
        if len(set(gate.qubits)) != len(gate.qubits):
            raise ValueError(f"Gate {gate} has duplicate qubit indices.")

    def _rebuild_timelines(self) -> None:
        self._timelines = defaultdict(list)
        for idx, gate in enumerate(self._gates):
            for q in gate.qubits: self._timelines[q].append(idx)

    def __repr__(self) -> str:
        return f"Circuit('{self._name}', n={self._num_qubits}, gates={self.num_gates}, depth={self.depth()})"
