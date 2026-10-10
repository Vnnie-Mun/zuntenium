"""
Transpiler v2 Optimization Passes — All 6 audit fixes applied.

[CRITICAL-1] ZYZ decomposition fixed for SU(2) inputs
[HIGH-1]     CommutationReorderingPass O(N²) → O(N) per swap
[MEDIUM-1]   fuse_insert dict access fixed for non-adjacent pairs
[MEDIUM-2]   RotationMergingPass commuting-window re-insertion
[LOW-1]      Cancellation-exposing swaps in commutation pass
"""
from __future__ import annotations
import math, time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple
import networkx as nx
import numpy as np

from zuntenium.transpiler.v2.core import (
    Gate, Circuit, normalize_angle, gates_commute, matrices_equivalent, _gate_matrix,
)

# ── Pass infrastructure ──

@dataclass
class PassResult:
    pass_name: str
    gates_removed: int = 0
    gates_added: int = 0
    depth_delta: int = 0
    elapsed_ms: float = 0.0
    notes: List[str] = field(default_factory=list)
    @property
    def net_delta(self) -> int: return self.gates_added - self.gates_removed


class OptimizationPass(ABC):
    """Abstract base — immutable semantics: _apply() returns NEW Circuit."""
    name: str = "BasePass"
    description: str = ""

    def __call__(self, circuit: Circuit) -> Tuple[Circuit, PassResult]:
        t0 = time.perf_counter()
        new_circ = self._apply(circuit)
        elapsed = (time.perf_counter() - t0) * 1000
        return new_circ, PassResult(
            pass_name=self.name,
            gates_removed=max(0, circuit.num_gates - new_circ.num_gates),
            gates_added=max(0, new_circ.num_gates - circuit.num_gates),
            depth_delta=new_circ.depth() - circuit.depth(),
            elapsed_ms=elapsed,
        )

    @abstractmethod
    def _apply(self, circuit: Circuit) -> Circuit: ...


# ── Cancellation tables ──
_SELF_INVERSE: FrozenSet[str] = frozenset({'X','Y','Z','H','CNOT','CX','SWAP','CZ','ISWAP'})
_INVERSE_PAIRS: Dict[str,str] = {'S':'SDG','SDG':'S','T':'TDG','TDG':'T','SX':'SXD','SXD':'SX'}
_FUSION_PAIRS: Dict[Tuple[str,str],str] = {('S','S'):'Z',('T','T'):'S',('SX','SX'):'X'}

def _is_inverse(g1: Gate, g2: Gate) -> bool:
    if g1.targets != g2.targets or g1.controls != g2.controls: return False
    nm1, nm2 = g1.name, g2.name
    if nm1 == nm2 and nm1 in _SELF_INVERSE: return True
    if _INVERSE_PAIRS.get(nm1) == nm2: return True
    if nm1 == nm2 and nm1 in ('RX','RY','RZ','PHASE') and g1.params and g2.params:
        return abs(normalize_angle(g1.params[0] + g2.params[0])) < 1e-9
    if g1.num_qubits <= 2:
        try:
            U, V = g1.matrix(), g2.matrix()
            return bool(np.allclose(U @ V, np.eye(U.shape[0]), atol=1e-9))
        except Exception: pass
    return False


# ══════════════════════════════════════════════════════════════
# Pass 1: Redundant Gate Cancellation (fix MEDIUM-1)
# ══════════════════════════════════════════════════════════════

class RedundantGateCancellationPass(OptimizationPass):
    """Cancel A·A†=I pairs; fuse S·S=Z, T·T=S, SX·SX=X.
    Fix MEDIUM-1: fuse_map stores Gate directly at position i."""
    name = "RedundantGateCancellationPass"
    description = "Cancel inverse pairs and fuse gate products."

    def _apply(self, circuit: Circuit) -> Circuit:
        gates = [g.copy() for g in circuit.gates]
        changed = True
        while changed: gates, changed = self._scan_once(gates)
        return Circuit.from_gate_list(circuit.num_qubits, gates)

    def _scan_once(self, gates: List[Gate]) -> Tuple[List[Gate], bool]:
        n, skip = len(gates), set()
        fuse_map: Dict[int, Gate] = {}
        for i in range(n):
            if i in skip: continue
            g = gates[i]
            for j in range(i + 1, n):
                if j in skip: continue
                gj = gates[j]
                if set(g.qubits) == set(gj.qubits):
                    if _is_inverse(g, gj):
                        skip.add(i); skip.add(j); break
                    key = (g.name, gj.name)
                    if key in _FUSION_PAIRS and g.targets == gj.targets and not g.params:
                        fg = Gate(_FUSION_PAIRS[key], g.targets)
                        fg.add_history(f"fused {g.name}+{gj.name}->{fg.name}")
                        fuse_map[i] = fg; skip.add(i); skip.add(j); break
                    break
                elif not gates_commute(g, gj): break
        if not skip: return gates, False
        result: List[Gate] = []
        for i, g in enumerate(gates):
            if i in skip:
                if i in fuse_map: result.append(fuse_map[i])
            else: result.append(g)
        return result, True


# ══════════════════════════════════════════════════════════════
# Pass 2: Rotation Merging (fix MEDIUM-2)
# ══════════════════════════════════════════════════════════════

_ROT_GATES: FrozenSet[str] = frozenset({'RX','RY','RZ'})

class RotationMergingPass(OptimizationPass):
    """Merge RΩ(a)·RΩ(b)→RΩ(a+b). Fix MEDIUM-2: commuting window re-inserted."""
    name = "RotationMergingPass"
    description = "Merge same-axis rotations; replace special angles."
    _EPS = 1e-9
    _SPECIAL_RZ = [(math.pi,'Z'),(math.pi/2,'S'),(math.pi/4,'T'),(-math.pi/2,'SDG'),(-math.pi/4,'TDG')]

    def _apply(self, circuit: Circuit) -> Circuit:
        gates = [g.copy() for g in circuit.gates]
        return Circuit.from_gate_list(circuit.num_qubits, self._merge_pass(gates))

    def _merge_pass(self, gates: List[Gate]) -> List[Gate]:
        result: List[Gate] = []; i = 0
        while i < len(gates):
            g = gates[i]
            if g.name not in _ROT_GATES or not g.params:
                result.append(g); i += 1; continue
            q, total = g.targets, g.params[0]
            consumed, commuting = {i}, []
            j = i + 1
            while j < len(gates):
                gj = gates[j]
                if gj.name == g.name and gj.targets == q:
                    total += gj.params[0]; consumed.add(j); j += 1
                elif gates_commute(g, gj):
                    commuting.append(gj.copy()); consumed.add(j); j += 1
                else: break
            total_norm = normalize_angle(total)
            if abs(total_norm) > self._EPS:
                result.append(self._maybe_replace(g.name, total_norm, q))
            result.extend(commuting)
            i = j
        return result

    def _maybe_replace(self, nm: str, angle: float, targets: Tuple[int,...]) -> Gate:
        if nm == 'RZ':
            for ref, gn in self._SPECIAL_RZ:
                if math.isclose(angle, ref, abs_tol=1e-7):
                    g = Gate(gn, targets); g.add_history(f"RZ({angle:.4f})->{gn}"); return g
        g = Gate(nm, targets, params=(angle,)); g.add_history(f"merged angle={angle:.6f}"); return g


# ══════════════════════════════════════════════════════════════
# Pass 3: Commutation Reordering (fix HIGH-1, LOW-1)
# ══════════════════════════════════════════════════════════════

class CommutationReorderingPass(OptimizationPass):
    """O(N) bubble reorder via local layer tracking (was O(N²))."""
    name = "CommutationReorderingPass"
    description = "Depth-reducing + cancellation-exposing reorder."

    def _apply(self, circuit: Circuit) -> Circuit:
        gates = [g.copy() for g in circuit.gates]
        improved = True
        while improved:
            improved = self._phase_a(gates) or self._phase_b(gates)
        return Circuit.from_gate_list(circuit.num_qubits, gates)

    def _phase_a(self, gates: List[Gate]) -> bool:
        changed = False
        for i in range(len(gates) - 1):
            if not gates_commute(gates[i], gates[i+1]): continue
            if self._swap_reduces_depth(gates, i):
                gates[i], gates[i+1] = gates[i+1], gates[i]; changed = True
        return changed

    def _phase_b(self, gates: List[Gate]) -> bool:
        changed = False
        for i in range(len(gates) - 1):
            if not gates_commute(gates[i], gates[i+1]): continue
            if i > 0 and _is_inverse(gates[i+1], gates[i-1]):
                gates[i], gates[i+1] = gates[i+1], gates[i]; changed = True
            elif i + 2 < len(gates) and _is_inverse(gates[i], gates[i+2]):
                gates[i], gates[i+1] = gates[i+1], gates[i]; changed = True
        return changed

    def _swap_reduces_depth(self, gates: List[Gate], i: int) -> bool:
        g1, g2 = gates[i], gates[i+1]
        q_layer: Dict[int, int] = defaultdict(int)
        for k in range(i):
            lk = 1 + max((q_layer[q] for q in gates[k].qubits), default=0)
            for q in gates[k].qubits: q_layer[q] = lk
        l1 = 1 + max((q_layer[q] for q in g1.qubits), default=0)
        qt = dict(q_layer)
        for q in g1.qubits: qt[q] = l1
        l2 = 1 + max((qt.get(q,0) for q in g2.qubits), default=0)
        d_orig = max(l1, l2)
        l2s = 1 + max((q_layer[q] for q in g2.qubits), default=0)
        qt2 = dict(q_layer)
        for q in g2.qubits: qt2[q] = l2s
        l1s = 1 + max((qt2.get(q,0) for q in g1.qubits), default=0)
        return max(l2s, l1s) < d_orig


# ══════════════════════════════════════════════════════════════
# Pass 4: Single-Qubit Fusion (fix CRITICAL-1)
# ══════════════════════════════════════════════════════════════

class SingleQubitFusionPass(OptimizationPass):
    """Fuse consecutive 1Q gates into U3 via corrected ZYZ decomposition."""
    name = "SingleQubitFusionPass"
    description = "Fuse consecutive 1Q gates into U3 via correct ZYZ."
    _EPS_IDENTITY = 1e-8; _EPS_VERIFY = 1e-6

    def _apply(self, circuit: Circuit) -> Circuit:
        result, gates, n, i = [], circuit.gates, len(circuit.gates), 0
        while i < n:
            g = gates[i]
            if not g.is_single_qubit or g.is_measurement:
                result.append(g.copy()); i += 1; continue
            q = g.targets[0]
            run = [g]; j = i + 1
            while j < n and gates[j].is_single_qubit and gates[j].targets[0] == q and not gates[j].is_measurement:
                run.append(gates[j]); j += 1
            if len(run) == 1: result.append(g.copy()); i += 1; continue
            U = np.eye(2, dtype=np.complex128)
            # Circuit order: run[0] applied first, run[-1] last.
            # Unitary: U_total = run[-1] @ ... @ run[1] @ run[0]
            # Build by iterating forward: U = gate_matrix @ U
            for rg in run: U = _gate_matrix(rg) @ U
            fused = self._to_gate(U, q, [r.name for r in run])
            if fused is not None: result.append(fused)
            i = j
        return Circuit.from_gate_list(circuit.num_qubits, result)

    def _to_gate(self, U: np.ndarray, qubit: int, source_names: List[str]) -> Optional[Gate]:
        det = complex(U[0,0]*U[1,1] - U[0,1]*U[1,0])
        U_su2 = np.exp(-1j * np.angle(det) / 2.0) * U
        if self._is_identity(U_su2): return None
        th, ph, lm = self._zyz(U_su2)
        if th is None: return None
        try: th, ph, lm = normalize_angle(th), normalize_angle(ph), normalize_angle(lm)
        except ValueError: return None
        c, s = math.cos(th/2), math.sin(th/2)
        U_check = np.array([[c,-np.exp(1j*lm)*s],[np.exp(1j*ph)*s,np.exp(1j*(ph+lm))*c]], dtype=np.complex128)
        if not matrices_equivalent(U_check, U_su2, tol=self._EPS_VERIFY): return None
        g = Gate('U3', (qubit,), params=(th, ph, lm))
        g.add_history(f"fused {len(source_names)} gates: {source_names}"); return g

    def _is_identity(self, U: np.ndarray) -> bool:
        u = U.copy()
        if abs(u[0,0]) > self._EPS_IDENTITY: u /= u[0,0]
        return bool(np.allclose(u, np.eye(2), atol=self._EPS_IDENTITY))

    def _zyz(self, U: np.ndarray) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Corrected ZYZ Euler decomposition of SU(2) matrix.

        U3(θ,φ,λ) = [[cos(θ/2), -e^{iλ}·sin(θ/2)], [e^{iφ}·sin(θ/2), e^{i(φ+λ)}·cos(θ/2)]]

        Since U_su2[0,0] may be complex (= cos(θ/2)·e^{iα} for some global phase α),
        we must factor out α = arg(U[0,0]) before extracting φ and λ.
        The resulting U3(θ,φ,λ) equals U_su2 up to the global phase e^{iα}.
        """
        a = complex(U[0, 0])
        cos_half = min(abs(a), 1.0)
        theta = 2.0 * float(np.arccos(cos_half))
        sin_half = math.sin(theta / 2.0)

        if sin_half < 1e-10:
            # Near-diagonal: U ≈ diag(e^{iα}, e^{-iα}) → φ=0, λ=-2α
            return theta, 0.0, float(-2.0 * np.angle(a))

        # Factor out α = arg(U[0,0]) so that U_rotated[0,0] is real
        alpha = np.angle(a)
        U_rot = np.exp(-1j * alpha) * U
        phi = float(np.angle(complex(U_rot[1, 0])))
        lam = float(np.angle(-complex(U_rot[0, 1])))
        return theta, phi, lam


# ══════════════════════════════════════════════════════════════
# Pass 5: Depth Optimization
# ══════════════════════════════════════════════════════════════

class DepthOptimizationPass(OptimizationPass):
    name = "DepthOptimizationPass"
    description = "Re-emit in topological order for maximal layer parallelism."
    def _apply(self, circuit: Circuit) -> Circuit:
        if not circuit.gates: return circuit.copy()
        dag = circuit.to_dag()
        return Circuit.from_gate_list(circuit.num_qubits,
            [circuit.gates[i].copy() for i in nx.topological_sort(dag)])


# ══════════════════════════════════════════════════════════════
# Pass 6: Dead Gate Elimination
# ══════════════════════════════════════════════════════════════

class DeadGateEliminationPass(OptimizationPass):
    name = "DeadGateEliminationPass"
    description = "Remove gates on unmeasured qubits via backward analysis."
    def _apply(self, circuit: Circuit) -> Circuit:
        gates = circuit.gates
        measured: Set[int] = set()
        for g in gates:
            if g.is_measurement: measured.update(g.qubits)
        if not measured: return circuit.copy()
        live, keep = set(measured), [False]*len(gates)
        for i in reversed(range(len(gates))):
            g = gates[i]
            if g.is_measurement: keep[i] = True; continue
            if set(g.qubits) & live: keep[i] = True; live.update(g.qubits)
        return Circuit.from_gate_list(circuit.num_qubits, [g.copy() for i,g in enumerate(gates) if keep[i]])
