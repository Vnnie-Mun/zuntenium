"""
Rule-based optimization passes (Phase 1) for the Zuntenium AI Transpiler.

Five composable passes, each implementing ``run(gates, hw) → gates``:

1. **PauliCancellationPass** — cancel inverse-pair gates with look-ahead.
2. **RotationFusionPass** — merge same-axis rotations.
3. **SingleQubitUnifyPass** — fold 1Q gate runs into a single U3.
4. **CommutativeReorderingPass** — bubble-sort for depth reduction.
5. **SWAPRoutingPass** — insert SWAPs for hardware connectivity.

The ``RuleEngine`` sequences all passes and iterates until convergence.

Author:  ZUNTENIUM Core Team
License: Apache 2.0
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from zuntenium.transpiler.ai.ir import (
    CircuitDAG,
    Gate,
    HardwareConfig,
    circuit_to_dag,
)


# ═══════════════════════════════════════════════════════════════════════════
# Abstract base
# ═══════════════════════════════════════════════════════════════════════════

class AITranspilerPass(ABC):
    """Base class for AI transpiler optimisation passes."""

    name: str = "BasePass"

    @abstractmethod
    def run(
        self,
        gates: List[Gate],
        hw: Optional[HardwareConfig] = None,
    ) -> List[Gate]:
        """Apply this pass and return the (possibly modified) gate list."""
        ...

    def __repr__(self) -> str:
        return f"<{self.name}>"


# ═══════════════════════════════════════════════════════════════════════════
# Commutation tables
# ═══════════════════════════════════════════════════════════════════════════

_SINGLE_QUBIT_COMMUTES: Dict[Tuple[str, str], bool] = {}
for _pair in [
    ("X", "X"), ("Y", "Y"), ("Z", "Z"),
    ("Z", "RZ"), ("RZ", "Z"), ("RZ", "RZ"),
    ("Z", "S"), ("S", "Z"), ("Z", "T"), ("T", "Z"),
    ("S", "T"), ("T", "S"), ("S", "S"), ("T", "T"),
    ("X", "RX"), ("RX", "X"), ("RX", "RX"),
    ("Y", "RY"), ("RY", "Y"), ("RY", "RY"),
    ("H", "H"),
]:
    _SINGLE_QUBIT_COMMUTES[_pair] = True
    _SINGLE_QUBIT_COMMUTES[(_pair[1], _pair[0])] = True


def gates_commute(g1: Gate, g2: Gate) -> bool:
    """Return ``True`` iff *g1* and *g2* can be swapped without changing
    the circuit unitary.

    Rules:
        1. Disjoint qubit support → always commute (tensor product).
        2. Same single-qubit → lookup commutation table.
        3. CNOT(c,t) commutes with Z/S/T on control, X/RX on target.
        4. Otherwise → conservative ``False``.
    """
    q1, q2 = set(g1.qubits), set(g2.qubits)
    if q1.isdisjoint(q2):
        return True
    if g1.num_qubits == 1 and g2.num_qubits == 1:
        return _SINGLE_QUBIT_COMMUTES.get((g1.name, g2.name), False)
    if g1.name in ("CNOT", "CX") and g2.num_qubits == 1:
        c, t = g1.qubits[0], g1.qubits[1]
        if g2.qubits[0] == c and g2.name in ("Z", "RZ", "S", "T", "SDG", "TDG"):
            return True
        if g2.qubits[0] == t and g2.name in ("X", "RX"):
            return True
    if g2.name in ("CNOT", "CX") and g1.num_qubits == 1:
        c, t = g2.qubits[0], g2.qubits[1]
        if g1.qubits[0] == c and g1.name in ("Z", "RZ", "S", "T", "SDG", "TDG"):
            return True
        if g1.qubits[0] == t and g1.name in ("X", "RX"):
            return True
    return False


# ═══════════════════════════════════════════════════════════════════════════
# 1Q gate matrices (fallback — no dependency on zuntenium gate objects)
# ═══════════════════════════════════════════════════════════════════════════

class _GateMatrices:
    """Minimal 2×2 gate matrices for SingleQubitUnifyPass."""

    X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / math.sqrt(2)
    S = np.array([[1, 0], [0, 1j]], dtype=np.complex128)
    T = np.array([[1, 0], [0, np.exp(1j * math.pi / 4)]], dtype=np.complex128)
    Sdg = np.array([[1, 0], [0, -1j]], dtype=np.complex128)
    Tdg = np.array([[1, 0], [0, np.exp(-1j * math.pi / 4)]], dtype=np.complex128)

    @staticmethod
    def rx(t: float) -> np.ndarray:
        c, s = math.cos(t / 2), math.sin(t / 2)
        return np.array([[c, -1j * s], [-1j * s, c]], dtype=np.complex128)

    @staticmethod
    def ry(t: float) -> np.ndarray:
        c, s = math.cos(t / 2), math.sin(t / 2)
        return np.array([[c, -s], [s, c]], dtype=np.complex128)

    @staticmethod
    def rz(t: float) -> np.ndarray:
        return np.array(
            [[np.exp(-1j * t / 2), 0], [0, np.exp(1j * t / 2)]],
            dtype=np.complex128,
        )

    @staticmethod
    def u3(t: float, p: float, l: float) -> np.ndarray:
        c, s = math.cos(t / 2), math.sin(t / 2)
        return np.array(
            [[c, -np.exp(1j * l) * s],
             [np.exp(1j * p) * s, np.exp(1j * (p + l)) * c]],
            dtype=np.complex128,
        )

    _FIXED: Dict[str, np.ndarray] = {}


_GateMatrices._FIXED = {
    "X": _GateMatrices.X, "Y": _GateMatrices.Y, "Z": _GateMatrices.Z,
    "H": _GateMatrices.H, "S": _GateMatrices.S, "T": _GateMatrices.T,
    "SDG": _GateMatrices.Sdg, "TDG": _GateMatrices.Tdg,
}


def _gate_matrix_1q(g: Gate) -> np.ndarray:
    """Return the 2×2 unitary matrix for any single-qubit gate."""
    nm = g.name.upper()
    fixed = _GateMatrices._FIXED.get(nm)
    if fixed is not None:
        return fixed
    if nm == "RX":
        return _GateMatrices.rx(g.params[0])
    if nm == "RY":
        return _GateMatrices.ry(g.params[0])
    if nm == "RZ":
        return _GateMatrices.rz(g.params[0])
    if nm == "U3":
        return _GateMatrices.u3(*g.params[:3])
    return np.eye(2, dtype=np.complex128)


# ═══════════════════════════════════════════════════════════════════════════
# Pass 1 — Pauli Cancellation
# ═══════════════════════════════════════════════════════════════════════════

class PauliCancellationPass(AITranspilerPass):
    """Cancel A·A† = I for self-inverse and inverse-pair gates.

    Identities:
        X·X = I, Y·Y = I, Z·Z = I, H·H = I
        CNOT·CNOT = I, SWAP·SWAP = I, CZ·CZ = I
        S·S† = I, T·T† = I
        RX(θ)·RX(−θ) = I  (with mod 2π check)

    Algorithm: left-to-right scan with commutative look-ahead.
    Complexity: O(n²) worst-case; O(n) on well-structured circuits.
    """

    name = "PauliCancellationPass"

    _INVERSES: Dict[str, str] = {
        "X": "X", "Y": "Y", "Z": "Z", "H": "H",
        "CNOT": "CNOT", "CX": "CX", "SWAP": "SWAP", "CZ": "CZ",
        "S": "SDG", "SDG": "S", "T": "TDG", "TDG": "T",
    }

    def run(
        self,
        gates: List[Gate],
        hw: Optional[HardwareConfig] = None,
    ) -> List[Gate]:
        changed = True
        result = list(gates)
        while changed:
            result, changed = self._scan(result)
        return result

    def _scan(self, gates: List[Gate]) -> Tuple[List[Gate], bool]:
        skip: Set[int] = set()
        changed = False
        for i in range(len(gates)):
            if i in skip:
                continue
            j = self._find_partner(gates, i, skip)
            if j is not None and self._cancels(gates[i], gates[j]):
                skip.add(i)
                skip.add(j)
                changed = True
        result = [gates[i] for i in range(len(gates)) if i not in skip]
        return result, changed

    def _find_partner(
        self,
        gates: List[Gate],
        start: int,
        skip: Set[int],
    ) -> Optional[int]:
        g = gates[start]
        for j in range(start + 1, len(gates)):
            if j in skip:
                continue
            gj = gates[j]
            if set(gj.qubits) == set(g.qubits):
                return j
            if not gates_commute(g, gj):
                break
        return None

    def _cancels(self, g1: Gate, g2: Gate) -> bool:
        if g1.qubits != g2.qubits:
            return False
        inv = self._INVERSES.get(g1.name)
        if inv == g2.name:
            return True
        if (
            g1.name == g2.name
            and g1.name in ("RX", "RY", "RZ")
            and g1.params
            and g2.params
        ):
            total = g1.params[0] + g2.params[0]
            return (
                math.isclose(total % (2 * math.pi), 0.0, abs_tol=1e-9)
                or math.isclose(total, 0.0, abs_tol=1e-9)
            )
        return False


# ═══════════════════════════════════════════════════════════════════════════
# Pass 2 — Rotation Fusion
# ═══════════════════════════════════════════════════════════════════════════

class RotationFusionPass(AITranspilerPass):
    """Merge consecutive same-axis rotations.

    Identity: exp(−iθ₁P/2)·exp(−iθ₂P/2) = exp(−i(θ₁+θ₂)P/2)
    because a Pauli generator P commutes with itself.

    Post-fusion angles are normalised to ``(−π, π]``.
    If ``|θ_total| < ε`` the gate is dropped entirely.
    """

    name = "RotationFusionPass"
    _ROT = {"RX", "RY", "RZ"}
    _EPS = 1e-9

    def run(
        self,
        gates: List[Gate],
        hw: Optional[HardwareConfig] = None,
    ) -> List[Gate]:
        result: List[Gate] = []
        i = 0
        while i < len(gates):
            g = gates[i]
            if g.name not in self._ROT:
                result.append(g)
                i += 1
                continue
            total = g.params[0]
            j = i + 1
            while j < len(gates):
                gj = gates[j]
                if gj.name == g.name and gj.qubits == g.qubits:
                    total += gj.params[0]
                    j += 1
                elif gates_commute(g, gj):
                    j += 1
                else:
                    break
            total_norm = (total + math.pi) % (2 * math.pi) - math.pi
            if abs(total_norm) > self._EPS:
                result.append(Gate(g.name, g.qubits, (total_norm,)))
            i = j
        return result


# ═══════════════════════════════════════════════════════════════════════════
# Pass 3 — Single-Qubit Unification
# ═══════════════════════════════════════════════════════════════════════════

class SingleQubitUnifyPass(AITranspilerPass):
    """Fold any run of consecutive 1Q gates on the same qubit into one U3.

    Steps:
        1. U_total = U_k · … · U_1  (matrix product, right-to-left)
        2. If U_total ≈ e^{iα}·I → remove entire run.
        3. Else decompose via ZYZ: U = RZ(φ)·RY(θ)·RZ(λ).
    """

    name = "SingleQubitUnifyPass"
    _EPS = 1e-8

    def run(
        self,
        gates: List[Gate],
        hw: Optional[HardwareConfig] = None,
    ) -> List[Gate]:
        result: List[Gate] = []
        i = 0
        while i < len(gates):
            g = gates[i]
            if not g.is_single_qubit():
                result.append(g)
                i += 1
                continue
            q = g.qubits[0]
            run = [g]
            j = i + 1
            while (
                j < len(gates)
                and gates[j].is_single_qubit()
                and gates[j].qubits[0] == q
            ):
                run.append(gates[j])
                j += 1
            if len(run) == 1:
                result.append(g)
                i += 1
                continue
            U = np.eye(2, dtype=np.complex128)
            for rg in reversed(run):
                U = _gate_matrix_1q(rg) @ U
            if self._is_identity(U):
                i = j
                continue
            theta, phi, lam = self._zyz(U)
            result.append(Gate("U3", (q,), (theta, phi, lam)))
            i = j
        return result

    def _is_identity(self, U: np.ndarray) -> bool:
        u = U.copy()
        if abs(u[0, 0]) > self._EPS:
            u /= u[0, 0]
        return bool(np.allclose(u, np.eye(2), atol=self._EPS))

    @staticmethod
    def _zyz(U: np.ndarray) -> Tuple[float, float, float]:
        det = U[0, 0] * U[1, 1] - U[0, 1] * U[1, 0]
        U2 = np.exp(-1j * np.angle(det) / 2) * U
        a = U2[0, 0]
        theta = 2.0 * math.acos(min(abs(a), 1.0))
        if abs(math.sin(theta / 2)) < 1e-10:
            return 0.0, float(-2 * np.angle(a)), 0.0
        phi_plus_lam = float(-2 * np.angle(a))
        phi_minus_lam = float(2 * np.angle(U2[1, 0]) + math.pi)
        return (
            float(theta),
            (phi_plus_lam + phi_minus_lam) / 2,
            (phi_plus_lam - phi_minus_lam) / 2,
        )


# ═══════════════════════════════════════════════════════════════════════════
# Pass 4 — Commutative Reordering
# ═══════════════════════════════════════════════════════════════════════════

class CommutativeReorderingPass(AITranspilerPass):
    """Reduce circuit depth by greedy bubble-sort of commuting gates.

    For each adjacent pair (i, i+1) that commutes, try swapping and keep
    the swap if DAG depth decreases.  Terminates when no swap improves
    depth.  Complexity: O(n² · D) where D is the cost of depth().
    """

    name = "CommutativeReorderingPass"

    def run(
        self,
        gates: List[Gate],
        hw: Optional[HardwareConfig] = None,
    ) -> List[Gate]:
        result = list(gates)
        improved = True
        while improved:
            improved = False
            for i in range(len(result) - 1):
                if gates_commute(result[i], result[i + 1]):
                    d_before = circuit_to_dag(result).depth()
                    result[i], result[i + 1] = result[i + 1], result[i]
                    if circuit_to_dag(result).depth() < d_before:
                        improved = True
                    else:
                        result[i], result[i + 1] = result[i + 1], result[i]
        return result


# ═══════════════════════════════════════════════════════════════════════════
# Pass 5 — SWAP Routing
# ═══════════════════════════════════════════════════════════════════════════

class SWAPRoutingPass(AITranspilerPass):
    """Insert SWAP gates for hardware coupling-map satisfaction.

    Maintains a logical→physical qubit mapping ``l2p`` that is updated
    after each SWAP insertion.  Each SWAP decomposes to 3 CNOTs on real
    hardware.
    """

    name = "SWAPRoutingPass"

    def run(
        self,
        gates: List[Gate],
        hw: Optional[HardwareConfig] = None,
    ) -> List[Gate]:
        if hw is None:
            return gates
        result: List[Gate] = []
        l2p = {i: i for i in range(hw.num_qubits)}
        for gate in gates:
            if gate.num_qubits == 1:
                result.append(
                    Gate(gate.name, (l2p[gate.qubits[0]],), gate.params)
                )
                continue
            if gate.num_qubits == 2:
                c, t = gate.qubits
                cp, tp = l2p[c], l2p[t]
                if hw.are_connected(cp, tp):
                    result.append(
                        Gate(gate.name, (cp, tp), gate.params)
                    )
                else:
                    for qa, qb in hw.shortest_swap_path(cp, tp):
                        result.append(Gate("SWAP", (qa, qb)))
                        la = next(k for k, v in l2p.items() if v == qa)
                        lb = next(k for k, v in l2p.items() if v == qb)
                        l2p[la], l2p[lb] = qb, qa
                    result.append(
                        Gate(gate.name, (l2p[c], l2p[t]), gate.params)
                    )
            else:
                result.append(gate)
        return result


# ═══════════════════════════════════════════════════════════════════════════
# Rule Engine — ordered multi-pass pipeline
# ═══════════════════════════════════════════════════════════════════════════

class RuleEngine:
    """Ordered multi-pass rule-based optimiser.

    Sequences the five passes and repeats until gate count converges
    (up to ``max_iterations``).
    """

    def __init__(self) -> None:
        self._passes: List[AITranspilerPass] = [
            PauliCancellationPass(),
            RotationFusionPass(),
            SingleQubitUnifyPass(),
            CommutativeReorderingPass(),
        ]
        self._routing = SWAPRoutingPass()

    def optimize(
        self,
        gates: List[Gate],
        hw: Optional[HardwareConfig] = None,
        max_iterations: int = 5,
    ) -> Tuple[List[Gate], List[str]]:
        """Run the full Phase 1 pipeline, returning ``(gates, log)``."""
        current = list(gates)
        log: List[str] = []
        for iteration in range(max_iterations):
            prev = len(current)
            for p in self._passes:
                before = len(current)
                current = p.run(current, hw)
                d = before - len(current)
                if d:
                    log.append(f"iter{iteration + 1}:{p.name}: -{d}")
            if hw is not None:
                before = len(current)
                current = self._routing.run(current, hw)
                d = len(current) - before
                if d:
                    log.append(f"iter{iteration + 1}:SWAPRouting: +{d}")
            if len(current) >= prev:
                break
        return current, log
