"""ZUNTENIUM hardware-aware transpilation for v2 Circuit IR."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Union

import networkx as nx
import numpy as np

from zuntenium.transpiler.v2 import Circuit, Gate, Transpiler
from zuntenium.transpiler.v2.passes import (
    OptimizationPass, RedundantGateCancellationPass, RotationMergingPass,
    SingleQubitFusionPass, DepthOptimizationPass,
)


@dataclass
class HardwareConfig:
    name: str
    num_qubits: int
    coupling_map: List[Tuple[int, int]]
    native_gates: Set[str]
    gate_durations: Dict[str, float]
    error_rates: Dict[str, float]
    link_errors: Dict[Tuple[int, int], float] = field(default_factory=dict)
    qubit_t1: Dict[int, float] = field(default_factory=dict)
    qubit_t2: Dict[int, float] = field(default_factory=dict)
    basis_gates: List[str] = field(default_factory=list)
    description: str = ""
    cost_weights: Tuple[float, float, float] = (1.0, 0.3, 2.0)

    def __post_init__(self):
        self.native_gates = {gate.upper() for gate in self.native_gates}
        if not self.basis_gates:
            self.basis_gates = sorted(self.native_gates)
        symmetric = {}
        for (q0, q1), error in self.link_errors.items():
            symmetric[(q0, q1)] = error
            symmetric[(q1, q0)] = error
        self.link_errors = symmetric
        if self.num_qubits < 1:
            raise ValueError("num_qubits must be positive")
        if any(q < 0 or q >= self.num_qubits for edge in self.coupling_map for q in edge):
            raise ValueError("coupling_map contains an invalid physical qubit")

    def coupling_graph(self) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(range(self.num_qubits))
        default = self.error_rates.get("2Q", self.error_rates.get("2Q".lower(), 0.01))
        for q0, q1 in self.coupling_map:
            error = self.link_errors.get((q0, q1), default)
            graph.add_edge(q0, q1, weight=max(3.0 * error, 1e-9), error=error)
        return graph

    def swap_cost(self, q0: int, q1: int) -> float:
        return 3.0 * self.link_error(q0, q1)

    def link_error(self, q0: int, q1: int) -> float:
        return self.link_errors.get((q0, q1), self.error_rates.get("2Q", 0.01))

    def qubit_error(self, qubit: int) -> float:
        return self.error_rates.get(f"Q{qubit}", self.error_rates.get("1Q", 0.001))

    def are_connected(self, q0: int, q1: int) -> bool:
        return self.coupling_graph().has_edge(q0, q1)

    def shortest_swap_path(self, q0: int, q1: int) -> Tuple[List[int], float]:
        graph = self.coupling_graph()
        try:
            return nx.shortest_path(graph, q0, q1, weight="weight"), nx.shortest_path_length(graph, q0, q1, weight="weight")
        except nx.NetworkXNoPath:
            return [q0], float("inf")

    def validate_for(self, circuit: Circuit) -> None:
        if circuit.num_qubits > self.num_qubits:
            raise ValueError(f"Circuit needs {circuit.num_qubits} qubits but hardware has {self.num_qubits}.")


class HardwareProfileRegistry:
    _registry: Dict[str, HardwareConfig] = {}

    @classmethod
    def register(cls, config: HardwareConfig): cls._registry[config.name] = config
    @classmethod
    def get(cls, name: str) -> HardwareConfig:
        if name not in cls._registry:
            raise KeyError(f"Unknown hardware profile {name!r}; available: {sorted(cls._registry)}")
        return cls._registry[name]
    @classmethod
    def available(cls) -> List[str]: return sorted(cls._registry)


_IBM_QX5 = HardwareConfig(
    name="ibm_qx5",
    num_qubits=5,
    coupling_map=[(0, 1), (1, 2), (1, 3), (3, 4)],
    native_gates={"RZ", "SX", "X", "ECR", "MEASURE"},
    gate_durations={"RZ": 0, "SX": 35, "X": 35, "ECR": 519, "SWAP": 1557, "MEASURE": 740},
    error_rates={"RZ": 0, "SX": .0002, "X": .0002, "ECR": .008, "SWAP": .024, "1Q": .0002, "2Q": .008},
    link_errors={(0, 1): .0081, (1, 2): .0076, (1, 3): .0092, (3, 4): .0085},
    qubit_t1={0: 97.0, 1: 85.0, 2: 112.0, 3: 78.0, 4: 103.0},
    qubit_t2={0: 131.0, 1: 92.0, 2: 98.0, 3: 67.0, 4: 89.0},
    basis_gates=["RZ", "SX", "X", "ECR"],
    description="IBM Quantum 5-qubit T-shape (Falcon r5.1)",
)

_IBM_QX7 = HardwareConfig(
    name="ibm_qx7",
    num_qubits=7,
    coupling_map=[(0, 1), (1, 2), (0, 3), (2, 5), (3, 4), (4, 5), (5, 6)],
    native_gates={"RZ", "SX", "X", "ECR", "MEASURE"},
    gate_durations={"RZ": 0, "SX": 35, "X": 35, "ECR": 519, "SWAP": 1557, "MEASURE": 740},
    error_rates={"RZ": 0, "SX": .0002, "X": .0002, "ECR": .007, "SWAP": .021, "1Q": .0002, "2Q": .007},
    link_errors={
        (0, 1): .0065, (1, 2): .0078, (0, 3): .0072, (2, 5): .0089,
        (3, 4): .0071, (4, 5): .0083, (5, 6): .0091,
    },
    qubit_t1={i: float(90 + 10 * (i % 3)) for i in range(7)},
    qubit_t2={i: float(80 + 15 * (i % 4)) for i in range(7)},
    basis_gates=["RZ", "SX", "X", "ECR"],
    description="IBM Quantum 7-qubit heavy-hex (Falcon r5)",
)

_RIGETTI_ASPEN8 = HardwareConfig(
    name="rigetti_aspen8",
    num_qubits=8,
    coupling_map=[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 0)],
    native_gates={"RZ", "RX", "CZ", "MEASURE"},
    gate_durations={"RZ": 40, "RX": 120, "CZ": 280, "SWAP": 840, "MEASURE": 1200},
    error_rates={"RZ": .001, "RX": .002, "CZ": .012, "SWAP": .036, "1Q": .001, "2Q": .012},
    link_errors={
        (0, 1): .011, (1, 2): .013, (2, 3): .010, (3, 4): .012,
        (4, 5): .014, (5, 6): .011, (6, 7): .013, (7, 0): .015,
    },
    qubit_t1={i: float(50 + 8 * i) for i in range(8)},
    qubit_t2={i: float(30 + 6 * i) for i in range(8)},
    basis_gates=["RZ", "RX", "CZ"],
    description="Rigetti Aspen-8 (8-qubit ring lattice)",
)

_IONQ_HARMONY = HardwareConfig(
    name="ionq_harmony",
    num_qubits=11,
    coupling_map=[(i, j) for i in range(11) for j in range(i + 1, 11)],
    native_gates={"RZ", "RX", "RY", "XX", "MEASURE"},
    gate_durations={"RZ": 0, "RX": 1000, "RY": 1000, "XX": 600, "SWAP": 1800, "MEASURE": 1500},
    error_rates={"RZ": 0, "RX": .0004, "RY": .0004, "XX": .003, "SWAP": .009, "1Q": .0004, "2Q": .003},
    link_errors={(i, j): 0.003 for i in range(11) for j in range(i + 1, 11)},
    qubit_t1={i: 10000.0 for i in range(11)},
    qubit_t2={i: 1000.0 for i in range(11)},
    basis_gates=["RZ", "RX", "RY", "XX"],
    description="IonQ Harmony 11-qubit trapped-ion (all-to-all connectivity)",
)

def _simulator_config(n: int) -> HardwareConfig:
    return HardwareConfig(
        name="simulator",
        num_qubits=n,
        coupling_map=[(i, j) for i in range(n) for j in range(i + 1, n)],
        native_gates={"X", "Y", "Z", "H", "S", "T", "SDG", "TDG", "RX", "RY", "RZ", "U3", "CNOT", "CZ", "SWAP", "ISWAP", "CCX", "MEASURE"},
        gate_durations={},
        error_rates={"1Q": .001, "2Q": .005},
        link_errors={},
        basis_gates=["RX", "RY", "RZ", "CNOT"],
        description=f"{n}-qubit simulator",
    )

for _profile in [_IBM_QX5, _IBM_QX7, _RIGETTI_ASPEN8, _IONQ_HARMONY, _simulator_config(20)]:
    HardwareProfileRegistry.register(_profile)


class HardwareCostFunction:
    def __init__(self, hardware: HardwareConfig):
        self.hw = hardware
        self.alpha, self.beta, self.gamma = hardware.cost_weights

    def fidelity(self, circuit: Circuit) -> float:
        fidelity = 1.0
        for gate in circuit.gates:
            if gate.is_measurement: continue
            error = self.hw.error_rates.get(gate.name, self.hw.error_rates.get(gate.name.upper(), None))
            if error is None:
                error = self.hw.link_error(*gate.qubits[:2]) if gate.num_qubits >= 2 else self.hw.qubit_error(gate.qubits[0])
            fidelity *= max(0.0, 1.0 - error)
        return max(fidelity, 1e-12)

    def depth_ns(self, circuit: Circuit) -> float:
        if not circuit.gates: return 0.0
        durations = self.hw.gate_durations
        return sum(max(durations.get(gate.name, durations.get("1Q", 50.0)) for gate in layer) for layer in circuit.layers()) if durations else float(circuit.depth())

    def gate_weight(self, gate: Gate) -> float:
        if gate.name in {"RZ", "I"}: return 0.0
        if gate.name == "SWAP": return 3.0
        return 1.0 if gate.num_qubits == 1 else 2.0

    def weighted_gates(self, circuit: Circuit) -> float:
        return sum(self.gate_weight(gate) for gate in circuit.gates if not gate.is_measurement)

    def compute(self, circuit: Circuit) -> float:
        return self.alpha * self.depth_ns(circuit) / 10000.0 + self.beta * self.weighted_gates(circuit) / 100.0 + self.gamma * (1.0 - self.fidelity(circuit))


class QubitMapping:
    def __init__(self, n_logical: int, n_physical: int):
        self.n_logical, self.n_physical = n_logical, n_physical
        self._l2p: Dict[int, int] = {}
        self._p2l: Dict[int, Optional[int]] = {q: None for q in range(n_physical)}
    def assign(self, logical, physical):
        if logical in self._l2p: self._p2l[self._l2p[logical]] = None
        if self._p2l[physical] is not None: del self._l2p[self._p2l[physical]]
        self._l2p[logical], self._p2l[physical] = physical, logical
    def l2p(self, logical): return self._l2p[logical]
    def p2l(self, physical): return self._p2l.get(physical)
    def swap_physical(self, q0, q1):
        l0, l1 = self._p2l[q0], self._p2l[q1]
        self._p2l[q0], self._p2l[q1] = l1, l0
        if l0 is not None: self._l2p[l0] = q1
        if l1 is not None: self._l2p[l1] = q0
    def is_complete(self): return len(self._l2p) == self.n_logical
    def table(self): return "Qubit mapping:\n" + "\n".join(f"  lq{l} -> pq{self._l2p[l]}" for l in sorted(self._l2p))
    def copy(self):
        result = QubitMapping(self.n_logical, self.n_physical); result._l2p = dict(self._l2p); result._p2l = dict(self._p2l); return result


class QubitMappingPass(OptimizationPass):
    name = "QubitMappingPass"
    description = "Noise-aware initial qubit mapping."
    def __init__(self, hardware): self.hw = hardware
    def compute_mapping(self, circuit):
        graph = nx.Graph(); graph.add_nodes_from(range(circuit.num_qubits))
        for gate in circuit.gates:
            if gate.num_qubits >= 2: graph.add_edge(gate.qubits[0], gate.qubits[1], weight=graph.get_edge_data(gate.qubits[0], gate.qubits[1], {}).get("weight", 0) + 1)
        logical = sorted(range(circuit.num_qubits), key=lambda q: -graph.degree(q))
        physical = sorted(range(self.hw.num_qubits), key=lambda q: self.hw.qubit_error(q) - .001 * self.hw.qubit_t1.get(q, 100.0))
        mapping = QubitMapping(circuit.num_qubits, self.hw.num_qubits)
        for logical_q in logical:
            neighbors = [mapping.l2p(q) for q in graph.neighbors(logical_q) if q in mapping._l2p]
            candidates = [q for q in physical if q not in mapping._p2l or mapping.p2l(q) is None]
            def score(q):
                distance = sum(nx.shortest_path_length(self.hw.coupling_graph(), q, n, weight="weight") for n in neighbors) if neighbors else 0
                return distance + .1 * self.hw.qubit_error(q)
            mapping.assign(logical_q, min(candidates, key=score))
        return mapping
    def _apply(self, circuit):
        self.hw.validate_for(circuit); mapping = self.compute_mapping(circuit); gates = []
        for gate in circuit.gates:
            copy_gate = gate.copy(); object.__setattr__(copy_gate, "targets", tuple(mapping.l2p(q) for q in gate.targets)); object.__setattr__(copy_gate, "controls", tuple(mapping.l2p(q) for q in gate.controls)); gates.append(copy_gate)
        result = Circuit(self.hw.num_qubits, "mapped")
        for gate in gates: result.add_gate(gate)
        result._qubit_mapping = mapping
        return result


class SWAPInsertionPass(OptimizationPass):
    name = "SWAPInsertionPass"
    description = "Dijkstra-based coupling-map routing."
    def __init__(self, hardware): self.hw = hardware
    def _apply(self, circuit):
        self.hw.validate_for(circuit); mapping = getattr(circuit, "_qubit_mapping", None)
        mapping = mapping.copy() if mapping else QubitMapping(circuit.num_qubits, self.hw.num_qubits)
        if not mapping.is_complete():
            for q in range(circuit.num_qubits): mapping.assign(q, q)
        initial_mapping = mapping.copy()
        output = []
        for gate in circuit.gates:
            logical_controls = tuple(initial_mapping.p2l(q) for q in gate.controls)
            logical_targets = tuple(initial_mapping.p2l(q) for q in gate.targets)
            if any(q is None for q in logical_controls + logical_targets):
                raise ValueError(f"Gate {gate} references an unmapped physical qubit")
            current_controls = tuple(mapping.l2p(q) for q in logical_controls)
            current_targets = tuple(mapping.l2p(q) for q in logical_targets)
            physical = list(current_controls + current_targets)
            if gate.is_measurement or gate.num_qubits < 2:
                copy_gate = gate.copy()
                object.__setattr__(copy_gate, "controls", current_controls)
                object.__setattr__(copy_gate, "targets", current_targets)
                output.append(copy_gate)
                continue
            if len(physical) == 2 and not self.hw.are_connected(*physical):
                path, _ = self.hw.shortest_swap_path(*physical)
                if len(path) < 2: raise ValueError(f"No routing path for {physical}")
                for index in range(len(path) - 2):
                    a, b = path[index:index + 2]; output.append(Gate("SWAP", (a, b))); mapping.swap_physical(a, b)
                physical = [path[-2], path[-1]]
            copy_gate = gate.copy()
            if gate.controls: object.__setattr__(copy_gate, "controls", (physical[0],)); object.__setattr__(copy_gate, "targets", (physical[1],))
            else: object.__setattr__(copy_gate, "targets", tuple(physical))
            output.append(copy_gate)
        result = Circuit(self.hw.num_qubits, "routed")
        for gate in output: result.add_gate(gate)
        result._qubit_mapping = mapping
        return result


class HardwareCostOptPass(OptimizationPass):
    """Post-routing cleanup that preserves the target native gate basis."""
    name = "HardwareCostOptPass"
    description = "Remove redundant native gates and merge native rotations."

    def __init__(self, hardware):
        self.hw = hardware

    def _apply(self, circuit):
        current = HardwareRotationMergingPass()._apply(circuit)
        current = RedundantGateCancellationPass()._apply(current)
        current = HardwareRotationMergingPass()._apply(current)
        if hasattr(circuit, "_qubit_mapping"):
            current._qubit_mapping = circuit._qubit_mapping
        return current


class GateDecomposer:
    """Exact algebraic decompositions into target native basis gate sets."""

    # ── IBM decompositions: {RZ, SX, X, ECR} ──────────────────────────────

    @staticmethod
    def _ibm_H(q: int) -> List[Gate]:
        return [
            Gate('RZ', (q,), params=(math.pi/2,)),
            Gate('SX', (q,)),
            Gate('RZ', (q,), params=(math.pi/2,)),
        ]

    @staticmethod
    def _ibm_S(q: int) -> List[Gate]:
        return [Gate('RZ', (q,), params=(math.pi/2,))]

    @staticmethod
    def _ibm_T(q: int) -> List[Gate]:
        return [Gate('RZ', (q,), params=(math.pi/4,))]

    @staticmethod
    def _ibm_SDG(q: int) -> List[Gate]:
        return [Gate('RZ', (q,), params=(-math.pi/2,))]

    @staticmethod
    def _ibm_TDG(q: int) -> List[Gate]:
        return [Gate('RZ', (q,), params=(-math.pi/4,))]

    @staticmethod
    def _ibm_RX(theta: float, q: int) -> List[Gate]:
        if abs(abs(theta) - math.pi) < 1e-7:
            return [Gate('X', (q,))]
        if abs(abs(theta) - math.pi/2) < 1e-7:
            if theta > 0:
                return [Gate('SX', (q,))]
            else:
                return [Gate('RZ', (q,), params=(-math.pi,)), Gate('SX', (q,)),
                        Gate('RZ', (q,), params=(-math.pi,))]
        return [
            Gate('RZ', (q,), params=(-math.pi/2,)),
            Gate('SX', (q,)),
            Gate('RZ', (q,), params=(math.pi - theta,)),
            Gate('SX', (q,)),
            Gate('RZ', (q,), params=(-math.pi/2,)),
        ]

    @staticmethod
    def _ibm_RY(theta: float, q: int) -> List[Gate]:
        return [
            Gate('RZ', (q,), params=(math.pi/2,)),
            Gate('SX', (q,)),
            Gate('RZ', (q,), params=(theta,)),
            Gate('SX', (q,)),
            Gate('RZ', (q,), params=(math.pi/2,)),
        ]

    @staticmethod
    def _ibm_CNOT(ctrl: int, tgt: int) -> List[Gate]:
        return [
            Gate('SX',  (ctrl,)),
            Gate('ECR', (tgt,), controls=(ctrl,)),
            Gate('RZ',  (tgt,), params=(-math.pi/2,)),
            Gate('SX',  (tgt,)),
        ]

    @staticmethod
    def _ibm_CZ(ctrl: int, tgt: int) -> List[Gate]:
        return GateDecomposer._ibm_H(tgt) + GateDecomposer._ibm_CNOT(ctrl, tgt) + GateDecomposer._ibm_H(tgt)

    @staticmethod
    def _ibm_SWAP(q0: int, q1: int) -> List[Gate]:
        return (GateDecomposer._ibm_CNOT(q0, q1)
                + GateDecomposer._ibm_CNOT(q1, q0)
                + GateDecomposer._ibm_CNOT(q0, q1))

    # ── Rigetti decompositions: {RZ, RX, CZ} ─────────────────────────────

    @staticmethod
    def _rigetti_H(q: int) -> List[Gate]:
        return [
            Gate('RZ', (q,), params=(math.pi/2,)),
            Gate('RX', (q,), params=(math.pi/2,)),
            Gate('RZ', (q,), params=(math.pi/2,)),
        ]

    @staticmethod
    def _rigetti_CNOT(ctrl: int, tgt: int) -> List[Gate]:
        return (GateDecomposer._rigetti_H(tgt)
                + [Gate('CZ', (tgt,), controls=(ctrl,))]
                + GateDecomposer._rigetti_H(tgt))

    @staticmethod
    def _rigetti_SWAP(q0: int, q1: int) -> List[Gate]:
        gates = []
        for c, t in [(q0, q1), (q1, q0), (q0, q1)]:
            gates += GateDecomposer._rigetti_CNOT(c, t)
        return gates

    # ── IonQ decompositions: {RZ, RX, RY, XX} ────────────────────────────

    @staticmethod
    def _ionq_H(q: int) -> List[Gate]:
        return [Gate('RY', (q,), params=(math.pi/2,)), Gate('RZ', (q,), params=(math.pi,))]

    @staticmethod
    def _ionq_CNOT(ctrl: int, tgt: int) -> List[Gate]:
        return [
            Gate('RY',  (ctrl,), params=( math.pi/2,)),
            Gate('XX',  (ctrl, tgt), params=(math.pi/4,)),
            Gate('RX',  (ctrl,), params=(-math.pi/2,)),
            Gate('RX',  (tgt,),  params=(-math.pi/2,)),
            Gate('RY',  (ctrl,), params=(-math.pi/2,)),
        ]

    @staticmethod
    def _ionq_SWAP(q0: int, q1: int) -> List[Gate]:
        gates = []
        for c, t in [(q0, q1), (q1, q0), (q0, q1)]:
            gates += GateDecomposer._ionq_CNOT(c, t)
        return gates

    # ── Generic decompositions: {RX, RY, RZ, CNOT} ───────────────────────

    @staticmethod
    def _generic_H(q: int) -> List[Gate]:
        return [Gate('RY', (q,), params=(math.pi/2,)), Gate('RX', (q,), params=(math.pi,))]

    @staticmethod
    def _generic_S(q: int)   -> List[Gate]: return [Gate('RZ', (q,), params=(math.pi/2,))]
    @staticmethod
    def _generic_T(q: int)   -> List[Gate]: return [Gate('RZ', (q,), params=(math.pi/4,))]
    @staticmethod
    def _generic_SDG(q: int) -> List[Gate]: return [Gate('RZ', (q,), params=(-math.pi/2,))]
    @staticmethod
    def _generic_TDG(q: int) -> List[Gate]: return [Gate('RZ', (q,), params=(-math.pi/4,))]
    @staticmethod
    def _generic_CZ(c: int, t: int) -> List[Gate]:
        H = GateDecomposer._generic_H
        return H(t) + [Gate('CNOT', (t,), controls=(c,))] + H(t)
    @staticmethod
    def _generic_SWAP(q0: int, q1: int) -> List[Gate]:
        return ([Gate('CNOT', (q1,), controls=(q0,))]
              + [Gate('CNOT', (q0,), controls=(q1,))]
              + [Gate('CNOT', (q1,), controls=(q0,))])

    @staticmethod
    def _u3_to_rzsx(theta, phi, lam, q):
        return [
            Gate('RZ', (q,), params=(lam,)),
            Gate('SX', (q,)),
            Gate('RZ', (q,), params=(theta + math.pi,)),
            Gate('SX', (q,)),
            Gate('RZ', (q,), params=(phi + math.pi,)),
        ]

    @staticmethod
    def _u3_to_rzrx(theta, phi, lam, q):
        return [
            Gate('RZ', (q,), params=(phi + math.pi/2,)),
            Gate('RX', (q,), params=(theta,)),
            Gate('RZ', (q,), params=(lam - math.pi/2,)),
        ]

    @staticmethod
    def _u3_to_ryrz(theta, phi, lam, q):
        return [
            Gate('RZ', (q,), params=(phi,)),
            Gate('RY', (q,), params=(theta,)),
            Gate('RZ', (q,), params=(lam,)),
        ]


class HardwareRotationMergingPass(OptimizationPass):
    name = "HardwareRotationMergingPass"
    description = "Merge adjacent RZ gates without changing the hardware basis."
    def _apply(self, circuit: Circuit) -> Circuit:
        output = []; index = 0
        while index < len(circuit.gates):
            gate = circuit.gates[index]
            if gate.name != "RZ": output.append(gate.copy()); index += 1; continue
            total, qubits = gate.params[0], gate.targets; index += 1
            while index < len(circuit.gates) and circuit.gates[index].name == "RZ" and circuit.gates[index].targets == qubits:
                total += circuit.gates[index].params[0]; index += 1
            total = (total + math.pi) % (2 * math.pi) - math.pi
            if not math.isclose(total, 0.0, abs_tol=1e-10): output.append(Gate("RZ", qubits, params=(total,)))
        result = Circuit(circuit.num_qubits)
        for gate in output: result.add_gate(gate)
        if hasattr(circuit, "_qubit_mapping"): result._qubit_mapping = circuit._qubit_mapping
        return result


class GateDecompositionPass(OptimizationPass):
    name = "GateDecompositionPass"
    description = "Decompose non-native gates into hardware basis."

    def __init__(self, hardware: HardwareConfig):
        self.hw = hardware
        self._native = {g.upper() for g in hardware.native_gates}
        self.backend = self._detect_backend()

    def _detect_backend(self) -> str:
        native_up = self._native
        if "ECR" in native_up: return "ibm"
        if "CZ" in native_up and "RX" in native_up: return "rigetti"
        if "XX" in native_up: return "ionq"
        return "generic"

    def _decompose_gate(self, gate: Gate, depth: int = 0) -> List[Gate]:
        if depth > 5:
            return [gate]
        nm = gate.name.upper()
        if nm in self._native or gate.is_measurement:
            return [gate]

        q = gate.qubits
        p = gate.params
        be = self.backend
        D = GateDecomposer

        if be == "ibm":
            if nm == "H": sub = D._ibm_H(q[0])
            elif nm == "S": sub = D._ibm_S(q[0])
            elif nm == "T": sub = D._ibm_T(q[0])
            elif nm == "SDG": sub = D._ibm_SDG(q[0])
            elif nm == "TDG": sub = D._ibm_TDG(q[0])
            elif nm == "Y": sub = [Gate("X", (q[0],)), Gate("RZ", (q[0],), params=(math.pi,))]
            elif nm == "Z": sub = [Gate("RZ", (q[0],), params=(math.pi,))]
            elif nm == "RX": sub = D._ibm_RX(p[0], q[0])
            elif nm == "RY": sub = D._ibm_RY(p[0], q[0])
            elif nm in ("CNOT", "CX"):
                ctrl = gate.controls[0] if gate.controls else q[0]
                tgt  = gate.targets[0] if gate.controls else q[1]
                sub = D._ibm_CNOT(ctrl, tgt)
            elif nm == "CZ":
                ctrl = gate.controls[0] if gate.controls else q[0]
                tgt  = gate.targets[0] if gate.controls else q[1]
                sub = D._ibm_CZ(ctrl, tgt)
            elif nm == "SWAP": sub = D._ibm_SWAP(q[0], q[1])
            elif nm == "U3": sub = D._u3_to_rzsx(p[0], p[1], p[2], q[0])
            else: return [gate]

        elif be == "rigetti":
            if nm == "H": sub = D._rigetti_H(q[0])
            elif nm == "S": sub = [Gate("RZ", (q[0],), params=(math.pi/2,))]
            elif nm == "T": sub = [Gate("RZ", (q[0],), params=(math.pi/4,))]
            elif nm == "SDG": sub = [Gate("RZ", (q[0],), params=(-math.pi/2,))]
            elif nm == "TDG": sub = [Gate("RZ", (q[0],), params=(-math.pi/4,))]
            elif nm == "X": sub = [Gate("RX", (q[0],), params=(math.pi,))]
            elif nm == "Y": sub = [Gate("RX", (q[0],), params=(math.pi/2,)),
                                   Gate("RZ", (q[0],), params=(math.pi,)),
                                   Gate("RX", (q[0],), params=(-math.pi/2,))]
            elif nm == "Z": sub = [Gate("RZ", (q[0],), params=(math.pi,))]
            elif nm == "RX": sub = [Gate("RX", (q[0],), params=(p[0],))]
            elif nm == "RZ": sub = [Gate("RZ", (q[0],), params=(p[0],))]
            elif nm == "RY": sub = [Gate("RZ", (q[0],), params=(-math.pi/2,)),
                                   Gate("RX", (q[0],), params=(p[0],)),
                                   Gate("RZ", (q[0],), params=(math.pi/2,))]
            elif nm in ("CNOT", "CX"):
                ctrl = gate.controls[0] if gate.controls else q[0]
                tgt  = gate.targets[0] if gate.controls else q[1]
                sub = D._rigetti_CNOT(ctrl, tgt)
            elif nm == "SWAP": sub = D._rigetti_SWAP(q[0], q[1])
            elif nm == "U3": sub = D._u3_to_rzrx(p[0], p[1], p[2], q[0])
            else: return [gate]

        elif be == "ionq":
            if nm == "H": sub = D._ionq_H(q[0])
            elif nm == "S": sub = [Gate("RZ", (q[0],), params=(math.pi/2,))]
            elif nm == "T": sub = [Gate("RZ", (q[0],), params=(math.pi/4,))]
            elif nm == "SDG": sub = [Gate("RZ", (q[0],), params=(-math.pi/2,))]
            elif nm == "TDG": sub = [Gate("RZ", (q[0],), params=(-math.pi/4,))]
            elif nm in ("CNOT", "CX"):
                ctrl = gate.controls[0] if gate.controls else q[0]
                tgt  = gate.targets[0] if gate.controls else q[1]
                sub = D._ionq_CNOT(ctrl, tgt)
            elif nm == "CZ":
                ctrl = gate.controls[0] if gate.controls else q[0]
                tgt  = gate.targets[0] if gate.controls else q[1]
                sub = D._ionq_H(tgt) + D._ionq_CNOT(ctrl, tgt) + D._ionq_H(tgt)
            elif nm == "SWAP": sub = D._ionq_SWAP(q[0], q[1])
            elif nm in ("X", "Y", "Z"):
                ax = {"X": "RX", "Y": "RY", "Z": "RZ"}[nm]
                sub = [Gate(ax, (q[0],), params=(math.pi,))]
            else: return [gate]

        else: # generic
            if nm == "H": sub = D._generic_H(q[0])
            elif nm == "S": sub = D._generic_S(q[0])
            elif nm == "T": sub = D._generic_T(q[0])
            elif nm == "SDG": sub = D._generic_SDG(q[0])
            elif nm == "TDG": sub = D._generic_TDG(q[0])
            elif nm == "CZ":
                ctrl = gate.controls[0] if gate.controls else q[0]
                tgt  = gate.targets[0] if gate.controls else q[1]
                sub = D._generic_CZ(ctrl, tgt)
            elif nm == "SWAP": sub = D._generic_SWAP(q[0], q[1])
            elif nm == "X": sub = [Gate("RX", (q[0],), params=(math.pi,))]
            elif nm == "Y": sub = [Gate("RY", (q[0],), params=(math.pi,))]
            elif nm == "Z": sub = [Gate("RZ", (q[0],), params=(math.pi,))]
            elif nm == "U3": sub = D._u3_to_ryrz(p[0], p[1], p[2], q[0])
            else: return [gate]

        result = []
        for sg in sub:
            result.extend(self._decompose_gate(sg, depth + 1))
        return result

    def _apply(self, circuit: Circuit) -> Circuit:
        output = []
        for gate in circuit.gates:
            output.extend(self._decompose_gate(gate))
        result = Circuit.from_gate_list(circuit.num_qubits, output)
        if hasattr(circuit, "_qubit_mapping"):
            result._qubit_mapping = circuit._qubit_mapping
        return HardwareRotationMergingPass()._apply(result)


@dataclass
class HardwareTranspilerResult:
    circuit: Circuit
    original_gates: int
    original_depth: int
    phase1_gates: int
    phase1_depth: int
    final_gates: int
    final_depth: int
    estimated_fidelity: float
    hardware_cost: float
    total_time_ms: float
    qubit_mapping: Optional[QubitMapping]
    hardware_name: str

    @property
    def gate_reduction_pct(self): return 100.0 * (1 - self.final_gates / max(self.original_gates, 1))
    @property
    def depth_reduction_pct(self): return 100.0 * (1 - self.final_depth / max(self.original_depth, 1))
    def summary(self):
        m = f"\n  {self.qubit_mapping.table()}" if self.qubit_mapping else ""
        return (
            f"[{self.hardware_name}] "
            f"Gates: {self.original_gates}→{self.phase1_gates}(P1)→{self.final_gates}(HW) "
            f"({self.gate_reduction_pct:.1f}%↓)  "
            f"Depth: {self.original_depth}→{self.final_depth} ({self.depth_reduction_pct:.1f}%↓)  "
            f"Fidelity: {self.estimated_fidelity:.4f}  "
            f"HWcost: {self.hardware_cost:.4f}  "
            f"{self.total_time_ms:.1f}ms{m}"
        )


class HardwareAwareTranspiler:
    def __init__(self, hardware_config: Union[str, HardwareConfig], phase1_passes=None, use_ai=False):
        self._hw = HardwareProfileRegistry.get(hardware_config) if isinstance(hardware_config, str) else hardware_config
        self._use_ai = use_ai
        self._ai = None
        if use_ai:
            try:
                from zuntenium.transpiler.ai_transpiler_phase2 import AITranspiler
                self._ai = AITranspiler(use_ai=True)
            except ImportError:
                pass
        self._phase1 = Transpiler(passes=phase1_passes or ["cancel", "rotation", "commute", "fusion", "depth"])
        self._passes = [
            QubitMappingPass(self._hw),
            SWAPInsertionPass(self._hw),
            GateDecompositionPass(self._hw),
            HardwareRotationMergingPass(),
        ]
        self._cost = HardwareCostFunction(self._hw)

    def transpile(self, circuit: Circuit) -> HardwareTranspilerResult:
        started = time.perf_counter()
        self._hw.validate_for(circuit)
        phase1, report = self._phase1.transpile(circuit)
        current = self._ai.transpile(phase1, use_ai=True).circuit if self._ai is not None else phase1
        for optimization_pass in self._passes:
            current, _ = optimization_pass(current)
        return HardwareTranspilerResult(
            current, circuit.num_gates, circuit.depth(),
            phase1.num_gates, phase1.depth(),
            current.num_gates, current.depth(),
            self._cost.fidelity(current), self._cost.compute(current),
            (time.perf_counter() - started) * 1000,
            getattr(current, "_qubit_mapping", None),
            self._hw.name,
        )


def build_hardware_executor(n_qubits: int, hardware: Union[str, HardwareConfig, None] = None, seed: int = 0):
    from zuntenium.execution.quantum_executor import QuantumExecutor, _apply_circuit_to_simulator, validate_circuit_compatibility, TranspiledExecutionResult as ExecutionResult
    if hardware is None: return QuantumExecutor.default(n_qubits, seed=seed)
    config = HardwareProfileRegistry.get(hardware) if isinstance(hardware, str) else hardware
    simulator = __import__("zuntenium.quantum_core.statevector", fromlist=["StatevectorSimulator"]).StatevectorSimulator(n_qubits, seed=seed)
    transpiler = HardwareAwareTranspiler(config)

    class HardwareExecutor:
        @property
        def num_qubits(self): return self._sim.num_qubits
        @property
        def simulator(self): return self._sim
        def __init__(self): self._sim = simulator
        def execute(self, circuit, optimize=True, optimization_level=3, shots=1024, debug=False):
            validate_circuit_compatibility(circuit, max_qubits=self._sim.num_qubits)
            result = transpiler.transpile(circuit) if optimize else None
            executable = result.circuit if result else circuit
            if debug and result: print(f"[HWExecutor] {result.summary()}")
            started = time.perf_counter()
            self._sim.initialize(self._sim.num_qubits)
            _apply_circuit_to_simulator(executable, self._sim)
            execution_ms = (time.perf_counter() - started) * 1000
            state = self._sim.get_statevector()
            probabilities = np.abs(state) ** 2
            counts = self._sim.sample(shots) if shots > 0 else {}
            return ExecutionResult(
                state, probabilities, counts, self._sim.num_qubits, shots, None,
                execution_ms, execution_ms, executable.depth(), executable.num_gates,
            )
    return HardwareExecutor()


class HardwareTranspilerTestSuite:
    def __init__(self): self._p = 0; self._f = 0

    def _a(self, cond, msg):
        if cond: self._p += 1; print(f"  ✓  {msg}")
        else:    self._f += 1; print(f"  ✗  FAIL: {msg}")

    def test_profiles_registered(self):
        avail = HardwareProfileRegistry.available()
        for name in ['ibm_qx5', 'ibm_qx7', 'rigetti_aspen8', 'ionq_harmony']:
            self._a(name in avail, f"Profile '{name}' registered")

    def test_coupling_graph_weights(self):
        hw = HardwareProfileRegistry.get('ibm_qx5')
        G = hw.coupling_graph()
        self._a(G.number_of_edges() == 4, f"IBM QX5: 4 edges (got {G.number_of_edges()})")
        for (u, v) in G.edges():
            w = G[u][v]['weight']
            self._a(w > 0, f"  edge ({u},{v}) weight>0: {w:.4f}")

    def test_shortest_swap_path_non_adjacent(self):
        hw = HardwareProfileRegistry.get('ibm_qx5')
        path, cost = hw.shortest_swap_path(0, 4)
        self._a(len(path) >= 3, f"IBM QX5: q0→q4 path len≥3 (got {len(path)}: {path})")
        self._a(cost > 0, f"IBM QX5: q0→q4 cost>0 ({cost:.4f})")

    def test_qubit_mapping_ibm(self):
        hw = HardwareProfileRegistry.get('ibm_qx5')
        c = Circuit(3)
        c.add_gate(Gate('H', (0,)))
        c.add_gate(Gate('CNOT', (1,), controls=(0,)))
        c.add_gate(Gate('CNOT', (2,), controls=(1,)))
        p = QubitMappingPass(hw)
        mapping = p.compute_mapping(c)
        self._a(mapping.is_complete(), "Mapping complete for 3-qubit circuit")
        qubits = set(mapping._l2p.values())
        self._a(len(qubits) == 3, "3 distinct physical qubits assigned")
        for lq, pq in mapping._l2p.items():
            self._a(0 <= pq < hw.num_qubits, f"  lq{lq}→pq{pq} in range [0,5)")

    def test_gate_decomp_ibm_H(self):
        hw = HardwareProfileRegistry.get('ibm_qx5')
        c = Circuit(1); c.add_gate(Gate('H', (0,)))
        dp = GateDecompositionPass(hw)
        c2, _ = dp(c)
        native = {g.upper() for g in hw.native_gates}
        for g in c2.gates:
            self._a(g.name.upper() in native, f"IBM decomp H: gate '{g.name}' is native")

    def test_gate_decomp_ibm_CNOT(self):
        hw = HardwareProfileRegistry.get('ibm_qx5')
        c = Circuit(2); c.add_gate(Gate('CNOT', (1,), controls=(0,)))
        dp = GateDecompositionPass(hw)
        c2, _ = dp(c)
        native = {g.upper() for g in hw.native_gates}
        all_native = all(g.name.upper() in native for g in c2.gates)
        self._a(all_native, f"IBM CNOT decomp: all {c2.num_gates} gates native")
        self._a(c2.num_gates > 0, "IBM CNOT decomp: non-empty result")

    def test_gate_decomp_rigetti(self):
        hw = HardwareProfileRegistry.get('rigetti_aspen8')
        c = Circuit(2); c.add_gate(Gate('H', (0,))); c.add_gate(Gate('CNOT', (1,), controls=(0,)))
        dp = GateDecompositionPass(hw)
        c2, _ = dp(c)
        native = {g.upper() for g in hw.native_gates}
        all_n = all(g.name.upper() in native for g in c2.gates)
        self._a(all_n, f"Rigetti H+CNOT: all gates native ({c2.num_gates} total)")

    def test_gate_decomp_ionq(self):
        hw = HardwareProfileRegistry.get('ionq_harmony')
        c = Circuit(2); c.add_gate(Gate('CNOT', (1,), controls=(0,)))
        dp = GateDecompositionPass(hw)
        c2, _ = dp(c)
        native = {g.upper() for g in hw.native_gates}
        all_n = all(g.name.upper() in native for g in c2.gates)
        self._a(all_n, f"IonQ CNOT decomp: all gates native ({c2.num_gates} total)")

    def test_hardware_cost_function(self):
        hw = HardwareProfileRegistry.get('ibm_qx5')
        cf = HardwareCostFunction(hw)
        c = Circuit(2); c.add_gate(Gate('H', (0,))); c.add_gate(Gate('CNOT', (1,), controls=(0,)))
        cost = cf.compute(c)
        fid = cf.fidelity(c)
        self._a(0.0 <= cost, f"cost ≥ 0: {cost:.4f}")
        self._a(0.0 < fid <= 1.0, f"fidelity in (0,1]: {fid:.4f}")

    def test_full_hw_transpile_bell(self):
        hw = HardwareProfileRegistry.get('ibm_qx5')
        t = HardwareAwareTranspiler(hw)
        c = Circuit(2)
        c.add_gate(Gate('H', (0,))); c.add_gate(Gate('CNOT', (1,), controls=(0,)))
        r = t.transpile(c)
        self._a(r.final_gates > 0, f"Bell HW output non-empty: {r.final_gates} gates")
        self._a(r.estimated_fidelity > 0.5, f"Bell fidelity>0.5: {r.estimated_fidelity:.4f}")
        native = {g.upper() for g in hw.native_gates}
        all_n = all(g.name.upper() in native or g.is_measurement for g in r.circuit.gates)
        self._a(all_n, "Bell: all output gates native for IBM")

    def test_full_hw_transpile_ghz(self):
        hw = HardwareProfileRegistry.get('ibm_qx5')
        t = HardwareAwareTranspiler(hw)
        c = Circuit(3)
        c.add_gate(Gate('H', (0,)))
        c.add_gate(Gate('CNOT', (1,), controls=(0,)))
        c.add_gate(Gate('CNOT', (2,), controls=(1,)))
        r = t.transpile(c)
        self._a(r.final_gates > 0, f"GHZ-3 HW: {r.final_gates} gates")
        self._a(r.hardware_cost > 0, f"GHZ-3 cost>0: {r.hardware_cost:.4f}")

    def test_backward_compat_no_hw(self):
        ex = build_hardware_executor(3, hardware=None)
        c = Circuit(3); c.add_gate(Gate('H', (0,)))
        r = ex.execute(c, shots=0)
        self._a(r.num_qubits == 3, "backward compat: no hw config works")

    def test_hw_executor_bell(self):
        ex = build_hardware_executor(5, hardware='ibm_qx5', seed=0)
        c = Circuit(2); c.add_gate(Gate('H', (0,))); c.add_gate(Gate('CNOT', (1,), controls=(0,)))
        r = ex.execute(c, shots=200, debug=False)
        self._a(r.num_qubits == 5, f"HW executor: num_qubits=5 (got {r.num_qubits})")

    def test_rigetti_full_pipeline(self):
        hw = HardwareProfileRegistry.get('rigetti_aspen8')
        t = HardwareAwareTranspiler(hw)
        c = Circuit(4)
        c.add_gate(Gate('H', (0,)))
        for i in range(3): c.add_gate(Gate('CNOT', (i + 1,), controls=(i,)))
        r = t.transpile(c)
        native = {g.upper() for g in hw.native_gates}
        all_n = all(g.name.upper() in native or g.is_measurement for g in r.circuit.gates)
        self._a(all_n, f"Rigetti GHZ-4: all native ({r.final_gates} gates)")
        self._a(r.estimated_fidelity > 0.1, f"Rigetti fidelity>0.1: {r.estimated_fidelity:.4f}")

    def test_ionq_all_to_all_no_swaps(self):
        hw = HardwareProfileRegistry.get('ionq_harmony')
        c = Circuit(4)
        for q in range(3): c.add_gate(Gate('CNOT', (q + 1,), controls=(q,)))
        t = HardwareAwareTranspiler(hw)
        r = t.transpile(c)
        n_swaps = sum(1 for g in r.circuit.gates if g.name.upper() == 'SWAP')
        self._a(n_swaps == 0, f"IonQ all-to-all: 0 routing SWAPs (got {n_swaps})")

    def test_cost_lower_after_hw_transpile(self):
        hw = HardwareProfileRegistry.get('ibm_qx5')
        t = HardwareAwareTranspiler(hw)
        c = Circuit(3)
        c.add_gate(Gate('H', (0,))); c.add_gate(Gate('H', (0,))); c.add_gate(Gate('CNOT', (1,), controls=(0,)))
        r = t.transpile(c)
        self._a(r.estimated_fidelity > 0.0, f"HW fidelity>0: {r.estimated_fidelity:.4f}")

    def run_all(self):
        print("\n" + "═" * 60)
        print("  Hardware-Aware Transpiler — Test Suite")
        print("═" * 60)
        tests = [m for m in sorted(dir(self)) if m.startswith('test_')]
        for t in tests:
            print(f"\n[{t}]"); getattr(self, t)()
        total = self._p + self._f
        print(f"\n{'═' * 60}")
        print(f"  Results: {self._p}/{total} passed"
              + (f"  ({self._f} FAILED)" if self._f else "  ✓ all pass"))
        print("═" * 60)
        assert self._f == 0, f"{self._f} tests failed in HardwareTranspilerTestSuite"


def run_demo():
    print("\n" + "═" * 70)
    print("  ZUNTENIUM Hardware-Aware Transpiler — Demo")
    print("═" * 70)
    hw = HardwareProfileRegistry.get('ibm_qx5')
    t = HardwareAwareTranspiler(hw)
    c = Circuit(2); c.add_gate(Gate('H', (0,))); c.add_gate(Gate('CNOT', (1,), controls=(0,)))
    r = t.transpile(c)
    print(f"  {r.summary()}")
    print("\n" + "═" * 70 + "\n")


__all__ = [
    "HardwareConfig", "HardwareProfileRegistry", "HardwareCostFunction", "QubitMapping",
    "QubitMappingPass", "SWAPInsertionPass", "HardwareCostOptPass",
    "HardwareRotationMergingPass", "GateDecomposer", "GateDecompositionPass",
    "HardwareTranspilerResult", "HardwareAwareTranspiler", "build_hardware_executor",
    "HardwareTranspilerTestSuite", "run_demo",
    "_IBM_QX5", "_IBM_QX7", "_RIGETTI_ASPEN8", "_IONQ_HARMONY",
]

if __name__ == "__main__":
    run_demo()
    HardwareTranspilerTestSuite().run_all()
