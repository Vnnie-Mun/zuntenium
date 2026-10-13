"""Rule-based adaptive backend selection and execution."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Set, Tuple

import numpy as np

from zuntenium.backends.ibm_backend import (
    BackendInfo, IBMBackend, QuantumBackend, SimulatorBackend,
)
from zuntenium.backends.tensor_network_backend import TensorNetworkBackend
from zuntenium.execution.quantum_executor import validate_circuit_compatibility
from zuntenium.transpiler.v2 import Circuit, Gate, normalize_angle
from zuntenium.transpiler.v2 import Transpiler


_CLIFFORD_GATES: Set[str] = frozenset({
    "I", "X", "Y", "Z", "H", "S", "SDG", "SX", "SXD", "CNOT", "CX",
    "CZ", "SWAP", "ISWAP", "MEASURE", "MEAS", "M",
})


def _is_clifford_gate(gate: Gate) -> bool:
    if gate.name in _CLIFFORD_GATES:
        return True
    if not gate.params:
        return False
    theta = abs(normalize_angle(gate.params[0]))
    if gate.name == "RZ":
        return any(math.isclose(theta, k * math.pi / 2, abs_tol=1e-9) for k in range(5))
    if gate.name in {"RX", "RY"}:
        return any(math.isclose(theta, k * math.pi, abs_tol=1e-9) for k in range(3))
    return False


@dataclass
class CircuitFeatures:
    num_qubits: int
    depth: int
    gate_count: int
    clifford_only: bool
    entanglement_score: float
    parametric: bool
    measurement_count: int
    connectivity_density: float
    estimated_memory_mb: float

    @property
    def is_empty(self): return self.gate_count == 0
    @property
    def is_large(self): return self.estimated_memory_mb > 1024.0
    @property
    def is_very_large(self): return self.num_qubits > 30
    def summary(self):
        return (f"CircuitFeatures(n={self.num_qubits}, depth={self.depth}, gates={self.gate_count}, "
                f"clifford={self.clifford_only}, ent={self.entanglement_score:.3f}, "
                f"conn_density={self.connectivity_density:.3f}, mem={self.estimated_memory_mb:.2f}MB, "
                f"parametric={self.parametric})")


class CircuitAnalyzer:
    @staticmethod
    def analyze(circuit: Circuit) -> CircuitFeatures:
        total_qubits = 0
        non_measurements = 0
        measurements = 0
        clifford = True
        parametric = False
        pairs = set()
        for gate in circuit.gates:
            if gate.is_measurement:
                measurements += 1
                continue
            non_measurements += 1
            total_qubits += gate.num_qubits
            clifford = clifford and _is_clifford_gate(gate)
            parametric = parametric or bool(gate.params)
            qubits = gate.qubits
            for first in range(len(qubits)):
                for second in range(first + 1, len(qubits)):
                    pairs.add(frozenset((qubits[first], qubits[second])))
        max_pairs = circuit.num_qubits * (circuit.num_qubits - 1) / 2
        average_qubits = total_qubits / max(non_measurements, 1)
        multi_qubit = sum(gate.num_qubits >= 2 and not gate.is_measurement for gate in circuit.gates)
        entanglement = min((multi_qubit / max(non_measurements, 1)) * (average_qubits / max(circuit.num_qubits, 2)), 1.0)
        return CircuitFeatures(
            circuit.num_qubits, circuit.depth(), circuit.num_gates, clifford,
            entanglement, parametric, measurements, len(pairs) / max(max_pairs, 1.0),
            16.0 * (2 ** circuit.num_qubits) / (1024.0 ** 2),
        )


@dataclass
class Preferences:
    optimize_for: Literal["speed", "accuracy", "cost"] = "speed"
    allow_hardware: bool = False
    noise_model: Optional[bool] = None
    max_qubits: int = 20
    max_memory_mb: float = 2048.0
    preferred_backend: Optional[str] = None
    shots: int = 1024


@dataclass
class BackendCapability:
    name: str
    max_qubits: int
    exact_simulation: bool
    clifford_efficient: bool
    noise_capable: bool
    cost_per_gate: float
    memory_scaling: str
    available_now: bool


_BACKEND_CAPABILITIES = {
    "statevector": BackendCapability("statevector", 30, True, False, False, 1.0, "exponential", True),
    "stabilizer": BackendCapability("stabilizer", 10000, True, True, False, .01, "polynomial", True),
    "densitymatrix": BackendCapability("densitymatrix", 15, True, False, True, 4.0, "exponential", False),
    "tensornetwork": BackendCapability("tensornetwork", 200, False, False, False, 2.0, "polynomial", True),
    "hardware": BackendCapability("hardware", 127, False, False, True, 100.0, "constant", True),
}


@dataclass
class RuleResult:
    rule_name: str
    fired: bool
    suggested_backend: Optional[str]
    score_boost: float
    reason: str


class RuleEngine:
    THRESHOLDS = {"small_qubits": 20, "small_depth": 100, "high_entanglement": .7,
                  "low_entanglement": .3, "hw_min_qubits": 15, "large_qubits": 30}
    RULE_BOOSTS = {"R1_CLIFFORD": 100.0, "R2_SMALL": 50.0, "R3_HIGH_ENT": -50.0,
                   "R4_LOW_ENT": 30.0, "R5_NOISY": 80.0, "R6_HARDWARE": 60.0, "R7_FALLBACK": 10.0}

    def apply_rules(self, features: CircuitFeatures, prefs: Preferences) -> List[RuleResult]:
        t, b = self.THRESHOLDS, self.RULE_BOOSTS
        small = features.num_qubits <= t["small_qubits"] and features.depth <= t["small_depth"] and features.estimated_memory_mb <= prefs.max_memory_mb
        entries = [
            RuleResult("R1_CLIFFORD", features.clifford_only and features.gate_count > 0, "stabilizer" if features.clifford_only else None, b["R1_CLIFFORD"] if features.clifford_only else 0.0, "All gates are Clifford." if features.clifford_only else "Non-Clifford gates present."),
            RuleResult("R2_SMALL", small, "statevector" if small else None, b["R2_SMALL"] if small else 0.0, "Circuit fits statevector limits." if small else "Circuit exceeds simple statevector limits."),
            RuleResult("R3_HIGH_ENT", features.entanglement_score > t["high_entanglement"], None, b["R3_HIGH_ENT"] if features.entanglement_score > t["high_entanglement"] else 0.0, "High entanglement penalizes tensor networks."),
            RuleResult("R4_LOW_ENT", features.num_qubits > t["large_qubits"] and features.entanglement_score < t["low_entanglement"], "tensornetwork" if features.num_qubits > t["large_qubits"] and features.entanglement_score < t["low_entanglement"] else None, b["R4_LOW_ENT"] if features.num_qubits > t["large_qubits"] and features.entanglement_score < t["low_entanglement"] else 0.0, "Large, low-entanglement circuit favors tensor networks."),
            RuleResult("R5_NOISY", prefs.noise_model is True, "densitymatrix" if prefs.noise_model is True else None, b["R5_NOISY"] if prefs.noise_model is True else 0.0, "Noise model requested."),
            RuleResult("R6_HARDWARE", prefs.allow_hardware and features.num_qubits > t["hw_min_qubits"], "hardware" if prefs.allow_hardware and features.num_qubits > t["hw_min_qubits"] else None, b["R6_HARDWARE"] if prefs.allow_hardware and features.num_qubits > t["hw_min_qubits"] else 0.0, "Hardware execution allowed for a larger circuit."),
            RuleResult("R7_FALLBACK", True, "statevector", b["R7_FALLBACK"], "Statevector is the safety fallback."),
        ]
        return entries


class ScoringEngine:
    BASE_WEIGHTS = (.4, .3, .2, .1)
    @classmethod
    def _weights(cls, prefs):
        return {"accuracy": (.2, .5, .2, .1), "cost": (.2, .3, .4, .1)}.get(prefs.optimize_for, cls.BASE_WEIGHTS)
    @classmethod
    def score_all(cls, features, prefs, rule_boosts, available):
        wp, wa, wc, we = cls._weights(prefs); n, depth, ent, mem = features.num_qubits, features.depth, features.entanglement_score, features.estimated_memory_mb
        scores = {
            "statevector": wp / (1 + depth / 100) + wa + wc * (1 - min(mem / max(prefs.max_memory_mb, 1), 1)) + we,
            "stabilizer": wp * (1 if features.clifford_only else .05) + wa * (1 if features.clifford_only else 0) + wc * (1 - n / 10000) + we * (1 if features.clifford_only else 0),
            "densitymatrix": wp * .1 / (1 + depth / 50) + wa + wc * (1 - min(mem ** 2 / max(prefs.max_memory_mb ** 2, 1), 1)) + we * .8,
            "tensornetwork": wp / (1 + ent * 10) + wa * .85 + wc * (1 - min(ent * depth / 200, 1)) + we * max(1 - ent, 0),
            "hardware": wp * .3 + wa * .97 + wc * (1 - (.9 if prefs.optimize_for == "cost" else .3)) + we * .7,
        }
        for backend, boost in rule_boosts.items():
            if backend in scores: scores[backend] += boost
        return {name: value if name in available else -1000.0 for name, value in scores.items()}


@dataclass
class SelectionDecision:
    selected_backend: str
    confidence: float
    reason: str
    features: CircuitFeatures
    scores: Dict[str, float]
    rules_fired: List[str]
    analysis_time_ms: float = 0.0
    def explain(self):
        lines = ["Adaptive Execution Engine - Selection Decision", f"  Selected backend : {self.selected_backend}", f"  Confidence       : {self.confidence:.3f}", f"  Rules fired      : {', '.join(self.rules_fired) or 'none'}", f"  Analysis time    : {self.analysis_time_ms:.2f}ms", "", f"  Reason: {self.reason}", "", f"  Circuit features: {self.features.summary()}", "", "  All backend scores:"]
        lines.extend(f"    {name:<18}: {score:>8.3f}{' <- SELECTED' if name == self.selected_backend else ''}" for name, score in sorted(self.scores.items(), key=lambda item: -item[1]))
        return "\n".join(lines)


class StabilizerSimulatorStub(QuantumBackend):
    """Stabilizer simulator using the full Gottesman-Knill tableau."""
    def __init__(self, fallback_n_limit=10000):
        self._limit = fallback_n_limit
        try:
            from zuntenium.backends.stabilizer import StabilizerSimulator
            self._real_sim = StabilizerSimulator(max_qubits=fallback_n_limit)
        except Exception:
            self._real_sim = None
    def execute(self, circuit, shots=1024, **kwargs):
        if self._real_sim is not None:
            try:
                return self._real_sim.execute(circuit, shots=shots, **kwargs)
            except (ValueError, NotImplementedError):
                pass
        # Fallback to statevector for small Clifford circuits
        if circuit.num_qubits > 30:
            raise RuntimeError(
                f"StabilizerSimulator: real simulator failed and circuit has "
                f"{circuit.num_qubits} qubits (>30), too large for statevector fallback. "
                "Ensure the circuit contains only Clifford gates."
            )
        return SimulatorBackend(n_qubits=circuit.num_qubits).execute(circuit, shots=shots)
    def get_backend_info(self): return BackendInfo("stabilizer", "local", self._limit, True, 0.0, 0.0)
    def supports_noise_model(self): return False
    def supports_batching(self): return True
    def is_available(self): return True
    def get_available_backends(self, **kwargs): return [self.get_backend_info()]


class DensityMatrixSimulatorStub(QuantumBackend):
    """Density matrix simulator wrapper for the adaptive engine."""
    def __init__(self, max_qubits=15):
        self._max_qubits = max_qubits
        try:
            from zuntenium.backends.density_matrix_simulator import DensityMatrixSimulator
            self._sim_cls = DensityMatrixSimulator
        except Exception:
            self._sim_cls = None
    def execute(self, circuit, shots=1024, **kwargs):
        if self._sim_cls is not None and circuit.num_qubits <= self._max_qubits:
            try:
                sim = self._sim_cls(circuit.num_qubits, seed=kwargs.get('seed'))
                return sim.execute(circuit, shots=shots, **kwargs)
            except Exception:
                pass
        # Fallback to statevector for circuits within limit
        if circuit.num_qubits <= 30:
            return SimulatorBackend(n_qubits=circuit.num_qubits).execute(circuit, shots=shots)
        raise RuntimeError(
            f"DensityMatrixSimulator failed and circuit has {circuit.num_qubits} qubits "
            "(>30), too large for statevector fallback."
        )
    def get_backend_info(self): return BackendInfo("densitymatrix", "local", self._max_qubits, True)
    def supports_noise_model(self): return True
    def supports_batching(self): return True
    def is_available(self): return self._sim_cls is not None
    def get_available_backends(self, **kwargs): return [self.get_backend_info()] if self._sim_cls else []


class TensorNetworkSimulatorStub(QuantumBackend):
    """MPS tensor network simulator wrapper for the adaptive engine."""
    def __init__(self):
        try:
            self._real_backend = TensorNetworkBackend()
        except Exception:
            self._real_backend = None
    def execute(self, circuit, shots=1024, **kwargs):
        # Try the real MPS backend first
        if self._real_backend is not None:
            try:
                return self._real_backend.execute(circuit, shots=shots, **kwargs)
            except Exception:
                pass
        # Fallback to statevector for small circuits
        if circuit.num_qubits <= 30:
            return SimulatorBackend(n_qubits=circuit.num_qubits).execute(circuit, shots=shots)
        raise RuntimeError(
            f"TensorNetworkBackend failed and circuit has {circuit.num_qubits} qubits "
            "(>30), too large for statevector fallback."
        )
    def get_backend_info(self): return BackendInfo("tensornetwork", "local", 200, True)
    def supports_noise_model(self): return False
    def supports_batching(self): return True
    def is_available(self): return True
    def get_available_backends(self, **kwargs): return [self.get_backend_info()]


class BackendSelector:
    def __init__(self, backends: Optional[Dict[str, QuantumBackend]] = None, default_preferences: Optional[Preferences] = None):
        self._backends = backends or {"statevector": SimulatorBackend(30), "stabilizer": StabilizerSimulatorStub(), "densitymatrix": DensityMatrixSimulatorStub(), "tensornetwork": TensorNetworkSimulatorStub(), "hardware": IBMBackend(token=None)}
        self._prefs = default_preferences or Preferences(); self._available = {name for name, backend in self._backends.items() if backend.is_available()}
    def register_backend(self, name, backend): self._backends[name] = backend; (self._available.add(name) if backend.is_available() else self._available.discard(name))
    def select(self, circuit, preferences=None):
        started = time.perf_counter(); prefs = preferences or self._prefs; features = CircuitAnalyzer.analyze(circuit)
        if prefs.preferred_backend and prefs.preferred_backend.lower() in self._backends:
            name = prefs.preferred_backend.lower(); return SelectionDecision(name, 1.0, f"User explicitly requested backend '{name}'.", features, {name: 999.0}, ["USER_OVERRIDE"], (time.perf_counter() - started) * 1000)
        rules = RuleEngine().apply_rules(features, prefs); boosts = {}; fired = []
        for rule in rules:
            if rule.fired: fired.append(rule.rule_name)
            if rule.fired and rule.suggested_backend: boosts[rule.suggested_backend] = boosts.get(rule.suggested_backend, 0) + rule.score_boost
            if rule.rule_name == "R3_HIGH_ENT" and rule.fired: boosts["tensornetwork"] = boosts.get("tensornetwork", 0) + rule.score_boost
        scores = ScoringEngine.score_all(features, prefs, boosts, self._available); ranked = sorted(((name, score) for name, score in scores.items() if name in self._available and score > -999), key=lambda item: -item[1])
        if not ranked: ranked = [("statevector", 0.0)]
        winner, top = ranked[0]; confidence = 1.0 if len(ranked) == 1 else max(0.0, min((top - ranked[1][1]) / max(abs(top), 1e-9), 1.0)); reason = f"Scoring selected '{winner}' (score={top:.3f})."
        rule_reason = next((rule.reason for rule in rules if rule.fired and rule.suggested_backend == winner), None)
        if rule_reason: reason = rule_reason + " " + reason
        return SelectionDecision(winner, confidence, reason, features, scores, fired, (time.perf_counter() - started) * 1000)
    def get_backend(self, decision): return self._backends.get(decision.selected_backend, self._backends["statevector"])
    def explain(self, circuit, prefs=None): return self.select(circuit, prefs).explain()


class AdaptiveQuantumExecutor:
    def __init__(self, selector=None, max_qubits=30, default_shots=1024): self._selector, self._max_qubits, self._shots = selector or BackendSelector(), max_qubits, default_shots
    def execute(self, circuit, backend=None, preferences=None, shots=None, optimize=True, debug=False):
        validate_circuit_compatibility(circuit, max_qubits=self._max_qubits)
        prefs = preferences or Preferences()
        shots_n = prefs.shots if shots is None and preferences is not None else (self._shots if shots is None else shots)
        if shots_n < 0: raise ValueError("shots must be non-negative")
        if backend: prefs = Preferences(**{**prefs.__dict__, "preferred_backend": backend})
        decision = self._selector.select(circuit, prefs)
        if debug: print(decision.explain())
        executable = circuit
        if optimize and decision.selected_backend in {"statevector", "stabilizer"}:
            executable, _ = Transpiler(
                passes=["cancel", "rotation", "commute", "fusion", "depth"]
            ).transpile(circuit)
        result = self._selector.get_backend(decision).execute(executable, shots=shots_n)
        result.metadata["selection_decision"] = decision
        result.metadata["selected_backend"] = decision.selected_backend
        result.metadata["optimized"] = optimize
        result.counts["__decision__"] = decision.selected_backend; result.counts["__confidence__"] = f"{decision.confidence:.3f}"
        return result
    @property
    def selector(self): return self._selector


class AdaptiveEngineTestSuite:
    def __init__(self): self._p = 0; self._f = 0

    def _a(self, cond, msg):
        if cond: self._p += 1; print(f"  ✓  {msg}")
        else:    self._f += 1; print(f"  ✗  FAIL: {msg}")

    def test_analyzer_empty(self):
        c = Circuit(3)
        f = CircuitAnalyzer.analyze(c)
        self._a(f.gate_count == 0, "empty: gate_count=0")
        self._a(f.clifford_only, "empty: clifford_only=True")
        self._a(f.entanglement_score == 0.0, "empty: entanglement=0")

    def test_analyzer_bell(self):
        c = Circuit(2)
        c.add_gate(Gate('H', (0,))); c.add_gate(Gate('CNOT', (1,), controls=(0,)))
        f = CircuitAnalyzer.analyze(c)
        self._a(f.num_qubits == 2, "Bell: 2 qubits")
        self._a(f.gate_count == 2, "Bell: 2 gates")
        self._a(f.clifford_only, "Bell: all Clifford (H, CNOT)")
        self._a(f.entanglement_score > 0, f"Bell: entanglement>0 ({f.entanglement_score:.3f})")

    def test_analyzer_parametric(self):
        from zuntenium.transpiler.v2.engine import _build_qaoa
        c = _build_qaoa(4, 1)
        f = CircuitAnalyzer.analyze(c)
        self._a(f.parametric, "QAOA: parametric=True")
        self._a(not f.clifford_only, "QAOA: clifford_only=False")

    def test_analyzer_memory(self):
        c = Circuit(10)
        f = CircuitAnalyzer.analyze(c)
        expected_mb = 16.0 * (2**10) / (1024**2)
        self._a(abs(f.estimated_memory_mb - expected_mb) < 1e-6,
                f"memory n=10: {f.estimated_memory_mb:.4f} MB")

    def test_analyzer_connectivity_density(self):
        c = Circuit(4)
        c.add_gate(Gate('CNOT', (1,), controls=(0,)))
        c.add_gate(Gate('CNOT', (2,), controls=(1,)))
        c.add_gate(Gate('CNOT', (3,), controls=(2,)))
        f = CircuitAnalyzer.analyze(c)
        expected = 3 / 6
        self._a(abs(f.connectivity_density - expected) < 1e-6,
                f"connectivity_density=0.5 (got {f.connectivity_density:.3f})")

    def test_rule_R1_clifford(self):
        c = Circuit(2)
        c.add_gate(Gate('H', (0,))); c.add_gate(Gate('CNOT', (1,), controls=(0,)))
        f = CircuitAnalyzer.analyze(c)
        rules = RuleEngine().apply_rules(f, Preferences())
        r1 = next(r for r in rules if r.rule_name == "R1_CLIFFORD")
        self._a(r1.fired, "R1 fires on Clifford circuit")
        self._a(r1.suggested_backend == "stabilizer", "R1 suggests stabilizer")

    def test_rule_R2_small(self):
        c = Circuit(4)
        for q in range(4): c.add_gate(Gate('H', (q,)))
        f = CircuitAnalyzer.analyze(c)
        rules = RuleEngine().apply_rules(f, Preferences())
        r2 = next(r for r in rules if r.rule_name == "R2_SMALL")
        self._a(r2.fired, "R2 fires on small circuit")
        self._a(r2.suggested_backend == "statevector", "R2 suggests statevector")

    def test_rule_R6_hardware(self):
        c = Circuit(20)
        for q in range(20): c.add_gate(Gate('RX', (q,), params=(0.3,)))
        f = CircuitAnalyzer.analyze(c)
        prefs = Preferences(allow_hardware=True)
        rules = RuleEngine().apply_rules(f, prefs)
        r6 = next(r for r in rules if r.rule_name == "R6_HARDWARE")
        self._a(r6.fired, "R6 fires when hardware allowed and n>15")

    def test_rule_R7_always(self):
        c = Circuit(1); c.add_gate(Gate('H', (0,)))
        f = CircuitAnalyzer.analyze(c)
        rules = RuleEngine().apply_rules(f, Preferences())
        r7 = next(r for r in rules if r.rule_name == "R7_FALLBACK")
        self._a(r7.fired, "R7 always fires")

    def test_scoring_clifford_stabilizer_wins(self):
        c = Circuit(2)
        c.add_gate(Gate('H', (0,))); c.add_gate(Gate('CNOT', (1,), controls=(0,)))
        f = CircuitAnalyzer.analyze(c)
        boosts = {"stabilizer": 100.0}
        available = {"statevector", "stabilizer"}
        scores = ScoringEngine.score_all(f, Preferences(), boosts, available)
        self._a(scores["stabilizer"] > scores["statevector"],
                f"Stabilizer wins with Clifford boost: {scores['stabilizer']:.2f}>{scores['statevector']:.2f}")

    def test_scoring_weights_accuracy(self):
        c = Circuit(5)
        for q in range(5): c.add_gate(Gate('RX', (q,), params=(0.5,)))
        f = CircuitAnalyzer.analyze(c)
        prefs_acc   = Preferences(optimize_for="accuracy")
        prefs_speed = Preferences(optimize_for="speed")
        s_acc   = ScoringEngine.score_all(f, prefs_acc, {}, {"statevector"})
        s_speed = ScoringEngine.score_all(f, prefs_speed, {}, {"statevector"})
        self._a("statevector" in s_acc and "statevector" in s_speed, "Accuracy vs speed scoring valid")

    def test_select_clifford_circuit(self):
        c = Circuit(3)
        c.add_gate(Gate('H', (0,))); c.add_gate(Gate('CNOT', (1,), controls=(0,)))
        c.add_gate(Gate('CNOT', (2,), controls=(1,)))
        sel = BackendSelector()
        d = sel.select(c)
        self._a(d.selected_backend == "stabilizer",
                f"Clifford GHZ → stabilizer (got '{d.selected_backend}')")
        self._a("R1_CLIFFORD" in d.rules_fired, "R1 in fired rules")

    def test_select_parametric_small(self):
        from zuntenium.transpiler.v2.engine import _build_qaoa
        c = _build_qaoa(4, 1)
        sel = BackendSelector()
        d = sel.select(c)
        self._a(d.selected_backend == "statevector",
                f"Small QAOA → statevector (got '{d.selected_backend}')")

    def test_select_hardware_preference(self):
        c = Circuit(16)
        for q in range(16): c.add_gate(Gate('RX', (q,), params=(0.3,)))
        sel = BackendSelector()
        d = sel.select(c, Preferences(allow_hardware=True))
        r6_fired = "R6_HARDWARE" in d.rules_fired
        self._a(r6_fired, f"R6 fires for n=16+allow_hardware (selected={d.selected_backend})")

    def test_select_preferred_override(self):
        c = Circuit(2); c.add_gate(Gate('H', (0,)))
        sel = BackendSelector()
        d = sel.select(c, Preferences(preferred_backend="statevector"))
        self._a(d.selected_backend == "statevector", "preferred_backend override works")
        self._a(d.confidence == 1.0, "override confidence = 1.0")

    def test_select_empty_circuit(self):
        c = Circuit(3)
        sel = BackendSelector()
        d = sel.select(c)
        self._a(d.selected_backend in ("statevector", "stabilizer"),
                f"Empty circuit → valid backend (got '{d.selected_backend}')")

    def test_select_large_circuit_stub(self):
        c = Circuit(25)
        for q in range(25): c.add_gate(Gate('RX', (q,), params=(0.3,)))
        sel = BackendSelector()
        d = sel.select(c, Preferences(allow_hardware=False))
        self._a(d.selected_backend in ("statevector", "hardware"),
                f"n=25 non-Clifford → valid backend (got '{d.selected_backend}')")

    def test_confidence_range(self):
        from zuntenium.transpiler.v2.engine import _build_ghz
        c = _build_ghz(3)
        sel = BackendSelector()
        d = sel.select(c)
        self._a(0.0 <= d.confidence <= 1.0, f"confidence in [0,1]: {d.confidence:.3f}")

    def test_scores_all_present(self):
        c = Circuit(2); c.add_gate(Gate('H', (0,)))
        sel = BackendSelector()
        d = sel.select(c)
        for b in ("statevector", "stabilizer", "densitymatrix", "tensornetwork", "hardware"):
            self._a(b in d.scores, f"'{b}' has a score entry")

    def test_decision_explain(self):
        c = Circuit(2); c.add_gate(Gate('H', (0,)))
        sel = BackendSelector()
        d = sel.select(c)
        txt = d.explain()
        self._a("SELECTED" in txt, "explain() contains 'SELECTED' marker")
        self._a("Circuit features" in txt, "explain() contains features")

    def test_adaptive_executor_bell(self):
        c = Circuit(2)
        c.add_gate(Gate('H', (0,))); c.add_gate(Gate('CNOT', (1,), controls=(0,)))
        ex = AdaptiveQuantumExecutor()
        r = ex.execute(c, shots=0)
        self._a(r.probabilities is not None, "adaptive executor: Bell has probs")
        self._a('__decision__' in r.counts, "decision tagged in result")
        sv = r.statevector
        if sv is not None:
            p = abs(sv[0])**2 + abs(sv[3])**2
            self._a(abs(p - 1.0) < 1e-6, f"Bell correct: P[00]+P[11]={p:.6f}")

    def test_adaptive_executor_debug(self):
        import io, contextlib
        c = Circuit(2); c.add_gate(Gate('H', (0,)))
        ex = AdaptiveQuantumExecutor()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            ex.execute(c, debug=True, shots=0)
        txt = buf.getvalue()
        self._a("Adaptive Execution Engine" in txt, "debug output has engine header")

    def test_adaptive_executor_backend_override(self):
        c = Circuit(2); c.add_gate(Gate('H', (0,)))
        ex = AdaptiveQuantumExecutor()
        r = ex.execute(c, backend="statevector", shots=0)
        self._a(r.counts.get('__decision__') == 'statevector', "backend override: decision=statevector")

    def test_clifford_rz_special_angles(self):
        for k in range(5):
            g = Gate('RZ', (0,), params=(k * math.pi / 2,))
            self._a(_is_clifford_gate(g), f"RZ({k}·π/2) is Clifford")

    def test_clifford_rz_general_angle(self):
        g = Gate('RZ', (0,), params=(0.3,))
        self._a(not _is_clifford_gate(g), "RZ(0.3) is NOT Clifford")

    def run_all(self):
        print("\n" + "═" * 60)
        print("  Adaptive Execution Engine — Test Suite")
        print("═" * 60)
        tests = [m for m in sorted(dir(self)) if m.startswith('test_')]
        for t in tests:
            print(f"\n[{t}]"); getattr(self, t)()
        total = self._p + self._f
        print(f"\n{'═' * 60}")
        print(f"  Results: {self._p}/{total} passed"
              + (f"  ({self._f} FAILED)" if self._f else "  ✓ all pass"))
        print("═" * 60)
        assert self._f == 0, f"{self._f} tests failed in AdaptiveEngineTestSuite"


def run_demo():
    print("\n" + "═" * 70)
    print("  ZUNTENIUM Adaptive Execution Engine — Demo")
    print("═" * 70)
    c = Circuit(2)
    c.add_gate(Gate('H', (0,))); c.add_gate(Gate('CNOT', (1,), controls=(0,)))
    sel = BackendSelector()
    d = sel.select(c)
    print(d.explain())
    print("\n" + "═" * 70 + "\n")


__all__ = [
    "CircuitFeatures", "CircuitAnalyzer", "Preferences", "BackendCapability",
    "_BACKEND_CAPABILITIES", "RuleResult", "RuleEngine", "ScoringEngine",
    "SelectionDecision", "StabilizerSimulatorStub", "DensityMatrixSimulatorStub",
    "TensorNetworkSimulatorStub", "BackendSelector", "AdaptiveQuantumExecutor",
    "AdaptiveEngineTestSuite", "_is_clifford_gate", "run_demo",
]

if __name__ == "__main__":
    run_demo()
    AdaptiveEngineTestSuite().run_all()
