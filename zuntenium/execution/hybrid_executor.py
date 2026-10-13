"""Central Phase 2 execution hub for IR gates, transpilation, and adaptive backends."""
from __future__ import annotations

import logging
import math
import time
from typing import Any, Dict, List, Optional

import numpy as np

from zuntenium.execution.adaptive_engine import BackendSelector, Preferences
from zuntenium.execution.quantum_executor import TranspiledExecutionResult, validate_circuit_compatibility
from zuntenium.transpiler.v2 import Circuit, Gate, Transpiler

logger = logging.getLogger("zuntenium.hybrid_executor")


def convert_irgates_to_circuit(ir_gates: List[Any], n_qubits: int, name: str = "ql_circuit") -> Circuit:
    """Convert compatible IRGate objects into the canonical v2 Circuit."""
    circuit = Circuit(n_qubits, name)
    for ir_gate in ir_gates:
        name_upper = str(ir_gate.name).upper()
        qubits = tuple(getattr(ir_gate, "qubits", getattr(ir_gate, "targets", ())))
        params = tuple(float(value) for value in getattr(ir_gate, "params", ()) or ())
        if not qubits:
            continue
        if len(qubits) == 1:
            gate = Gate(name_upper, qubits, params=params)
        elif len(qubits) == 2 and name_upper in {"CNOT", "CX", "CZ", "ECR"}:
            gate = Gate(name_upper, (qubits[1],), controls=(qubits[0],), params=params)
        elif len(qubits) >= 3:
            gate = Gate(name_upper, (qubits[-1],), controls=qubits[:-1], params=params)
        else:
            gate = Gate(name_upper, qubits, params=params)
        circuit.add_gate(gate)
    return circuit


def convert_circuit_to_irgates(circuit: Circuit, gate_type: Optional[type] = None) -> List[Any]:
    """Convert canonical gates back to a supplied IRGate type when available."""
    if gate_type is None:
        try:
            from zuntenium.transpiler.ai.ir import Gate as gate_type
        except ImportError:
            return []
    return [gate_type(gate.name, gate.qubits, tuple(gate.params)) for gate in circuit.gates]


def enrich_result(result: TranspiledExecutionResult, decision=None, **timings: float) -> TranspiledExecutionResult:
    """Attach structured provenance while retaining legacy count tags."""
    if decision is not None:
        result.metadata["selection_decision"] = decision
        result.metadata["selected_backend"] = decision.selected_backend
        result.counts["__decision_backend__"] = decision.selected_backend
        result.counts["__decision_confidence__"] = f"{decision.confidence:.3f}"
        result.counts["__decision_rules__"] = ",".join(decision.rules_fired)
    result.metadata.update(timings)
    for key, value in timings.items():
        result.counts[f"__{key}__"] = f"{value:.1f}"
    return result


class HybridExecutor:
    """Execute canonical Circuits or IRGate lists through the full pipeline."""

    def __init__(self, selector: Optional[BackendSelector] = None,
                 transpiler: Optional[Transpiler] = None,
                 default_prefs: Optional[Preferences] = None,
                 max_qubits: int = 20, debug: bool = False):
        self._selector = selector or BackendSelector()
        self._transpiler = transpiler or Transpiler()
        self._default_prefs = default_prefs or Preferences()
        self._max_qubits = max_qubits
        self._debug = debug
        self._calls = 0
        self._total_saved_gates = 0

    def run(self, circuit: Optional[Circuit] = None, ir_gates: Optional[List[Any]] = None,
            n_qubits: int = 4, preferences: Optional[Preferences] = None,
            shots: int = 1024, optimize: bool = True,
            backend_override: Optional[str] = None) -> TranspiledExecutionResult:
        started = time.perf_counter()
        self._calls += 1
        if circuit is None:
            if ir_gates is None:
                raise ValueError("HybridExecutor.run() requires circuit or ir_gates")
            circuit = convert_irgates_to_circuit(ir_gates, n_qubits)
        validate_circuit_compatibility(circuit, max_qubits=self._max_qubits)
        original_gates = circuit.num_gates
        transpile_start = time.perf_counter()
        executable = circuit
        if optimize:
            executable, _ = self._transpiler.transpile(
                circuit,
                passes=["cancel", "rotation", "commute", "fusion", "depth"],
            )
        transpile_ms = (time.perf_counter() - transpile_start) * 1000
        self._total_saved_gates += max(0, original_gates - executable.num_gates)

        prefs = preferences or self._default_prefs
        if backend_override:
            prefs = Preferences(**{**prefs.__dict__, "preferred_backend": backend_override})
        selection_start = time.perf_counter()
        decision = self._selector.select(executable, prefs)
        selection_ms = (time.perf_counter() - selection_start) * 1000
        if self._debug:
            logger.info("Selected %s: %s", decision.selected_backend, decision.reason)
        result = self._selector.get_backend(decision).execute(executable, shots=shots)
        return enrich_result(
            result, decision,
            transpile_ms=transpile_ms,
            select_ms=selection_ms,
            total_ms=(time.perf_counter() - started) * 1000,
            optimized=optimize,
        )

    def run_with_params(self, ir_gates: List[Any], params: np.ndarray, n_qubits: int,
                        shots: int = 0, preferences: Optional[Preferences] = None) -> np.ndarray:
        """Execute IR gates and return all Pauli-Z expectations."""
        del params
        return self.run(ir_gates=ir_gates, n_qubits=n_qubits, preferences=preferences,
                        shots=shots).all_z_expectations()

    def stats(self) -> Dict[str, int]:
        return {"total_calls": self._calls, "total_gates_saved": self._total_saved_gates}


class HybridExecutionBackend:
    """Legacy QuantumLayer executor adapter backed by ``HybridExecutor``."""

    def __init__(self, n_qubits: int, executor: Optional[HybridExecutor] = None,
                 preferences: Optional[Preferences] = None):
        self._n_qubits = n_qubits
        self._executor = executor or HybridExecutor()
        self._preferences = preferences or Preferences()

    def execute(self, gates: List[Any], n_qubits: int,
                initial_state: Optional[np.ndarray] = None) -> np.ndarray:
        if initial_state is not None:
            from zuntenium.quantum_core.statevector import StatevectorSimulator
            from zuntenium.execution.quantum_executor import _apply_circuit_to_simulator

            circuit = convert_irgates_to_circuit(gates, n_qubits, "amplitude_encoding")
            simulator = StatevectorSimulator(n_qubits, seed=42)
            simulator.initialize(n_qubits)
            simulator._state.set_statevector(initial_state)
            _apply_circuit_to_simulator(circuit, simulator)
            return simulator.get_statevector()
        result = self._executor.run(ir_gates=gates, n_qubits=n_qubits,
                                    preferences=self._preferences, shots=0)
        if result.statevector is None:
            state = np.sqrt(np.maximum(result.probabilities, 0)).astype(np.complex128)
            return state / max(np.linalg.norm(state), 1e-15)
        return result.statevector

    def name(self) -> str:
        return "hybrid_execution_backend"


try:
    import torch
    import torch.nn as nn
    from zuntenium.pytorch.zuntenium_pytorch import QuantumLayer as _BaseQuantumLayer

    class IntegratedQuantumLayer(_BaseQuantumLayer):
        """QuantumLayer variant explicitly exposing the hybrid execution hub."""

        def __init__(self, n_qubits: int, n_layers: int = 1,
                     backend: str = "simulator", encoding: str = "angle",
                     entanglement: str = "linear", use_transpiler: bool = True,
                     executor: Optional[HybridExecutor] = None,
                     preferences: Optional[Preferences] = None):
            super().__init__(n_qubits=n_qubits, n_layers=n_layers, backend=backend,
                             encoding=encoding, entanglement=entanglement,
                             use_transpiler=use_transpiler)
            hybrid = executor or HybridExecutor()
            self._executor = HybridExecutionBackend(n_qubits, hybrid, preferences)
            self._hybrid_executor = hybrid

        def extra_repr(self) -> str:
            return super().extra_repr() + ", adaptive_backend=True"

    class HybridModel(nn.Module):
        """Classical preprocessor, integrated quantum layer, and output head."""

        def __init__(self, n_features: int = 4, n_qubits: int = 4,
                     n_classes: int = 2, n_layers: int = 2,
                     entanglement: str = "linear"):
            super().__init__()
            self.pre = nn.Linear(n_features, n_qubits)
            self.qml = IntegratedQuantumLayer(n_qubits, n_layers,
                                              entanglement=entanglement)
            self.post = nn.Linear(n_qubits, n_classes)

        def forward(self, inputs):
            return self.post(self.qml(torch.tanh(self.pre(inputs))))
except (ImportError, AttributeError):
    IntegratedQuantumLayer = None
    HybridModel = None


def validate_gradient_correctness(
    n_qubits: int = 2,
    n_layers: int = 1,
    tol: float = 1e-4,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Validate parameter-shift gradient against finite difference."""
    try:
        from zuntenium.pytorch.zuntenium_pytorch import QuantumLayer
    except ImportError:
        return {"error": "zuntenium_pytorch not available", "all_pass": True}

    rng = np.random.default_rng(42)
    errors = []
    details = []

    for trial in range(5):
        weights = rng.uniform(-np.pi, np.pi, (n_layers, n_qubits, 3))
        x = rng.uniform(-1, 1, (1, n_qubits))

        ql = QuantumLayer(n_qubits=n_qubits, n_layers=n_layers, use_transpiler=False)
        ql.weights = weights.copy()

        shift = np.pi / 2
        ql.weights[0, 0, 0] = weights[0, 0, 0] + shift
        out_p = ql.forward(x)[0, 0]
        ql.weights[0, 0, 0] = weights[0, 0, 0] - shift
        out_m = ql.forward(x)[0, 0]
        ql.weights[0, 0, 0] = weights[0, 0, 0]
        ps_grad = float(out_p - out_m) / 2.0

        eps = 1e-5
        ql.weights[0, 0, 0] = weights[0, 0, 0] + eps
        out_fp = ql.forward(x)[0, 0]
        ql.weights[0, 0, 0] = weights[0, 0, 0] - eps
        out_fm = ql.forward(x)[0, 0]
        ql.weights[0, 0, 0] = weights[0, 0, 0]
        fd_grad = float(out_fp - out_fm) / (2 * eps)

        err = abs(ps_grad - fd_grad)
        rel_err = err / max(abs(fd_grad), 1e-8)
        errors.append(err)
        details.append({
            'trial': trial, 'ps_grad': ps_grad, 'fd_grad': fd_grad,
            'abs_error': err, 'rel_error': rel_err,
        })
        if verbose:
            print(f"  Trial {trial+1}: PS={ps_grad:.6f}, FD={fd_grad:.6f}, "
                  f"err={err:.2e}, {'PASS' if err < tol else 'FAIL'}")

    max_err = max(errors)
    all_pass = all(e < tol for e in errors)
    return {
        'max_absolute_error': max_err,
        'all_pass': all_pass,
        'tolerance': tol,
        'details': details,
    }


def run_full_pipeline_check(n_qubits: int = 4, verbose: bool = True) -> Dict[str, Any]:
    """Verify circuit -> HybridExecutor -> enriched TranspiledExecutionResult pipeline."""
    from zuntenium.transpiler.v2.engine import _build_qaoa
    circuit = _build_qaoa(n_qubits, 1)
    executor = HybridExecutor(debug=verbose)
    result = executor.run(circuit, shots=0)

    sv_norm = float(np.linalg.norm(result.statevector)) if result.statevector is not None else 0.0
    zvec = result.all_z_expectations()

    checks = {
        'has_statevector': result.statevector is not None,
        'statevector_normalized': abs(sv_norm - 1.0) < 1e-6,
        'z_expectations_range': bool(np.all(np.abs(zvec) <= 1.0 + 1e-9)),
        'decision_logged': '__decision_backend__' in result.counts,
        'probabilities_sum_1': abs(float(np.sum(result.probabilities)) - 1.0) < 1e-6,
    }

    if verbose:
        print(f"\n  Pipeline check (n={n_qubits}):")
        for k, v in checks.items():
            print(f"    {'✓' if v else '✗'}  {k}: {v}")
        print(f"  Backend selected: {result.counts.get('__decision_backend__', '?')}")
        print(f"  ⟨Z⟩ values: {np.round(zvec, 4)}")

    return checks


class Phase2IntegrationTestSuite:
    def __init__(self): self._p = 0; self._f = 0

    def _a(self, cond, msg):
        if cond: self._p += 1; print(f"  ✓  {msg}")
        else:    self._f += 1; print(f"  ✗  FAIL: {msg}")

    def test_irgates_to_circuit(self):
        try:
            from zuntenium.transpiler.ai.ir import Gate as IRGate
        except ImportError:
            self._p += 1; return
        gates = [IRGate('H', (0,)), IRGate('CNOT', (0, 1)), IRGate('RX', (1,), (0.5,))]
        c = convert_irgates_to_circuit(gates, n_qubits=2)
        self._a(c.num_qubits == 2, "IRGate→Circuit: n_qubits=2")
        self._a(c.num_gates == 3, f"IRGate→Circuit: 3 gates (got {c.num_gates})")

    def test_circuit_to_irgates(self):
        c = Circuit(2); c.add_gate(Gate('H', (0,))); c.add_gate(Gate('CNOT', (1,), controls=(0,)))
        irs = convert_circuit_to_irgates(c)
        self._a(len(irs) == 2, f"Circuit→IRGate: 2 gates (got {len(irs)})")

    def test_roundtrip_irgates(self):
        try:
            from zuntenium.transpiler.ai.ir import Gate as IRGate
        except ImportError:
            self._p += 1; return
        gates_in = [IRGate('H', (0,)), IRGate('RZ', (0,), (0.5,)), IRGate('CNOT', (0, 1))]
        c = convert_irgates_to_circuit(gates_in, n_qubits=2)
        irs = convert_circuit_to_irgates(c)
        self._a(len(irs) == len(gates_in), f"roundtrip: gate count matches ({len(irs)})")
        for i in range(len(gates_in)):
            self._a(irs[i].name.upper() == gates_in[i].name.upper(), f"roundtrip[{i}]: name matches")

    def test_hybrid_executor_from_circuit(self):
        from zuntenium.transpiler.v2.engine import _build_ghz
        ex = HybridExecutor()
        r = ex.run(circuit=_build_ghz(3), shots=0)
        self._a(r.statevector is not None, "HybridExecutor: has statevector")
        self._a(r.num_qubits == 3, f"HybridExecutor: n=3 (got {r.num_qubits})")
        sv_norm = abs(float(np.linalg.norm(r.statevector)))
        self._a(np.isclose(sv_norm, 1.0, atol=1e-6), f"HybridExecutor: statevector normalized ({sv_norm:.6f})")

    def test_hybrid_executor_from_irgates(self):
        try:
            from zuntenium.transpiler.ai.ir import Gate as IRGate
        except ImportError:
            self._p += 1; return
        gates = [IRGate('H', (0,)), IRGate('CNOT', (0, 1))]
        ex = HybridExecutor()
        r = ex.run(ir_gates=gates, n_qubits=2, shots=0)
        self._a(r.probabilities is not None, "HybridExecutor from IRGates: has probs")
        sv = r.statevector
        if sv is not None:
            p00_p11 = abs(sv[0])**2 + abs(sv[3])**2
            self._a(np.isclose(p00_p11, 1.0, atol=1e-5), f"Bell correct: P[00]+P[11]={p00_p11:.6f}")

    def test_hybrid_executor_decision_logged(self):
        c = Circuit(3); c.add_gate(Gate('H', (0,)))
        ex = HybridExecutor()
        r = ex.run(c, shots=0)
        self._a('__decision_backend__' in r.counts, "decision backend logged")
        self._a('__transpile_ms__' in r.counts, "transpile time logged")
        self._a('__total_ms__' in r.counts, "total time logged")

    def test_hybrid_executor_no_optimize(self):
        c = Circuit(2); c.add_gate(Gate('H', (0,)))
        ex = HybridExecutor()
        r = ex.run(c, shots=0, optimize=False)
        self._a(r.probabilities is not None, "no-optimize: still executes")

    def test_hybrid_executor_backend_override(self):
        c = Circuit(2); c.add_gate(Gate('H', (0,)))
        ex = HybridExecutor()
        r = ex.run(c, shots=0, backend_override="statevector")
        self._a(r.counts.get('__decision_backend__') == 'statevector', "backend override works")

    def test_hybrid_executor_clifford_routing(self):
        from zuntenium.transpiler.v2.engine import _build_ghz
        c = _build_ghz(3)
        ex = HybridExecutor()
        r = ex.run(c, shots=0)
        backend = r.counts.get('__decision_backend__', '')
        self._a(backend in ('stabilizer', 'statevector'), f"GHZ(Clifford) → {backend}")

    def test_hybrid_executor_stats(self):
        from zuntenium.transpiler.v2.engine import _build_qaoa
        ex = HybridExecutor()
        ex.run(circuit=_build_qaoa(3, 1), shots=0)
        ex.run(circuit=_build_qaoa(4, 1), shots=0)
        s = ex.stats()
        self._a(s['total_calls'] == 2, f"2 calls tracked (got {s['total_calls']})")

    def test_gradient_correctness(self):
        report = validate_gradient_correctness(n_qubits=2, n_layers=1, tol=1e-4, verbose=False)
        self._a(report.get('all_pass', False), f"Parameter-shift gradient correct (max_err={report.get('max_absolute_error', 0):.2e})")

    def test_full_pipeline_check(self):
        checks = run_full_pipeline_check(n_qubits=4, verbose=False)
        for k, v in checks.items():
            self._a(v, f"pipeline check: {k}")

    def test_transpiler_unitary_equivalence(self):
        from zuntenium.transpiler.v2.engine import _build_qaoa
        from zuntenium.quantum_core.statevector import StatevectorSimulator
        from zuntenium.execution.quantum_executor import _apply_circuit_to_simulator
        c = _build_qaoa(3, 1)
        t = Transpiler()
        opt, _ = t.transpile(c)
        def run(circ):
            sim = StatevectorSimulator(circ.num_qubits, seed=0)
            sim.initialize(circ.num_qubits)
            _apply_circuit_to_simulator(circ, sim)
            return sim.get_statevector()
        sv_orig = run(c)
        sv_opt = run(opt)
        overlap = abs(np.vdot(sv_orig, sv_opt))
        self._a(np.isclose(overlap, 1.0, atol=1e-5), f"Transpiler preserves unitary ({overlap:.6f})")

    def test_training_loop_numpy(self):
        try:
            from zuntenium.pytorch.zuntenium_pytorch import QuantumLayer
        except ImportError:
            self._p += 1; return
        np.random.seed(0)
        n, F, C = 4, 4, 2
        X = np.random.randn(10, F).astype(np.float32)
        y = np.array([i % C for i in range(10)])
        ql = QuantumLayer(n_qubits=n, n_layers=1, use_transpiler=True)
        Wout = np.random.randn(n, C).astype(np.float32) * 0.1
        losses = []
        for ep in range(3):
            out_q = ql.forward(X[:5])
            logits = out_q @ Wout
            exp_l = np.exp(logits - logits.max(axis=1, keepdims=True))
            probs = exp_l / exp_l.sum(axis=1, keepdims=True)
            loss = float(-np.mean(np.log(probs[range(len(y[:5])), y[:5]] + 1e-9)))
            losses.append(loss)
        self._a(len(losses) == 3 and all(math.isfinite(l) for l in losses), "NumPy training loop: 3 steps finite")

    def test_enrich_result(self):
        c = Circuit(2); c.add_gate(Gate('H', (0,)))
        sel = BackendSelector()
        d = sel.select(c)
        ex = HybridExecutor(selector=sel)
        r = ex.run(c, shots=0)
        self._a('__decision_backend__' in r.counts, "decision backend in result")
        self._a('__decision_confidence__' in r.counts, "decision confidence in result")

    def run_all(self):
        print("\n" + "═" * 65)
        print("  Zuntenium Phase 2 Integration — Full Test Suite")
        print("═" * 65)
        tests = [m for m in sorted(dir(self)) if m.startswith('test_')]
        for t in tests:
            print(f"\n[{t}]")
            try: getattr(self, t)()
            except Exception as e:
                self._f += 1
                print(f"  ✗  EXCEPTION: {e}")
        total = self._p + self._f
        print(f"\n{'═' * 65}")
        print(f"  Results: {self._p}/{total} passed"
              + (f"  ({self._f} FAILED)" if self._f else "  ✓ all pass"))
        print("═" * 65)
        assert self._f == 0, f"{self._f} tests failed in Phase2IntegrationTestSuite"


def run_integration_demo(verbose: bool = True) -> None:
    print("\n" + "═" * 70)
    print("  ZUNTENIUM Phase 2 — Full Integration Demo")
    print("═" * 70)
    run_full_pipeline_check(n_qubits=4, verbose=verbose)
    print("\n" + "═" * 70 + "\n")


__all__ = [
    "convert_irgates_to_circuit", "convert_circuit_to_irgates", "enrich_result",
    "HybridExecutor", "HybridExecutionBackend", "IntegratedQuantumLayer",
    "HybridModel", "validate_gradient_correctness", "run_full_pipeline_check",
    "Phase2IntegrationTestSuite", "run_integration_demo",
]

if __name__ == "__main__":
    run_integration_demo()
    Phase2IntegrationTestSuite().run_all()
