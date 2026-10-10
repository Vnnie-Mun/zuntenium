"""
Transpiler v2 Engine — PassRegistry, TranspilerPipeline, Transpiler, benchmarks.
"""
from __future__ import annotations
import math, time, timeit
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Type
import networkx as nx
import numpy as np

from zuntenium.transpiler.v2.core import Gate, Circuit, _gate_matrix
from zuntenium.transpiler.v2.passes import (
    OptimizationPass, PassResult,
    RedundantGateCancellationPass, RotationMergingPass,
    CommutationReorderingPass, SingleQubitFusionPass,
    DepthOptimizationPass, DeadGateEliminationPass,
)


class PassRegistry:
    """Dynamic pass registry with AI agent injection point."""
    _BUILTINS: Dict[str, Type[OptimizationPass]] = {
        'cancel': RedundantGateCancellationPass, 'redundant': RedundantGateCancellationPass,
        'rotation': RotationMergingPass, 'commute': CommutationReorderingPass,
        'commutation': CommutationReorderingPass, 'fusion': SingleQubitFusionPass,
        'depth': DepthOptimizationPass, 'dead': DeadGateEliminationPass,
    }
    _custom: Dict[str, OptimizationPass] = {}

    @classmethod
    def register(cls, name: str, pass_instance: OptimizationPass) -> None:
        cls._custom[name.lower()] = pass_instance

    @classmethod
    def get(cls, name: str) -> OptimizationPass:
        key = name.lower()
        if key in cls._custom: return cls._custom[key]
        if key in cls._BUILTINS: return cls._BUILTINS[key]()
        raise ValueError(f"Unknown pass '{name}'. Built-ins: {sorted(cls._BUILTINS)}.")

    @classmethod
    def default_passes(cls) -> List[OptimizationPass]:
        return [cls.get(k) for k in ['cancel','rotation','commute','fusion','depth','dead']]

    @classmethod
    def available(cls) -> List[str]:
        return sorted(set(cls._BUILTINS) | set(cls._custom))


@dataclass
class OptimizationReport:
    original_gates: int; optimized_gates: int; original_depth: int; optimized_depth: int
    estimated_fidelity: float; total_time_ms: float; passes_run: List[PassResult]
    critical_path_len: int = 0; parallelism_ratio: float = 0.0

    @property
    def gate_reduction_pct(self) -> float:
        return 100.0 * (1 - self.optimized_gates / max(self.original_gates, 1))
    @property
    def depth_reduction_pct(self) -> float:
        return 100.0 * (1 - self.optimized_depth / max(self.original_depth, 1))

    def summary(self) -> str:
        return (f"Gates: {self.original_gates}->{self.optimized_gates} "
                f"({self.gate_reduction_pct:.1f}%down)  "
                f"Depth: {self.original_depth}->{self.optimized_depth} "
                f"({self.depth_reduction_pct:.1f}%down)  "
                f"Fidelity: {self.estimated_fidelity:.4f}  Time: {self.total_time_ms:.1f}ms")

    def passes_summary(self) -> str:
        lines = ["Pass results:"]
        for r in self.passes_run:
            if r.gates_removed or r.gates_added:
                sign = "-" if r.net_delta <= 0 else "+"
                lines.append(f"  {r.pass_name:<38s}: {sign}{abs(r.net_delta):3d} gates, "
                             f"ddepth={r.depth_delta:+d}, {r.elapsed_ms:.1f}ms")
        return "\n".join(lines)


def _estimate_fidelity(circuit: Circuit, p1q=0.001, p2q=0.01) -> float:
    f = 1.0
    for g in circuit.gates:
        if g.is_measurement: continue
        p = g.metadata.get('error_rate', p1q if g.num_qubits == 1 else p2q)
        f *= (1.0 - p)
    return max(f, 0.0)


def _critical_path_len(circuit: Circuit) -> int:
    if not circuit.gates: return 0
    return len(nx.dag_longest_path(circuit.to_dag()))


class TranspilerPipeline:
    def __init__(self, passes: Optional[List[OptimizationPass]] = None, max_iterations: int = 5):
        self._passes = passes if passes is not None else PassRegistry.default_passes()
        self._max_iter = max_iterations

    def add_pass(self, p: OptimizationPass, position: int = -1) -> None:
        if position < 0: self._passes.append(p)
        else: self._passes.insert(position, p)

    def remove_pass(self, name: str) -> None:
        self._passes = [p for p in self._passes if p.name != name]

    def run(self, circuit: Circuit, debug: bool = False) -> Tuple[Circuit, List[PassResult]]:
        current, all_results = circuit, []
        for iteration in range(self._max_iter):
            prev = current.num_gates
            for p in self._passes:
                new_circ, result = p(current)
                if debug and (result.gates_removed or result.gates_added):
                    print(f"  [{p.name}] iter={iteration+1}: "
                          f"{current.num_gates}->{new_circ.num_gates} gates, "
                          f"ddepth={result.depth_delta:+d}, {result.elapsed_ms:.1f}ms")
                if result.gates_removed > 0 or result.gates_added > 0:
                    all_results.append(result)
                current = new_circ
            if current.num_gates >= prev: break
        return current, all_results


class Transpiler:
    """ZUNTENIUM Quantum Transpiler v2 — production entry point."""

    def __init__(self, passes: Optional[List[str]] = None,
                 backend_config: Optional[Dict[str, Any]] = None, max_iterations: int = 5):
        self._backend_config = backend_config
        pass_objs = PassRegistry.default_passes() if passes is None else [PassRegistry.get(n) for n in passes]
        self._pipeline = TranspilerPipeline(pass_objs, max_iterations)

    def transpile(self, circuit: Circuit, backend_config: Optional[Dict[str, Any]] = None,
                  passes: Optional[List[str]] = None, debug: bool = False) -> Tuple[Circuit, OptimizationReport]:
        pipeline = TranspilerPipeline([PassRegistry.get(n) for n in passes], self._pipeline._max_iter) if passes else self._pipeline
        if backend_config: self._backend_config = backend_config
        t0 = time.perf_counter(); d_before = circuit.depth()
        if debug: print(f"\n[Transpiler v2] Input: {circuit}")
        optimized, pass_results = pipeline.run(circuit, debug=debug)
        total_ms = (time.perf_counter() - t0) * 1000
        report = OptimizationReport(
            original_gates=circuit.num_gates, optimized_gates=optimized.num_gates,
            original_depth=d_before, optimized_depth=optimized.depth(),
            estimated_fidelity=_estimate_fidelity(optimized), total_time_ms=total_ms,
            passes_run=pass_results, critical_path_len=_critical_path_len(optimized),
            parallelism_ratio=(optimized.num_gates / max(optimized.depth(), 1)),
        )
        if debug:
            print(f"\n[Transpiler v2] Output: {optimized}")
            print(f"\n{report.summary()}")
            print(report.passes_summary())
        return optimized, report

    def add_pass(self, p: OptimizationPass, position: int = -1) -> None:
        self._pipeline.add_pass(p, position)


# ── Benchmark circuit builders ──

def _build_bell(extra: int = 3) -> Circuit:
    c = Circuit(2, "bell")
    for _ in range(extra): c.add_gate(Gate('H',(0,)))
    c.add_gate(Gate('CNOT',(1,), controls=(0,))); return c

def _build_ghz(n: int) -> Circuit:
    c = Circuit(n, f"ghz_{n}"); c.add_gate(Gate('H',(0,)))
    for i in range(n-1): c.add_gate(Gate('CNOT',(i+1,), controls=(i,)))
    return c

def _build_qaoa(n: int, p: int) -> Circuit:
    c = Circuit(n, f"qaoa_n{n}_p{p}"); rng = np.random.default_rng(42)
    for q in range(n): c.add_gate(Gate('H',(q,)))
    for _ in range(p):
        for i in range(n-1):
            c.add_gate(Gate('CNOT',(i+1,), controls=(i,)))
            c.add_gate(Gate('RZ',(i+1,), params=(float(rng.uniform(0,1)),)))
            c.add_gate(Gate('CNOT',(i+1,), controls=(i,)))
        for q in range(n): c.add_gate(Gate('RX',(q,), params=(float(rng.uniform(0,1)),)))
    return c

def _build_random(n: int, d: int) -> Circuit:
    c = Circuit(n, f"random_n{n}_d{d}"); rng = np.random.default_rng(0)
    P1 = ['X','Y','Z','H','S','T','RX','RY','RZ']; P2 = ['CNOT','CZ','SWAP']
    for _ in range(d):
        if n > 1 and rng.random() < 0.3:
            q0 = int(rng.integers(0,n)); q1 = int(rng.integers(0,n))
            while q1 == q0: q1 = int(rng.integers(0,n))
            nm = P2[int(rng.integers(0,len(P2)))]
            c.add_gate(Gate(nm,(q1,), controls=(q0,)) if nm=='CNOT' else Gate(nm,(q0,q1)))
        else:
            q = int(rng.integers(0,n)); nm = P1[int(rng.integers(0,len(P1)))]
            p = (float(rng.uniform(0,2*math.pi)),) if nm in ('RX','RY','RZ') else ()
            c.add_gate(Gate(nm,(q,), params=p))
    return c

def _build_adversarial(n: int) -> Circuit:
    c = Circuit(n, "adversarial")
    for q in range(n):
        c.add_gate(Gate('H',(q,))); c.add_gate(Gate('H',(q,)))
        c.add_gate(Gate('RZ',(q,), params=(0.5,))); c.add_gate(Gate('RZ',(q,), params=(-0.5,)))
        c.add_gate(Gate('T',(q,))); c.add_gate(Gate('TDG',(q,)))
        c.add_gate(Gate('X',(q,))); c.add_gate(Gate('X',(q,)))
    return c

def run_benchmarks(verbose: bool = True) -> Dict[str, Any]:
    t = Transpiler(); results = {}
    circuits = [("Bell+3H", _build_bell(3)), ("GHZ-10", _build_ghz(10)),
                ("QAOA-4-p2", _build_qaoa(4,2)), ("Random-8-50g", _build_random(8,50)),
                ("Random-20-200g", _build_random(20,200)), ("Adversarial-10", _build_adversarial(10))]
    if verbose:
        print("\n" + "="*80); print("  BENCHMARK - Transpiler v2"); print("="*80)
    for name, circ in circuits:
        opt, rp = t.transpile(circ)
        results[name] = {'g_before': rp.original_gates, 'g_after': rp.optimized_gates,
                         'ms': rp.total_time_ms}
        if verbose:
            print(f"  {name:<20} {rp.original_gates:>5} -> {rp.optimized_gates:>5} "
                  f"({rp.gate_reduction_pct:.1f}%) {rp.total_time_ms:.1f}ms")
    if verbose: print("="*80)
    return results
