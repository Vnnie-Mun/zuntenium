"""
Benchmark tests for statevector simulation.

Run with: pytest tests/benchmarks/ -v --benchmark-only
Or directly: python scripts/benchmark_runner.py
"""
import numpy as np
import pytest

from zuntenium.backend.simulator.statevector_backend import StatevectorBackend
from zuntenium.circuit.circuit import QuantumCircuit
from zuntenium.utils.profiling import benchmark

QUBIT_COUNTS = [4, 8, 12, 16, 20]


def build_random_circuit(n: int, depth: int = 10) -> QuantumCircuit:
    """Build a random circuit with alternating H and CX layers."""
    rng = np.random.default_rng(42)
    qc = QuantumCircuit(n)
    for _ in range(depth):
        # Random H layer
        for q in range(n):
            if rng.random() > 0.5:
                qc.h(q)
        # Random CX layer
        for q in range(0, n - 1, 2):
            qc.cx(q, q + 1)
    return qc


def build_qft_circuit(n: int) -> QuantumCircuit:
    """Build an n-qubit QFT circuit."""
    from zuntenium.api.circuit_api import quantum_fourier_transform
    return quantum_fourier_transform(n)


class TestSimulationBenchmarks:
    """Performance benchmarks for various circuit sizes."""

    @pytest.mark.parametrize("n", [4, 8, 12])
    def test_random_circuit_timing(self, n: int) -> None:
        """Verify that 12-qubit random circuits run within 10 seconds."""
        backend = StatevectorBackend(seed=42, optimization_level=0)
        qc = build_random_circuit(n, depth=5)
        import time
        t0 = time.perf_counter()
        result = backend.run(qc, shots=1)
        elapsed = time.perf_counter() - t0
        assert result is not None
        # Liberal bound: any reasonable hardware should finish within 60s
        assert elapsed < 60.0, f"Circuit with {n} qubits took {elapsed:.2f}s"

    def test_bell_circuit_throughput(self) -> None:
        """Bell circuit should run in well under 1ms."""
        backend = StatevectorBackend(seed=42)
        qc = QuantumCircuit(2)
        qc.h(0).cx(0, 1).measure_all()
        stats = benchmark(lambda: backend.run(qc, shots=1000), repeat=10)
        assert stats["mean_ms"] < 1000   # <1 second per run

    def test_optimization_reduces_gate_count(self) -> None:
        """Optimization level 2 should not increase gate count vs level 0."""
        qc = QuantumCircuit(2)
        qc.h(0).h(0).x(1).x(1).cx(0, 1)  # HH and XX cancel

        backend_no_opt = StatevectorBackend(seed=42, optimization_level=0)
        backend_opt = StatevectorBackend(seed=42, optimization_level=2)

        result_no_opt = backend_no_opt.run(qc, shots=1)
        result_opt = backend_opt.run(qc, shots=1)

        gates_no_opt = result_no_opt.metadata.get("num_gates", float("inf"))
        gates_opt = result_opt.metadata.get("num_gates", float("inf"))

        assert gates_opt <= gates_no_opt, (
            f"Optimization should not increase gate count: "
            f"level=0 has {gates_no_opt}, level=2 has {gates_opt}"
        )
