"""
bench_statevector.py — Benchmark: N-qubit gate time vs qubit count.

Measures single-qubit gate application time from 4 to 20 qubits.
"""
from __future__ import annotations

import time
import numpy as np

from zuntenium.quantum_core.statevector.statevector import StateVector
from zuntenium.quantum_core.gates.single_qubit import HGate


def bench_single_qubit_gate_scaling():
    """Benchmark H gate on qubit 0 for increasing system sizes."""
    gate = HGate()
    results = []
    for n in range(4, 21, 2):
        sv = StateVector(n, backend="numpy")
        start = time.perf_counter()
        for _ in range(100):
            sv.apply_gate(gate, [0])
        elapsed = (time.perf_counter() - start) / 100
        results.append((n, elapsed))
        print(f"  {n:2d} qubits: {elapsed*1e6:8.1f} µs / gate")
    return results


def bench_measurement_sampling():
    """Benchmark measurement sampling for increasing qubit counts."""
    from zuntenium.quantum_core.executor.measurement import sample_statevector
    results = []
    for n in range(4, 17, 2):
        sv = StateVector(n, backend="numpy")
        sv.apply_gate(HGate(), [0])
        amplitudes = sv.get_amplitudes()
        qubits = list(range(n))
        start = time.perf_counter()
        for _ in range(10):
            sample_statevector(amplitudes, qubits, n, 1024)
        elapsed = (time.perf_counter() - start) / 10
        results.append((n, elapsed))
        print(f"  {n:2d} qubits: {elapsed*1e3:8.2f} ms / 1024 shots")
    return results


if __name__ == "__main__":
    print("=== Single-qubit gate scaling ===")
    bench_single_qubit_gate_scaling()
    print()
    print("=== Measurement sampling ===")
    bench_measurement_sampling()
