"""
bench_gate_ops.py — Benchmark: single vs two-qubit gate throughput.

Measures raw gate application throughput on fixed-size statevectors.
"""
from __future__ import annotations

import time
import numpy as np

from zuntenium.quantum_core.statevector.statevector import StateVector
from zuntenium.quantum_core.gates.single_qubit import HGate, XGate, RzGate
from zuntenium.quantum_core.gates.two_qubit import CXGate, CZGate, SWAPGate


def bench_single_qubit_gates(num_qubits: int = 12, reps: int = 500):
    """Throughput for various single-qubit gates."""
    gates = [("H", HGate()), ("X", XGate()), ("Rz(0.5)", RzGate(0.5))]
    print(f"  Single-qubit gates on {num_qubits}-qubit state ({reps} reps):")
    for name, gate in gates:
        sv = StateVector(num_qubits, backend="numpy")
        start = time.perf_counter()
        for _ in range(reps):
            sv.apply_gate(gate, [0])
        elapsed = time.perf_counter() - start
        throughput = reps / elapsed
        print(f"    {name:>10s}: {throughput:8.0f} gates/sec  ({elapsed/reps*1e6:.1f} µs)")


def bench_two_qubit_gates(num_qubits: int = 12, reps: int = 200):
    """Throughput for various two-qubit gates."""
    gates = [("CX", CXGate()), ("CZ", CZGate()), ("SWAP", SWAPGate())]
    print(f"  Two-qubit gates on {num_qubits}-qubit state ({reps} reps):")
    for name, gate in gates:
        sv = StateVector(num_qubits, backend="numpy")
        start = time.perf_counter()
        for _ in range(reps):
            sv.apply_gate(gate, [0, 1])
        elapsed = time.perf_counter() - start
        throughput = reps / elapsed
        print(f"    {name:>10s}: {throughput:8.0f} gates/sec  ({elapsed/reps*1e6:.1f} µs)")


if __name__ == "__main__":
    print("=== Gate Operations Benchmark ===")
    bench_single_qubit_gates()
    print()
    bench_two_qubit_gates()
