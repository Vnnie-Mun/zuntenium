"""
bench_circuit_depth.py — Benchmark: 100-gate circuit end-to-end time.

Measures full pipeline: build → transpile → execute for deep circuits.
"""
from __future__ import annotations

import time
import numpy as np

from zuntenium import QuantumCircuit, execute, get_backend
from zuntenium.transpiler.pipeline import TranspilerPipeline


def bench_deep_circuit(num_qubits: int = 8, num_gates: int = 100, shots: int = 1024):
    """Build and execute a circuit with many gates."""
    print(f"  {num_gates}-gate circuit on {num_qubits} qubits:")

    # Build
    t0 = time.perf_counter()
    qc = QuantumCircuit(num_qubits)
    for i in range(num_gates):
        q = i % num_qubits
        if i % 3 == 0:
            qc.h(q)
        elif i % 3 == 1:
            qc.rx(0.5, q)
        else:
            qc.cx(q, (q + 1) % num_qubits)
    qc.measure_all()
    t_build = time.perf_counter() - t0
    print(f"    Build:     {t_build*1e3:8.2f} ms")

    # Transpile
    t0 = time.perf_counter()
    ir = qc.to_ir()
    pipeline = TranspilerPipeline(level=2)
    opt_ir = pipeline.run(ir)
    t_transpile = time.perf_counter() - t0
    print(f"    Transpile: {t_transpile*1e3:8.2f} ms  "
          f"({len(ir.instructions)} → {len(opt_ir.instructions)} gates)")

    # Execute
    backend = get_backend("statevector_simulator")
    t0 = time.perf_counter()
    result = execute(qc, backend=backend, shots=shots)
    t_exec = time.perf_counter() - t0
    print(f"    Execute:   {t_exec*1e3:8.2f} ms  ({shots} shots)")
    print(f"    Total:     {(t_build + t_transpile + t_exec)*1e3:8.2f} ms")

    return t_build, t_transpile, t_exec


def bench_scaling():
    """Run with increasing circuit sizes."""
    for n_gates in [50, 100, 200, 500]:
        bench_deep_circuit(num_qubits=8, num_gates=n_gates)
        print()


if __name__ == "__main__":
    print("=== Circuit Depth Benchmark ===")
    bench_scaling()
