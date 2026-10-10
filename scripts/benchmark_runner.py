#!/usr/bin/env python3
"""
scripts/benchmark_runner.py
Runs the ZUNTENIUM benchmark suite and exports results to JSON.

Usage:
    python scripts/benchmark_runner.py --output results.json --qubit-range 2,20 --shots 1024
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))


def _bench_statevector(qubit_range: range, shots: int) -> list[dict[str, Any]]:
    """Benchmark statevector gate application across qubit counts."""
    results = []
    try:
        from zuntenium import QuantumCircuit, get_backend

        backend = get_backend("statevector_simulator")
        for n in qubit_range:
            qc = QuantumCircuit(n)
            for q in range(n):
                qc.h(q)
            if n >= 2:
                qc.cx(0, 1)

            start = time.perf_counter()
            _ = backend.run(qc, shots=shots)
            elapsed = time.perf_counter() - start

            results.append({
                "benchmark": "statevector_apply_gate",
                "num_qubits": n,
                "shots": shots,
                "time_seconds": round(elapsed, 6),
                "ops_per_second": round(1.0 / elapsed, 2),
            })
            print(f"  n={n:2d} qubits: {elapsed*1000:.3f} ms")
    except ImportError as e:
        print(f"WARN: Could not import zuntenium: {e}")
    return results


def _bench_circuit_depth(shots: int) -> list[dict[str, Any]]:
    """Benchmark circuits with increasing gate depth."""
    results = []
    try:
        from zuntenium import QuantumCircuit, get_backend

        backend = get_backend("statevector_simulator")
        for depth in [10, 50, 100, 200]:
            qc = QuantumCircuit(4)
            for _ in range(depth):
                for q in range(4):
                    qc.h(q)
                qc.cx(0, 1).cx(2, 3)

            start = time.perf_counter()
            _ = backend.run(qc, shots=shots)
            elapsed = time.perf_counter() - start

            results.append({
                "benchmark": "circuit_depth_scaling",
                "gate_depth": depth,
                "num_qubits": 4,
                "time_seconds": round(elapsed, 6),
            })
            print(f"  depth={depth:4d}: {elapsed*1000:.3f} ms")
    except ImportError as e:
        print(f"WARN: Could not import zuntenium: {e}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="ZUNTENIUM Benchmark Runner")
    parser.add_argument("--output", default="benchmark_results.json",
                        help="Output JSON file path")
    parser.add_argument("--qubit-range", default="2,16",
                        help="Comma-separated start,end for qubit sweep (default: 2,16)")
    parser.add_argument("--shots", type=int, default=1024,
                        help="Number of measurement shots (default: 1024)")
    args = parser.parse_args()

    q_start, q_end = (int(x) for x in args.qubit_range.split(","))
    qubit_range = range(q_start, q_end + 1)

    print("═" * 52)
    print(f" ZUNTENIUM Benchmark Runner")
    print(f" Qubits: {q_start}–{q_end} | Shots: {args.shots}")
    print("═" * 52)

    all_results: dict[str, Any] = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "shots": args.shots,
            "qubit_range": [q_start, q_end],
        },
        "benchmarks": [],
    }

    print("\n[1/2] Statevector benchmarks:")
    all_results["benchmarks"].extend(_bench_statevector(qubit_range, args.shots))

    print("\n[2/2] Circuit depth benchmarks:")
    all_results["benchmarks"].extend(_bench_circuit_depth(args.shots))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_results, indent=2))

    print(f"\n✓ Results written to: {out_path.resolve()}")
    print("═" * 52)


if __name__ == "__main__":
    main()
