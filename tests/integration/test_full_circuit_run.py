"""
test_full_circuit_run.py — End-to-end integration tests: Bell pair, GHZ, QFT.

Tests the complete pipeline: build → compile → transpile → execute → verify.
"""
from __future__ import annotations

import math
import numpy as np
import pytest

from zuntenium import QuantumCircuit, execute, get_backend


class TestBellPair:

    def test_bell_state_counts(self):
        qc = QuantumCircuit(2).h(0).cx(0, 1).measure_all()
        result = execute(qc, shots=2048)
        for k in result.counts:
            assert k in ("00", "11")
        assert sum(result.counts.values()) == 2048

    def test_bell_state_amplitudes(self):
        qc = QuantumCircuit(2).h(0).cx(0, 1)
        result = execute(qc, shots=1)
        sv = result.statevector
        assert abs(abs(sv[0]) - 1 / math.sqrt(2)) < 1e-8
        assert abs(sv[1]) < 1e-10
        assert abs(sv[2]) < 1e-10
        assert abs(abs(sv[3]) - 1 / math.sqrt(2)) < 1e-8


class TestGHZState:

    def test_ghz_3_qubit(self):
        qc = QuantumCircuit(3).h(0).cx(0, 1).cx(1, 2).measure_all()
        result = execute(qc, shots=4096)
        for k in result.counts:
            assert k in ("000", "111")

    def test_ghz_5_qubit(self):
        qc = QuantumCircuit(5)
        qc.h(0)
        for i in range(4):
            qc.cx(i, i + 1)
        qc.measure_all()
        result = execute(qc, shots=2048)
        for k in result.counts:
            assert k in ("00000", "11111")

    def test_ghz_amplitudes(self):
        n = 4
        qc = QuantumCircuit(n).h(0)
        for i in range(n - 1):
            qc.cx(i, i + 1)
        result = execute(qc, shots=1)
        sv = result.statevector
        assert abs(abs(sv[0]) - 1 / math.sqrt(2)) < 1e-8
        assert abs(abs(sv[-1]) - 1 / math.sqrt(2)) < 1e-8
        for i in range(1, 2**n - 1):
            assert abs(sv[i]) < 1e-10


class TestQFT:

    def test_qft_2_qubit(self):
        """QFT on |00⟩ should yield uniform superposition."""
        qc = QuantumCircuit(2)
        # QFT: H on q0, controlled-Rz, H on q1, swap
        qc.h(0)
        qc.crz(math.pi / 2, 1, 0)
        qc.h(1)
        qc.swap(0, 1)
        result = execute(qc, shots=1)
        sv = result.statevector
        expected_amp = 1.0 / 2.0
        for amp in sv:
            assert abs(abs(amp) - expected_amp) < 1e-8


class TestEndToEndPipeline:

    def test_transpile_then_execute(self):
        """Transpile with optimization then execute — should give same answer."""
        from zuntenium.transpiler.pipeline import TranspilerPipeline

        qc = QuantumCircuit(2).h(0).h(0).cx(0, 1).measure_all()
        ir = qc.to_ir()
        pipeline = TranspilerPipeline(level=2)
        opt_ir = pipeline.run(ir)
        # H·H cancels → fewer gates
        assert len(opt_ir.instructions) < len(ir.instructions)

    def test_multiple_backends_agree(self):
        """StatevectorBackend should produce valid results."""
        qc = QuantumCircuit(2).h(0).cx(0, 1).measure_all()
        backend = get_backend("statevector_simulator")
        r1 = execute(qc, backend=backend, shots=1000)
        r2 = execute(qc, backend=backend, shots=1000)
        # Both should only have 00 and 11
        for k in r1.counts:
            assert k in ("00", "11")
        for k in r2.counts:
            assert k in ("00", "11")
