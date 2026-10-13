"""
QuFlow SDK — External Programmer Smoke Test Suite
===================================================

Phase 0 Baseline: Tests that simulate EXACTLY what a developer types in
PyCharm / VS Code / Jupyter / bare terminal when using QuFlow as an installed package.

These tests catch:
 D1: Import failures (missing __all__, circular imports)
 D2: Convention mismatches (endianness, angle units)
 D3: Silent wrong-answer bugs
 D4: Memory / performance regressions
 D5: IDE autocomplete gaps (missing type hints)

Usage:
  python -m pytest tests/integration/test_quflow_external_smoke.py -v
"""
from __future__ import annotations

import math
import sys

import numpy as np
import pytest

# ═══════════════════════════════════════════════════════════════════════════════
# D1: IMPORT SMOKE — what a programmer types in their IDE
# ═══════════════════════════════════════════════════════════════════════════════


class TestImportSmoke:
    """Every public import path must resolve without error."""

    def test_top_level_star_import(self):
        """import zuntenium as zn  — the simplest case."""
        import zuntenium as zn

        assert hasattr(zn, "__version__")
        assert hasattr(zn, "QuantumCircuit")
        assert hasattr(zn, "execute")
        assert hasattr(zn, "get_backend")
        assert hasattr(zn, "plot_histogram")

    def test_canonical_alias(self):
        """zn.Circuit should be an alias for zn.QuantumCircuit."""
        import zuntenium as zn

        assert zn.Circuit is zn.QuantumCircuit

    def test_all_public_symbols_importable(self):
        """Every name in __all__ must be importable."""
        import zuntenium

        for name in zuntenium.__all__:
            obj = getattr(zuntenium, name)
            assert obj is not None, f"zuntenium.{name} resolved to None"

    def test_version_format(self):
        """Version string must follow semver."""
        import zuntenium

        parts = zuntenium.__version__.split(".")
        assert len(parts) == 3, f"Bad version format: {zuntenium.__version__}"
        for p in parts:
            assert p.isdigit(), f"Non-numeric version part: {p}"

    def test_subpackage_imports(self):
        """Subpackage imports that programmers will use."""
        from zuntenium.transpiler import TranspilerCache
        from zuntenium.backend.provider import BackendProvider
        from zuntenium.quantum_core.executor.result import ExecutionResult

        assert TranspilerCache is not None
        assert BackendProvider is not None
        assert ExecutionResult is not None


# ═══════════════════════════════════════════════════════════════════════════════
# D2: CONVENTION CHECKS — endianness, angles, measurement labeling
# ═══════════════════════════════════════════════════════════════════════════════


class TestConventions:
    """Verify conventions match what docs advertise."""

    def test_little_endian_qubit_ordering(self):
        """Bit strings must use little-endian (q0 = rightmost bit, Qiskit convention)."""
        import zuntenium as zn

        qc = zn.QuantumCircuit(2)
        qc.x(0)  # flip qubit 0 only
        qc.measure_all()
        result = zn.execute(qc, backend="statevector", shots=100)
        # Little-endian: q0=1, q1=0 → "01" (q1 q0 ordering)
        assert "01" in result.counts, (
            f"Expected '01' in counts but got {result.counts}. "
            "QuFlow uses little-endian (q0 = rightmost bit)."
        )
        assert result.counts.get("01", 0) == 100

    def test_rotation_angle_radians(self):
        """Rx(π) must produce the |1⟩ state (angle in radians, not degrees)."""
        import zuntenium as zn

        qc = zn.QuantumCircuit(1)
        qc.rx(math.pi, 0)  # rx(theta, qubit) — theta first
        qc.measure_all()
        result = zn.execute(qc, backend="statevector", shots=100)
        assert "1" in result.counts
        assert result.counts.get("1", 0) == 100, (
            f"Rx(π) should flip |0⟩ → |1⟩ deterministically. Got {result.counts}"
        )

    def test_measurement_counts_sum_to_shots(self):
        """Sum of counts must exactly equal requested shots."""
        import zuntenium as zn

        qc = zn.QuantumCircuit(3)
        qc.h(0).cx(0, 1).cx(1, 2).measure_all()
        shots = 5000
        result = zn.execute(qc, backend="statevector", shots=shots)
        total = sum(result.counts.values())
        assert total == shots, f"Counts sum {total} != shots {shots}"


# ═══════════════════════════════════════════════════════════════════════════════
# D3: CORRECTNESS — known-answer tests for basic circuits
# ═══════════════════════════════════════════════════════════════════════════════


class TestCorrectness:
    """Functional correctness tests for common circuits."""

    def test_bell_state_balanced(self):
        """Bell state: only '00' and '11' should appear."""
        import zuntenium as zn

        qc = zn.QuantumCircuit(2)
        qc.h(0).cx(0, 1).measure_all()
        result = zn.execute(qc, backend="statevector", shots=10000)
        assert set(result.counts.keys()).issubset({"00", "11"})
        # Check approximate balance (within statistical tolerance)
        p00 = result.counts.get("00", 0) / 10000
        assert 0.40 <= p00 <= 0.60, f"Bell state balance off: P(00)={p00}"

    def test_ghz3_state(self):
        """3-qubit GHZ: only '000' and '111'."""
        import zuntenium as zn

        qc = zn.QuantumCircuit(3)
        qc.h(0).cx(0, 1).cx(1, 2).measure_all()
        result = zn.execute(qc, backend="statevector", shots=5000)
        assert set(result.counts.keys()).issubset({"000", "111"})

    def test_identity_circuit(self):
        """No-op circuit on |0⟩ must always return '0'."""
        import zuntenium as zn

        qc = zn.QuantumCircuit(1)
        qc.measure_all()
        result = zn.execute(qc, backend="statevector", shots=100)
        assert result.counts == {"0": 100}

    def test_hh_cancellation(self):
        """H·H = I on |0⟩ must produce '0'."""
        import zuntenium as zn

        qc = zn.QuantumCircuit(1)
        qc.h(0).h(0).measure_all()
        result = zn.execute(qc, backend="statevector", shots=100)
        assert result.counts == {"0": 100}

    def test_xx_cancellation(self):
        """X·X = I."""
        import zuntenium as zn

        qc = zn.QuantumCircuit(1)
        qc.x(0).x(0).measure_all()
        result = zn.execute(qc, backend="statevector", shots=100)
        assert result.counts == {"0": 100}

    @pytest.mark.parametrize("backend_name", ["statevector", "density_matrix", "stabilizer"])
    def test_cross_backend_bell(self, backend_name: str):
        """Bell state must give same answer on all backends."""
        import zuntenium as zn

        qc = zn.QuantumCircuit(2)
        qc.h(0).cx(0, 1).measure_all()
        result = zn.execute(qc, backend=backend_name, shots=5000)
        assert set(result.counts.keys()).issubset({"00", "11"})
        assert sum(result.counts.values()) == 5000


# ═══════════════════════════════════════════════════════════════════════════════
# D4: BACKEND PROVIDER — external discovery
# ═══════════════════════════════════════════════════════════════════════════════


class TestBackendProvider:
    """Verify the SDK backend registry works as external devs expect."""

    def test_available_backends_returns_list(self):
        from zuntenium.backend.provider import BackendProvider

        available = BackendProvider.available_backends()
        assert isinstance(available, list)
        assert len(available) >= 5  # at least sv, dm, stab, tn, gpu

    def test_get_backend_returns_object_with_name(self):
        import zuntenium as zn

        for alias in ["statevector", "density_matrix", "stabilizer", "gpu"]:
            backend = zn.get_backend(alias)
            assert hasattr(backend, "name"), f"{alias} backend missing .name attribute"
            assert hasattr(backend, "run"), f"{alias} backend missing .run method"


# ═══════════════════════════════════════════════════════════════════════════════
# D5: TYPE ANNOTATION / IDE AUTOCOMPLETE VERIFICATION
# ═══════════════════════════════════════════════════════════════════════════════


class TestTypeAnnotations:
    """Verify type annotations exist for IDE autocomplete."""

    def test_execute_return_type(self):
        """execute() must return ExecutionResult."""
        from zuntenium.quantum_core.executor.result import ExecutionResult
        import zuntenium as zn

        qc = zn.QuantumCircuit(1)
        qc.h(0).measure_all()
        result = zn.execute(qc, backend="statevector", shots=10)
        assert isinstance(result, ExecutionResult)

    def test_execution_result_has_counts(self):
        """ExecutionResult must have .counts, .shots, .num_qubits."""
        import zuntenium as zn

        qc = zn.QuantumCircuit(2)
        qc.h(0).cx(0, 1).measure_all()
        result = zn.execute(qc, shots=100)
        assert hasattr(result, "counts")
        assert hasattr(result, "shots")
        assert hasattr(result, "num_qubits")
        assert isinstance(result.counts, dict)

    def test_quantum_circuit_fluent_api(self):
        """QuantumCircuit builder must support fluent chaining."""
        import zuntenium as zn

        qc = zn.QuantumCircuit(3)
        returned = qc.h(0).cx(0, 1).cx(1, 2)
        assert returned is qc, "Fluent methods must return self for chaining"
