"""Integration tests for StatevectorBackend."""
import math

import numpy as np
import pytest

from zuntenium.api.quantum_api import execute, get_backend
from zuntenium.backend.simulator.statevector_backend import StatevectorBackend
from zuntenium.circuit.circuit import QuantumCircuit


@pytest.fixture
def backend() -> StatevectorBackend:
    return StatevectorBackend(seed=7)


class TestBellStateExecution:
    """Bell state preparation and measurement statistics."""

    def test_bell_state_counts(self, backend: StatevectorBackend) -> None:
        qc = QuantumCircuit(2)
        qc.h(0).cx(0, 1).measure_all()
        result = backend.run(qc, shots=4096)
        assert set(result.counts.keys()) <= {"00", "11"}
        total = sum(result.counts.values())
        assert total == 4096
        # Bell state should be ~50/50 between "00" and "11"
        for outcome in ["00", "11"]:
            fraction = result.counts.get(outcome, 0) / total
            assert 0.40 < fraction < 0.60, f"Unexpected fraction for {outcome}: {fraction}"

    def test_statevector_is_correct(self, backend: StatevectorBackend) -> None:
        qc = QuantumCircuit(2)
        qc.h(0).cx(0, 1)
        result = backend.run(qc, shots=1)
        assert result.statevector is not None
        expected = np.array([1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)], dtype=np.complex128)
        np.testing.assert_allclose(
            np.abs(result.statevector), np.abs(expected), atol=1e-10
        )


class TestGHZStateExecution:
    """GHZ state: (|000⟩ + |111⟩)/√2."""

    def test_ghz_only_produces_00x_or_11x(self, backend: StatevectorBackend) -> None:
        qc = QuantumCircuit(3)
        qc.h(0).cx(0, 1).cx(1, 2).measure_all()
        result = backend.run(qc, shots=2048)
        for outcome in result.counts.keys():
            assert outcome in {"000", "111"}, f"Unexpected outcome: {outcome}"


class TestSingleQubitRotations:
    """Test Rx/Ry/Rz rotations produce correct expectation values."""

    def test_rx_pi_over_2_equal_superposition(self, backend: StatevectorBackend) -> None:
        qc = QuantumCircuit(1)
        qc.rx(math.pi / 2, 0).measure_all()
        result = backend.run(qc, shots=8192)
        total = sum(result.counts.values())
        p0 = result.counts.get("0", 0) / total
        p1 = result.counts.get("1", 0) / total
        # Rx(π/2) from |0⟩ → cos(π/4)|0⟩ - i·sin(π/4)|1⟩ → P(0) = P(1) ≈ 0.5
        assert abs(p0 - 0.5) < 0.05
        assert abs(p1 - 0.5) < 0.05

    def test_x_gate_flips_state(self, backend: StatevectorBackend) -> None:
        qc = QuantumCircuit(1)
        qc.x(0).measure_all()
        result = backend.run(qc, shots=100)
        assert result.counts.get("1", 0) == 100
        assert result.counts.get("0", 0) == 0


class TestAPIFunctions:
    """Tests for the top-level execute() and get_backend() API."""

    def test_execute_default_backend(self) -> None:
        qc = QuantumCircuit(1)
        qc.x(0).measure_all()
        result = execute(qc, shots=50)
        assert result.counts.get("1", 0) == 50

    def test_get_backend_returns_statevector(self) -> None:
        b = get_backend("statevector_simulator")
        assert isinstance(b, StatevectorBackend)

    def test_execute_with_explicit_backend(self) -> None:
        b = StatevectorBackend(seed=123)
        qc = QuantumCircuit(2)
        qc.h(0).cx(0, 1).measure_all()
        result = execute(qc, backend=b, shots=1000)
        assert len(result.counts) >= 1
        assert sum(result.counts.values()) == 1000


class TestExecutionMetadata:
    """Tests for metadata in ExecutionResult."""

    def test_result_has_backend_metadata(self, backend: StatevectorBackend) -> None:
        qc = QuantumCircuit(1)
        qc.h(0).measure_all()
        result = backend.run(qc, shots=100)
        assert "backend" in result.metadata
        assert "exec_time_ms" in result.metadata

    def test_result_num_qubits_correct(self, backend: StatevectorBackend) -> None:
        for n in [1, 2, 3, 5]:
            qc = QuantumCircuit(n)
            qc.h(0).measure_all()
            result = backend.run(qc, shots=10)
            assert result.num_qubits == n

    def test_result_shots_correct(self, backend: StatevectorBackend) -> None:
        qc = QuantumCircuit(1)
        qc.h(0).measure_all()
        for shots in [1, 100, 1024]:
            result = backend.run(qc, shots=shots)
            assert result.shots == shots
