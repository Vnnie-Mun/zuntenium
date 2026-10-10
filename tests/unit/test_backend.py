"""
test_backend.py — Unit tests for StatevectorBackend, UnitaryBackend, SnapshotBackend.

Tests backend.run() → result shape, provider registry, and validation.
"""
from __future__ import annotations

import numpy as np
import pytest

from zuntenium import QuantumCircuit, get_backend, execute
from zuntenium.backend.provider import BackendProvider
from zuntenium.backend.simulator.statevector_backend import StatevectorBackend
from zuntenium.backend.simulator.unitary_backend import UnitaryBackend
from zuntenium.backend.simulator.snapshot_backend import SnapshotBackend


class TestStatevectorBackend:

    def test_run_bell_state(self):
        qc = QuantumCircuit(2)
        qc.h(0).cx(0, 1).measure_all()
        backend = StatevectorBackend()
        result = backend.run(qc, shots=1024)
        assert result.num_qubits == 2
        assert result.shots == 1024
        assert sum(result.counts.values()) == 1024

    def test_statevector_present(self):
        qc = QuantumCircuit(1).h(0)
        backend = StatevectorBackend()
        result = backend.run(qc, shots=100)
        assert result.statevector is not None
        assert len(result.statevector) == 2

    def test_max_qubits(self):
        backend = StatevectorBackend()
        assert backend.max_qubits == 30

    def test_is_simulator(self):
        backend = StatevectorBackend()
        assert backend.is_simulator is True

    def test_result_probabilities(self):
        qc = QuantumCircuit(1).x(0).measure_all()
        backend = StatevectorBackend()
        result = backend.run(qc, shots=512)
        probs = result.probabilities
        assert "1" in probs or "01" in probs or any("1" in k for k in probs)


class TestBackendProvider:

    def test_get_statevector_simulator(self):
        backend = get_backend("statevector_simulator")
        assert backend is not None
        assert backend.is_simulator

    def test_available_backends_not_empty(self):
        backends = BackendProvider.available_backends()
        assert len(backends) > 0
        assert "zuntenium_statevector_simulator" in backends

    def test_unknown_backend_raises(self):
        with pytest.raises(Exception):
            get_backend("nonexistent_backend_xyz")


class TestExecuteAPI:

    def test_execute_returns_result(self):
        qc = QuantumCircuit(2).h(0).cx(0, 1).measure_all()
        result = execute(qc, shots=256)
        assert result.shots == 256
        assert isinstance(result.counts, dict)

    def test_execute_with_named_backend(self):
        qc = QuantumCircuit(1).x(0).measure_all()
        backend = get_backend("statevector_simulator")
        result = execute(qc, backend=backend, shots=100)
        assert result.num_qubits == 1
