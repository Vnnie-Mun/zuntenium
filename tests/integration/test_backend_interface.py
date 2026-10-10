"""
test_backend_interface.py — Integration: real vs mock hardware backend protocol.

Tests that MockHardwareBackend exposes the same interface as AbstractHardwareBackend.
"""
from __future__ import annotations

import pytest

from zuntenium import QuantumCircuit
from zuntenium.backend.hardware.mock.mock_hardware import MockHardwareBackend
from zuntenium.backend.base_backend import BackendStatus


class TestMockHardwareBackend:

    def test_run_returns_result(self):
        backend = MockHardwareBackend(seed=42)
        qc = QuantumCircuit(2).h(0).cx(0, 1).measure_all()
        result = backend.run(qc, shots=512)
        assert result.num_qubits == 2
        assert sum(result.counts.values()) == 512

    def test_submit_and_fetch_job(self):
        backend = MockHardwareBackend()
        qc = QuantumCircuit(1).x(0).measure_all()
        job = backend.submit_job(qc, shots=100)
        assert job.is_done()
        result = backend.fetch_result(job.job_id)
        assert result.shots == 100

    def test_cancel_job(self):
        backend = MockHardwareBackend()
        qc = QuantumCircuit(1).h(0).measure_all()
        job = backend.submit_job(qc)
        assert backend.cancel_job(job.job_id) is True
        assert backend.cancel_job("nonexistent") is False

    def test_status_operational(self):
        backend = MockHardwareBackend()
        status = backend.status()
        assert isinstance(status, BackendStatus)
        assert status.operational is True

    def test_validate_circuit_too_many_qubits(self):
        backend = MockHardwareBackend()
        qc = QuantumCircuit(10).h(0)
        validation = backend.validate_circuit(qc)
        assert validation.valid is False
        assert len(validation.errors) > 0

    def test_validate_circuit_ok(self):
        backend = MockHardwareBackend()
        qc = QuantumCircuit(3).h(0).cx(0, 1)
        validation = backend.validate_circuit(qc)
        assert validation.valid is True

    def test_metadata_includes_backend_name(self):
        backend = MockHardwareBackend()
        qc = QuantumCircuit(1).x(0).measure_all()
        result = backend.run(qc, shots=50)
        assert result.metadata.get("backend") == "mock_hardware"

    def test_fetch_nonexistent_job_raises(self):
        backend = MockHardwareBackend()
        with pytest.raises(KeyError):
            backend.fetch_result("nonexistent-job-id")
