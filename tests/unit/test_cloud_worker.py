"""
Unit tests for Zuntenium Cloud Worker execution pipeline.

Verifies:
  - Canonical circuit reconstruction from JSON
  - Backend registry resolution (SimulatorBackend, TensorNetwork, Stabilizer, GPU)
  - End-to-end execution on all available local backends
  - Proper result unpacking (counts, statevector, probabilities)
"""

import pytest
from zuntenium_cloud.worker import (
    _build_circuit,
    _get_backend_registry,
    _run_circuit_on_backend,
)


class TestCloudWorkerExecution:

    def test_build_circuit(self):
        circuit_json = {
            "num_qubits": 2,
            "name": "bell_test",
            "gates": [
                {"name": "H", "qubits": [0]},
                {"name": "CX", "qubits": [0, 1]},
            ],
        }
        circ = _build_circuit(circuit_json)
        assert circ.num_qubits == 2
        assert circ.num_gates == 2
        assert circ.depth() >= 1

    def test_backend_registry_contains_simulators(self):
        reg = _get_backend_registry()
        # Must contain standard local simulation backends
        assert "statevector" in reg
        assert "tensornetwork" in reg
        assert "stabilizer" in reg
        assert "gpu_statevector" in reg

    def test_run_circuit_on_statevector(self):
        circ = _build_circuit({
            "num_qubits": 2,
            "gates": [
                {"name": "h", "qubits": [0]},
                {"name": "cx", "qubits": [0, 1]},
            ],
        })
        reg = _get_backend_registry()
        result = _run_circuit_on_backend(circ, "statevector", reg, {"shots": 100})
        counts = result.get("counts", {})
        assert "00" in counts or "11" in counts
        assert sum(counts.values()) == 100
        assert result.get("statevector") is not None

    def test_run_circuit_on_tensornetwork(self):
        circ = _build_circuit({
            "num_qubits": 2,
            "gates": [
                {"name": "x", "qubits": [0]},
                {"name": "cx", "qubits": [0, 1]},
            ],
        })
        reg = _get_backend_registry()
        result = _run_circuit_on_backend(circ, "tensornetwork", reg, {"shots": 50})
        counts = result.get("counts", {})
        assert counts.get("11") == 50

    def test_run_circuit_on_stabilizer(self):
        circ = _build_circuit({
            "num_qubits": 2,
            "gates": [
                {"name": "h", "qubits": [0]},
                {"name": "cx", "qubits": [0, 1]},
            ],
        })
        reg = _get_backend_registry()
        result = _run_circuit_on_backend(circ, "stabilizer", reg, {"shots": 100})
        counts = result.get("counts", {})
        assert "00" in counts or "11" in counts
        assert sum(counts.values()) == 100

    def test_run_circuit_on_gpu_statevector(self):
        circ = _build_circuit({
            "num_qubits": 2,
            "gates": [
                {"name": "h", "qubits": [0]},
                {"name": "cx", "qubits": [0, 1]},
            ],
        })
        reg = _get_backend_registry()
        result = _run_circuit_on_backend(circ, "gpu_statevector", reg, {"shots": 100})
        counts = result.get("counts", {})
        assert "00" in counts or "11" in counts
        assert sum(counts.values()) == 100
        assert result.get("statevector") is not None

    def test_run_circuit_empty_or_none(self):
        reg = _get_backend_registry()
        res = _run_circuit_on_backend(None, "statevector", reg, {"shots": 100})
        assert res["counts"] == {}
        assert res["statevector"] is None

    def test_run_circuit_fallback_on_unknown_backend(self):
        circ = _build_circuit({
            "num_qubits": 2,
            "gates": [
                {"name": "x", "qubits": [0]},
            ],
        })
        reg = _get_backend_registry()
        res = _run_circuit_on_backend(circ, "nonexistent_backend_xyz", reg, {"shots": 50})
        counts = res.get("counts", {})
        assert sum(counts.values()) == 50

    def test_run_circuit_parameterized_gates(self):
        circ = _build_circuit({
            "num_qubits": 2,
            "gates": [
                {"name": "rx", "qubits": [0], "params": [3.14159265]},
                {"name": "ry", "qubits": [1], "params": [1.57079632]},
            ],
        })
        reg = _get_backend_registry()
        res = _run_circuit_on_backend(circ, "statevector", reg, {"shots": 100})
        counts = res.get("counts", {})
        assert sum(counts.values()) == 100
