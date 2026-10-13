"""
test_external_sdk_usage.py — Unit tests verifying external SDK programmer experience.

Tests all programmer entrypoints, top-level exports, string backend dispatch in execute(),
BackendProvider discovery for extended backends, and histogram visualization helpers.
"""
import pytest

import zuntenium as zq
from zuntenium import (
    BaseGate,
    CouplingMap,
    DensityMatrixSimulator,
    ExecutionResult,
    GPUStatevectorSimulator,
    Gate,
    Instruction,
    MeasurementResult,
    QuantumCircuit,
    StabilizerSimulator,
    execute,
    get_backend,
    plot_histogram,
)
from zuntenium.backend.provider import BackendProvider


def test_top_level_exports():
    """Verify all public symbols are importable from top-level package."""
    assert DensityMatrixSimulator is not None
    assert StabilizerSimulator is not None
    assert GPUStatevectorSimulator is not None
    assert Gate is not None
    assert Instruction is not None
    assert BaseGate is not None
    assert CouplingMap is not None
    assert ExecutionResult is not None
    assert MeasurementResult is not None


def test_backend_provider_extended_discovery():
    """Verify all extended simulators are registered and discoverable."""
    available = BackendProvider.available_backends()
    assert "zuntenium_density_matrix_simulator" in available
    assert "zuntenium_stabilizer_simulator" in available
    assert "zuntenium_tensor_network_simulator" in available
    assert "zuntenium_gpu_statevector_simulator" in available


def test_execute_string_backend_dispatch():
    """Verify execute() works with string names for all backends."""
    qc = QuantumCircuit(2)
    qc.h(0).cx(0, 1).measure_all()

    backends = ["statevector", "density_matrix", "stabilizer", "tensor_network", "gpu"]
    for b_name in backends:
        result = execute(qc, backend=b_name, shots=200)
        assert isinstance(result, ExecutionResult)
        assert result.shots == 200
        assert sum(result.counts.values()) == 200
        assert set(result.counts.keys()).issubset({"00", "11"})


def test_get_backend_aliases():
    """Verify alias resolution for backend names."""
    aliases = {
        "densitymatrix": "zuntenium_density_matrix_simulator",
        "density_matrix": "zuntenium_density_matrix_simulator",
        "clifford": "zuntenium_stabilizer_simulator",
        "stabilizer": "zuntenium_stabilizer_simulator",
        "mps": "zuntenium_tensor_network_simulator",
        "tensornetwork": "zuntenium_tensor_network_simulator",
        "gpu": "zuntenium_gpu_statevector_simulator",
        "gpu_statevector": "zuntenium_gpu_statevector_simulator",
    }
    for alias, canonical in aliases.items():
        b = get_backend(alias)
        assert b.name == canonical


def test_plot_histogram_dict_and_result():
    """Verify plot_histogram accepts both dict and ExecutionResult."""
    # Should not raise TypeError or AttributeError
    plot_histogram({"00": 50, "11": 50})

    res = ExecutionResult(counts={"00": 60, "11": 40}, num_qubits=2, shots=100)
    plot_histogram(res)
