"""Test suite root conftest.py."""
import pytest


@pytest.fixture(scope="session")
def statevector_backend():
    from zuntenium.backend.simulator.statevector_backend import StatevectorBackend
    return StatevectorBackend(seed=42)

@pytest.fixture
def bell_circuit():
    from zuntenium.circuit.circuit import QuantumCircuit
    qc = QuantumCircuit(2, name="bell")
    qc.h(0).cx(0, 1).measure_all()
    return qc

@pytest.fixture
def ghz_circuit():
    from zuntenium.circuit.circuit import QuantumCircuit
    qc = QuantumCircuit(3, name="ghz")
    qc.h(0).cx(0, 1).cx(1, 2).measure_all()
    return qc
