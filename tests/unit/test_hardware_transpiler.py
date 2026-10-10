"""
Unit tests for ZUNTENIUM Hardware-Aware Transpiler (Phase 3).
"""
import pytest
from zuntenium.transpiler.hardware_transpiler import (
    HardwareAwareTranspiler,
    HardwareConfig,
    HardwareProfileRegistry,
    HardwareTranspilerTestSuite,
    build_hardware_executor,
)
from zuntenium.transpiler.v2 import Circuit, Gate


@pytest.fixture
def transpiler_suite():
    return HardwareTranspilerTestSuite()


def test_profiles_registered(transpiler_suite):
    transpiler_suite.test_profiles_registered()
    assert transpiler_suite._f == 0


def test_coupling_graph_weights(transpiler_suite):
    transpiler_suite.test_coupling_graph_weights()
    assert transpiler_suite._f == 0


def test_qubit_mapping_ibm(transpiler_suite):
    transpiler_suite.test_qubit_mapping_ibm()
    assert transpiler_suite._f == 0


def test_shortest_swap_path_non_adjacent(transpiler_suite):
    transpiler_suite.test_shortest_swap_path_non_adjacent()
    assert transpiler_suite._f == 0


def test_gate_decomp_ibm_H(transpiler_suite):
    transpiler_suite.test_gate_decomp_ibm_H()
    assert transpiler_suite._f == 0


def test_gate_decomp_ibm_CNOT(transpiler_suite):
    transpiler_suite.test_gate_decomp_ibm_CNOT()
    assert transpiler_suite._f == 0


def test_gate_decomp_rigetti(transpiler_suite):
    transpiler_suite.test_gate_decomp_rigetti()
    assert transpiler_suite._f == 0


def test_gate_decomp_ionq(transpiler_suite):
    transpiler_suite.test_gate_decomp_ionq()
    assert transpiler_suite._f == 0


def test_hardware_cost_function(transpiler_suite):
    transpiler_suite.test_hardware_cost_function()
    assert transpiler_suite._f == 0


def test_full_hw_transpile_bell(transpiler_suite):
    transpiler_suite.test_full_hw_transpile_bell()
    assert transpiler_suite._f == 0


def test_full_hw_transpile_ghz(transpiler_suite):
    transpiler_suite.test_full_hw_transpile_ghz()
    assert transpiler_suite._f == 0


def test_cost_lower_after_hw_transpile(transpiler_suite):
    transpiler_suite.test_cost_lower_after_hw_transpile()
    assert transpiler_suite._f == 0


def test_backward_compat_no_hw(transpiler_suite):
    transpiler_suite.test_backward_compat_no_hw()
    assert transpiler_suite._f == 0


def test_hw_executor_bell(transpiler_suite):
    transpiler_suite.test_hw_executor_bell()
    assert transpiler_suite._f == 0


def test_ionq_all_to_all_no_swaps(transpiler_suite):
    transpiler_suite.test_ionq_all_to_all_no_swaps()
    assert transpiler_suite._f == 0


def test_rigetti_full_pipeline(transpiler_suite):
    transpiler_suite.test_rigetti_full_pipeline()
    assert transpiler_suite._f == 0


def test_custom_hardware_config():
    config = HardwareConfig(
        name="test_backend",
        num_qubits=3,
        coupling_map=[(0, 1), (1, 2)],
        native_gates={"RZ", "SX", "X", "ECR"},
        gate_durations={"RZ": 0.0, "SX": 20.0, "X": 20.0, "ECR": 200.0},
        error_rates={"RZ": 0.0, "SX": 1e-4, "X": 1e-4, "ECR": 1e-2, "1Q": 1e-4, "2Q": 1e-2},
        link_errors={(0, 1): 1e-2, (1, 2): 1e-2},
        qubit_t1={0: 100.0, 1: 100.0, 2: 100.0},
        qubit_t2={0: 80.0, 1: 80.0, 2: 80.0},
    )
    transpiler = HardwareAwareTranspiler(config)
    circuit = Circuit(3)
    circuit.add_gate(Gate("H", (0,)))
    circuit.add_gate(Gate("CNOT", (2,), controls=(0,)))
    result = transpiler.transpile(circuit)
    assert result.circuit is not None
    assert result.hardware_name == "test_backend"
