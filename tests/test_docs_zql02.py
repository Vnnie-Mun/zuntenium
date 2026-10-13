"""
test_docs_zql02.py — Verification tests for ZQL-02, ZQL-02A, and ZQL-02B guides.

Validates that code snippets and data structure examples in the ZQL-02 curriculum
execute correctly against the active Zuntenium SDK.
"""
from typing import Dict, List, Optional, Sequence
import numpy as np
import pytest

from zuntenium.quantum_core.statevector.statevector_state import StatevectorState
from zuntenium.quantum_core.statevector.gate_applicator import GateApplicator
from zuntenium.quantum_core.statevector.gate_library import GateLibrary
from zuntenium.quantum_core.statevector.measurement_engine import MeasurementEngine
from zuntenium.quantum_core.statevector.simulator_backend import SimulatorBackend
from zuntenium.quantum_core.statevector.statevector_simulator import StatevectorSimulator
from zuntenium.backends.density_matrix_simulator import DensityMatrixSimulator
from zuntenium.quantum_core.statevector.density_matrix import DensityMatrix
from zuntenium.transpiler.v2.core import Circuit, Gate
from zuntenium.transpiler.coupling_map import CouplingMap


def test_docs_zql02a_modular_statevector_operations():
    """Verify modular operations: StatevectorState + GateApplicator + MeasurementEngine."""
    state = StatevectorState(2)
    measurement_engine = MeasurementEngine()

    h_mat = GateLibrary.get("H")
    GateApplicator.apply(state.data, h_mat, qubits=[0])

    cx_mat = GateLibrary.get("CX")
    GateApplicator.apply(state.data, cx_mat, qubits=[0, 1])

    probs = MeasurementEngine.probabilities(state.data)
    assert np.isclose(probs[0], 0.5)
    assert np.isclose(probs[3], 0.5)
    assert np.isclose(probs[1], 0.0)
    assert np.isclose(probs[2], 0.0)

    counts = MeasurementEngine.sample_counts(state.data, shots=500)
    assert sum(counts.values()) == 500
    assert set(counts.keys()).issubset({"00", "11"})


def test_docs_zql02a_density_matrix_kraus_noise():
    """Verify DensityMatrixSimulator with Kraus noise channels from ZQL-02A."""
    sim = DensityMatrixSimulator(1)
    sim.apply_gate(GateLibrary.get("H"), qubits=[0])
    assert np.isclose(sim.purity(), 1.0)

    gamma = 0.2
    K0 = np.array([[1.0, 0.0], [0.0, np.sqrt(1.0 - gamma)]], dtype=np.complex128)
    K1 = np.array([[0.0, np.sqrt(gamma)], [0.0, 0.0]], dtype=np.complex128)

    sim.apply_noise_channel([K0, K1], qubit=0)
    assert sim.purity() < 1.0
    assert sim.von_neumann_entropy() > 0.0


def test_docs_zql02a_custom_logging_backend():
    """Verify custom simulator subclassing SimulatorBackend from ZQL-02A."""
    class LoggingSimulator(SimulatorBackend):
        def __init__(self, num_qubits: int):
            self._sim = StatevectorSimulator(num_qubits)
            self.log: List[str] = []

        def initialize(self, num_qubits: int) -> None:
            self.log.append(f"INIT n={num_qubits}")
            self._sim.initialize(num_qubits)

        def apply_gate(self, matrix: np.ndarray, qubits: Sequence[int]) -> None:
            self.log.append(f"GATE on qubits {qubits}")
            self._sim.apply_gate(matrix, qubits)

        def get_probabilities(self) -> np.ndarray:
            return self._sim.get_probabilities()

        def measure(
            self,
            qubits: Sequence[int],
            collapse: bool = True,
        ) -> Dict[str, int]:
            self.log.append(f"MEASURE qubits={qubits} collapse={collapse}")
            return self._sim.measure(qubits=qubits, collapse=collapse)

        def sample(self, shots: int, qubits: Optional[Sequence[int]] = None) -> Dict[str, int]:
            self.log.append(f"SAMPLE shots={shots}")
            return self._sim.sample(shots=shots, qubits=qubits)

        def get_statevector(self) -> np.ndarray:
            return self._sim.get_statevector()

    sim = LoggingSimulator(2)
    sim.apply_gate(GateLibrary.get("H"), qubits=[0])
    sim.apply_gate(GateLibrary.get("CX"), qubits=[0, 1])
    counts = sim.sample(shots=100)

    assert len(sim.log) == 3
    assert sim.log[0] == "GATE on qubits [0]"
    assert sim.log[1] == "GATE on qubits [0, 1]"
    assert sim.log[2] == "SAMPLE shots=100"
    assert sum(counts.values()) == 100


def test_docs_zql02b_circuit_ir_and_gate_data_structure():
    """Verify Circuit and Gate dataclass behavior from ZQL-02B."""
    circ = Circuit(num_qubits=2)
    g1 = Gate("H", targets=(0,))
    g2 = Gate("CX", targets=(1,), controls=(0,))
    circ.add_gate(g1)
    circ.add_gate(g2)

    assert circ.num_qubits == 2
    assert circ.num_gates == 2
    assert g1.structural_eq(Gate("H", targets=(0,)))
    assert not g1.structural_eq(g2)


def test_docs_zql02b_density_matrix_and_coupling_map():
    """Verify DensityMatrix and CouplingMap data structures from ZQL-02B."""
    dm = DensityMatrix(num_qubits=2)
    assert dm.matrix.shape == (4, 4)
    assert np.isclose(dm.purity(), 1.0)

    edges = [(0, 1), (1, 2)]
    cmap = CouplingMap(num_qubits=3, edges=edges)
    assert cmap.distance(0, 2) == 2
    assert set(cmap.neighbors(1)) == {0, 2}
