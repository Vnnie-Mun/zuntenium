"""
test_density_matrix_simulator.py — Unit tests for DensityMatrixSimulator backend.

Tests:
- Initialization and memory boundary guard (MAX_DM_QUBITS)
- Pure state unitary evolution (X, H, CNOT -> Bell state)
- Noise channel application (Kraus operators, purity decrease)
- Measurement sampling and statistics
- Execution via transpiler v2 Circuit interface
"""
import numpy as np
import pytest

from zuntenium.backends.density_matrix_simulator import DensityMatrixSimulator, MAX_DM_QUBITS
from zuntenium.quantum_core.statevector.gate_library import GateLibrary
from zuntenium.transpiler.v2.core import Circuit, Gate


class TestDensityMatrixSimulator:

    def test_init_and_max_qubits_guard(self):
        sim = DensityMatrixSimulator(2)
        assert sim.num_qubits == 2
        assert np.isclose(sim.purity(), 1.0)
        assert np.isclose(sim.von_neumann_entropy(), 0.0)

        # Above MAX_DM_QUBITS should raise ValueError
        with pytest.raises(ValueError, match="supports at most"):
            DensityMatrixSimulator(MAX_DM_QUBITS + 1)

    def test_single_qubit_evolution(self):
        sim = DensityMatrixSimulator(1)
        # Apply X gate
        x_mat = GateLibrary.get("X")
        sim.apply_gate(x_mat, [0])

        probs = sim.get_probabilities()
        # |1><1| -> index 1 has prob 1.0
        assert np.isclose(probs[1], 1.0)
        assert np.isclose(probs[0], 0.0)
        assert np.isclose(sim.purity(), 1.0)

    def test_bell_state_generation(self):
        sim = DensityMatrixSimulator(2)
        h_mat = GateLibrary.get("H")
        cx_mat = GateLibrary.get("CX")

        sim.apply_gate(h_mat, [0])
        sim.apply_gate(cx_mat, [0, 1])

        probs = sim.get_probabilities()
        # Bell state (|00> + |11>) / sqrt(2)
        # indices 0 (|00>) and 3 (|11>) should have probability 0.5
        assert np.isclose(probs[0], 0.5)
        assert np.isclose(probs[3], 0.5)
        assert np.isclose(probs[1], 0.0)
        assert np.isclose(probs[2], 0.0)
        assert np.isclose(sim.purity(), 1.0)

        # Sampling test
        counts = sim.sample(shots=500)
        assert sum(counts.values()) == 500
        assert set(counts.keys()).issubset({"00", "11"})

    def test_kraus_noise_channel(self):
        sim = DensityMatrixSimulator(1)
        # Apply amplitude damping noise:
        # K0 = [[1, 0], [0, sqrt(1-gamma)]]
        # K1 = [[0, sqrt(gamma)], [0, 0]]
        gamma = 0.3
        K0 = np.array([[1.0, 0.0], [0.0, np.sqrt(1.0 - gamma)]], dtype=np.complex128)
        K1 = np.array([[0.0, np.sqrt(gamma)], [0.0, 0.0]], dtype=np.complex128)

        # First put qubit in superposition (|0> + |1>) / sqrt(2)
        sim.apply_gate(GateLibrary.get("H"), [0])
        assert np.isclose(sim.purity(), 1.0)

        # Apply noise channel
        sim.apply_noise_channel([K0, K1], qubit=0)

        # State becomes mixed -> purity < 1.0, entropy > 0.0
        assert sim.purity() < 1.0
        assert sim.von_neumann_entropy() > 0.0

    def test_circuit_execution_interface(self):
        circ = Circuit(2)
        circ.add_gate(Gate("H", (0,)))
        circ.add_gate(Gate("CX", (1,), controls=(0,)))

        sim = DensityMatrixSimulator(2)
        result = sim.execute(circ, shots=100)

        assert result.num_qubits == 2
        assert result.shots == 100
        assert result.metadata["backend"] == "density_matrix"
        assert np.isclose(result.metadata["purity"], 1.0)
        assert sum(result.counts.values()) == 100
