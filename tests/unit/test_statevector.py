"""Unit tests for StateVector."""
import numpy as np
import pytest

from zuntenium.quantum_core.gates import CXGate, HGate, XGate
from zuntenium.quantum_core.statevector.statevector import StateVector


class TestStateVectorInitialization:
    """Tests for StateVector creation and initial state."""

    def test_single_qubit_initial_state(self) -> None:
        sv = StateVector(1)
        amps = sv.get_amplitudes()
        expected = np.array([1.0, 0.0], dtype=np.complex128)
        np.testing.assert_allclose(amps, expected, atol=1e-12)

    def test_two_qubit_initial_state(self) -> None:
        sv = StateVector(2)
        amps = sv.get_amplitudes()
        expected = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex128)
        np.testing.assert_allclose(amps, expected, atol=1e-12)

    def test_initial_state_normalized(self) -> None:
        for n in range(1, 8):
            sv = StateVector(n)
            norm = np.sum(np.abs(sv.get_amplitudes()) ** 2)
            assert abs(norm - 1.0) < 1e-12, f"Not normalized for {n} qubits"

    def test_invalid_qubit_count(self) -> None:
        with pytest.raises(ValueError):
            StateVector(0)


class TestStateVectorGateApplication:
    """Tests for gate application."""

    def test_hadamard_creates_superposition(self) -> None:
        sv = StateVector(1)
        sv.apply_gate(HGate(), [0])
        amps = sv.get_amplitudes()
        expected = np.array([1 / np.sqrt(2), 1 / np.sqrt(2)], dtype=np.complex128)
        np.testing.assert_allclose(amps, expected, atol=1e-12)

    def test_x_gate_flips_state(self) -> None:
        sv = StateVector(1)
        sv.apply_gate(XGate(), [0])
        amps = sv.get_amplitudes()
        np.testing.assert_allclose(amps, np.array([0.0, 1.0]), atol=1e-12)

    def test_bell_state_preparation(self) -> None:
        sv = StateVector(2)
        sv.apply_gate(HGate(), [0])
        sv.apply_gate(CXGate(), [0, 1])
        amps = sv.get_amplitudes()
        expected = np.array([1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)], dtype=np.complex128)
        np.testing.assert_allclose(amps, expected, atol=1e-10)

    def test_norm_preserved_after_gate(self) -> None:
        sv = StateVector(3)
        sv.apply_gate(HGate(), [0])
        sv.apply_gate(HGate(), [1])
        sv.apply_gate(CXGate(), [0, 2])
        norm = float(np.sum(np.abs(sv.get_amplitudes()) ** 2))
        assert abs(norm - 1.0) < 1e-12

    def test_double_hadamard_is_identity(self) -> None:
        """Applying H twice should return to |0...0⟩."""
        sv = StateVector(2)
        initial = sv.get_amplitudes().copy()
        sv.apply_gate(HGate(), [0])
        sv.apply_gate(HGate(), [0])
        np.testing.assert_allclose(sv.get_amplitudes(), initial, atol=1e-12)


class TestStateVectorProperties:
    """Tests for StateVector measurement-related properties."""

    def test_probabilities_sum_to_one(self) -> None:
        sv = StateVector(3)
        sv.apply_gate(HGate(), [0])
        sv.apply_gate(HGate(), [1])
        probs = np.abs(sv.get_amplitudes()) ** 2
        assert abs(probs.sum() - 1.0) < 1e-12

    def test_probability_of_zero_state(self) -> None:
        sv = StateVector(1)
        probs = np.abs(sv.get_amplitudes()) ** 2
        assert abs(probs[0] - 1.0) < 1e-12
        assert abs(probs[1] - 0.0) < 1e-12
