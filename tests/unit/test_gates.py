"""Unit tests for quantum gates."""
import math

import numpy as np
import pytest

from zuntenium.quantum_core.gates import (
    CXGate,
    CZGate,
    HGate,
    RxGate,
    RyGate,
    RzGate,
    SGate,
    SWAPGate,
    TGate,
    ToffoliGate,
    U3Gate,
    XGate,
    YGate,
    ZGate,
)
from zuntenium.utils.validation.gate_validator import check_dimension, check_unitarity

ALL_SINGLE_QUBIT_GATES = [
    HGate(), XGate(), YGate(), ZGate(), SGate(), TGate(),
    RxGate(0.5), RyGate(1.0), RzGate(math.pi / 4),
    U3Gate(0.3, 0.2, 0.1),
]

ALL_TWO_QUBIT_GATES = [
    CXGate(), CZGate(), SWAPGate(),
]

ALL_THREE_QUBIT_GATES = [
    ToffoliGate(),
]


class TestGateUnitarity:
    """All gates must produce unitary matrices."""

    @pytest.mark.parametrize("gate", ALL_SINGLE_QUBIT_GATES)
    def test_single_qubit_gates_are_unitary(self, gate: object) -> None:
        assert check_unitarity(gate), f"{gate} is not unitary"  # type: ignore[arg-type]

    @pytest.mark.parametrize("gate", ALL_TWO_QUBIT_GATES)
    def test_two_qubit_gates_are_unitary(self, gate: object) -> None:
        assert check_unitarity(gate)  # type: ignore[arg-type]

    @pytest.mark.parametrize("gate", ALL_THREE_QUBIT_GATES)
    def test_three_qubit_gates_are_unitary(self, gate: object) -> None:
        assert check_unitarity(gate)  # type: ignore[arg-type]


class TestGateDimensions:
    """Gate matrix dimensions must match num_qubits."""

    @pytest.mark.parametrize("gate", ALL_SINGLE_QUBIT_GATES)
    def test_single_qubit_dimension(self, gate: object) -> None:
        assert check_dimension(gate)  # type: ignore[arg-type]
        assert gate.matrix().shape == (2, 2)  # type: ignore[union-attr]

    @pytest.mark.parametrize("gate", ALL_TWO_QUBIT_GATES)
    def test_two_qubit_dimension(self, gate: object) -> None:
        assert gate.matrix().shape == (4, 4)  # type: ignore[union-attr]

    @pytest.mark.parametrize("gate", ALL_THREE_QUBIT_GATES)
    def test_three_qubit_dimension(self, gate: object) -> None:
        assert gate.matrix().shape == (8, 8)  # type: ignore[union-attr]


class TestHadamardGate:
    """Tests specific to the Hadamard gate."""

    def test_h_is_self_inverse(self) -> None:
        h = HGate().matrix()
        result = h @ h
        np.testing.assert_allclose(result, np.eye(2), atol=1e-12)

    def test_h_maps_zero_to_plus(self) -> None:
        h = HGate().matrix()
        zero = np.array([1.0, 0.0], dtype=np.complex128)
        plus = h @ zero
        expected = np.array([1 / math.sqrt(2), 1 / math.sqrt(2)], dtype=np.complex128)
        np.testing.assert_allclose(plus, expected, atol=1e-12)

    def test_h_maps_one_to_minus(self) -> None:
        h = HGate().matrix()
        one = np.array([0.0, 1.0], dtype=np.complex128)
        minus = h @ one
        expected = np.array([1 / math.sqrt(2), -1 / math.sqrt(2)], dtype=np.complex128)
        np.testing.assert_allclose(minus, expected, atol=1e-12)


class TestPauliGates:
    """Tests specific to Pauli X, Y, Z."""

    def test_x_is_self_inverse(self) -> None:
        x = XGate().matrix()
        np.testing.assert_allclose(x @ x, np.eye(2), atol=1e-12)

    def test_y_is_self_inverse(self) -> None:
        y = YGate().matrix()
        np.testing.assert_allclose(y @ y, np.eye(2), atol=1e-12)

    def test_z_is_self_inverse(self) -> None:
        z = ZGate().matrix()
        np.testing.assert_allclose(z @ z, np.eye(2), atol=1e-12)

    def test_xyz_anticommutation(self) -> None:
        """XY + YX = 0 (anti-commute)."""
        x = XGate().matrix()
        y = YGate().matrix()
        anti = x @ y + y @ x
        np.testing.assert_allclose(anti, np.zeros((2, 2)), atol=1e-12)


class TestRotationGates:
    """Tests for parametric rotation gates."""

    def test_rx_at_zero_is_identity(self) -> None:
        rx = RxGate(0.0).matrix()
        np.testing.assert_allclose(rx, np.eye(2), atol=1e-12)

    def test_rx_at_pi_is_x(self) -> None:
        rx = RxGate(math.pi).matrix()
        x = XGate().matrix()
        np.testing.assert_allclose(np.abs(rx), np.abs(x), atol=1e-10)

    def test_rz_at_pi_is_z(self) -> None:
        rz = RzGate(math.pi).matrix()
        z = ZGate().matrix()
        # Global phase may differ
        ratio = rz[0, 0] / z[0, 0]
        np.testing.assert_allclose(rz / ratio, z, atol=1e-10)

    def test_rotation_continuity(self) -> None:
        """RxGate(θ) matrix entries vary continuously with θ."""
        for theta in np.linspace(0, 2 * math.pi, 20):
            gate = RxGate(theta)
            assert check_unitarity(gate)


class TestCNOTGate:
    """Tests for the CNOT gate."""

    def test_cnot_is_self_inverse(self) -> None:
        cx = CXGate().matrix()
        np.testing.assert_allclose(cx @ cx, np.eye(4), atol=1e-12)

    def test_cnot_flips_target_when_control_is_one(self) -> None:
        cx = CXGate().matrix()
        control_one_target_zero = np.array([0, 0, 1, 0], dtype=np.complex128)  # |10⟩
        result = cx @ control_one_target_zero
        expected = np.array([0, 0, 0, 1], dtype=np.complex128)  # |11⟩
        np.testing.assert_allclose(result, expected, atol=1e-12)
