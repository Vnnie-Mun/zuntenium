"""Unit tests for QuantumCircuit."""
import pytest

from zuntenium.circuit.circuit import QuantumCircuit
from zuntenium.quantum_core.exceptions import QubitIndexError


class TestQuantumCircuitConstruction:
    """Tests for QuantumCircuit builder API."""

    def test_create_single_qubit_circuit(self) -> None:
        qc = QuantumCircuit(1)
        assert qc.num_qubits == 1
        assert qc.num_gates == 0

    def test_create_multi_qubit(self) -> None:
        qc = QuantumCircuit(5)
        assert qc.num_qubits == 5

    def test_fluent_chaining(self) -> None:
        qc = QuantumCircuit(2)
        result = qc.h(0).cx(0, 1)
        assert result is qc
        assert qc.num_gates == 2

    def test_measure_all_adds_classical_bits(self) -> None:
        qc = QuantumCircuit(3)
        qc.h(0).h(1).h(2).measure_all()
        assert qc.num_clbits >= 3

    def test_circuit_name(self) -> None:
        qc = QuantumCircuit(2, name="test_circuit")
        assert qc.name == "test_circuit"

    def test_invalid_qubit_count(self) -> None:
        with pytest.raises(ValueError):
            QuantumCircuit(0)

    def test_out_of_range_qubit(self) -> None:
        qc = QuantumCircuit(2)
        with pytest.raises(QubitIndexError):
            qc.h(2)   # valid: 0, 1 only

    def test_all_single_qubit_gates(self) -> None:
        qc = QuantumCircuit(1)
        for gate_fn in [qc.x, qc.y, qc.z, qc.s, qc.t]:
            gate_fn(0)
        for gate_fn, arg in [(qc.rx, 0.5), (qc.ry, 1.0), (qc.rz, 1.5)]:
            gate_fn(arg, 0)
        assert qc.num_gates == 8

    def test_all_two_qubit_gates(self) -> None:
        qc = QuantumCircuit(2)
        qc.cx(0, 1).cz(0, 1).cy(0, 1).swap(0, 1)
        assert qc.num_gates == 4

    def test_toffoli_gate(self) -> None:
        qc = QuantumCircuit(3)
        qc.ccx(0, 1, 2)
        assert qc.num_gates == 1

    def test_fredkin_gate(self) -> None:
        qc = QuantumCircuit(3)
        qc.cswap(0, 1, 2)
        assert qc.num_gates == 1


class TestCircuitDepth:
    """Tests for circuit depth computation."""

    def test_empty_circuit_depth_zero(self) -> None:
        qc = QuantumCircuit(3)
        assert qc.depth == 0

    def test_sequential_depth(self) -> None:
        qc = QuantumCircuit(1)
        qc.h(0).x(0).y(0)
        assert qc.depth == 3

    def test_parallel_depth(self) -> None:
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.h(1)
        qc.h(2)   # parallel, depth = 1
        assert qc.depth == 1

    def test_mixed_depth(self) -> None:
        qc = QuantumCircuit(2)
        qc.h(0)         # layer 1
        qc.cx(0, 1)    # layer 2 (uses both)
        qc.h(0)        # layer 3
        assert qc.depth >= 2


class TestCircuitCompose:
    """Tests for circuit composition."""

    def test_compose_appends_instructions(self) -> None:
        qc1 = QuantumCircuit(2)
        qc1.h(0)
        qc2 = QuantumCircuit(2)
        qc2.cx(0, 1)
        qc1.compose(qc2)
        assert qc1.num_gates == 2

    def test_compose_incompatible_raises(self) -> None:
        qc1 = QuantumCircuit(1)
        qc2 = QuantumCircuit(3)
        with pytest.raises(Exception):
            qc1.compose(qc2)


class TestCircuitIR:
    """Tests for circuit → IR compilation."""

    def test_compile_produces_ir(self) -> None:
        from zuntenium.circuit.ir.quantum_ir import QuantumIR
        qc = QuantumCircuit(2)
        qc.h(0).cx(0, 1)
        ir = qc.to_ir()
        assert isinstance(ir, QuantumIR)
        assert ir.num_qubits == 2

    def test_ir_instruction_count(self) -> None:
        qc = QuantumCircuit(2)
        qc.h(0).cx(0, 1).measure(0, 0)
        ir = qc.to_ir()
        assert len(ir) == 3

    def test_barrier_in_ir(self) -> None:
        qc = QuantumCircuit(2)
        qc.h(0).barrier(0, 1).cx(0, 1)
        ir = qc.to_ir()
        ops = [n.operation for n in ir.instructions]
        assert "barrier" in ops

    def test_reset_in_ir(self) -> None:
        qc = QuantumCircuit(1)
        qc.x(0).reset(0)
        ir = qc.to_ir()
        ops = [n.operation for n in ir.instructions]
        assert "reset" in ops


class TestParameterBinding:
    """Tests for parametric circuit parameter binding."""

    def test_bind_parameters_returns_new_circuit(self) -> None:
        qc = QuantumCircuit(1)
        qc.rx(0.0, 0)
        bound = qc.bind_parameters([1.5])
        assert bound is not qc
