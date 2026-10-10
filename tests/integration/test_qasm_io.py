"""Integration tests for QASM I/O round-trip."""
import math

from zuntenium.circuit.circuit import QuantumCircuit
from zuntenium.circuit.ir.serialization.json_serializer import ir_to_json, json_to_ir
from zuntenium.circuit.ir.serialization.qasm_parser import parse_qasm2
from zuntenium.circuit.ir.serialization.qasm_serializer import ir_to_qasm2


class TestQASM2Serialization:
    """Tests for QASM 2.0 serialization."""

    def test_bell_circuit_to_qasm(self) -> None:
        qc = QuantumCircuit(2)
        qc.h(0).cx(0, 1).measure_all()
        ir = qc.to_ir()
        qasm = ir_to_qasm2(ir)
        assert "OPENQASM 2.0" in qasm
        assert "qreg q[2]" in qasm
        assert "h q[0]" in qasm
        assert "cx q[0], q[1]" in qasm
        assert "measure q[0]" in qasm

    def test_parametric_gate_in_qasm(self) -> None:
        qc = QuantumCircuit(1)
        qc.rx(math.pi / 4, 0)
        ir = qc.to_ir()
        qasm = ir_to_qasm2(ir)
        assert "rx(" in qasm

    def test_barrier_in_qasm(self) -> None:
        qc = QuantumCircuit(2)
        qc.h(0).barrier(0, 1).cx(0, 1)
        ir = qc.to_ir()
        qasm = ir_to_qasm2(ir)
        assert "barrier" in qasm

    def test_reset_in_qasm(self) -> None:
        qc = QuantumCircuit(1)
        qc.x(0).reset(0)
        ir = qc.to_ir()
        qasm = ir_to_qasm2(ir)
        assert "reset" in qasm


class TestQASM2Parsing:
    """Tests for QASM 2.0 parsing."""

    def test_parse_bell_circuit(self) -> None:
        qasm = (
            'OPENQASM 2.0;\n'
            'include "qelib1.inc";\n'
            'qreg q[2];\n'
            'creg c[2];\n'
            'h q[0];\n'
            'cx q[0], q[1];\n'
            'measure q[0] -> c[0];\n'
            'measure q[1] -> c[1];\n'
        )
        ir = parse_qasm2(qasm)
        assert ir.num_qubits == 2
        assert ir.num_gates() == 2
        gate_names = {n.gate_name() for n in ir.instructions if n.is_gate()}
        assert "H" in gate_names
        assert "CX" in gate_names

    def test_parse_single_qubit_gates(self) -> None:
        qasm = (
            'OPENQASM 2.0;\n'
            'qreg q[1];\n'
            'h q[0];\n'
            'x q[0];\n'
            'y q[0];\n'
            'z q[0];\n'
        )
        ir = parse_qasm2(qasm)
        assert ir.num_gates() == 4

    def test_parse_parametric_gate(self) -> None:
        qasm = (
            'OPENQASM 2.0;\n'
            'qreg q[1];\n'
            'rx(1.5707963268) q[0];\n'
        )
        ir = parse_qasm2(qasm)
        assert ir.num_gates() == 1
        node = ir.instructions[0]
        assert node.gate_name() == "Rx"
        assert abs(node.params[0] - 1.5707963268) < 1e-8


class TestJSONRoundTrip:
    """Tests for JSON serialization/deserialization round-trip."""

    def test_bell_circuit_json_roundtrip(self) -> None:
        qc = QuantumCircuit(2)
        qc.h(0).cx(0, 1).measure_all()
        ir = qc.to_ir()
        json_str = ir_to_json(ir)
        recovered = json_to_ir(json_str)
        assert recovered.num_qubits == ir.num_qubits
        assert recovered.num_gates() == ir.num_gates()

    def test_all_gates_json_roundtrip(self) -> None:
        qc = QuantumCircuit(3)
        qc.h(0).x(1).cx(0, 1).rz(0.5, 2).ccx(0, 1, 2)
        ir = qc.to_ir()
        json_str = ir_to_json(ir)
        recovered = json_to_ir(json_str)
        assert recovered.num_gates() == ir.num_gates()

    def test_json_contains_version(self) -> None:
        import json
        qc = QuantumCircuit(1)
        qc.h(0)
        ir = qc.to_ir()
        data = json.loads(ir_to_json(ir))
        assert "version" in data
