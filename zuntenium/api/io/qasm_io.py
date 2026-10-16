"""
QASM I/O — load_qasm() and save_qasm() public API.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from zuntenium.circuit.circuit import QuantumCircuit


def load_qasm(path: str | Path) -> QuantumCircuit:
    """
    Load a QuantumCircuit from an OpenQASM 2.0 file.

    Args:
        path: Path to .qasm file.

    Returns:
        QuantumCircuit populated from the QASM source.
    """
    from zuntenium.circuit.circuit import QuantumCircuit
    from zuntenium.circuit.ir.serialization.qasm_parser import parse_qasm2

    qasm_str = Path(path).read_text(encoding="utf-8")
    ir = parse_qasm2(qasm_str)

    qc = QuantumCircuit(ir.num_qubits, ir.num_clbits)
    for node in ir.instructions:
        if node.is_gate() and node.gate is not None:
            qc.append(node.gate, node.qubits)
        elif node.operation == "measure":
            for q, c in zip(node.qubits, node.clbits):
                qc.measure(q, c)
        elif node.operation == "barrier":
            qc.barrier(*node.qubits)
        elif node.operation == "reset":
            for q in node.qubits:
                qc.reset(q)
    return qc


def save_qasm(circuit: QuantumCircuit, path: str | Path, version: int = 2) -> None:
    """
    Save a QuantumCircuit to an OpenQASM file.

    Args:
        circuit: The circuit to export.
        path: Output file path.
        version: 2 (default) or 3.
    """
    from zuntenium.circuit.ir.serialization.qasm_serializer import ir_to_qasm2, ir_to_qasm3

    ir = circuit.to_ir()
    qasm_str = ir_to_qasm3(ir) if version == 3 else ir_to_qasm2(ir)
    Path(path).write_text(qasm_str, encoding="utf-8")
