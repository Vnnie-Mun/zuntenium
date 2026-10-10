"""
JSON I/O — load_circuit() and save_circuit() public API.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from zuntenium.circuit.circuit import QuantumCircuit


def save_circuit(circuit: QuantumCircuit, path: str | Path) -> None:
    """
    Save a QuantumCircuit to a JSON file.

    Args:
        circuit: Circuit to serialize.
        path: Output .json file path.
    """
    from zuntenium.circuit.ir.serialization.json_serializer import ir_to_json
    ir = circuit.to_ir()
    Path(path).write_text(ir_to_json(ir))


def load_circuit(path: str | Path) -> QuantumCircuit:
    """
    Load a QuantumCircuit from a JSON file.

    Args:
        path: Path to .json file.

    Returns:
        Restored QuantumCircuit.
    """
    from zuntenium.circuit.circuit import QuantumCircuit
    from zuntenium.circuit.ir.serialization.json_serializer import json_to_ir

    json_str = Path(path).read_text()
    ir = json_to_ir(json_str)

    qc = QuantumCircuit(ir.num_qubits, ir.num_clbits)
    for node in ir.instructions:
        if node.is_gate() and node.gate is not None:
            qc.append(node.gate, node.qubits)
        elif node.operation == "measure":
            for q, c in zip(node.qubits, node.clbits):
                qc.measure(q, c)
        elif node.operation == "barrier":
            qc.barrier(*node.qubits)
    return qc
