"""
OpenQASM 2/3 serializer.

Converts a QuantumIR to an OpenQASM string for export.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from zuntenium.circuit.ir.quantum_ir import QuantumIR

# Gate name → QASM 2 opcode mapping
_QASM2_GATE_MAP = {
    "H": "h", "X": "x", "Y": "y", "Z": "z",
    "S": "s", "T": "t",
    "Rx": "rx", "Ry": "ry", "Rz": "rz",
    "U1": "u1", "U2": "u2", "U3": "u3",
    "CX": "cx", "CZ": "cz", "CY": "cy",
    "SWAP": "swap", "ISWAP": "iswap",
    "CRx": "crx", "CRy": "cry", "CRz": "crz",
    "CCX": "ccx", "CSWAP": "cswap",
}


def ir_to_qasm2(ir: QuantumIR) -> str:
    """
    Serialize a QuantumIR to OpenQASM 2.0 string.

    Args:
        ir: The QuantumIR to serialize.

    Returns:
        OpenQASM 2.0 string.
    """
    lines = [
        'OPENQASM 2.0;',
        'include "qelib1.inc";',
        f'qreg q[{ir.num_qubits}];',
    ]
    if ir.num_clbits > 0:
        lines.append(f'creg c[{ir.num_clbits}];')
    lines.append("")

    for node in ir.instructions:
        if node.operation == "barrier":
            qb = ", ".join(f"q[{q}]" for q in node.qubits)
            lines.append(f"barrier {qb};")
        elif node.operation == "measure":
            for q, c in zip(node.qubits, node.clbits):
                lines.append(f"measure q[{q}] -> c[{c}];")
        elif node.operation == "reset":
            for q in node.qubits:
                lines.append(f"reset q[{q}];")
        elif node.is_gate() and node.gate is not None:
            gate_name = _QASM2_GATE_MAP.get(node.gate.name, node.gate.name.lower())
            qubits_str = ", ".join(f"q[{q}]" for q in node.qubits)
            if node.params:
                params_str = ", ".join(f"{p:.10f}" for p in node.params)
                lines.append(f"{gate_name}({params_str}) {qubits_str};")
            else:
                lines.append(f"{gate_name} {qubits_str};")

    return "\n".join(lines) + "\n"


def ir_to_qasm3(ir: QuantumIR) -> str:
    """
    Serialize a QuantumIR to OpenQASM 3.0 string.

    v0.1: Basic subset. Full QASM 3 support in v0.2.
    """
    lines = [
        "OPENQASM 3.0;",
        f"qubit[{ir.num_qubits}] q;",
    ]
    if ir.num_clbits > 0:
        lines.append(f"bit[{ir.num_clbits}] c;")
    lines.append("")

    for node in ir.instructions:
        if node.operation == "measure":
            for q, c in zip(node.qubits, node.clbits):
                lines.append(f"c[{c}] = measure q[{q}];")
        elif node.operation == "reset":
            for q in node.qubits:
                lines.append(f"reset q[{q}];")
        elif node.is_gate() and node.gate is not None:
            gate_name = _QASM2_GATE_MAP.get(node.gate.name, node.gate.name.lower())
            qubits_str = ", ".join(f"q[{q}]" for q in node.qubits)
            if node.params:
                params_str = ", ".join(f"{p:.10f}" for p in node.params)
                lines.append(f"{gate_name}({params_str}) {qubits_str};")
            else:
                lines.append(f"{gate_name} {qubits_str};")

    return "\n".join(lines) + "\n"
