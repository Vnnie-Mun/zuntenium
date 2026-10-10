"""
OpenQASM 2.0 parser — converts QASM string to QuantumIR.

v0.1: Supports the standard gate set from qelib1.inc.
v0.2: Full QASM 3.0 support including custom gate definitions.
"""
from __future__ import annotations

import re

from zuntenium.circuit.ir.ir_node import IRNode
from zuntenium.circuit.ir.quantum_ir import QuantumIR
from zuntenium.quantum_core.exceptions import QASMParseError

# Supported gate name → (gate_class, num_params)
_GATE_REGISTRY: dict[str, tuple[type, int]] = {}

def _build_registry() -> None:
    from zuntenium.quantum_core.gates import (
        CRxGate,
        CRyGate,
        CRzGate,
        CXGate,
        CYGate,
        CZGate,
        FredkinGate,
        HGate,
        RxGate,
        RyGate,
        RzGate,
        SGate,
        SWAPGate,
        TGate,
        ToffoliGate,
        U1Gate,
        U2Gate,
        U3Gate,
        XGate,
        YGate,
        ZGate,
    )
    global _GATE_REGISTRY
    _GATE_REGISTRY = {
        "h": (HGate, 0), "x": (XGate, 0), "y": (YGate, 0), "z": (ZGate, 0),
        "s": (SGate, 0), "t": (TGate, 0),
        "rx": (RxGate, 1), "ry": (RyGate, 1), "rz": (RzGate, 1),
        "u1": (U1Gate, 1), "u2": (U2Gate, 2), "u3": (U3Gate, 3),
        "cx": (CXGate, 0), "cz": (CZGate, 0), "cy": (CYGate, 0),
        "swap": (SWAPGate, 0),
        "crx": (CRxGate, 1), "cry": (CRyGate, 1), "crz": (CRzGate, 1),
        "ccx": (ToffoliGate, 0), "cswap": (FredkinGate, 0),
    }


def parse_qasm2(qasm_str: str) -> QuantumIR:
    """
    Parse an OpenQASM 2.0 string into a QuantumIR.

    Args:
        qasm_str: OpenQASM 2.0 source string.

    Returns:
        QuantumIR with all parsed instructions.

    Raises:
        QASMParseError: On syntax errors or unsupported gates.
    """
    if not _GATE_REGISTRY:
        _build_registry()

    num_qubits = 0
    num_clbits = 0
    nodes: list[IRNode] = []

    lines = [raw_line.strip() for raw_line in qasm_str.splitlines()]
    for lineno, line in enumerate(lines, 1):
        line = line.split("//")[0].strip()   # strip comments
        if not line or line.startswith("OPENQASM") or line.startswith("include"):
            continue

        # qreg / creg
        m = re.match(r"qreg\s+\w+\[(\d+)\];", line)
        if m:
            num_qubits = int(m.group(1))
            continue
        m = re.match(r"creg\s+\w+\[(\d+)\];", line)
        if m:
            num_clbits = int(m.group(1))
            continue

        # measure q[i] -> c[j]
        m = re.match(r"measure\s+\w+\[(\d+)\]\s*->\s*\w+\[(\d+)\];", line)
        if m:
            nodes.append(IRNode(
                operation="measure",
                qubits=[int(m.group(1))],
                clbits=[int(m.group(2))],
            ))
            continue

        # barrier
        if line.startswith("barrier"):
            qubits = [int(x) for x in re.findall(r"\[(\d+)\]", line)]
            nodes.append(IRNode(operation="barrier", qubits=qubits))
            continue

        # reset
        m = re.match(r"reset\s+\w+\[(\d+)\];", line)
        if m:
            nodes.append(IRNode(operation="reset", qubits=[int(m.group(1))]))
            continue

        # Gate: name[(params)] qubit[i], qubit[j];
        m = re.match(
            r"([a-zA-Z][a-zA-Z0-9_]*)"
            r"(?:\(([^)]*)\))?"
            r"\s+((?:\w+\[\d+\](?:\s*,\s*)?)+);",
            line,
        )
        if m:
            gate_name = m.group(1).lower()
            raw_params = m.group(2) or ""
            raw_qubits = m.group(3)

            params = [float(p.strip()) for p in raw_params.split(",") if p.strip()] if raw_params else []
            qubits = [int(x) for x in re.findall(r"\[(\d+)\]", raw_qubits)]

            if gate_name not in _GATE_REGISTRY:
                raise QASMParseError(f"Line {lineno}: Unknown gate '{gate_name}'")

            gate_cls, n_params = _GATE_REGISTRY[gate_name]
            gate = gate_cls(*params) if n_params > 0 else gate_cls()
            nodes.append(IRNode(operation="gate", gate=gate, qubits=qubits, params=params))
            continue

    if num_qubits == 0:
        raise QASMParseError("No qreg declaration found in QASM input.")

    return QuantumIR(
        num_qubits=num_qubits,
        num_clbits=num_clbits,
        instructions=nodes,
        metadata={"source_format": "qasm2"},
    )
