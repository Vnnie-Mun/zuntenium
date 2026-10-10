"""
JSON serializer for QuantumIR.

Provides a compact, human-readable JSON format for API transport,
circuit storage, and inter-process communication.
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from zuntenium.circuit.ir.quantum_ir import QuantumIR


def ir_to_dict(ir: QuantumIR) -> dict[str, Any]:
    """Serialize a QuantumIR to a JSON-compatible dict."""
    instructions = []
    for node in ir.instructions:
        entry: dict[str, Any] = {
            "op": node.operation,
            "qubits": node.qubits,
        }
        if node.clbits:
            entry["clbits"] = node.clbits
        if node.params:
            entry["params"] = [round(p, 10) for p in node.params]
        if node.gate is not None:
            entry["gate"] = node.gate.name
        instructions.append(entry)

    return {
        "version": "0.1",
        "num_qubits": ir.num_qubits,
        "num_clbits": ir.num_clbits,
        "instructions": instructions,
        "metadata": ir.metadata,
    }


def ir_to_json(ir: QuantumIR, indent: int = 2) -> str:
    """Serialize a QuantumIR to a JSON string."""
    return json.dumps(ir_to_dict(ir), indent=indent)


def dict_to_ir(data: dict[str, Any]) -> QuantumIR:
    """
    Deserialize a QuantumIR from a JSON-compatible dict.

    Args:
        data: Dict produced by ``ir_to_dict()``.

    Returns:
        Reconstructed QuantumIR.
    """
    from zuntenium.circuit.ir.ir_node import IRNode
    from zuntenium.circuit.ir.quantum_ir import QuantumIR
    from zuntenium.quantum_core.gates import (
        CRxGate,
        CRyGate,
        CRzGate,
        CXGate,
        CYGate,
        CZGate,
        FredkinGate,
        HGate,
        ISWAPGate,
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

    _GATE_MAP = {
        "H": HGate, "X": XGate, "Y": YGate, "Z": ZGate,
        "S": SGate, "T": TGate,
        "Rx": RxGate, "Ry": RyGate, "Rz": RzGate,
        "U1": U1Gate, "U2": U2Gate, "U3": U3Gate,
        "CX": CXGate, "CZ": CZGate, "CY": CYGate,
        "SWAP": SWAPGate, "ISWAP": ISWAPGate,
        "CRx": CRxGate, "CRy": CRyGate, "CRz": CRzGate,
        "CCX": ToffoliGate, "CSWAP": FredkinGate,
    }

    nodes: list[IRNode] = []
    for entry in data.get("instructions", []):
        op = entry["op"]
        gate = None
        params = entry.get("params", [])
        if op == "gate" and "gate" in entry:
            gate_cls = _GATE_MAP.get(entry["gate"])
            if gate_cls is not None:
                gate = gate_cls(*params) if params else gate_cls()
        nodes.append(IRNode(
            operation=op,
            gate=gate,
            qubits=entry.get("qubits", []),
            clbits=entry.get("clbits", []),
            params=params,
        ))

    return QuantumIR(
        num_qubits=data["num_qubits"],
        num_clbits=data.get("num_clbits", 0),
        instructions=nodes,
        metadata=data.get("metadata", {}),
    )


def json_to_ir(json_str: str) -> QuantumIR:
    """Deserialize a QuantumIR from a JSON string."""
    return dict_to_ir(json.loads(json_str))
