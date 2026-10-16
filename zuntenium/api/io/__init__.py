"""api/io — QASM and JSON import/export convenience functions."""
from __future__ import annotations

from zuntenium.circuit.ir.serialization.json_serializer import ir_to_json, json_to_ir
from zuntenium.circuit.ir.serialization.qasm_parser import parse_qasm2
from zuntenium.circuit.ir.serialization.qasm_serializer import ir_to_qasm2


def load_qasm(path: str) -> object:
    """Load a QASM2 file and return a QuantumIR."""
    from pathlib import Path
    return parse_qasm2(Path(path).read_text(encoding="utf-8"))


def save_qasm(ir: object, path: str) -> None:
    """Serialize a QuantumIR to a QASM2 file."""
    from pathlib import Path
    Path(path).write_text(ir_to_qasm2(ir), encoding="utf-8")  # type: ignore[arg-type]


__all__ = ["load_qasm", "save_qasm", "parse_qasm2", "ir_to_qasm2", "ir_to_json", "json_to_ir"]
