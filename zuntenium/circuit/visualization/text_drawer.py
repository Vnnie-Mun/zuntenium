"""
TextDrawer — ASCII circuit diagram renderer.

Produces a human-readable text representation of a QuantumCircuit.

Example output for Bell circuit::

    q[0]: ─── H ─── ●  ───
    q[1]: ──────── CX  ───
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from zuntenium.circuit.circuit import QuantumCircuit


class TextDrawer:
    """
    Renders a QuantumCircuit as an ASCII diagram.

    Args:
        circuit: The circuit to draw.
        fold_at: Wrap diagram after this many columns (0 = no wrap).
    """

    def __init__(self, circuit: QuantumCircuit, fold_at: int = 0) -> None:
        self._circuit = circuit
        self._fold_at = fold_at

    def draw(self) -> str:
        """Return the ASCII circuit diagram as a string."""
        n = self._circuit.num_qubits
        instructions = self._circuit.instructions

        # Each qubit gets a wire; we build column-by-column
        wires: list[list[str]] = [[] for _ in range(n)]

        for instr in instructions:
            label = self._gate_label(instr)
            width = len(label) + 2   # padding

            affected = set(instr.qubits)
            # For gates spanning multiple qubits, fill intermediate wires
            if instr.qubits:
                q_min, q_max = min(instr.qubits), max(instr.qubits)
            else:
                q_min, q_max = 0, n - 1

            for q in range(n):
                if q in affected:
                    if q == instr.qubits[0] and len(instr.qubits) > 1:
                        wires[q].append(f"─ ● {'─' * (width - 4)}")
                    else:
                        wires[q].append(f" {label} ")
                elif q_min < q < q_max:
                    wires[q].append("─" * width + "│")
                else:
                    wires[q].append("─" * (width + 2))

        # Assemble
        lines: list[str] = []
        for q in range(n):
            prefix = f"q[{q}]: "
            wire_str = "".join(wires[q]) if wires[q] else "────"
            lines.append(prefix + "─" + wire_str + "─")

        if self._circuit.num_clbits > 0:
            lines.append("")
            for c in range(self._circuit.num_clbits):
                lines.append(f"c[{c}]: ═" + "═" * 20)

        return "\n".join(lines)

    def _gate_label(self, instr: object) -> str:
        from zuntenium.circuit.instruction import Instruction
        assert isinstance(instr, Instruction)
        if instr.operation == "measure":
            return "M"
        if instr.operation == "reset":
            return "RST"
        if instr.operation == "barrier":
            return "░"
        if instr.gate is not None:
            name = instr.gate.name
            if instr.params:
                pstr = ",".join(f"{p:.2f}" for p in instr.params)
                return f"{name}({pstr})"
            return name
        return instr.operation.upper()
