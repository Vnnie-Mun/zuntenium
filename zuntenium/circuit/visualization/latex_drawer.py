"""
LatexDrawer — Full quantikz LaTeX circuit renderer.

Produces compilable LaTeX source using the quantikz package.
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from zuntenium.circuit.circuit import QuantumCircuit


class LatexDrawer:
    """
    Renders a QuantumCircuit as a LaTeX quantikz diagram.

    Args:
        circuit: The circuit to draw.
        scale: tikz scale factor (default 1.0).

    Example::

        drawer = LatexDrawer(qc)
        latex = drawer.draw()
        print(latex)
        drawer.to_pdf("circuit.pdf")
    """

    _PREAMBLE = r"""\documentclass[border=2pt]{standalone}
\usepackage{quantikz}
\begin{document}
"""
    _POSTAMBLE = r"""
\end{document}
"""

    def __init__(self, circuit: QuantumCircuit, scale: float = 1.0) -> None:
        self._circuit = circuit
        self._scale = scale

    def draw(self) -> str:
        """Return a complete compilable LaTeX string for the circuit."""
        n = self._circuit.num_qubits
        instructions = self._circuit.instructions

        # Build per-qubit column lists
        wires: list[list[str]] = [[] for _ in range(n)]
        col = 0
        for instr in instructions:
            col_cells = self._render_instruction(instr, n)
            for q in range(n):
                wires[q].append(col_cells[q])
            col += 1

        # Assemble quantikz rows
        rows = []
        for q in range(n):
            cells = " & ".join(wires[q]) if wires[q] else r"\qw"
            rows.append(rf"\lstick{{$q_{{{q}}}$}} & {cells} & \qw")

        tikz_body = " \\\\\n".join(rows)

        body = (
            f"\\begin{{quantikz}}[scale={self._scale}]\n"
            f"{tikz_body}\n"
            r"\end{quantikz}"
        )
        return self._PREAMBLE + body + self._POSTAMBLE

    def _render_instruction(self, instr: object, n: int) -> list[str]:
        """Return list of n LaTeX cell strings for one instruction column."""
        from zuntenium.circuit.instruction import Instruction
        assert isinstance(instr, Instruction)
        cells = [r"\qw"] * n

        op = instr.operation
        qubits = instr.qubits

        if op == "barrier":
            for q in range(n):
                cells[q] = r"\barrier{}" if q == 0 else r"\qw"
            return cells

        if op == "measure":
            q = qubits[0]
            cells[q] = r"\meter{}"
            return cells

        if op == "reset":
            q = qubits[0]
            cells[q] = r"\gate{|0\rangle}"
            return cells

        if instr.gate is None:
            return cells

        name = instr.gate.name
        params = instr.params

        # Single-qubit case
        if len(qubits) == 1:
            q = qubits[0]
            if params:
                pstr = ",".join(f"{p:.3f}" for p in params)
                cells[q] = rf"\gate{{{name}({pstr})}}"
            else:
                cells[q] = rf"\gate{{{name}}}"
            return cells

        # Two-qubit: control + target
        if len(qubits) == 2:
            ctrl, tgt = qubits[0], qubits[1]
            offset = tgt - ctrl
            if name.upper() in ("CX", "CNOT"):
                cells[ctrl] = rf"\ctrl{{{offset}}}"
                cells[tgt] = r"\targ{}"
            elif name.upper() == "CZ":
                cells[ctrl] = rf"\ctrl{{{offset}}}"
                cells[tgt] = r"\control{}"
            elif name.upper() == "SWAP":
                cells[ctrl] = rf"\swap{{{offset}}}"
                cells[tgt] = r"\targX{}"
            else:
                cells[ctrl] = rf"\ctrl{{{offset}}}"
                cells[tgt] = rf"\gate{{{name}}}"
            # Fill intermediary wires
            lo, hi = min(ctrl, tgt), max(ctrl, tgt)
            for q in range(lo + 1, hi):
                cells[q] = r"\qw"
            return cells

        # 3-qubit (Toffoli etc.)
        if len(qubits) == 3:
            c0, c1, tgt = qubits[0], qubits[1], qubits[2]
            cells[c0] = rf"\ctrl{{{tgt - c0}}}"
            cells[c1] = rf"\ctrl{{{tgt - c1}}}"
            cells[tgt] = r"\targ{}"
            return cells

        return cells

    def to_pdf(self, output_path: str) -> None:
        """
        Compile the LaTeX circuit to a PDF (requires pdflatex / lualatex).

        Args:
            output_path: Path to save the PDF (e.g. 'circuit.pdf').
        """
        latex_src = self.draw()
        out_path = Path(output_path)
        with tempfile.TemporaryDirectory() as tmpdir:
            tex_file = Path(tmpdir) / "circuit.tex"
            tex_file.write_text(latex_src)
            result = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", str(tex_file)],
                cwd=tmpdir,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"pdflatex failed:\n{result.stderr}\n"
                    "Install quantikz: tlmgr install quantikz"
                )
            pdf = Path(tmpdir) / "circuit.pdf"
            if pdf.exists():
                out_path.parent.mkdir(parents=True, exist_ok=True)
                pdf.rename(out_path)
