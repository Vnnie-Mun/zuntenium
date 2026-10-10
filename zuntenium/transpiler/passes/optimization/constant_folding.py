"""
ConstantFoldingPass — Evaluate compile-time parameter expressions.

In v0.1: replaces parametric gates whose parameters are literal floats
(not symbolic) with their evaluated matrix-equivalent fixed gates.

v0.2: Will fold symbolic expressions (e.g., θ = π/2) using sympy.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

from zuntenium.circuit.ir.ir_node import IRNode
from zuntenium.transpiler.passes.base_pass import TranspilerPass

if TYPE_CHECKING:
    from zuntenium.circuit.ir.quantum_ir import QuantumIR

_HALF_PI = math.pi / 2
_TOL = 1e-9


def _is_close(a: float, b: float) -> bool:
    return abs(a - b) < _TOL


class ConstantFoldingPass(TranspilerPass):
    """
    Replace parametric rotation gates with known equivalents when parameters
    are compile-time constants.

    Examples:
        Rx(π)   → X  (up to global phase)
        Ry(π)   → Y
        Rz(π)   → Z
        Rz(π/2) → S
        Rz(π/4) → T
        Rx(0)   → Identity (dropped)
    """

    name = "ConstantFolding"
    description = "Replaces parametric gates with fixed gates for known angle values."

    def run(self, ir: QuantumIR, **kwargs: object) -> QuantumIR:
        from zuntenium.circuit.ir.quantum_ir import QuantumIR
        from zuntenium.quantum_core.gates import (
            SGate,
            TGate,
            XGate,
            YGate,
            ZGate,
        )
        from zuntenium.quantum_core.gates.base_gate import BaseGate

        folded: list[IRNode] = []

        for node in ir.instructions:
            if not node.is_gate() or node.gate is None or not node.params:
                folded.append(node)
                continue

            gate_name = node.gate.name
            theta = node.params[0]
            replacement: BaseGate | None = None

            if gate_name == "Rx":
                if _is_close(abs(theta), math.pi):
                    replacement = XGate()
                elif _is_close(theta, 0):
                    continue   # Identity, drop
            elif gate_name == "Ry":
                if _is_close(abs(theta), math.pi):
                    replacement = YGate()
                elif _is_close(theta, 0):
                    continue
            elif gate_name == "Rz":
                if _is_close(abs(theta), math.pi):
                    replacement = ZGate()
                elif _is_close(abs(theta), _HALF_PI):
                    replacement = SGate()
                elif _is_close(abs(theta), _HALF_PI / 2):
                    replacement = TGate()
                elif _is_close(theta, 0):
                    continue

            if replacement is not None:
                folded.append(IRNode(
                    operation="gate",
                    gate=replacement,
                    qubits=node.qubits,
                    params=[],
                ))
            else:
                folded.append(node)

        return QuantumIR(
            num_qubits=ir.num_qubits,
            num_clbits=ir.num_clbits,
            instructions=folded,
            metadata={**ir.metadata, "constant_folding": True},
        )

    def is_idempotent(self) -> bool:
        return True
