"""
RotationMergingPass — Merge adjacent Rz (and Rx/Ry) rotations on the same qubit.

Two consecutive Rz(θ₁) · Rz(θ₂) = Rz(θ₁+θ₂).
Angle sum is normalised to (-π, π].
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

from zuntenium.circuit.ir.ir_node import IRNode
from zuntenium.transpiler.passes.base_pass import TranspilerPass

if TYPE_CHECKING:
    from zuntenium.circuit.ir.quantum_ir import QuantumIR

_ROTATION_GATES = {"Rx", "Ry", "Rz"}


def _normalise_angle(theta: float) -> float:
    """Wrap angle into (-π, π]."""
    theta = theta % (2 * math.pi)
    if theta > math.pi:
        theta -= 2 * math.pi
    return theta


class RotationMergingPass(TranspilerPass):
    """
    Merge sequences of same-axis rotation gates acting on the same qubit.

    Rz(θ₁) · Rz(θ₂) → Rz(θ₁+θ₂)
    Removes gates whose merged angle is effectively zero (|θ| < tol).

    Args:
        tol: Angle threshold below which a rotation is removed (default 1e-10).
    """

    name = "RotationMerging"
    description = "Merges adjacent same-axis rotation gates: Rz(a)·Rz(b) → Rz(a+b)."

    def __init__(self, tol: float = 1e-10) -> None:
        self._tol = tol

    def run(self, ir: QuantumIR, **kwargs: object) -> QuantumIR:
        from zuntenium.circuit.ir.quantum_ir import QuantumIR
        from zuntenium.quantum_core.gates import RxGate, RyGate, RzGate

        _GATE_CLASS = {"Rx": RxGate, "Ry": RyGate, "Rz": RzGate}

        instructions = list(ir.instructions)
        merged: list[IRNode] = []

        i = 0
        while i < len(instructions):
            node = instructions[i]
            if (
                node.is_gate()
                and node.gate is not None
                and node.gate.name in _ROTATION_GATES
                and len(node.qubits) == 1
            ):
                gate_name = node.gate.name
                qubit = node.qubits[0]
                angle = node.params[0] if node.params else 0.0

                # Accumulate consecutive same-gate same-qubit rotations
                j = i + 1
                while j < len(instructions):
                    nxt = instructions[j]
                    if (
                        nxt.is_gate()
                        and nxt.gate is not None
                        and nxt.gate.name == gate_name
                        and nxt.qubits == [qubit]
                    ):
                        angle += nxt.params[0] if nxt.params else 0.0
                        j += 1
                    else:
                        break

                angle = _normalise_angle(angle)
                if abs(angle) > self._tol:
                    gate_cls = _GATE_CLASS[gate_name]
                    new_gate = gate_cls(angle)
                    merged.append(IRNode(
                        operation="gate",
                        gate=new_gate,
                        qubits=[qubit],
                        params=[angle],
                    ))
                # else: angle ≈ 0, drop the gate entirely
                i = j
            else:
                merged.append(node)
                i += 1

        return QuantumIR(
            num_qubits=ir.num_qubits,
            num_clbits=ir.num_clbits,
            instructions=merged,
            metadata={**ir.metadata, "rotation_merging": True},
        )
