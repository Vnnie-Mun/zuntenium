"""
SingleQubitUnifyPass — Compile any single-qubit gate sequence into a single U3.

Any sequence of single-qubit gates acting on the same qubit can be represented
as a single U3(θ, φ, λ) gate. This pass performs that unification by composing
the gate matrices and decomposing the result via ZYZ Euler angles.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from zuntenium.circuit.ir.ir_node import IRNode
from zuntenium.transpiler.passes.base_pass import TranspilerPass

if TYPE_CHECKING:
    from zuntenium.circuit.ir.quantum_ir import QuantumIR

_I = complex(0, 1)
_SINGLE_QUBIT_OPS = {
    "H", "X", "Y", "Z", "S", "T",
    "Rx", "Ry", "Rz", "U1", "U2", "U3",
}


def _zyz_decompose(u: np.ndarray) -> tuple[float, float, float]:
    """
    Decompose a 2×2 unitary U into ZYZ Euler angles (θ, φ, λ).

    U = e^{iα} · Rz(φ) · Ry(θ) · Rz(λ)

    Returns:
        (theta, phi, lam) angles for U3 gate.
    """
    # Normalise global phase
    if abs(u[0, 0]) > 1e-10:
        phase = u[0, 0] / abs(u[0, 0])
        u = u / phase

    theta = 2 * float(np.arccos(min(1.0, abs(u[0, 0]))))
    if abs(np.sin(theta / 2)) < 1e-10:
        phi = 0.0
        lam = float(np.angle(u[1, 1]))
    else:
        phi = float(np.angle(u[1, 0]) - np.angle(u[0, 0]))
        lam = float(np.angle(-u[0, 1]) - np.angle(u[0, 0]))

    return theta, phi, lam


class SingleQubitUnifyPass(TranspilerPass):
    """
    Merge consecutive single-qubit gates acting on the same qubit into one U3.

    This reduces gate count and circuit depth for any single-qubit sequence,
    at the cost of one matrix multiply per merged block.
    """

    name = "SingleQubitUnify"
    description = "Compiles consecutive single-qubit gates on the same qubit into a single U3."

    def run(self, ir: QuantumIR, **kwargs: object) -> QuantumIR:
        from zuntenium.circuit.ir.quantum_ir import QuantumIR
        from zuntenium.quantum_core.gates import U3Gate

        instructions = list(ir.instructions)
        result: list[IRNode] = []

        i = 0
        while i < len(instructions):
            node = instructions[i]
            if (
                node.is_gate()
                and node.gate is not None
                and node.gate.name in _SINGLE_QUBIT_OPS
                and len(node.qubits) == 1
            ):
                qubit = node.qubits[0]
                # Accumulate consecutive same-qubit single-qubit gates
                composite = node.gate.matrix().copy()
                j = i + 1
                while j < len(instructions):
                    nxt = instructions[j]
                    if (
                        nxt.is_gate()
                        and nxt.gate is not None
                        and nxt.gate.name in _SINGLE_QUBIT_OPS
                        and nxt.qubits == [qubit]
                    ):
                        composite = nxt.gate.matrix() @ composite
                        j += 1
                    else:
                        break

                # Only replace if we actually merged something
                if j > i + 1:
                    theta, phi, lam = _zyz_decompose(composite)
                    u3 = U3Gate(theta, phi, lam)
                    result.append(IRNode(
                        operation="gate",
                        gate=u3,
                        qubits=[qubit],
                        params=[theta, phi, lam],
                    ))
                else:
                    result.append(node)
                i = j
            else:
                result.append(node)
                i += 1

        return QuantumIR(
            num_qubits=ir.num_qubits,
            num_clbits=ir.num_clbits,
            instructions=result,
            metadata={**ir.metadata, "single_qubit_unify": True},
        )
