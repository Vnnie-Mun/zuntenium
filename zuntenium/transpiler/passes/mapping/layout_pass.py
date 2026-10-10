"""
LayoutPass — Initial qubit layout assignment.

Maps logical circuit qubits to physical device qubits before routing.
v0.1: trivial identity mapping (logical q_i → physical q_i).
v0.2: noise-aware layout using T1/T2 and gate fidelity data.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from zuntenium.transpiler.passes.base_pass import TranspilerPass

if TYPE_CHECKING:
    from zuntenium.circuit.ir.quantum_ir import QuantumIR


class LayoutPass(TranspilerPass):
    """
    Assigns logical qubits to physical qubits.

    v0.1: Identity mapping — logical q_i maps to physical q_i.
    v0.2: Noise-aware layout selection.

    The chosen layout is stored in ir.metadata['layout'].
    """

    name = "Layout"
    description = "Maps logical circuit qubits to physical device qubits."

    def run(self, ir: QuantumIR, **kwargs: object) -> QuantumIR:
        from zuntenium.transpiler.target import Target

        target_raw = kwargs.get("target")

        # v0.1: identity layout
        layout = {q: q for q in range(ir.num_qubits)}
        ir.metadata["layout"] = layout

        if (
            target_raw is not None
            and isinstance(target_raw, Target)
            and target_raw.num_qubits < ir.num_qubits
        ):
            from zuntenium.quantum_core.exceptions import CircuitCompilationError
            raise CircuitCompilationError(
                f"Circuit requires {ir.num_qubits} qubits but target only has "
                f"{target_raw.num_qubits}."
            )

        return ir

    def is_idempotent(self) -> bool:
        return True
