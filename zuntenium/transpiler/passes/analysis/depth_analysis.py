"""
DepthAnalysisPass — Compute circuit depth and annotate IR metadata.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from zuntenium.transpiler.passes.base_pass import TranspilerPass

if TYPE_CHECKING:
    from zuntenium.circuit.ir.quantum_ir import QuantumIR


class DepthAnalysisPass(TranspilerPass):
    """
    Analysis pass that computes and annotates circuit depth.

    Does not modify the IR; adds 'depth' and 'critical_path' to metadata.
    """

    name = "DepthAnalysis"
    description = "Computes circuit depth (critical path length) and annotates IR metadata."

    def run(self, ir: QuantumIR, **kwargs: object) -> QuantumIR:
        depth = ir.depth()
        layers = ir.to_layers()

        ir.metadata["depth"] = depth
        ir.metadata["num_layers"] = len(layers)
        ir.metadata["layer_sizes"] = [len(layer) for layer in layers]
        return ir

    def is_idempotent(self) -> bool:
        return True
