"""Transpiler v2 — Audited & Refactored."""
from zuntenium.transpiler.v2.core import Gate, Circuit, normalize_angle, gates_commute, matrices_equivalent
from zuntenium.transpiler.v2.passes import (
    RedundantGateCancellationPass, RotationMergingPass,
    CommutationReorderingPass, SingleQubitFusionPass,
    DepthOptimizationPass, DeadGateEliminationPass,
)
from zuntenium.transpiler.v2.engine import Transpiler, PassRegistry, OptimizationReport

__all__ = [
    "Gate", "Circuit", "normalize_angle", "gates_commute", "matrices_equivalent",
    "RedundantGateCancellationPass", "RotationMergingPass",
    "CommutationReorderingPass", "SingleQubitFusionPass",
    "DepthOptimizationPass", "DeadGateEliminationPass",
    "Transpiler", "PassRegistry", "OptimizationReport",
]
