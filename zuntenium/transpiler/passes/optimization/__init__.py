"""optimization passes — Gate cancellation, rotation merging, constant folding."""
from __future__ import annotations

from zuntenium.transpiler.passes.optimization.constant_folding import ConstantFoldingPass
from zuntenium.transpiler.passes.optimization.gate_cancellation import GateCancellationPass
from zuntenium.transpiler.passes.optimization.rotation_merging import RotationMergingPass
from zuntenium.transpiler.passes.optimization.single_qubit_unify import SingleQubitUnifyPass

__all__ = [
    "GateCancellationPass", "RotationMergingPass",
    "ConstantFoldingPass", "SingleQubitUnifyPass",
]
