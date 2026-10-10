"""
TranspilerPipeline — Ordered pass execution over QuantumIR.

Supports optimization levels 0–3 with pre-built pass sets, and
allows custom pass insertion at any position in the pipeline.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from zuntenium.transpiler.passes.base_pass import TranspilerPass

if TYPE_CHECKING:
    from zuntenium.circuit.ir.quantum_ir import QuantumIR
    from zuntenium.transpiler.target import Target


def _default_passes(level: int) -> list[TranspilerPass]:
    """Build the default pass list for the given optimization level."""
    from zuntenium.transpiler.passes.optimization.constant_folding import ConstantFoldingPass
    from zuntenium.transpiler.passes.optimization.gate_cancellation import GateCancellationPass
    from zuntenium.transpiler.passes.optimization.rotation_merging import RotationMergingPass
    from zuntenium.transpiler.passes.optimization.single_qubit_unify import SingleQubitUnifyPass

    if level == 0:
        return []
    if level == 1:
        return [GateCancellationPass()]
    if level == 2:
        return [GateCancellationPass(), RotationMergingPass(), SingleQubitUnifyPass()]
    # level 3+
    return [
        GateCancellationPass(),
        RotationMergingPass(),
        SingleQubitUnifyPass(),
        ConstantFoldingPass(),
    ]


class TranspilerPipeline:
    """
    Ordered sequence of transpilation passes applied to a QuantumIR.

    Supports optimization levels 0–3 with pre-built pass sets.

    Args:
        passes: Explicit list of TranspilerPass objects. If None, uses
                the default passes for the given ``level``.
        level: Optimization level 0–3 (used when passes=None).
               0 = no optimization (debug), 3 = maximum optimization.

    Example::

        pipeline = TranspilerPipeline(level=2)
        optimized_ir = pipeline.run(circuit.to_ir(), target=backend.target)
    """

    LEVEL_PASSES = {
        0: [],
        1: ["GateCancellation"],
        2: ["GateCancellation", "RotationMerging", "SingleQubitUnify"],
        3: ["GateCancellation", "RotationMerging", "SingleQubitUnify", "ConstantFolding"],
    }

    def __init__(
        self,
        passes: list[TranspilerPass] | None = None,
        level: int = 2,
    ) -> None:
        self._level = level
        self._passes: list[TranspilerPass] = (
            passes if passes is not None else _default_passes(level)
        )

    def run(
        self,
        ir: QuantumIR,
        target: Target | None = None,
    ) -> QuantumIR:
        """
        Apply all passes in sequence to the given IR.

        Args:
            ir: Input QuantumIR.
            target: Optional hardware target for device-aware passes.

        Returns:
            Optimized QuantumIR.
        """
        current_ir = ir
        for pass_ in self._passes:
            current_ir = pass_.run(current_ir, target=target)
        return current_ir

    def add_pass(
        self,
        pass_: TranspilerPass,
        position: int = -1,
    ) -> None:
        """
        Insert a pass into the pipeline.

        Args:
            pass_: The pass to add.
            position: Index to insert at. -1 = append at end.
        """
        if position == -1:
            self._passes.append(pass_)
        else:
            self._passes.insert(position, pass_)

    def remove_pass(self, name: str) -> bool:
        """Remove the first pass with the given name. Returns True if removed."""
        for i, p in enumerate(self._passes):
            if p.name == name:
                self._passes.pop(i)
                return True
        return False

    @property
    def passes(self) -> list[TranspilerPass]:
        return list(self._passes)

    @property
    def level(self) -> int:
        return self._level

    def __repr__(self) -> str:
        names = [p.name for p in self._passes]
        return f"TranspilerPipeline(level={self._level}, passes={names})"
