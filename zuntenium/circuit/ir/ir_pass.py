"""
IRPass — Abstract base class for QuantumIR transformation passes.

All transpiler passes and analysis passes inherit from this.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from zuntenium.circuit.ir.quantum_ir import QuantumIR


class IRPass(ABC):
    """
    Abstract base for all IR transformation and analysis passes.

    Passes are pure functions: given an IR, they return a (potentially
    modified) IR. They must not maintain state between invocations.

    Attributes:
        name: Unique identifier for this pass.
        description: Human-readable description.
        requires: Names of analysis passes that must run before this one.
    """

    name: str = ""
    description: str = ""
    requires: list[str] = []

    @abstractmethod
    def run(self, ir: QuantumIR, **kwargs: object) -> QuantumIR:
        """
        Apply this pass to the given IR.

        Args:
            ir: Input QuantumIR.
            **kwargs: Pass-specific options (e.g., target=Target(...)).

        Returns:
            Transformed QuantumIR (may be the same object if no changes).
        """
        ...

    def is_idempotent(self) -> bool:
        """Return True if running this pass twice has no additional effect."""
        return False

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
