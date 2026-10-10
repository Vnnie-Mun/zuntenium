"""
TranspilerPass — Abstract base class for all transpiler passes.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from zuntenium.circuit.ir.quantum_ir import QuantumIR


class TranspilerPass(ABC):
    """
    Abstract base for all transpiler passes.

    Must be stateless: a pass that is run twice on the same IR
    must produce the same result. Passes should not cache IR state.

    Attributes:
        name: Unique pass identifier.
        description: Human-readable description.
        requires: Names of passes that must precede this one.
    """

    name: str = ""
    description: str = ""
    requires: list[str] = []

    @abstractmethod
    def run(self, ir: QuantumIR, **kwargs: object) -> QuantumIR:
        """
        Apply this pass to the given IR.

        Args:
            ir: Input QuantumIR (treat as immutable; return a new or same obj).
            **kwargs: Pass-specific options. ``target`` is a common key.

        Returns:
            Transformed QuantumIR.
        """
        ...

    def is_idempotent(self) -> bool:
        """Return True if running this pass twice has no additional effect."""
        return False

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
