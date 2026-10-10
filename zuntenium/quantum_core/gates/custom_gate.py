"""
CustomGate — User-defined gate from an arbitrary unitary matrix.

Validates unitarity and wraps any numpy matrix as a first-class gate.
"""
from __future__ import annotations

import numpy as np

from zuntenium.quantum_core.exceptions import GateDimensionError
from zuntenium.quantum_core.gates.base_gate import BaseGate


class CustomGate(BaseGate):
    """
    A gate defined by an arbitrary unitary matrix provided by the user.

    Args:
        matrix: A 2^n × 2^n complex128 unitary matrix.
        name: Optional name for the gate (default: "Custom").
        validate: If True (default), checks unitarity on construction.

    Raises:
        GateDimensionError: If matrix is not square or not a power-of-2 dimension.
        GateNotUnitaryError: If matrix is not unitary (within tolerance).

    Example::

        import numpy as np
        u = np.array([[0, 1], [1, 0]], dtype=np.complex128)  # X gate
        custom_x = CustomGate(u, name="MyX")
        custom_x.validate_unitary()   # passes
    """

    def __init__(
        self,
        matrix: np.ndarray,
        name: str = "Custom",
        validate: bool = True,
    ) -> None:
        m = np.asarray(matrix, dtype=np.complex128)
        if m.ndim != 2 or m.shape[0] != m.shape[1]:
            raise GateDimensionError(name, -1, m.shape[0] if m.ndim >= 1 else 0)
        dim = m.shape[0]
        # Verify dimension is a power of 2
        n = int(np.log2(dim))
        if 2 ** n != dim:
            raise ValueError(
                f"CustomGate matrix dimension must be a power of 2, got {dim}."
            )
        self._matrix = m
        self.name = name
        self.num_qubits = n
        self.is_parametric = False

        if validate:
            self.validate_unitary()

    def matrix(self) -> np.ndarray:
        return self._matrix.copy()

    def compose(self, other: CustomGate) -> CustomGate:
        """Return a new CustomGate that is other applied first, then self."""
        assert self.num_qubits == other.num_qubits, "Qubit count must match for composition"
        composed = self._matrix @ other._matrix
        return CustomGate(composed, name=f"{self.name}·{other.name}", validate=False)

    def __repr__(self) -> str:
        return f"CustomGate(name='{self.name}', num_qubits={self.num_qubits})"
