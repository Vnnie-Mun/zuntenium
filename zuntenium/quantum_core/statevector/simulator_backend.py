"""
zuntenium/quantum_core/statevector/simulator_backend.py
=======================================================
Abstract base class for all ZUNTENIUM simulation backends.

Extracted from statevector_simulator.py as part of O6 God-class decomposition.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
StateArray = np.ndarray   # shape (2**n,), dtype complex128
MatrixNxN  = np.ndarray   # shape (N, N),   dtype complex128


class SimulatorBackend(ABC):
    """
    Abstract base class for all ZUNTENIUM simulation backends.

    Concrete backends (StatevectorSimulator, TensorNetworkEngine, GPUBackend)
    must implement every abstract method.  Higher-level code (CircuitExecutor,
    HybridSolver) depends ONLY on this interface, enabling transparent swapping.
    """

    @abstractmethod
    def initialize(self, num_qubits: int) -> None:
        """Reset the quantum state to |0...0⟩."""
        ...

    @abstractmethod
    def apply_gate(self, matrix: MatrixNxN, qubits: Sequence[int]) -> None:
        """Apply a unitary gate (as a matrix) to the specified qubit indices."""
        ...

    @abstractmethod
    def get_statevector(self) -> StateArray:
        """Return the current statevector as a 1-D complex128 ndarray."""
        ...

    @abstractmethod
    def get_probabilities(self) -> np.ndarray:
        """Return |α_x|² for all 2^n basis states."""
        ...

    @abstractmethod
    def measure(
        self,
        qubits: Sequence[int],
        collapse: bool = True,
    ) -> Dict[str, int]:
        """
        Sample one measurement outcome.

        Parameters
        ----------
        qubits  : which qubits to measure (subset of 0..n-1)
        collapse: if True, project the statevector onto the outcome subspace

        Returns
        -------
        dict mapping bitstring (str) → outcome integer (0 or 1) per qubit
        """
        ...

    @abstractmethod
    def sample(self, shots: int, qubits: Optional[Sequence[int]] = None) -> Dict[str, int]:
        """
        Draw `shots` independent measurement samples without collapsing state.

        Returns
        -------
        dict mapping outcome bitstring → count
        """
        ...

    def get_expectation_value(self, pauli_string: str) -> float:
        """
        Compute ⟨ψ|P|ψ⟩ for a Pauli string P (e.g. 'XZI').

        Default implementation via full statevector contraction.
        Override in TensorNetworkEngine for efficiency.

        The Pauli string is ordered as qubit (n-1) ... qubit 0 (MSB to LSB),
        consistent with standard quantum computing convention.
        """
        import numpy as np

        # Pauli matrices
        _PAULI = {
            'I': np.eye(2, dtype=np.complex128),
            'X': np.array([[0, 1], [1, 0]], dtype=np.complex128),
            'Y': np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
            'Z': np.array([[1, 0], [0, -1]], dtype=np.complex128),
        }

        pauli_string = pauli_string.upper().strip()
        n = len(pauli_string)

        # Build the full Pauli tensor product operator
        # P = P_{n-1} ⊗ P_{n-2} ⊗ ... ⊗ P_0
        op = np.array([[1.0]], dtype=np.complex128)
        for ch in pauli_string:
            if ch not in _PAULI:
                raise ValueError(
                    f"Invalid Pauli character '{ch}'. "
                    "Use I, X, Y, or Z."
                )
            op = np.kron(op, _PAULI[ch])

        # Get the statevector — this depends on the concrete subclass
        sv = None
        if hasattr(self, '_state') and hasattr(self._state, 'data'):
            sv = self._state.data
        elif hasattr(self, '_state') and hasattr(self._state, 'amplitudes'):
            sv = self._state.amplitudes
        elif hasattr(self, 'statevector'):
            sv = self.statevector
        elif hasattr(self, '_amplitudes'):
            sv = self._amplitudes

        if sv is None:
            raise RuntimeError(
                "Cannot compute expectation value: no statevector found. "
                "Run a circuit first."
            )

        sv = np.asarray(sv, dtype=np.complex128)
        if len(sv) != 2 ** n:
            raise ValueError(
                f"Pauli string length ({n}) doesn't match statevector "
                f"dimension ({len(sv)} = 2^{int(np.log2(len(sv)))}). "
                f"Use a {int(np.log2(len(sv)))}-character Pauli string."
            )

        # ⟨ψ|P|ψ⟩ = ψ† · P · ψ
        result = sv.conj() @ op @ sv
        return float(np.real(result))
