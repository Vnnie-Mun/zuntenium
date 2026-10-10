"""
DensityMatrix — Mixed state representation with partial trace support.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from zuntenium.quantum_core.statevector.statevector import StateVector


class DensityMatrix:
    """
    Represents a quantum state as a density matrix ρ (2^n × 2^n, complex).

    Supports pure states (from StateVector) and mixed states (from data).
    Full partial trace and Kraus channel application are implemented.

    Args:
        num_qubits: Number of qubits.
        data: Optional initial density matrix array. Defaults to |0...0⟩⟨0...0|.
    """

    def __init__(self, num_qubits: int, data: np.ndarray | None = None) -> None:
        self._num_qubits = num_qubits
        self._dim = 2 ** num_qubits
        if data is not None:
            if data.shape != (self._dim, self._dim):
                raise ValueError(
                    f"data must be ({self._dim},{self._dim}), got {data.shape}"
                )
            self._matrix = np.asarray(data, dtype=np.complex128)
        else:
            self._matrix = np.zeros((self._dim, self._dim), dtype=np.complex128)
            self._matrix[0, 0] = 1.0

    # ── Constructors ─────────────────────────────────────────────────────────

    @classmethod
    def from_statevector(cls, sv: StateVector) -> DensityMatrix:
        """Convert a pure StateVector to ρ = |ψ⟩⟨ψ|."""
        psi = sv.get_amplitudes().reshape(-1, 1)
        rho = psi @ psi.conj().T
        return cls(sv.num_qubits, data=rho)

    @classmethod
    def from_amplitudes(cls, amplitudes: np.ndarray) -> DensityMatrix:
        """Create DensityMatrix from a raw amplitude array."""
        n = int(np.log2(len(amplitudes)))
        psi = amplitudes.reshape(-1, 1)
        return cls(n, data=psi @ psi.conj().T)

    @classmethod
    def maximally_mixed(cls, num_qubits: int) -> DensityMatrix:
        """Create the maximally mixed state ρ = I/2^n."""
        dim = 2 ** num_qubits
        return cls(num_qubits, data=np.eye(dim, dtype=np.complex128) / dim)

    # ── Core properties ───────────────────────────────────────────────────────

    def trace(self) -> complex:
        """Return tr(ρ). Should be 1 for physical states."""
        return complex(np.trace(self._matrix))

    def purity(self) -> float:
        """Return tr(ρ²). = 1 for pure states, < 1/dim for maximally mixed."""
        return float(np.real(np.trace(self._matrix @ self._matrix)))

    def is_pure(self, tol: float = 1e-10) -> bool:
        """Return True if purity ≈ 1 (pure state)."""
        return abs(self.purity() - 1.0) < tol

    def von_neumann_entropy(self) -> float:
        """Return S(ρ) = -tr(ρ log ρ) in nats."""
        eigenvalues = np.linalg.eigvalsh(self._matrix)
        eigenvalues = np.maximum(eigenvalues, 0.0)  # clamp numerical negatives
        ev = eigenvalues[eigenvalues > 1e-15]
        return float(-np.sum(ev * np.log(ev)))

    def expectation_value(self, observable: np.ndarray) -> float:
        """Compute ⟨O⟩ = tr(ρO) for a Hermitian observable."""
        return float(np.real(np.trace(self._matrix @ observable)))

    # ── Partial trace ─────────────────────────────────────────────────────────

    def partial_trace(self, keep_qubits: list[int]) -> DensityMatrix:
        """
        Compute partial trace over all subsystems NOT in keep_qubits.

        Uses tensor reshaping: ρ is reshaped to (2,2,...) with 2n indices,
        then traced over the subsystems to discard.

        Args:
            keep_qubits: List of qubit indices to retain.

        Returns:
            Reduced DensityMatrix on the kept subsystems.

        Example::

            # Bell state partial trace — qubit 0 is maximally mixed
            dm_bell = DensityMatrix.from_amplitudes(np.array([1,0,0,1])/√2)
            rho_q0 = dm_bell.partial_trace([0])
            assert abs(rho_q0.purity() - 0.5) < 1e-10
        """
        n = self._num_qubits
        keep = sorted(keep_qubits)
        trace_out = [q for q in range(n) if q not in keep]
        n_keep = len(keep)

        # Reshape ρ to (2,) * 2n:  axes [0..n-1] = row indices, [n..2n-1] = col indices
        rho: Any = self._matrix.reshape([2] * (2 * n))

        # Trace over each chosen subsystem axis (in reverse order to preserve indices)
        for q in sorted(trace_out, reverse=True):
            # Take trace over axis q (row) and axis q+n (col)
            rho = np.trace(rho, axis1=q, axis2=q + rho.ndim // 2)

        # After tracing, remaining shape is (2,)*2*n_keep
        dim_out = 2 ** n_keep
        rho = rho.reshape(dim_out, dim_out)
        return DensityMatrix(n_keep, data=rho)

    # ── Channel application ───────────────────────────────────────────────────

    def apply_kraus(self, kraus_ops: list[np.ndarray]) -> DensityMatrix:
        """
        Apply a quantum channel via Kraus operators: ε(ρ) = Σ_k K_k ρ K_k†.

        Args:
            kraus_ops: List of 2×2 Kraus matrices (single-qubit channel).

        Returns:
            New DensityMatrix after channel application.
        """
        result = np.zeros_like(self._matrix)
        for K in kraus_ops:
            result += K @ self._matrix @ K.conj().T
        return DensityMatrix(self._num_qubits, data=result)

    # ── Access ────────────────────────────────────────────────────────────────

    @property
    def matrix(self) -> np.ndarray:
        return self._matrix.copy()

    @property
    def num_qubits(self) -> int:
        return self._num_qubits

    @property
    def dim(self) -> int:
        return self._dim

    def __repr__(self) -> str:
        return (
            f"DensityMatrix(num_qubits={self._num_qubits}, "
            f"purity={self.purity():.6f}, trace={self.trace().real:.6f})"
        )
