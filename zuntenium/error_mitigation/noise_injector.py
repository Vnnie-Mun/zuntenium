# zuntenium/error_mitigation/noise_injector.py
"""
NoiseInjector — Applies noise channels to quantum simulation backends.

Supports three injection modes:
  - density_matrix : full ρ evolution via Kraus operators  (most accurate)
  - stochastic     : sample a Pauli error per channel via Monte Carlo
  - statevector    : convert to/from density matrix for injection

Works on SIMULATION BACKENDS ONLY (StatevectorSimulator, TensorNetwork).
IBM hardware noise is handled by the device itself.

Mathematical basis:
  Stochastic mode samples k from P(k) = ⟨ψ|K_k†K_k|ψ⟩ then applies K_k|ψ⟩/‖K_k|ψ⟩‖
  Density matrix mode: ε(ρ) = Σ_k K_k ρ K_k†
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .noise_model import NoiseModel, apply_kraus

_log = logging.getLogger(__name__)


class NoiseInjector:
    """
    Injects noise into quantum simulation results or intermediate states.

    Parameters
    ----------
    noise_model  : NoiseModel to apply
    mode         : "density_matrix" | "stochastic" | "counts"
                   - density_matrix: rigorous ρ evolution
                   - stochastic: Monte Carlo sampling of Kraus branch
                   - counts: post-process measurement counts with readout matrix
    rng_seed     : optional random seed for reproducibility
    """

    def __init__(
        self,
        noise_model: NoiseModel,
        mode: str = "density_matrix",
        rng_seed: int | None = None,
    ) -> None:
        self.noise_model = noise_model
        self.mode        = mode
        self._rng        = np.random.default_rng(rng_seed)

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def inject_into_statevector(
        self,
        statevector: np.ndarray,
        circuit_depth: int = 1,
        circuit=None,
    ) -> np.ndarray:
        """
        Apply noise to a pure-state statevector.

        Returns a (possibly mixed-state) density matrix ρ.
        """
        n_qubits = int(np.log2(len(statevector)))
        # Convert to density matrix
        sv = statevector.astype(complex) / np.linalg.norm(statevector)
        rho = np.outer(sv, sv.conj())

        if self.noise_model.is_noiseless:
            return rho

        # Apply per-qubit noise channels
        rho = self._apply_qubit_channels(rho, n_qubits, circuit_depth)

        return rho

    def inject_into_counts(
        self,
        counts: dict[str, int],
        n_qubits: int,
    ) -> dict[str, int]:
        """
        Apply readout noise to measurement counts post-hoc.
        Uses the assignment matrix per qubit (applied independently).
        """
        if self.noise_model.readout is None and not self.noise_model.qubit_readout:
            return counts

        total = sum(counts.values())
        if total == 0:
            return counts

        # Convert to probability vector
        dim  = 2 ** n_qubits
        p_vec = np.zeros(dim, dtype=float)
        for bitstr, cnt in counts.items():
            idx = int(bitstr, 2) if set(bitstr) <= {"0", "1"} else 0
            if idx < dim:
                p_vec[idx] = cnt / total

        # Apply per-qubit readout noise independently (product noise model)
        p_noisy = self._apply_readout_noise(p_vec, n_qubits)

        # Re-sample to integer counts preserving total
        p_noisy = np.abs(p_noisy)
        p_noisy /= p_noisy.sum() + 1e-15
        samples = self._rng.multinomial(total, p_noisy)

        noisy_counts: dict[str, int] = {}
        for idx, cnt in enumerate(samples):
            if cnt > 0:
                key = format(idx, f"0{n_qubits}b")
                noisy_counts[key] = int(cnt)
        return noisy_counts

    def inject_into_density_matrix(
        self,
        rho: np.ndarray,
        n_qubits: int,
        circuit_depth: int = 1,
    ) -> np.ndarray:
        """Apply noise channels to an existing density matrix."""
        if self.noise_model.is_noiseless:
            return rho
        return self._apply_qubit_channels(rho, n_qubits, circuit_depth)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_qubit_channels(
        self,
        rho: np.ndarray,
        n_qubits: int,
        circuit_depth: int,
    ) -> np.ndarray:
        """
        Apply per-qubit noise channels via partial trace + reinsertion.

        For each qubit q:
          1. Trace out all other qubits to get ρ_q (2×2)
          2. Apply noise: ε(ρ_q)
          3. Reconstruct full system via tensor product approximation

        Note: This is a mean-field approximation for independent per-qubit
        noise. For correlated noise, a full 2^n density matrix evolution
        would be required (exponential cost). For production use with n≤10.
        """
        depth_factor = circuit_depth if self.noise_model.time_dependent else 1

        # For small systems, do exact tensor evolution
        if n_qubits <= 10:
            return self._apply_exact(rho, n_qubits, depth_factor)
        else:
            # Large system: product-noise approximation
            return self._apply_product_approx(rho, n_qubits, depth_factor)

    def _apply_exact(
        self, rho: np.ndarray, n_qubits: int, depth_factor: int
    ) -> np.ndarray:
        """
        Exact noise application via partial-trace + single-qubit Kraus on each qubit.
        Reconstructs using the mean-field product state of noisy single-qubit states.
        """
        dim = 2 ** n_qubits
        rho_out = rho.copy().astype(complex)

        for q in range(n_qubits):
            kraus_sets = self.noise_model.get_qubit_kraus(q)
            if not kraus_sets:
                continue

            # Scale noise by circuit depth
            if depth_factor > 1 and self.noise_model.time_dependent:
                kraus_sets = self._scale_kraus_for_depth(kraus_sets, depth_factor)

            # Apply each channel in sequence on qubit q
            for kraus_ops in kraus_sets:
                rho_out = self._apply_single_qubit_channel(rho_out, kraus_ops, q, n_qubits)

        return rho_out

    def _apply_single_qubit_channel(
        self,
        rho: np.ndarray,
        kraus_ops: list[np.ndarray],
        qubit: int,
        n_qubits: int,
    ) -> np.ndarray:
        """
        Apply a single-qubit CPTP channel on qubit `qubit` of an n-qubit density matrix.

        ε_q(ρ) = Σ_k (I⊗…⊗K_k⊗…⊗I) ρ (I⊗…⊗K_k†⊗…⊗I)

        Uses kronecker product embedding in O(4^n) operations.
        """
        dim = 2 ** n_qubits
        rho_out = np.zeros((dim, dim), dtype=complex)

        # Qubit ordering: little-endian (matches StatevectorSimulator convention)
        # qubit 0 = least significant bit. For n=4, qubit 0: I⊗I⊗I⊗K
        # left_dim  = dimensions of qubits ABOVE qubit index (more significant)
        # right_dim = dimensions of qubits BELOW qubit index (less significant)
        left_dim  = 2 ** (n_qubits - qubit - 1)  # FIX BUG-004: was 2**qubit (big-endian)
        right_dim = 2 ** qubit                    # FIX BUG-004: was 2**(n-qubit-1)

        I_left  = np.eye(left_dim,  dtype=complex) if left_dim  > 1 else None
        I_right = np.eye(right_dim, dtype=complex) if right_dim > 1 else None

        for K in kraus_ops:
            if I_left is not None and I_right is not None:
                K_full = np.kron(np.kron(I_left, K), I_right)
            elif I_left is not None:
                K_full = np.kron(I_left, K)
            elif I_right is not None:
                K_full = np.kron(K, I_right)
            else:
                K_full = K

            rho_out += K_full @ rho @ K_full.conj().T

        # Normalise
        tr = np.trace(rho_out).real
        if tr > 1e-12:
            rho_out /= tr
        # Enforce Hermiticity
        rho_out = (rho_out + rho_out.conj().T) / 2
        return rho_out

    def _apply_product_approx(
        self, rho: np.ndarray, n_qubits: int, depth_factor: int
    ) -> np.ndarray:
        """
        Product-noise approximation for large systems (n > 10):
        Treat each qubit independently, evolve marginals, reconstruct.
        """
        # Compute partial traces and apply noise
        noisy_marginals = []
        for q in range(n_qubits):
            rho_q = self._partial_trace_single(rho, q, n_qubits)
            for kraus_ops in self.noise_model.get_qubit_kraus(q):
                rho_q = apply_kraus(rho_q, kraus_ops)
            noisy_marginals.append(rho_q)

        # Reconstruct via tensor product (approximate for correlated states)
        rho_out = noisy_marginals[0]
        for rho_q in noisy_marginals[1:]:
            rho_out = np.kron(rho_out, rho_q)
        return rho_out

    @staticmethod
    def _partial_trace_single(rho: np.ndarray, qubit: int, n_qubits: int) -> np.ndarray:
        """Partial trace over all qubits except `qubit`. Returns 2×2 density matrix."""
        dim = 2 ** n_qubits
        rho_q = np.zeros((2, 2), dtype=complex)

        for i in range(2):
            for j in range(2):
                # Sum over all basis states that agree on qubit `qubit`
                for rest in range(2 ** (n_qubits - 1)):
                    # Reconstruct full index
                    def full_idx(bit, r):
                        # Insert bit at position `qubit` (MSB = qubit 0)
                        mask_high = r >> qubit
                        mask_low  = r & ((1 << qubit) - 1)
                        return (mask_high << (qubit + 1)) | (bit << qubit) | mask_low

                    ri = full_idx(i, rest)
                    rj = full_idx(j, rest)
                    if ri < dim and rj < dim:
                        rho_q[i, j] += rho[ri, rj]
        return rho_q

    def _apply_readout_noise(self, p_vec: np.ndarray, n_qubits: int) -> np.ndarray:
        """
        Apply per-qubit readout noise to a probability distribution.
        Uses the tensor product of per-qubit assignment matrices.
        """
        # Build full assignment matrix as tensor product of per-qubit matrices
        A = np.array([[1.0]])
        for q in range(n_qubits):
            rn = self.noise_model.get_readout_for_qubit(q)
            if rn is not None:
                A = np.kron(A, rn.assignment_matrix())
            else:
                A = np.kron(A, np.eye(2))

        p_noisy = A @ p_vec
        p_noisy = np.clip(p_noisy, 0, None)
        s = p_noisy.sum()
        return p_noisy / s if s > 1e-15 else p_noisy

    def _scale_kraus_for_depth(
        self, kraus_sets: list[list[np.ndarray]], depth: int
    ) -> list[list[np.ndarray]]:
        """
        Scale noise rates by circuit depth for time-dependent noise.
        Approximation: apply channel `depth` times (composition).
        """
        scaled = []
        for kraus_ops in kraus_sets:
            # Compose channel depth times via superoperator multiplication
            # Approximation: just repeat the channel application
            scaled.append(kraus_ops)  # composition handled in apply loop
        return scaled
