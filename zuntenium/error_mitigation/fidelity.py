# zuntenium/error_mitigation/fidelity.py
"""
FidelityEstimator — Exact fidelity computation for quantum states and processes.

Mathematical Definitions:

1. Pure-state fidelity (statevector overlap):
   F(|ψ⟩, |φ⟩) = |⟨ψ|φ⟩|²   ∈ [0, 1]

2. Mixed-state fidelity (Uhlmann):
   F(ρ, σ) = (tr√(√ρ σ √ρ))²   ∈ [0, 1]
   For σ = |ψ⟩⟨ψ| pure:  F = ⟨ψ|ρ|ψ⟩  (simplifies greatly)

3. Trace distance:
   T(ρ, σ) = ½‖ρ - σ‖₁ = ½ tr|ρ - σ|   ∈ [0, 1]
   Relation: 1 - F ≤ T ≤ √(1 - F)

4. Process fidelity (average gate fidelity):
   F_proc(ε, U) = (d·F_e(ε, U) + 1) / (d + 1)
   where d = 2^n, F_e = Tr(U†ε(I/d)U·...)/d²

5. Classical fidelity (probability distributions):
   F_class(p, q) = (Σ_k √(p_k · q_k))²  (Bhattacharyya coefficient)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

_log = logging.getLogger(__name__)


@dataclass
class FidelityReport:
    """
    Complete fidelity characterisation of one execution result.

    All fidelity values ∈ [0, 1] (higher = better).
    """
    # Primary fidelity (best available estimate)
    fidelity: float = 0.0

    # Decomposed metrics (populated when data available)
    statevector_fidelity:   float | None = None
    density_matrix_fidelity: float | None = None
    classical_fidelity:     float | None = None
    trace_distance:         float | None = None

    # Error rates
    effective_error_rate: float = 0.0    # 1 - fidelity
    infidelity:           float = 0.0    # 1 - fidelity (alias)

    # Confidence in the estimate
    confidence: float = 1.0             # ∈ [0,1]
    confidence_reason: str = ""

    # Metadata
    method_used: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.fidelity           = float(np.clip(self.fidelity, 0.0, 1.0))
        self.infidelity         = float(1 - self.fidelity)
        self.effective_error_rate = self.infidelity

    def __str__(self) -> str:
        td_str = f" T={self.trace_distance:.4f}" if self.trace_distance else ""
        return (f"FidelityReport(F={self.fidelity:.6f}"
                f"{td_str} conf={self.confidence:.2f} [{self.method_used}])")


class FidelityEstimator:
    """
    Computes quantum fidelity using the best available method.

    Priority:
      1. Exact statevector overlap (if both ideal and noisy statevectors available)
      2. Pure-reference density matrix fidelity ⟨ψ|ρ|ψ⟩
      3. Classical distribution fidelity (probability vectors from counts)
    """

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def estimate(
        self,
        ideal_statevector: np.ndarray | None = None,
        noisy_statevector: np.ndarray | None = None,
        ideal_density_matrix: np.ndarray | None = None,
        noisy_density_matrix: np.ndarray | None = None,
        ideal_probs: np.ndarray | None = None,
        noisy_probs: np.ndarray | None = None,
        shots: int = 0,
    ) -> FidelityReport:
        """
        Estimate fidelity using the best available data.

        Priority: statevectors > density matrices > probability distributions.
        """
        # 1. Statevector overlap (most accurate)
        if ideal_statevector is not None and noisy_statevector is not None:
            return self._sv_fidelity(ideal_statevector, noisy_statevector)

        # 2. Pure reference + noisy density matrix: F = ⟨ψ_ideal|ρ_noisy|ψ_ideal⟩
        if ideal_statevector is not None and noisy_density_matrix is not None:
            return self._pure_mixed_fidelity(ideal_statevector, noisy_density_matrix)

        # 3. Both density matrices: Uhlmann fidelity
        if ideal_density_matrix is not None and noisy_density_matrix is not None:
            return self._uhlmann_fidelity(ideal_density_matrix, noisy_density_matrix)

        # 4. Classical fidelity from probability vectors
        if ideal_probs is not None and noisy_probs is not None:
            return self._classical_fidelity(ideal_probs, noisy_probs, shots)

        # Nothing available
        _log.warning("FidelityEstimator: no data to compare, returning F=0")
        return FidelityReport(
            fidelity=0.0, confidence=0.0,
            confidence_reason="No reference data provided",
            method_used="none",
        )

    # ------------------------------------------------------------------
    # Statevector fidelity
    # ------------------------------------------------------------------

    def _sv_fidelity(
        self, sv_ideal: np.ndarray, sv_noisy: np.ndarray
    ) -> FidelityReport:
        """F(|ψ⟩, |φ⟩) = |⟨ψ|φ⟩|²"""
        psi = np.asarray(sv_ideal, dtype=complex).ravel()
        phi = np.asarray(sv_noisy, dtype=complex).ravel()

        n = max(len(psi), len(phi))
        if len(psi) != len(phi):
            psi = np.pad(psi, (0, n - len(psi)))
            phi = np.pad(phi, (0, n - len(phi)))

        psi /= np.linalg.norm(psi) + 1e-15
        phi /= np.linalg.norm(phi) + 1e-15

        overlap = np.abs(np.vdot(psi, phi)) ** 2
        F = float(np.clip(overlap, 0.0, 1.0))

        # Also compute as density matrix for trace distance
        rho_ideal = np.outer(psi, psi.conj())
        rho_noisy = np.outer(phi, phi.conj())
        T = self._trace_distance(rho_ideal, rho_noisy)

        return FidelityReport(
            fidelity=F,
            statevector_fidelity=F,
            trace_distance=T,
            confidence=1.0,
            confidence_reason="Exact statevector overlap",
            method_used="statevector_overlap",
        )

    # ------------------------------------------------------------------
    # Pure-reference mixed-state fidelity
    # ------------------------------------------------------------------

    def _pure_mixed_fidelity(
        self, sv_ideal: np.ndarray, rho_noisy: np.ndarray
    ) -> FidelityReport:
        """
        F = ⟨ψ_ideal|ρ_noisy|ψ_ideal⟩

        This is the simplification of Uhlmann fidelity when one state is pure:
        F(|ψ⟩⟨ψ|, ρ) = ⟨ψ|ρ|ψ⟩
        """
        psi = np.asarray(sv_ideal, dtype=complex).ravel()
        psi /= np.linalg.norm(psi) + 1e-15
        rho = np.asarray(rho_noisy, dtype=complex)

        F = float(np.real(psi.conj() @ rho @ psi))
        F = float(np.clip(F, 0.0, 1.0))

        # Trace distance via eigenvalues
        rho_ideal = np.outer(psi, psi.conj())
        T = self._trace_distance(rho_ideal, rho)

        return FidelityReport(
            fidelity=F,
            density_matrix_fidelity=F,
            trace_distance=T,
            confidence=0.95,
            confidence_reason="Pure-reference mixed-state ⟨ψ|ρ|ψ⟩",
            method_used="pure_mixed_fidelity",
        )

    # ------------------------------------------------------------------
    # Uhlmann fidelity (both mixed)
    # ------------------------------------------------------------------

    def _uhlmann_fidelity(
        self, rho: np.ndarray, sigma: np.ndarray
    ) -> FidelityReport:
        """
        F(ρ, σ) = (tr√(√ρ σ √ρ))²

        Computed via eigendecomposition:
        √ρ = U diag(√λ) U†
        F = ‖√ρ √σ‖_trace²  where ‖A‖_trace = tr√(A†A)
        """
        rho   = np.asarray(rho,   dtype=complex)
        sigma = np.asarray(sigma, dtype=complex)

        # Enforce Hermiticity and positive semi-definiteness
        rho   = (rho   + rho.conj().T)   / 2
        sigma = (sigma + sigma.conj().T) / 2

        try:
            # √ρ via eigendecomposition
            lam_r, U_r = np.linalg.eigh(rho)
            lam_r = np.maximum(lam_r, 0)
            sqrt_rho = U_r @ np.diag(np.sqrt(lam_r)) @ U_r.conj().T

            # √ρ σ √ρ
            M = sqrt_rho @ sigma @ sqrt_rho
            M = (M + M.conj().T) / 2

            lam_m = np.linalg.eigvalsh(M)
            lam_m = np.maximum(lam_m.real, 0)

            F = float(np.clip(np.sum(np.sqrt(lam_m)) ** 2, 0.0, 1.0))
        except np.linalg.LinAlgError as exc:
            _log.warning("Uhlmann fidelity failed (%s), using classical proxy", exc)
            diag_rho   = np.diag(rho).real
            diag_sigma = np.diag(sigma).real
            bc = float(np.sum(np.sqrt(np.abs(diag_rho * diag_sigma))))
            F  = float(np.clip(bc ** 2, 0.0, 1.0))

        T = self._trace_distance(rho, sigma)

        return FidelityReport(
            fidelity=F,
            density_matrix_fidelity=F,
            trace_distance=T,
            confidence=0.9,
            confidence_reason="Uhlmann (mixed-mixed) fidelity",
            method_used="uhlmann_fidelity",
        )

    # ------------------------------------------------------------------
    # Classical fidelity
    # ------------------------------------------------------------------

    def _classical_fidelity(
        self,
        p_ideal: np.ndarray,
        p_noisy: np.ndarray,
        shots: int,
    ) -> FidelityReport:
        """
        Bhattacharyya coefficient squared (classical fidelity):
        F_class(p, q) = (Σ_k √(p_k · q_k))²

        Shot-noise confidence:  σ ~ 1/√shots
        """
        n = max(len(p_ideal), len(p_noisy))
        pp = np.pad(p_ideal, (0, n - len(p_ideal))).astype(float)
        qq = np.pad(p_noisy, (0, n - len(p_noisy))).astype(float)

        pp /= pp.sum() + 1e-15
        qq /= qq.sum() + 1e-15

        bc = float(np.sum(np.sqrt(np.clip(pp * qq, 0, None))))
        F  = float(np.clip(bc ** 2, 0.0, 1.0))

        # Shot-noise confidence
        if shots > 0:
            conf = float(1 - 1 / np.sqrt(shots))
        else:
            conf = 0.5

        tvd = float(0.5 * np.sum(np.abs(pp - qq)))

        return FidelityReport(
            fidelity=F,
            classical_fidelity=F,
            trace_distance=tvd,
            confidence=conf,
            confidence_reason=f"Classical Bhattacharyya coefficient (shots={shots})",
            method_used="classical_fidelity",
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _trace_distance(rho: np.ndarray, sigma: np.ndarray) -> float:
        """T(ρ, σ) = ½ tr|ρ - σ| = ½ Σ|λ_i|"""
        diff = rho - sigma
        eigs = np.linalg.eigvalsh(diff)
        return float(np.clip(0.5 * np.sum(np.abs(eigs.real)), 0.0, 1.0))

    @staticmethod
    def improvement(before: FidelityReport, after: FidelityReport) -> dict[str, float]:
        """Compute absolute and relative fidelity improvement after mitigation."""
        delta_abs = after.fidelity - before.fidelity
        delta_rel = delta_abs / max(before.infidelity, 1e-10)
        return {
            "fidelity_before": before.fidelity,
            "fidelity_after":  after.fidelity,
            "delta_absolute":  float(delta_abs),
            "delta_relative":  float(delta_rel),
            "error_reduction": float(np.clip(delta_rel, -1, 10)),
        }
