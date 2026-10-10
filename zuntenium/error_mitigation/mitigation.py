# zuntenium/error_mitigation/mitigation.py
"""
Error Mitigation Strategies
============================

1. ZNEMitigator  — Zero Noise Extrapolation
   Mathematical basis:
     Execute circuit at noise scales c ∈ {1, 2, 3, …} by gate-folding:
       G → G (G†G)^{(c-1)/2}   (unitary identity insertions)
     Measure E(c) at each scale, fit model:
       Polynomial:   E(c) = a_0 + a_1·c + a_2·c² + …
       Exponential:  E(c) = A·e^{γc} + B
     Extrapolate to c → 0:   E_mitig = lim_{c→0} E(c)

2. PECMitigator  — Probabilistic Error Cancellation
   Mathematical basis:
     Express ideal operation I as quasi-probability mixture:
       I = Σ_i η_i ε_i    where Σ_i |η_i| = γ ≥ 1
     Unbiased estimator:
       ⟨O⟩_ideal = γ · E_{i~p_i}[sgn(η_i)·⟨O⟩_{ε_i}]
     Sampling overhead: γ² (variance inflation factor)

3. ReadoutMitigator  — Linear Inversion Readout Correction
   Mathematical basis:
     Calibrate assignment matrix A by preparing all 2^n basis states,
     measuring each, building:
       A[i,j] = P(measure i | prepared j)
     Correction:
       p_ideal = A^{-1} · p_noisy
     Regularization (Tikhonov) when cond(A) > threshold:
       p_ideal = (A^T A + λI)^{-1} A^T · p_noisy
     Constrain: project onto probability simplex (non-negative, sum=1)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

_log = logging.getLogger(__name__)


# ============================================================================
# MitigatedResult
# ============================================================================

@dataclass
class MitigatedResult:
    """
    Corrected execution result after error mitigation.

    Contains the original raw result, corrected probabilities,
    fidelity improvement metrics, and full audit trail.
    """
    # Corrected probability distribution
    corrected_probs: np.ndarray = field(default_factory=lambda: np.array([]))
    corrected_counts: dict[str, int] = field(default_factory=dict)

    # Raw (pre-mitigation) data
    raw_probs: np.ndarray = field(default_factory=lambda: np.array([]))
    raw_counts: dict[str, int] = field(default_factory=dict)

    # Fidelity metrics
    fidelity_before: float = 0.0
    fidelity_after:  float = 0.0
    fidelity_improvement: float = 0.0   # delta absolute

    # Method metadata
    method_used: str = "none"
    confidence: float = 1.0
    overhead_factor: float = 1.0   # sampling overhead (PEC: γ², ZNE: n_scales)
    n_qubits: int = 0

    # Detailed audit
    mitigation_log: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    # Original raw result (pass-through for downstream)
    raw_result: Any = None

    def to_counts(self, total_shots: int | None = None) -> dict[str, int]:
        """Convert corrected_probs back to integer counts."""
        if total_shots is None:
            total_shots = max(sum(self.raw_counts.values()), 1024)
        dim    = len(self.corrected_probs)
        n      = self.n_qubits or max(int(np.log2(dim)), 1)
        counts = {}
        samples = np.round(self.corrected_probs * total_shots).astype(int)
        for idx, cnt in enumerate(samples):
            if cnt > 0:
                key = format(idx, f"0{n}b")
                counts[key] = int(cnt)
        return counts

    def __str__(self) -> str:
        imp = self.fidelity_improvement
        sign = "+" if imp >= 0 else ""
        return (f"MitigatedResult(method={self.method_used}, "
                f"F: {self.fidelity_before:.4f}→{self.fidelity_after:.4f} "
                f"({sign}{imp:.4f}), conf={self.confidence:.3f})")


# ============================================================================
# Base class
# ============================================================================

class BaseMitigator(ABC):
    """Abstract base for all mitigation strategies."""

    @abstractmethod
    def mitigate(
        self,
        counts: dict[str, int],
        n_qubits: int,
        circuit=None,
        noise_model=None,
        ideal_probs: np.ndarray | None = None,
        **kwargs,
    ) -> MitigatedResult:
        ...

    @staticmethod
    def _counts_to_probs(counts: dict[str, int], dim: int) -> np.ndarray:
        total = max(sum(counts.values()), 1)
        p     = np.zeros(dim, dtype=float)
        for k, v in counts.items():
            key = str(k).strip()
            if all(c in "01" for c in key):
                idx = int(key, 2) % dim
                p[idx] += v / total
        return p

    @staticmethod
    def _probs_to_counts(probs: np.ndarray, total: int, n_qubits: int) -> dict[str, int]:
        counts = {}
        samples = np.round(probs * total).astype(int)
        for idx, cnt in enumerate(samples):
            if cnt > 0:
                counts[format(idx, f"0{n_qubits}b")] = int(cnt)
        return counts

    @staticmethod
    def _project_simplex(p: np.ndarray) -> np.ndarray:
        """
        Project onto probability simplex:  min ‖p - x‖²  s.t. Σx=1, x≥0
        Uses O(n log n) algorithm (Duchi et al., 2008).
        """
        p = np.asarray(p, dtype=float)
        n = len(p)
        u = np.sort(p)[::-1]
        cssv = np.cumsum(u)
        rho = np.nonzero(u > (cssv - 1) / (np.arange(n) + 1))[0][-1]
        theta = (cssv[rho] - 1) / (rho + 1)
        return np.maximum(p - theta, 0)


# ============================================================================
# ZNE Mitigator
# ============================================================================

class ZNEMitigator(BaseMitigator):
    """
    Zero Noise Extrapolation (ZNE).

    Scales noise via gate-folding: G → G(G†G)^k  at scale factors c = 1+2k.
    Then extrapolates expectation values to zero-noise limit.

    Parameters
    ----------
    scale_factors  : list of noise scales, e.g. [1, 2, 3] (must include 1)
    fit_type       : "polynomial" | "exponential" | "richardson"
    poly_degree    : polynomial degree for polynomial fit
    execute_fn     : callable(circuit, scale) → counts (injected by pipeline)
    observable     : "expectation_z" | "all_probs" | callable
    """

    def __init__(
        self,
        scale_factors: list[int] | None = None,
        fit_type: str = "polynomial",
        poly_degree: int = 2,
        execute_fn=None,
        observable: str = "all_probs",
    ) -> None:
        self.scale_factors = scale_factors or [1, 2, 3]
        self.fit_type      = fit_type
        self.poly_degree   = min(poly_degree, len(self.scale_factors) - 1)
        self.execute_fn    = execute_fn
        self.observable    = observable

    def mitigate(
        self,
        counts: dict[str, int],
        n_qubits: int,
        circuit=None,
        noise_model=None,
        ideal_probs: np.ndarray | None = None,
        scaled_counts_list: list[dict[str, int]] | None = None,
        **kwargs,
    ) -> MitigatedResult:
        """
        Apply ZNE mitigation.

        If ``scaled_counts_list`` is provided (pre-executed at each scale),
        uses those. Otherwise, falls back to single-scale extrapolation
        (trivially returns the original result) unless execute_fn is set.

        Parameters
        ----------
        scaled_counts_list : counts at each scale factor (optional)
                             len must match len(scale_factors)
        """
        dim    = 2 ** n_qubits
        p_raw  = self._counts_to_probs(counts, dim)
        total  = max(sum(counts.values()), 1)
        log    = []

        # Build probability matrix: rows = scales, cols = states
        if scaled_counts_list and len(scaled_counts_list) >= 2:
            probs_at_scales = np.array([
                self._counts_to_probs(c, dim) for c in scaled_counts_list
            ])
            scales_used = self.scale_factors[:len(probs_at_scales)]
            log.append(f"ZNE using {len(scales_used)} scale factors: {scales_used}")
        else:
            # Only scale=1 available: return raw result with warning
            _log.info("ZNE: only scale=1 available, returning raw result")
            return MitigatedResult(
                corrected_probs=p_raw,
                corrected_counts=counts,
                raw_probs=p_raw,
                raw_counts=counts,
                fidelity_before=self._estimate_fidelity(p_raw, ideal_probs),
                fidelity_after=self._estimate_fidelity(p_raw, ideal_probs),
                method_used="zne_single_scale",
                confidence=0.5,
                overhead_factor=float(len(self.scale_factors)),
                n_qubits=n_qubits,
                mitigation_log=["ZNE: insufficient scales, no extrapolation"],
            )

        # Extrapolate each probability bin independently
        p_extrapolated = np.zeros(dim, dtype=float)
        fit_errors     = np.zeros(dim, dtype=float)

        for k in range(dim):
            y_vals = probs_at_scales[:, k]
            y0, err = self._extrapolate(scales_used, y_vals)
            p_extrapolated[k] = y0
            fit_errors[k]     = err

        # Project onto probability simplex (enforce valid distribution)
        p_mitigated = self._project_simplex(p_extrapolated)

        # Fidelity comparison
        f_before = self._estimate_fidelity(p_raw, ideal_probs)
        f_after  = self._estimate_fidelity(p_mitigated, ideal_probs)

        fit_quality = 1.0 - float(np.clip(np.mean(fit_errors), 0, 1))
        confidence  = float(np.clip(0.6 + 0.4 * fit_quality, 0.0, 1.0))

        log.append(f"ZNE fit_type={self.fit_type}, poly_degree={self.poly_degree}")
        log.append(f"Fidelity: {f_before:.4f} → {f_after:.4f}")

        return MitigatedResult(
            corrected_probs=p_mitigated,
            corrected_counts=self._probs_to_counts(p_mitigated, total, n_qubits),
            raw_probs=p_raw,
            raw_counts=counts,
            fidelity_before=f_before,
            fidelity_after=f_after,
            fidelity_improvement=f_after - f_before,
            method_used=f"zne_{self.fit_type}",
            confidence=confidence,
            overhead_factor=float(len(scales_used)),
            n_qubits=n_qubits,
            mitigation_log=log,
            extra={
                "scale_factors": scales_used,
                "fit_errors": fit_errors.tolist(),
                "probs_at_scales": probs_at_scales.tolist(),
            },
        )

    def _extrapolate(
        self, scales: list[int], y_vals: np.ndarray
    ) -> tuple[float, float]:
        """
        Extrapolate to zero-noise limit.
        Returns (y0_estimate, fit_residual_norm).
        """
        x = np.array(scales, dtype=float)
        y = np.array(y_vals, dtype=float)

        if self.fit_type == "polynomial":
            try:
                coeffs = np.polyfit(x, y, self.poly_degree)
                y0     = float(np.polyval(coeffs, 0.0))
                y_fit  = np.polyval(coeffs, x)
                err    = float(np.mean((y_fit - y) ** 2))
            except Exception:
                y0, err = float(y[0]), 0.0

        elif self.fit_type == "exponential":
            try:
                # Fit y = A*exp(γ*x) + B  → linearize via log(y-B)
                # Use scipy.optimize.curve_fit
                from scipy.optimize import curve_fit

                def exp_model(x, A, gamma, B):
                    return A * np.exp(gamma * x) + B

                p0 = [y[0] - y[-1], -0.1, y[-1]]
                popt, _ = curve_fit(exp_model, x, y, p0=p0, maxfev=1000)
                y0  = float(exp_model(0.0, *popt))
                err = float(np.mean((exp_model(x, *popt) - y) ** 2))
            except Exception:
                # Fallback to polynomial
                coeffs = np.polyfit(x, y, min(2, len(x) - 1))
                y0  = float(np.polyval(coeffs, 0.0))
                err = 0.05

        elif self.fit_type == "richardson":
            # Richardson extrapolation (exact for polynomial order n-1):
            # E_0 = Σ_k c_k E(x_k)  where c_k = Π_{j≠k} (-x_j/(x_k-x_j))
            n  = len(x)
            c  = np.ones(n)
            for k in range(n):
                for j in range(n):
                    if j != k:
                        c[k] *= (-x[j]) / (x[k] - x[j])
            y0  = float(np.dot(c, y))
            err = 0.0  # exact for polynomial fit
        else:
            # Default: linear
            coeffs = np.polyfit(x, y, 1)
            y0     = float(np.polyval(coeffs, 0.0))
            err    = 0.05

        return y0, err

    @staticmethod
    def _estimate_fidelity(p: np.ndarray, q: np.ndarray | None) -> float:
        if q is None:
            return 0.0
        n = max(len(p), len(q))
        pp = np.pad(p, (0, n - len(p)))
        qq = np.pad(q, (0, n - len(q)))
        pp /= pp.sum() + 1e-15
        qq /= qq.sum() + 1e-15
        return float(np.clip(np.sum(np.sqrt(np.abs(pp * qq))) ** 2, 0, 1))


# ============================================================================
# PEC Mitigator
# ============================================================================

class PECMitigator(BaseMitigator):
    """
    Probabilistic Error Cancellation (PEC).

    Implements the quasi-probability decomposition framework.
    For each noisy channel ε, decomposes the ideal channel I as:
        I = Σ_i η_i ε_i    (quasi-probability representation)
    with sampling overhead γ = Σ_i |η_i|.

    In this implementation, the noise channel is modeled as:
        ε = (1-p)I + p · D  where D is a depolarizing perturbation
        η_0 = 1/(1-p),  η_1 = -p/(3(1-p))  per Pauli

    The unbiased estimator uses Monte Carlo sampling with sign tracking.

    Note: Full PEC requires re-executing the circuit many times with
    different quasi-probability samples. This implementation provides:
    1. The mathematical decomposition
    2. A post-hoc approximation using known noise structure
    3. The sampling overhead calculation
    """

    def __init__(
        self,
        noise_p: float = 0.01,
        n_samples: int = 1000,
    ) -> None:
        self.noise_p   = noise_p
        self.n_samples = n_samples

    def mitigate(
        self,
        counts: dict[str, int],
        n_qubits: int,
        circuit=None,
        noise_model=None,
        ideal_probs: np.ndarray | None = None,
        **kwargs,
    ) -> MitigatedResult:
        """
        Apply PEC via quasi-probability post-processing.

        Uses the noise model to invert the depolarizing channel analytically.
        For independent per-qubit depolarizing:
          p_ideal_k ≈ (p_noisy_k - p/3) / (1 - 4p/3)  per computational basis state

        Sampling overhead: γ = 1/(1 - 4p/3) for single-qubit depolarizing
        """
        dim   = 2 ** n_qubits
        p_raw = self._counts_to_probs(counts, dim)
        total = max(sum(counts.values()), 1)
        log   = []

        p_noise = self.noise_p
        if noise_model is not None and hasattr(noise_model, "depolarizing"):
            dep = noise_model.depolarizing
            if dep is not None:
                p_noise = dep.p

        # Analytical inversion of depolarizing channel
        # For n-qubit depolarizing: ε(ρ) = (1-p)ρ + p/4^n · I
        # Inversion: ρ_ideal = (ε(ρ) - p/4^n · I) / (1 - p)  (valid for p < 1)
        # In probability space (diagonal of density matrix):
        #   p_ideal_k = (p_noisy_k - p/4^n) / (1 - p)
        p_uniform = 1.0 / dim
        gamma = 1.0 / max(1 - p_noise, 1e-6)  # sampling overhead

        p_corrected = gamma * (p_raw - p_noise * p_uniform) + p_noise * p_uniform

        # Project onto simplex
        p_mitigated = self._project_simplex(p_corrected)

        f_before = self._estimate_fidelity(p_raw, ideal_probs)
        f_after  = self._estimate_fidelity(p_mitigated, ideal_probs)

        sampling_overhead = float(gamma ** 2)
        log.append(f"PEC: p_noise={p_noise:.4f}, γ={gamma:.3f}, overhead={sampling_overhead:.1f}x")

        return MitigatedResult(
            corrected_probs=p_mitigated,
            corrected_counts=self._probs_to_counts(p_mitigated, total, n_qubits),
            raw_probs=p_raw,
            raw_counts=counts,
            fidelity_before=f_before,
            fidelity_after=f_after,
            fidelity_improvement=f_after - f_before,
            method_used="pec",
            confidence=float(np.clip(1 - 2 * p_noise, 0.3, 0.95)),
            overhead_factor=sampling_overhead,
            n_qubits=n_qubits,
            mitigation_log=log,
            extra={"gamma": gamma, "noise_p": p_noise},
        )

    @staticmethod
    def quasi_probability_decomposition(p: float) -> dict[str, float]:
        """
        Return the quasi-probability coefficients for depolarizing with rate p.

        E^-1(I) = η_I · I + η_X · X + η_Y · Y + η_Z · Z  (single qubit)
        η_I = (3 - p) / (3 - 4p)
        η_P = -p / (3 - 4p)  for P ∈ {X, Y, Z}
        Reconstruction identity: η_I + 3·η_P = 1.0.
        """
        denom = max(3.0 - 4.0 * p, 1e-10)
        eta_i = (3.0 - p) / denom
        eta_p = -p / denom
        return {
            "I": eta_i,
            "X": eta_p,
            "Y": eta_p,
            "Z": eta_p,
            "gamma": abs(eta_i) + 3.0 * abs(eta_p),
        }

    @staticmethod
    def _estimate_fidelity(p, q):
        if q is None: return 0.0
        n = max(len(p), len(q))
        pp = np.pad(p, (0, n-len(p))); qq = np.pad(q, (0, n-len(q)))
        pp /= pp.sum()+1e-15; qq /= qq.sum()+1e-15
        return float(np.clip(np.sum(np.sqrt(np.abs(pp*qq)))**2, 0, 1))


# ============================================================================
# Readout Mitigator
# ============================================================================

class ReadoutMitigator(BaseMitigator):
    """
    Linear Inversion Readout Error Mitigation.

    Calibration phase (one-time cost):
      Prepare each of the 2^n computational basis states |b⟩,
      measure n_calibration_shots times,
      build assignment matrix A where A[i,j] = P(measure i | prepared j).

    Correction phase (per execution):
      p_ideal = A^{-1} p_noisy

    With Tikhonov regularization:
      p_ideal = (A^T A + λI)^{-1} A^T p_noisy

    Final step: project onto probability simplex.

    Parameters
    ----------
    n_qubits           : number of qubits
    noise_model        : NoiseModel (used to build calibration matrix analytically)
    calibration_matrix : pre-built A (optional, avoids recomputation)
    regularization_lam : Tikhonov λ (default 1e-5)
    condition_threshold: warn if cond(A) exceeds this (default 1e6)
    """

    def __init__(
        self,
        n_qubits: int = 1,
        noise_model=None,
        calibration_matrix: np.ndarray | None = None,
        regularization_lam: float = 1e-5,
        condition_threshold: float = 1e6,
    ) -> None:
        self.n_qubits            = n_qubits
        self.noise_model         = noise_model
        self._cal_matrix         = calibration_matrix
        self.regularization_lam  = regularization_lam
        self.condition_threshold = condition_threshold

    def build_calibration_matrix(
        self, noise_model=None, n_qubits: int | None = None
    ) -> np.ndarray:
        """
        Build assignment matrix A analytically from noise model.
        If no noise model, returns identity.

        A[i,j] = P(measure i | prepared j)
                = product of per-qubit assignment probabilities
        """
        nm = noise_model or self.noise_model
        n  = n_qubits or self.n_qubits
        dim = 2 ** n

        if nm is None or (nm.readout is None and not nm.qubit_readout):
            return np.eye(dim, dtype=float)

        # Build per-qubit assignment matrices
        A_per_qubit = []
        for q in range(n):
            rn = nm.get_readout_for_qubit(q)
            if rn is not None:
                A_per_qubit.append(rn.assignment_matrix())
            else:
                A_per_qubit.append(np.eye(2, dtype=float))

        # Full assignment matrix = tensor product of per-qubit matrices
        A = A_per_qubit[0]
        for Aq in A_per_qubit[1:]:
            A = np.kron(A, Aq)

        self._cal_matrix = A
        return A

    def mitigate(
        self,
        counts: dict[str, int],
        n_qubits: int,
        circuit=None,
        noise_model=None,
        ideal_probs: np.ndarray | None = None,
        **kwargs,
    ) -> MitigatedResult:
        """Apply readout mitigation using calibration matrix."""
        dim   = 2 ** n_qubits
        p_raw = self._counts_to_probs(counts, dim)
        total = max(sum(counts.values()), 1)
        log   = []

        # Build calibration matrix if needed (FIX BUG-005: update self.n_qubits)
        if self._cal_matrix is None or self._cal_matrix.shape[0] != dim:
            A = self.build_calibration_matrix(
                noise_model=noise_model, n_qubits=n_qubits
            )
            self.n_qubits = n_qubits  # FIX BUG-005: keep instance in sync
        else:
            A = self._cal_matrix

        # Check if correction is needed
        if np.allclose(A, np.eye(dim), atol=1e-8):
            log.append("Readout: identity calibration, no correction needed")
            p_mitigated = p_raw
            confidence  = 1.0
        else:
            p_mitigated, confidence = self._invert_matrix(A, p_raw, dim, log)

        # Project onto probability simplex
        p_mitigated = self._project_simplex(p_mitigated)

        f_before = self._estimate_fidelity(p_raw, ideal_probs)
        f_after  = self._estimate_fidelity(p_mitigated, ideal_probs)

        return MitigatedResult(
            corrected_probs=p_mitigated,
            corrected_counts=self._probs_to_counts(p_mitigated, total, n_qubits),
            raw_probs=p_raw,
            raw_counts=counts,
            fidelity_before=f_before,
            fidelity_after=f_after,
            fidelity_improvement=f_after - f_before,
            method_used="readout_mitigation",
            confidence=confidence,
            overhead_factor=1.0,
            n_qubits=n_qubits,
            mitigation_log=log,
            extra={"condition_number": float(np.linalg.cond(A))},
        )

    def _invert_matrix(
        self, A: np.ndarray, p: np.ndarray, dim: int, log: list[str]
    ) -> tuple[np.ndarray, float]:
        """
        Invert assignment matrix with optional Tikhonov regularization.
        Returns (corrected_probs, confidence).
        """
        cond = np.linalg.cond(A)
        log.append(f"Readout: cond(A)={cond:.2e}")

        if cond > self.condition_threshold:
            _log.warning(
                "Readout calibration matrix ill-conditioned (cond=%.2e > %.2e), "
                "using Tikhonov regularization", cond, self.condition_threshold
            )
            log.append(f"Using Tikhonov λ={self.regularization_lam:.2e}")
            # Tikhonov: (A^T A + λI)^{-1} A^T p
            lam = self.regularization_lam * np.max(np.abs(A)) ** 2
            ATA = A.T @ A + lam * np.eye(dim)
            try:
                p_corrected = np.linalg.solve(ATA, A.T @ p)
            except np.linalg.LinAlgError:
                p_corrected = p.copy()
            confidence = float(np.clip(1.0 - np.log10(cond) / 10.0, 0.3, 0.9))
        else:
            try:
                p_corrected = np.linalg.solve(A, p)
                confidence  = float(np.clip(1.0 - cond / self.condition_threshold, 0.5, 1.0))
            except np.linalg.LinAlgError:
                log.append("Matrix inversion failed, using least-squares")
                p_corrected, _, _, _ = np.linalg.lstsq(A, p, rcond=None)
                confidence = 0.6

        return p_corrected, confidence

    @staticmethod
    def _estimate_fidelity(p, q):
        if q is None: return 0.0
        n = max(len(p), len(q))
        pp = np.pad(p, (0, n-len(p))); qq = np.pad(q, (0, n-len(q)))
        pp /= pp.sum()+1e-15; qq /= qq.sum()+1e-15
        return float(np.clip(np.sum(np.sqrt(np.abs(pp*qq)))**2, 0, 1))


# ============================================================================
# ErrorMitigationEngine — central orchestrator
# ============================================================================

class ErrorMitigationEngine:
    """
    Central orchestrator for all mitigation strategies.

    Automatically selects and applies the best available strategy based on:
    - Backend type (statevector / hardware)
    - Noise model properties
    - Error detection report
    - User preference / AI optimizer hint

    Parameters
    ----------
    strategy         : "auto" | "zne" | "pec" | "readout" | "combined" | "none"
    noise_model      : NoiseModel to use
    zne_scale_factors: scale factors for ZNE
    zne_fit_type     : "polynomial" | "exponential" | "richardson"
    readout_reg_lam  : Tikhonov regularization for readout mitigation
    min_improvement  : minimum fidelity improvement to accept mitigation (default 0.0)
    """

    def __init__(
        self,
        strategy: str = "auto",
        noise_model=None,
        zne_scale_factors: list[int] | None = None,
        zne_fit_type: str = "polynomial",
        readout_reg_lam: float = 1e-5,
        min_improvement: float = 0.0,
    ) -> None:
        self.strategy             = strategy
        self.noise_model          = noise_model
        self.zne_scale_factors    = zne_scale_factors or [1, 2, 3]
        self.zne_fit_type         = zne_fit_type
        self.readout_reg_lam      = readout_reg_lam
        self.min_improvement      = min_improvement

        # Instantiate strategy objects
        self._zne = ZNEMitigator(
            scale_factors=self.zne_scale_factors,
            fit_type=self.zne_fit_type,
        )
        self._pec = PECMitigator(
            noise_p=noise_model.depolarizing.p if (
                noise_model and noise_model.depolarizing
            ) else 0.01
        )
        self._readout = ReadoutMitigator(
            noise_model=noise_model,
            regularization_lam=readout_reg_lam,
        )

    def apply(
        self,
        counts: dict[str, int],
        n_qubits: int,
        circuit=None,
        noise_model=None,
        detection_report=None,
        ideal_probs: np.ndarray | None = None,
        scaled_counts_list: list[dict[str, int]] | None = None,
        strategy_override: str | None = None,
    ) -> MitigatedResult:
        """
        Apply error mitigation to raw counts.

        Parameters
        ----------
        counts            : raw measurement counts
        n_qubits          : number of qubits
        circuit           : original circuit (for ZNE gate-folding info)
        noise_model       : override instance noise model
        detection_report  : DetectionReport from ErrorDetector (optional)
        ideal_probs       : reference ideal probabilities for fidelity tracking
        scaled_counts_list: for ZNE, pre-executed counts at each scale factor
        strategy_override : force a specific strategy
        """
        nm       = noise_model or self.noise_model
        strategy = strategy_override or self._select_strategy(nm, detection_report)

        _log.info("ErrorMitigationEngine: strategy=%s n_qubits=%d", strategy, n_qubits)

        if strategy == "none":
            dim   = 2 ** n_qubits
            p_raw = self._counts_to_probs(counts, dim)
            return MitigatedResult(
                corrected_probs=p_raw,
                corrected_counts=counts,
                raw_probs=p_raw,
                raw_counts=counts,
                fidelity_before=0.0,
                fidelity_after=0.0,
                method_used="none",
                n_qubits=n_qubits,
            )

        result = self._dispatch(
            strategy, counts, n_qubits, circuit, nm, ideal_probs, scaled_counts_list
        )

        # Safety check: if mitigation made things worse, revert
        if result.fidelity_improvement < -0.05 and ideal_probs is not None:
            _log.warning(
                "Mitigation degraded fidelity by %.4f, reverting to raw result",
                result.fidelity_improvement,
            )
            result.mitigation_log.append("⚠ REVERTED: mitigation degraded fidelity")
            dim   = 2 ** n_qubits
            p_raw = self._counts_to_probs(counts, dim)
            result.corrected_probs  = p_raw
            result.corrected_counts = counts
            result.method_used     += "_reverted"

        return result

    def _dispatch(
        self, strategy, counts, n_qubits, circuit, nm, ideal_probs, scaled_counts_list
    ) -> MitigatedResult:
        common = dict(
            counts=counts, n_qubits=n_qubits, circuit=circuit,
            noise_model=nm, ideal_probs=ideal_probs,
        )
        if strategy == "zne":
            return self._zne.mitigate(
                **common, scaled_counts_list=scaled_counts_list
            )
        elif strategy == "pec":
            return self._pec.mitigate(**common)
        elif strategy == "readout":
            return self._readout.mitigate(**common)
        elif strategy == "combined":
            # Readout first, then ZNE
            r1 = self._readout.mitigate(**common)
            common2 = {**common, "counts": r1.corrected_counts}
            r2 = self._zne.mitigate(**common2, scaled_counts_list=scaled_counts_list)
            r2.fidelity_before       = r1.fidelity_before
            r2.fidelity_improvement  = r2.fidelity_after - r2.fidelity_before
            r2.method_used           = "combined_readout+zne"
            r2.mitigation_log = r1.mitigation_log + r2.mitigation_log
            return r2
        else:
            return self._readout.mitigate(**common)

    def _select_strategy(self, nm, detection_report) -> str:
        """Auto-select strategy based on noise model and error detection."""
        if nm is None or nm.is_noiseless:
            return "none"

        has_readout = nm.readout is not None or bool(nm.qubit_readout)
        has_gate    = nm.depolarizing is not None or nm.gate_noise is not None

        error_score = getattr(detection_report, "error_score", 0.0) if detection_report else 0.0

        if error_score < 0.02:
            return "none"
        elif has_readout and has_gate and error_score > 0.1:
            return "combined"
        elif has_readout:
            return "readout"
        elif has_gate:
            return "zne"
        return "readout"

    @staticmethod
    def _counts_to_probs(counts: dict[str, int], dim: int) -> np.ndarray:
        total = max(sum(counts.values()), 1)
        p     = np.zeros(dim, dtype=float)
        for k, v in counts.items():
            key = str(k).strip()
            if all(c in "01" for c in key):
                idx = int(key, 2) % dim
                p[idx] += v / total
        return p
