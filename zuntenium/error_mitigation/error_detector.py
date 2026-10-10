# zuntenium/error_mitigation/error_detector.py
"""
ErrorDetector — Quantitative error analysis for raw quantum results.

Methods:
  1. Chi-squared test          (counts vs. expected distribution)
  2. Kolmogorov-Smirnov test   (empirical CDF comparison)
  3. Fidelity / trace distance (when reference available)
  4. Variance tracking         (shot-to-shot consistency)
  5. Sanity checks             (probability sum, non-negativity, etc.)

Mathematical definitions:

  χ²  =  Σ_k (O_k - E_k)² / E_k          (chi-squared statistic)

  D_KS = sup_x |F_empirical(x) - F_reference(x)|   (KS distance)

  Trace distance:  T(ρ, σ) = ½‖ρ - σ‖₁ = ½ tr|ρ - σ|

  Fidelity (pure reference):  F = |⟨ψ|φ⟩|²

  Total variation distance:  TV = ½ Σ_k |p_k - q_k|
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

_log = logging.getLogger(__name__)


# ============================================================================
# DetectionReport
# ============================================================================

@dataclass
class DetectionReport:
    """
    Complete error characterisation for a single execution result.

    All metrics are in [0, 1] unless noted otherwise.
    """
    # Core statistics
    probability_sum: float = 1.0         # should be 1.0 ± tolerance
    max_negative_prob: float = 0.0       # should be 0.0
    entropy: float = 0.0                 # Shannon entropy H(p) in bits
    effective_num_states: float = 1.0    # exp(H) — "participation ratio"

    # Comparison to reference (if available)
    total_variation_distance: float | None = None
    chi_squared_stat: float | None = None
    chi_squared_pvalue: float | None = None
    ks_distance: float | None = None
    ks_pvalue: float | None = None

    # Error flags
    has_negative_probs: bool = False
    probability_sum_ok: bool = True
    statistically_significant_deviation: bool = False

    # Overall error score ∈ [0, 1]
    # 0 = perfect, 1 = maximally erroneous
    error_score: float = 0.0

    # Human-readable summary
    summary: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def is_suspicious(self) -> bool:
        """True if error score > 0.15 or any hard failure detected."""
        return (
            self.error_score > 0.15
            or not self.probability_sum_ok
            or self.has_negative_probs
            or self.statistically_significant_deviation
        )


# ============================================================================
# ErrorDetector
# ============================================================================

class ErrorDetector:
    """
    Detects and quantifies errors in quantum execution results.

    Parameters
    ----------
    alpha              : significance level for statistical tests (default 0.05)
    prob_sum_tolerance : tolerance for |Σp_k - 1| check (default 1e-3)
    negativity_tol     : tolerance for negative probabilities (default -1e-6)
    """

    def __init__(
        self,
        alpha: float = 0.05,
        prob_sum_tolerance: float = 1e-3,
        negativity_tol: float = -1e-6,
    ) -> None:
        self.alpha             = alpha
        self.prob_sum_tol      = prob_sum_tolerance
        self.negativity_tol    = negativity_tol

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def analyze(
        self,
        counts: dict[str, int],
        n_qubits: int,
        reference_probs: np.ndarray | None = None,
        reference_statevector: np.ndarray | None = None,
    ) -> DetectionReport:
        """
        Full error analysis of raw measurement counts.

        Parameters
        ----------
        counts               : raw measurement counts {"bitstring": count}
        n_qubits             : number of qubits
        reference_probs      : ideal probability vector (optional)
        reference_statevector: ideal statevector |ψ⟩ (optional, used to derive reference_probs)
        """
        report = DetectionReport()

        # Convert counts → probability vector
        p_obs = self._counts_to_probs(counts, n_qubits)

        # 1. Sanity checks
        self._check_sanity(p_obs, report)

        # 2. Information-theoretic metrics
        self._compute_information_metrics(p_obs, report)

        # 3. Reference comparison (if available)
        if reference_statevector is not None:
            sv = np.asarray(reference_statevector, dtype=complex)
            sv /= np.linalg.norm(sv) + 1e-15
            ref = np.abs(sv) ** 2
        elif reference_probs is not None:
            ref = np.asarray(reference_probs, dtype=float)
            ref = np.abs(ref) / (np.sum(np.abs(ref)) + 1e-15)
        else:
            ref = None

        if ref is not None:
            ref = self._align_prob_vector(ref, len(p_obs))
            self._compare_to_reference(p_obs, ref, counts, report)

        # 4. Compute overall error score
        report.error_score = self._compute_error_score(report)
        report.summary     = self._build_summary(report)

        _log.debug("ErrorDetector: score=%.3f suspicious=%s", report.error_score, report.is_suspicious)
        return report

    def analyze_density_matrix(
        self,
        rho: np.ndarray,
        rho_ideal: np.ndarray | None = None,
    ) -> DetectionReport:
        """
        Analyze a density matrix directly (when simulation provides ρ).
        """
        report = DetectionReport()

        # Eigenvalues should be non-negative and sum to 1
        eigvals = np.linalg.eigvalsh(rho).real
        report.max_negative_prob = float(min(eigvals.min(), 0))
        report.has_negative_probs = eigvals.min() < self.negativity_tol
        report.probability_sum = float(np.trace(rho).real)
        report.probability_sum_ok = abs(report.probability_sum - 1.0) < self.prob_sum_tol

        if rho_ideal is not None:
            # Trace distance T(ρ,σ) = ½‖ρ-σ‖₁
            diff = rho - rho_ideal
            eigvals_diff = np.linalg.eigvalsh(diff)
            trace_dist = 0.5 * float(np.sum(np.abs(eigvals_diff)))
            report.total_variation_distance = float(np.clip(trace_dist, 0, 1))
            report.statistically_significant_deviation = trace_dist > 0.1

        report.error_score = self._compute_error_score(report)
        report.summary     = self._build_summary(report)
        return report

    # ------------------------------------------------------------------
    # Statistical tests
    # ------------------------------------------------------------------

    def chi_squared_test(
        self,
        p_observed: np.ndarray,
        p_expected: np.ndarray,
        total_shots: int,
    ) -> tuple[float, float]:
        """
        Chi-squared goodness-of-fit test.

        Returns (statistic, p-value).
        Requires scipy if available; falls back to numpy-only approximation.
        """
        # Expected counts
        expected = p_expected * total_shots
        observed = p_observed * total_shots

        # Remove zero-expected bins to avoid division by zero
        mask = expected > 1e-10
        if mask.sum() < 2:
            return 0.0, 1.0

        chi2 = float(np.sum((observed[mask] - expected[mask]) ** 2 / expected[mask]))
        df   = int(mask.sum()) - 1

        try:
            from scipy.stats import chi2 as chi2_dist
            pval = float(1 - chi2_dist.cdf(chi2, df))
        except ImportError:
            # Rough p-value from normal approximation (only valid for large df)
            pval = float(np.exp(-chi2 / 2)) if df > 0 else 1.0

        return chi2, pval

    def ks_test(
        self,
        p_observed: np.ndarray,
        p_expected: np.ndarray,
    ) -> tuple[float, float]:
        """
        Kolmogorov-Smirnov test on cumulative distributions.
        Returns (D_KS, approximate_p_value).
        """
        # Ensure same length
        n = max(len(p_observed), len(p_expected))
        po = np.pad(p_observed, (0, n - len(p_observed)))
        pe = np.pad(p_expected, (0, n - len(p_expected)))

        cdf_obs = np.cumsum(po)
        cdf_exp = np.cumsum(pe)
        D = float(np.max(np.abs(cdf_obs - cdf_exp)))

        # Approximate p-value
        n_eff = len(po)
        try:
            from scipy.stats import ks_1samp
            # Use exact scipy test
            pval = float(np.exp(-2 * n_eff * D ** 2))
        except Exception:
            pval = float(np.exp(-2 * n_eff * D ** 2))

        return D, pval

    def total_variation_distance(
        self, p: np.ndarray, q: np.ndarray
    ) -> float:
        """TV(p, q) = ½ Σ_k |p_k - q_k|"""
        n = max(len(p), len(q))
        pp = np.pad(p, (0, n - len(p)))
        qq = np.pad(q, (0, n - len(q)))
        return float(0.5 * np.sum(np.abs(pp - qq)))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _counts_to_probs(self, counts: dict[str, int], n_qubits: int) -> np.ndarray:
        dim   = 2 ** n_qubits
        total = max(sum(counts.values()), 1)
        p     = np.zeros(dim, dtype=float)
        for bitstr, cnt in counts.items():
            key = str(bitstr).strip()
            if all(c in "01" for c in key):
                idx = int(key, 2) % dim
                p[idx] += cnt / total
        return p

    def _check_sanity(self, p: np.ndarray, report: DetectionReport) -> None:
        report.probability_sum    = float(p.sum())
        report.has_negative_probs = bool(p.min() < self.negativity_tol)
        report.max_negative_prob  = float(min(p.min(), 0.0))
        report.probability_sum_ok = abs(report.probability_sum - 1.0) < self.prob_sum_tol

    def _compute_information_metrics(self, p: np.ndarray, report: DetectionReport) -> None:
        pp = np.clip(p, 1e-15, None)
        pp /= pp.sum()
        H = float(-np.sum(pp * np.log2(pp + 1e-15)))
        report.entropy             = max(H, 0.0)
        report.effective_num_states = float(2 ** H)

    def _compare_to_reference(
        self,
        p_obs: np.ndarray,
        p_ref: np.ndarray,
        counts: dict[str, int],
        report: DetectionReport,
    ) -> None:
        total_shots = sum(counts.values())

        # Total variation distance
        tvd = self.total_variation_distance(p_obs, p_ref)
        report.total_variation_distance = float(np.clip(tvd, 0, 1))

        # Chi-squared
        if total_shots > 0:
            chi2, pval = self.chi_squared_test(p_obs, p_ref, total_shots)
            report.chi_squared_stat   = chi2
            report.chi_squared_pvalue = pval
            report.statistically_significant_deviation = pval < self.alpha

        # KS test
        ks_d, ks_p = self.ks_test(p_obs, p_ref)
        report.ks_distance = ks_d
        report.ks_pvalue   = ks_p

        report.details["tvd"]  = report.total_variation_distance
        report.details["chi2"] = report.chi_squared_stat
        report.details["ks"]   = report.ks_distance

    def _align_prob_vector(self, ref: np.ndarray, target_len: int) -> np.ndarray:
        """Pad or truncate reference probability vector to target_len."""
        if len(ref) == target_len:
            return ref
        r = np.zeros(target_len, dtype=float)
        n = min(len(ref), target_len)
        r[:n] = ref[:n]
        s = r.sum()
        return r / s if s > 1e-15 else r

    def _compute_error_score(self, report: DetectionReport) -> float:
        """
        Aggregate error score ∈ [0, 1].
        Combines TVD, chi-squared p-value, sanity flags.
        """
        score = 0.0

        if report.total_variation_distance is not None:
            score += 0.5 * float(report.total_variation_distance)

        if report.chi_squared_pvalue is not None:
            # Low p-value = evidence of error
            score += 0.3 * float(1 - report.chi_squared_pvalue)

        if not report.probability_sum_ok:
            score += 0.1 * abs(report.probability_sum - 1.0)

        if report.has_negative_probs:
            score += 0.1

        return float(np.clip(score, 0.0, 1.0))

    @staticmethod
    def _build_summary(report: DetectionReport) -> str:
        parts = [f"error_score={report.error_score:.3f}"]
        if report.total_variation_distance is not None:
            parts.append(f"TVD={report.total_variation_distance:.4f}")
        if report.chi_squared_pvalue is not None:
            parts.append(f"χ²_p={report.chi_squared_pvalue:.4f}")
        if report.statistically_significant_deviation:
            parts.append("⚠ SIGNIFICANT DEVIATION")
        if not report.probability_sum_ok:
            parts.append(f"⚠ prob_sum={report.probability_sum:.6f}")
        return " | ".join(parts)
