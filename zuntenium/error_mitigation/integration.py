# zuntenium/error_mitigation/integration.py
"""
MitigationPipeline — Full integration with HybridExecutor / QuantumExecutor.

Sits between backend.execute() and result delivery to HybridSolver/QuantumLayer:

  raw_result = backend.execute(circuit, ...)
  if mitigation_enabled:
      corrected = MitigationPipeline.process(raw_result, noise_model, circuit)
  return corrected

Zero overhead when disabled (mitigation_enabled=False).
Automatic fallback: if pipeline raises, raw result is returned with warning.

AIOptimizer hook: pipeline.set_strategy(ai_decision.mitigation_strategy)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .noise_model    import NoiseModel
from .noise_injector import NoiseInjector
from .error_detector import ErrorDetector, DetectionReport
from .fidelity       import FidelityEstimator, FidelityReport
from .mitigation     import (
    ErrorMitigationEngine, MitigatedResult, ReadoutMitigator,
    ZNEMitigator, PECMitigator,
)

_log = logging.getLogger(__name__)


# ============================================================================
# ExecutionResult shim (compatible with zuntenium.quantum_core.ExecutionResult)
# ============================================================================

@dataclass
class ExecutionResult:
    """
    Unified execution result container.
    Compatible with zuntenium.quantum_core.ExecutionResult (duck-typed).
    """
    counts: dict[str, int] = field(default_factory=dict)
    statevector: np.ndarray | None = None
    density_matrix: np.ndarray | None = None
    shots: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    # Mitigation fields (populated by pipeline)
    mitigated: bool = False
    mitigation_result: MitigatedResult | None = None
    fidelity_report: FidelityReport | None = None
    detection_report: DetectionReport | None = None

    @property
    def probabilities(self) -> np.ndarray | None:
        """Return probability vector (from counts or statevector)."""
        if self.statevector is not None:
            return np.abs(self.statevector) ** 2
        if self.counts:
            total = sum(self.counts.values())
            if total > 0:
                n = max(
                    1, max(len(k) for k in self.counts) if self.counts else 1
                )
                dim = 2 ** n
                p   = np.zeros(dim, dtype=float)
                for k, v in self.counts.items():
                    if all(c in "01" for c in k):
                        p[int(k, 2) % dim] += v / total
                return p
        return None


# ============================================================================
# MitigationPipeline
# ============================================================================

class MitigationPipeline:
    """
    Full error mitigation pipeline for Zuntenium execution results.

    Parameters
    ----------
    noise_model       : NoiseModel to use for mitigation
    strategy          : "auto" | "zne" | "pec" | "readout" | "combined" | "none"
    zne_scale_factors : scale factors for ZNE (default [1, 2, 3])
    zne_fit_type      : "polynomial" | "exponential" | "richardson"
    inject_noise      : whether to inject simulated noise into statevectors
    noise_injection_mode: "density_matrix" | "stochastic" | "counts"
    log_results       : whether to log mitigation results (for AI engine)
    execution_logger  : AIEngine ExecutionLogger (optional)
    min_shots_for_mitigation: skip mitigation if shots < this (default 100)
    """

    def __init__(
        self,
        noise_model: NoiseModel | None = None,
        strategy: str = "auto",
        zne_scale_factors: list[int] | None = None,
        zne_fit_type: str = "polynomial",
        inject_noise: bool = False,
        noise_injection_mode: str = "density_matrix",
        log_results: bool = True,
        execution_logger=None,
        min_shots_for_mitigation: int = 100,
    ) -> None:
        self.noise_model     = noise_model
        self.strategy        = strategy
        self.inject_noise    = inject_noise
        self.log_results     = log_results
        self._exec_logger    = execution_logger
        self.min_shots       = min_shots_for_mitigation

        # Sub-components
        self._injector  = NoiseInjector(noise_model or NoiseModel(), mode=noise_injection_mode)
        self._detector  = ErrorDetector()
        self._fidelity  = FidelityEstimator()
        self._engine    = ErrorMitigationEngine(
            strategy=strategy,
            noise_model=noise_model,
            zne_scale_factors=zne_scale_factors or [1, 2, 3],
            zne_fit_type=zne_fit_type,
        )

        _log.info(
            "MitigationPipeline: strategy=%s inject_noise=%s",
            strategy, inject_noise,
        )

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def process(
        self,
        raw_result: ExecutionResult | dict | Any,
        circuit=None,
        ideal_result: ExecutionResult | None = None,
        scaled_results: list[ExecutionResult] | None = None,
        strategy_override: str | None = None,
    ) -> ExecutionResult:
        """
        Process a raw execution result through the full mitigation pipeline.

        Pipeline stages:
          1. [Optional] Noise injection into statevector → density matrix
          2. Error detection & statistical analysis
          3. Fidelity estimation (pre-mitigation)
          4. Mitigation strategy application
          5. Fidelity estimation (post-mitigation)
          6. Result packaging with full audit trail
          7. [Optional] Log to AIEngine ExecutionLogger

        Parameters
        ----------
        raw_result      : execution result from backend
        circuit         : executed circuit (for ZNE gate-folding metadata)
        ideal_result    : ideal (noiseless) result for fidelity reference
        scaled_results  : results at noise scale factors (for ZNE)
        strategy_override: force specific strategy (overrides AIOptimizer hint)

        Returns
        -------
        ExecutionResult with mitigation metadata attached
        """
        t0 = time.perf_counter()

        try:
            result = self._process_inner(
                raw_result, circuit, ideal_result, scaled_results, strategy_override
            )
        except Exception as exc:
            _log.error("MitigationPipeline.process() failed: %s — returning raw", exc)
            result = self._wrap_raw(raw_result)
            result.metadata["mitigation_error"] = str(exc)

        result.metadata["pipeline_latency_ms"] = (time.perf_counter() - t0) * 1000
        return result

    def set_strategy(self, strategy: str) -> None:
        """Update mitigation strategy (called by AIOptimizer)."""
        self.strategy = strategy
        self._engine.strategy = strategy
        _log.info("MitigationPipeline: strategy updated to '%s'", strategy)

    def set_noise_model(self, noise_model: NoiseModel) -> None:
        """Update noise model (e.g., from hardware calibration data)."""
        self.noise_model     = noise_model
        self._injector       = NoiseInjector(noise_model)
        self._engine.noise_model = noise_model
        self._engine._readout = ReadoutMitigator(noise_model=noise_model)
        if noise_model.depolarizing:
            self._engine._pec = PECMitigator(noise_p=noise_model.depolarizing.p)

    # ------------------------------------------------------------------
    # Internal stages
    # ------------------------------------------------------------------

    def _process_inner(
        self,
        raw_result,
        circuit,
        ideal_result,
        scaled_results,
        strategy_override,
    ) -> ExecutionResult:
        # Normalise input to ExecutionResult
        er = self._to_execution_result(raw_result)
        n  = self._infer_n_qubits(er, circuit)

        if n == 0:
            _log.warning("Cannot determine n_qubits, skipping mitigation")
            return er

        # 1. Optional noise injection (simulation only)
        if self.inject_noise and er.statevector is not None and self.noise_model:
            depth = self._get_depth(circuit)
            er.density_matrix = self._injector.inject_into_statevector(
                er.statevector, circuit_depth=depth
            )
            # FIX BUG-007: null the statevector — state is now mixed, not pure
            # Keeping it would cause fidelity estimator to use ideal sv as reference
            er.statevector = None
            # Update counts from noisy density matrix diagonal
            if er.shots > 0:
                noisy_probs = np.diag(er.density_matrix).real
                noisy_probs = np.abs(noisy_probs) / (np.sum(np.abs(noisy_probs)) + 1e-15)
                er.counts = self._probs_to_counts(noisy_probs, er.shots, n)

        # Inject noise into counts as well (readout model)
        if self.inject_noise and er.counts and self.noise_model:
            er.counts = self._injector.inject_into_counts(er.counts, n)

        # 2. Skip mitigation for very few shots
        total_shots = max(sum(er.counts.values()) if er.counts else 0, er.shots)
        if total_shots < self.min_shots and not er.statevector is not None:
            _log.debug("Too few shots (%d), skipping mitigation", total_shots)
            er.metadata["mitigation_skipped"] = "insufficient_shots"
            return er

        # 3. Get ideal reference
        ideal_sv    = None
        ideal_probs = None
        if ideal_result is not None:
            ideal_r = self._to_execution_result(ideal_result)
            ideal_sv = ideal_r.statevector
            if ideal_r.probabilities is not None:
                ideal_probs = ideal_r.probabilities
        elif er.statevector is not None and not self.inject_noise:
            # If no noise injected, statevector IS the ideal
            ideal_sv    = er.statevector
            ideal_probs = np.abs(er.statevector) ** 2

        # 4. Error detection
        detection = self._detector.analyze(
            counts=er.counts or {},
            n_qubits=n,
            reference_probs=ideal_probs,
        )
        er.detection_report = detection

        # 5. Pre-mitigation fidelity
        f_before = self._fidelity.estimate(
            ideal_statevector=ideal_sv,
            noisy_density_matrix=er.density_matrix,
            ideal_probs=ideal_probs,
            noisy_probs=self._counts_to_probs(er.counts, 2**n) if er.counts else None,
            shots=total_shots,
        )

        # 6. Apply mitigation
        scaled_counts_list = None
        if scaled_results:
            scaled_counts_list = [
                self._to_execution_result(r).counts or {}
                for r in scaled_results
            ]

        mit_result = self._engine.apply(
            counts=er.counts or {},
            n_qubits=n,
            circuit=circuit,
            noise_model=self.noise_model,
            detection_report=detection,
            ideal_probs=ideal_probs,
            scaled_counts_list=scaled_counts_list,
            strategy_override=strategy_override or self.strategy,
        )

        # 7. Post-mitigation fidelity
        f_after = self._fidelity.estimate(
            ideal_statevector=ideal_sv,
            ideal_probs=ideal_probs,
            noisy_probs=mit_result.corrected_probs,
            shots=total_shots,
        )

        improvement = FidelityEstimator.improvement(f_before, f_after)

        # 8. Build corrected ExecutionResult
        er_out = ExecutionResult(
            counts=mit_result.corrected_counts,
            statevector=er.statevector,         # pass-through
            density_matrix=er.density_matrix,
            shots=total_shots,
            metadata={
                **er.metadata,
                "mitigation_applied": True,
                "mitigation_method":  mit_result.method_used,
                "fidelity_before":    f_before.fidelity,
                "fidelity_after":     f_after.fidelity,
                "fidelity_improvement": improvement,
                "overhead_factor":    mit_result.overhead_factor,
                "mitigation_log":     mit_result.mitigation_log,
                "error_score":        detection.error_score,
                "detection_summary":  detection.summary,
            },
            mitigated=True,
            mitigation_result=mit_result,
            fidelity_report=f_after,
            detection_report=detection,
        )

        _log.info(
            "Mitigation complete: %s | F: %.4f → %.4f (Δ=%.4f) | score=%.3f",
            mit_result.method_used,
            f_before.fidelity,
            f_after.fidelity,
            improvement.get("delta_absolute", 0),
            detection.error_score,
        )

        return er_out

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _to_execution_result(r) -> ExecutionResult:
        """Convert various result formats to ExecutionResult."""
        if isinstance(r, ExecutionResult):
            return r
        if isinstance(r, dict):
            return ExecutionResult(
                counts=r.get("counts", {}),
                statevector=r.get("statevector"),
                density_matrix=r.get("density_matrix"),
                shots=r.get("shots", sum(r.get("counts", {}).values())),
                metadata=r.get("metadata", {}),
            )
        # Duck-type: try to extract common attributes
        er = ExecutionResult()
        for attr in ("counts", "statevector", "density_matrix", "shots", "metadata"):
            if hasattr(r, attr):
                setattr(er, attr, getattr(r, attr))
        return er

    @staticmethod
    def _wrap_raw(r) -> ExecutionResult:
        if isinstance(r, ExecutionResult):
            return r
        return MitigationPipeline._to_execution_result(r)

    @staticmethod
    def _infer_n_qubits(er: ExecutionResult, circuit=None) -> int:
        if circuit is not None and hasattr(circuit, "num_qubits"):
            return int(circuit.num_qubits)
        if er.statevector is not None:
            return max(1, int(np.log2(len(er.statevector))))
        if er.counts:
            keys = [k for k in er.counts if all(c in "01" for c in str(k))]
            if keys:
                return max(len(k) for k in keys)
        return 0

    @staticmethod
    def _get_depth(circuit) -> int:
        if circuit is None:
            return 1
        if hasattr(circuit, "depth"):
            d = circuit.depth
            return int(d() if callable(d) else d)
        return 1

    @staticmethod
    def _counts_to_probs(counts: dict | None, dim: int) -> np.ndarray:
        if not counts:
            return np.zeros(dim, dtype=float)
        total = max(sum(counts.values()), 1)
        p     = np.zeros(dim, dtype=float)
        for k, v in counts.items():
            key = str(k).strip()
            if all(c in "01" for c in key):
                idx = int(key, 2) % dim
                p[idx] += v / total
        return p

    @staticmethod
    def _probs_to_counts(probs: np.ndarray, shots: int, n: int) -> dict[str, int]:
        counts = {}
        samples = np.round(probs * shots).astype(int)
        for idx, cnt in enumerate(samples):
            if cnt > 0:
                counts[format(idx, f"0{n}b")] = int(cnt)
        return counts


# ============================================================================
# AIAwareMitigationPipeline (AIOptimizer hook)
# ============================================================================

class AIAwareMitigationPipeline(MitigationPipeline):
    """
    MitigationPipeline extended with AIOptimizer strategy selection.

    When an AIOptimizer is provided, its decision influences:
    - Which mitigation strategy to apply
    - Whether to enable noise injection
    - Noise model parameter overrides

    Parameters
    ----------
    ai_optimizer : AIOptimizer (from zuntenium.ai_engine)
    **kwargs     : forwarded to MitigationPipeline
    """

    def __init__(self, ai_optimizer=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._ai = ai_optimizer

    def process(self, raw_result, circuit=None, ideal_result=None,
                scaled_results=None, strategy_override=None) -> ExecutionResult:
        """Apply AI-guided mitigation strategy selection."""
        # If AI optimizer available, ask for mitigation hint
        if self._ai is not None and strategy_override is None:
            try:
                hint = self._get_ai_hint(circuit)
                if hint:
                    strategy_override = hint
                    _log.info("AI mitigation hint: %s", hint)
            except Exception as exc:
                _log.debug("AI hint failed: %s", exc)

        return super().process(
            raw_result, circuit=circuit,
            ideal_result=ideal_result,
            scaled_results=scaled_results,
            strategy_override=strategy_override,
        )

    def _get_ai_hint(self, circuit) -> str | None:
        """Extract mitigation strategy hint from AIOptimizer decision."""
        if circuit is None or self._ai is None:
            return None
        try:
            decision = self._ai.select(circuit)
            # Check if decision params contain mitigation hint
            return decision.params.get("mitigation_strategy")
        except Exception:
            return None


# ============================================================================
# HybridExecutor integration patch
# ============================================================================

def patch_hybrid_executor(
    executor_class,
    noise_model: NoiseModel | None = None,
    strategy: str = "auto",
):
    """
    Monkey-patch an existing HybridExecutor class to add mitigation support.

    Usage:
        from zuntenium.executor import HybridExecutor
        from zuntenium.error_mitigation.integration import patch_hybrid_executor
        patch_hybrid_executor(HybridExecutor, noise_model=medium_noise_model())

    Then:
        executor = HybridExecutor(mitigation_enabled=True)
        result = executor.run(circuit, shots=1024)
    """
    original_run = executor_class.run if hasattr(executor_class, "run") else None

    pipeline = MitigationPipeline(noise_model=noise_model, strategy=strategy)

    def run_with_mitigation(self, circuit, mitigation_enabled=False, **kwargs):
        result = original_run(self, circuit, **kwargs) if original_run else {}
        if mitigation_enabled:
            result = pipeline.process(result, circuit=circuit)
        return result

    executor_class.run = run_with_mitigation
    executor_class._mitigation_pipeline = pipeline
    _log.info("HybridExecutor patched with MitigationPipeline (strategy=%s)", strategy)


class MitigationAwareHybridExecutor:
    """
    Standalone HybridExecutor wrapper that adds mitigation as a first-class feature.

    Parameters
    ----------
    base_executor       : existing HybridExecutor instance
    mitigation_enabled  : default False (backward compatible)
    noise_model         : NoiseModel for mitigation
    strategy            : mitigation strategy
    ai_optimizer        : optional AIOptimizer for strategy selection
    inject_noise        : inject simulated noise (for simulation backends)
    """

    def __init__(
        self,
        base_executor=None,
        mitigation_enabled: bool = False,
        noise_model: NoiseModel | None = None,
        strategy: str = "auto",
        ai_optimizer=None,
        inject_noise: bool = False,
        zne_scale_factors: list[int] | None = None,
    ) -> None:
        self._base            = base_executor
        self.mitigation_enabled = mitigation_enabled
        self._pipeline = AIAwareMitigationPipeline(
            ai_optimizer=ai_optimizer,
            noise_model=noise_model,
            strategy=strategy,
            inject_noise=inject_noise,
            zne_scale_factors=zne_scale_factors or [1, 2, 3],
        )

    def run(
        self,
        circuit,
        mitigation_enabled: bool | None = None,
        scaled_circuits=None,
        ideal_result=None,
        **kwargs,
    ) -> ExecutionResult:
        """
        Execute circuit with optional error mitigation.

        Parameters
        ----------
        circuit           : quantum circuit
        mitigation_enabled: override instance default
        scaled_circuits   : list of circuits at noise scale factors (for ZNE)
        ideal_result      : ideal reference result (for fidelity tracking)
        **kwargs          : forwarded to base executor
        """
        enabled = mitigation_enabled if mitigation_enabled is not None else self.mitigation_enabled

        # Execute base circuit
        if self._base is not None:
            try:
                raw = self._base.run(circuit, **kwargs)
            except Exception as exc:
                _log.error("Base executor failed: %s", exc)
                raw = {}
        else:
            raw = {}

        if not enabled:
            return self._pipeline._to_execution_result(raw)

        # Execute at scale factors if needed for ZNE
        scaled_results = None
        if scaled_circuits and self._base is not None:
            scaled_results = []
            for sc in scaled_circuits:
                try:
                    scaled_results.append(self._base.run(sc, **kwargs))
                except Exception as exc:
                    _log.warning("Scaled circuit execution failed: %s", exc)

        return self._pipeline.process(
            raw,
            circuit=circuit,
            ideal_result=ideal_result,
            scaled_results=scaled_results,
        )

    def set_noise_model(self, noise_model: NoiseModel) -> None:
        self._pipeline.set_noise_model(noise_model)

    def set_strategy(self, strategy: str) -> None:
        self._pipeline.set_strategy(strategy)
