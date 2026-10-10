# zuntenium/executor_ai_integration.py
"""
Integration patch for HybridExecutor and QuantumExecutor.

Adds ``optimization_mode`` parameter:
  "rule" (default) → existing BackendSelector behaviour (zero breaking change)
  "ai"             → AIOptimizer.select() called first; rule-based fallback always available

Usage
-----
    from zuntenium.executor_ai_integration import AIAwareHybridExecutor

    executor = AIAwareHybridExecutor(
        optimization_mode="ai",
        model_path="models/ai_engine",
        use_rl=True,
    )
    result = executor.run(circuit, shots=1024)

The integration is designed as a subclass / mixin so the original
HybridExecutor is never modified — zero breaking changes.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stub base classes (replaced by real imports when running inside Zuntenium)
# ---------------------------------------------------------------------------
# These allow this file to be read and tested in isolation.

try:
    from zuntenium.execution.hybrid_executor import HybridExecutor as _BaseHybridExecutor
    from zuntenium.execution.quantum_executor import QuantumExecutor as _BaseQuantumExecutor
    from zuntenium.execution.adaptive_engine import BackendSelector as _BackendSelector
    from zuntenium.transpiler.v2 import Transpiler as _Transpiler
    _REAL_IMPORTS = True
except ImportError:
    _REAL_IMPORTS = False
    _log.warning("zuntenium.executor not found — using stub base classes.")

    class _BaseHybridExecutor:                              # type: ignore[no-redef]
        def __init__(self, *a, **kw): pass
        def run(self, circuit, **kw):
            return {"counts": {}, "statevector": None}

    class _BaseQuantumExecutor:                             # type: ignore[no-redef]
        def __init__(self, *a, **kw): pass
        def execute(self, circuit, **kw):
            return {"counts": {}}

    class _BackendSelector:                                 # type: ignore[no-redef]
        def select(self, circuit): return type("D", (), {"backend": "statevector"})()

    class _Transpiler:                                      # type: ignore[no-redef]
        def transpile(self, circuit, **kw): return circuit


from zuntenium.ai_engine.features import FeatureExtractor
from zuntenium.ai_engine.logger import ExecutionLogger
from zuntenium.ai_engine.online_loop import OnlineLearningLoop
from zuntenium.ai_engine.optimizer import AIOptimizer
from zuntenium.ai_engine.decision import (
    AIDecision, BackendChoice, TranspilationStrategy,
)


# ---------------------------------------------------------------------------
# AIAwareQuantumExecutor
# ---------------------------------------------------------------------------

class AIAwareQuantumExecutor(_BaseQuantumExecutor):
    """
    QuantumExecutor extended with AI optimization support.

    When ``optimization_mode="ai"`` the executor:
      1. Calls ai_optimizer.select() to get an AIDecision.
      2. Uses the AI-chosen backend and transpilation strategy.
      3. After execution, calls ai_optimizer.record_outcome() for online learning.

    Parameters
    ----------
    optimization_mode : "rule" | "ai"
    ai_optimizer      : AIOptimizer | None  (auto-created if None and mode="ai")
    model_path        : path for model persistence
    use_rl            : enable PPO RL agent
    **kwargs          : forwarded to base QuantumExecutor.__init__
    """

    def __init__(
        self,
        *args,
        optimization_mode: str = "rule",
        ai_optimizer: AIOptimizer | None = None,
        model_path: str | Path | None = None,
        use_rl: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.optimization_mode = optimization_mode

        if optimization_mode == "ai":
            rule_selector = getattr(self, "_backend_selector", None) or _BackendSelector()
            self._ai_optimizer = ai_optimizer or AIOptimizer(
                rule_selector=rule_selector,
                model_path=model_path,
                use_rl=use_rl,
            )
            self._learning_loop = OnlineLearningLoop(self._ai_optimizer)
            self._learning_loop.start()
            _log.info("AIAwareQuantumExecutor: AI mode active.")
        else:
            self._ai_optimizer  = None
            self._learning_loop = None

    def execute(self, circuit, **kwargs) -> dict:
        """
        Execute with optional AI routing.
        Falls back to parent implementation if AI not active.
        """
        if self.optimization_mode != "ai" or self._ai_optimizer is None:
            return super().execute(circuit, **kwargs)

        t_start  = time.perf_counter()
        decision = self._ai_optimizer.select(circuit, preferences=kwargs)
        _log.info("AI routing: %s", decision)

        # Apply AI decision
        result, error = self._execute_with_decision(circuit, decision, kwargs)

        exec_time = time.perf_counter() - t_start
        success   = error is None
        fidelity  = self._estimate_fidelity(result)

        # Record outcome for online learning
        self._ai_optimizer.record_outcome(
            circuit, decision,
            execution_time_s=exec_time,
            fidelity=fidelity,
            success=success,
            error_message=str(error) if error else None,
        )
        if self._learning_loop:
            self._learning_loop.notify_execution()

        if not success:
            raise error  # type: ignore[misc]
        return result

    def _execute_with_decision(
        self, circuit, decision: AIDecision, kwargs: dict
    ) -> tuple[dict, Exception | None]:
        """Route execution based on AIDecision, with fallback."""
        try:
            backend_name = decision.backend_str
            params       = {**decision.params, **kwargs}

            # Apply transpilation
            transpiled = self._apply_transpilation(circuit, decision)

            # Select backend and run
            return self._run_on_backend(transpiled, backend_name, params), None

        except Exception as exc:
            _log.error("Execution with AI decision failed: %s — falling back to base", exc)
            try:
                return super().execute(circuit, **kwargs), None
            except Exception as exc2:
                return {}, exc2

    def _apply_transpilation(self, circuit, decision: AIDecision):
        """Apply the chosen transpilation strategy."""
        strategy = decision.transpilation_strategy

        if strategy == TranspilationStrategy.NONE:
            return circuit

        transpiler = getattr(self, "_transpiler", None)
        if transpiler is None:
            return circuit  # No transpiler available

        try:
            if strategy == TranspilationStrategy.RULE_ONLY:
                return transpiler.transpile(circuit, mode="rule")
            elif strategy == TranspilationStrategy.RL_REFINE:
                return transpiler.transpile(circuit, mode="rl")
            elif strategy == TranspilationStrategy.HARDWARE:
                return transpiler.transpile(circuit, mode="hardware")
        except Exception as exc:
            _log.warning("Transpilation failed (%s), using original circuit", exc)
        return circuit

    def _run_on_backend(self, circuit, backend_name: str, params: dict) -> dict:
        """Dispatch to the appropriate backend by name."""
        # Delegate to parent's backend registry if available
        if hasattr(self, "_backends") and backend_name in self._backends:
            backend = self._backends[backend_name]
            return backend.run(circuit, **params)

        # Fallback: let parent handle it
        return super().execute(circuit, **params)

    @staticmethod
    def _estimate_fidelity(result: Any) -> float:
        """Extract fidelity from either a result mapping or execution result."""
        if not result:
            return 0.0
        if isinstance(result, dict):
            return float(result.get("fidelity", 1.0))
        metadata = getattr(result, "metadata", {}) or {}
        return float(metadata.get("fidelity", getattr(result, "fidelity", 1.0)))

    def shutdown(self) -> None:
        """Graceful shutdown: flush learning loop and save models."""
        if self._learning_loop:
            _log.info("Flushing learning loop before shutdown...")
            self._learning_loop.flush()
            self._learning_loop.stop()


# ---------------------------------------------------------------------------
# AIAwareHybridExecutor (primary interface)
# ---------------------------------------------------------------------------

class AIAwareHybridExecutor(_BaseHybridExecutor):
    """
    HybridExecutor extended with AI optimization.

    Primary entry point for users of the Zuntenium AI Engine.

    Parameters
    ----------
    optimization_mode : "rule" | "ai"
        "rule" = legacy behaviour (backward-compatible)
        "ai"   = AI-driven backend + transpilation selection
    ai_optimizer : AIOptimizer | None
        Inject a pre-configured optimizer; auto-created if None.
    model_path : str | Path | None
        Path for loading/saving model checkpoints.
    use_rl : bool
        Enable PPO RL agent (Phase 2). Requires PyTorch.
    learning_loop : bool
        Start background learning loop (default True when mode="ai").
    retrain_every : int
        Supervised retrain cadence (executions).
    rl_update_every : int
        PPO update cadence (executions).
    **kwargs : forwarded to base HybridExecutor.__init__
    """

    def __init__(
        self,
        *args,
        optimization_mode: str = "rule",
        ai_optimizer: AIOptimizer | None = None,
        model_path: str | Path | None = None,
        use_rl: bool = False,
        learning_loop: bool = True,
        retrain_every: int = 50,
        rl_update_every: int = 10,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.optimization_mode = optimization_mode
        self._ai_optimizer: AIOptimizer | None = None
        self._learning_loop: OnlineLearningLoop | None = None

        if optimization_mode == "ai":
            rule_sel = getattr(self, "_backend_selector", None) or _BackendSelector()
            self._ai_optimizer = ai_optimizer or AIOptimizer(
                rule_selector=rule_sel,
                model_path=model_path,
                use_rl=use_rl,
            )
            if learning_loop:
                self._learning_loop = OnlineLearningLoop(
                    self._ai_optimizer,
                    retrain_every=retrain_every,
                    rl_update_every=rl_update_every,
                )
                self._learning_loop.start()

            _log.info(
                "AIAwareHybridExecutor ready | RL=%s | loop=%s | path=%s",
                use_rl, learning_loop, model_path,
            )

    def run(self, circuit, **kwargs) -> dict:
        """
        Run circuit with optional AI routing.

        Extra kwargs (passed as preferences to AIOptimizer):
          shots, hardware_available, queue_time_estimate_s,
          noise_model_requested, real_hardware
        """
        if self.optimization_mode != "ai" or self._ai_optimizer is None:
            return super().run(circuit, **kwargs)

        t_start  = time.perf_counter()
        decision = self._ai_optimizer.select(circuit, preferences=kwargs)
        _log.info("[AI] %s", decision)

        result, error = self._dispatch(circuit, decision, kwargs)

        exec_time = time.perf_counter() - t_start
        success   = error is None

        self._ai_optimizer.record_outcome(
            circuit, decision,
            execution_time_s=exec_time,
            fidelity=self._estimate_fidelity(result),
            success=success,
            error_message=str(error) if error else None,
        )
        if self._learning_loop:
            self._learning_loop.notify_execution()

        if not success:
            raise error  # type: ignore[misc]
        return result

    def _dispatch(
        self, circuit, decision: AIDecision, kwargs: dict
    ) -> tuple[dict, Exception | None]:
        """Route circuit to the backend indicated by AIDecision."""
        try:
            return self._run_decision(circuit, decision, kwargs), None
        except Exception as exc:
            _log.error("AI dispatch failed (%s), falling back to base.run()", exc)
            try:
                return super().run(circuit, **kwargs), None
            except Exception as exc2:
                return {}, exc2

    def _run_decision(self, circuit, decision: AIDecision, kwargs: dict) -> dict:
        """
        FIX BUG-009: Route to the backend indicated by AIDecision.
        Checks backend registry first, falls back to base executor.
        """
        _log.debug(
            "Executing: backend=%s strategy=%s params=%s",
            decision.backend_str, decision.strategy_str, decision.params,
        )
        # Merge AI params with caller kwargs (caller wins)
        merged = {**decision.params, **kwargs}

        # Route to specific backend if registry exists
        backend_registry = getattr(self, "_backends", None) or getattr(self._base if hasattr(self, "_base") else self, "_backends", None)
        if backend_registry and decision.backend_str in backend_registry:
            backend = backend_registry[decision.backend_str]
            _log.info("AI routing → backend registry: %s", decision.backend_str)
            return backend.run(circuit, **merged)

        # Special-case: TensorNetworkBackend for large circuits
        if decision.backend_str == "tensornetwork" and decision.params.get("max_bond_dim"):
            try:
                import sys, os
                sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
                from zuntenium.backends.tensor_network import TensorNetworkBackend
                tn = TensorNetworkBackend(max_bond_dim=int(decision.params.get("max_bond_dim", 64)))
                return tn.run(circuit, shots=merged.get("shots", 1024))
            except Exception as exc:
                _log.warning("TN backend failed (%s), falling to base", exc)

        # Default: base executor
        return super().run(circuit, **merged)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """Graceful shutdown with model persistence."""
        if self._learning_loop:
            _log.info("AIAwareHybridExecutor: flushing learning loop…")
            self._learning_loop.flush()
            self._learning_loop.stop()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.shutdown()

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    def get_decision_history(self, n: int = 20) -> list[dict]:
        """Return last N execution records from the logger."""
        if self._ai_optimizer is None:
            return []
        logger = self._ai_optimizer._logger
        with logger._conn() as conn:
            rows = conn.execute(
                """
                SELECT datetime_utc, decision_backend, decision_strategy,
                       decision_confidence, execution_time_s, success, reward
                FROM executions ORDER BY timestamp DESC LIMIT ?
                """,
                (n,),
            ).fetchall()
        return [dict(r) for r in rows]

    def force_retrain(self) -> dict:
        """Force immediate supervised retrain (blocking)."""
        if self._ai_optimizer:
            return self._ai_optimizer.retrain_supervised(min_samples=1)
        return {}
