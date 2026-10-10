# zuntenium/ai_engine/optimizer.py
"""
AIOptimizer — The core intelligence of the Zuntenium AI Optimization Engine.

Sits directly above BackendSelector and is called first by HybridExecutor
when optimization_mode="ai".

Decision Pipeline
-----------------
  Circuit + context
      │
      ▼
  FeatureExtractor.extract()
      │
      ▼
  ┌──────────────────────────────────────────┐
  │        AIOptimizer.select()              │
  │  1. SupervisedModel.predict()  (Phase 1) │
  │  2. PPOAgent.predict()         (Phase 2) │
  │  3. Confidence gate → fallback if <0.7   │
  │  4. Build AIDecision + explanation        │
  └──────────────────────────────────────────┘
      │
      ▼
  AIDecision → HybridExecutor

Safety Contract
---------------
  - confidence < CONFIDENCE_THRESHOLD → rule-based fallback
  - any exception at any stage → emergency fallback
  - always returns a valid, executable AIDecision
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from .decision import (
    AIDecision, BackendChoice, TranspilationStrategy, DecisionSource,
    make_emergency_decision,
)
from .features import FeatureVector, FeatureExtractor
from .logger import ExecutionLogger
from .models import (
    SupervisedModel, PPOAgent, ReplayBuffer,
    BACKEND_LABELS, STRATEGY_LABELS,
    encode_label, decode_label,
)

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONFIDENCE_THRESHOLD = 0.70   # below this → rule-based fallback
MAX_INFERENCE_MS     = 10.0   # warn if inference exceeds this
_DEFAULT_BOND_DIM    = 64     # default MPS bond dimension


# ---------------------------------------------------------------------------
# AIOptimizer
# ---------------------------------------------------------------------------

class AIOptimizer:
    """
    The production-ready AI decision engine for the Zuntenium platform.

    Parameters
    ----------
    rule_selector : BackendSelector (or duck-typed equivalent)
        The existing rule-based selector — used as trusted fallback.
    logger : ExecutionLogger
        Persistent execution history store.
    feature_extractor : FeatureExtractor | None
        If None, a fresh FeatureExtractor is constructed (with logger injected).
    model_path : str | Path | None
        Directory for model checkpoints.  None → in-memory only (no persistence).
    use_rl : bool
        Whether to also run the PPO agent (Phase 2).  Requires PyTorch.
    confidence_threshold : float
        Confidence gate below which rule-based fallback is triggered.
    """

    def __init__(
        self,
        rule_selector=None,
        logger: ExecutionLogger | None = None,
        feature_extractor: FeatureExtractor | None = None,
        model_path: str | Path | None = None,
        use_rl: bool = False,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
    ) -> None:
        self._rule_selector        = rule_selector
        self._logger               = logger or ExecutionLogger()
        self._model_path           = Path(model_path) if model_path else None
        self._use_rl               = use_rl
        self._confidence_threshold = confidence_threshold
        self._n_executions         = 0

        # Feature extractor
        self._extractor = feature_extractor or FeatureExtractor(
            execution_logger=self._logger
        )

        # Models
        state_dim = FeatureVector().DIM
        self._supervised = self._load_or_create_supervised(state_dim)
        self._rl_agent   = self._load_or_create_rl(state_dim) if use_rl else None
        self._replay     = ReplayBuffer(capacity=10_000, state_dim=state_dim)

        _log.info(
            "AIOptimizer ready | RL=%s | threshold=%.2f | model_path=%s",
            use_rl, confidence_threshold, model_path,
        )

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def select(
        self,
        circuit,
        preferences: dict | None = None,
    ) -> AIDecision:
        """
        Produce the optimal AIDecision for executing ``circuit``.

        Parameters
        ----------
        circuit     : zuntenium.quantum_core.Circuit
        preferences : optional runtime hints from caller
                      (hardware_available, shots, noise_model_requested, …)

        Returns
        -------
        AIDecision — always valid, never raises.
        """
        t0 = time.perf_counter()
        try:
            decision = self._select_inner(circuit, preferences or {})
        except Exception as exc:
            _log.exception("AIOptimizer.select() failed: %s — using emergency fallback", exc)
            decision = make_emergency_decision()

        decision.decision_latency_ms = (time.perf_counter() - t0) * 1000
        if decision.decision_latency_ms > MAX_INFERENCE_MS:
            _log.warning(
                "Inference took %.1f ms (threshold %s ms)", decision.decision_latency_ms, MAX_INFERENCE_MS
            )
        _log.info("Decision: %s", decision)
        return decision

    # ------------------------------------------------------------------
    # Online learning hooks (called by OnlineLearningLoop / HybridExecutor)
    # ------------------------------------------------------------------

    def record_outcome(
        self,
        circuit,
        decision: AIDecision,
        *,
        execution_time_s: float,
        fidelity: float = 1.0,
        success: bool = True,
        error_message: str | None = None,
    ) -> None:
        """
        Log execution result and push experience to replay buffer.
        Called by HybridExecutor after each run.
        """
        try:
            features = self._extractor.extract(circuit)
            row_id = self._logger.log_execution(
                circuit_name=getattr(circuit, "name", "unknown"),
                num_qubits=int(getattr(circuit, "num_qubits", 0)),
                features=features,
                decision=decision,
                execution_time_s=execution_time_s,
                fidelity=fidelity,
                success=success,
                error_message=error_message,
            )
            self._n_executions += 1

            # Push to RL replay buffer
            if self._rl_agent is not None:
                action_idx = encode_label(
                    decision.backend.value,
                    decision.transpilation_strategy.value,
                )
                reward = ExecutionLogger._compute_reward(
                    execution_time_s=execution_time_s,
                    fidelity=fidelity,
                    success=success,
                    fallback_used=decision.fallback_used,
                )
                self._replay.push(
                    state=features.to_array().astype(np.float32),
                    action=action_idx,
                    reward=reward,
                    next_state=features.to_array().astype(np.float32),
                    done=False,
                )
            _log.debug("Recorded outcome row_id=%s", row_id)

        except Exception as exc:
            _log.warning("record_outcome failed (non-fatal): %s", exc)

    def retrain_supervised(self, min_samples: int = 50) -> dict:
        """
        Retrain the supervised model from execution history.
        FIX BUG-002/003: use full joint (backend|strategy) label, min_samples 20->50.
        """
        data = self._logger.get_training_data(limit=5_000)
        if len(data) < min_samples:
            _log.info("Skipping retrain: only %d samples (need %d)", len(data), min_samples)
            return {"skipped": True, "n_samples": len(data)}

        X = np.array([d["features_array"] for d in data], dtype=np.float32)
        y = np.array([
            encode_label(d["label"], d.get("strategy", "rule_only"))
            for d in data
        ], dtype=np.int64)

        metrics = self._supervised.train(X, y)
        if self._model_path:
            self._supervised.save(self._model_path / "supervised")
        _log.info("Retrained supervised model: %s", metrics)
        return metrics

    def update_rl(self) -> dict:
        """Run one PPO update step from the replay buffer."""
        if self._rl_agent is None:
            return {}
        metrics = self._rl_agent.update(self._replay)
        if self._model_path and metrics:
            self._rl_agent.save(self._model_path / "ppo")
        return metrics

    # ------------------------------------------------------------------
    # Internal decision logic
    # ------------------------------------------------------------------

    def _select_inner(self, circuit, preferences: dict) -> AIDecision:
        # 1. Extract features
        features = self._extractor.extract(circuit, **self._parse_prefs(preferences))
        arr      = features.to_array()

        # 2. Supervised model prediction
        sup_label, sup_conf = self._supervised.predict(arr)

        # 3. RL agent prediction (if enabled and trained)
        rl_label, rl_conf = None, 0.0
        if self._rl_agent is not None:
            rl_label, rl_conf = self._rl_agent.predict(arr)

        # 4. Ensemble: prefer RL if it has higher confidence, else supervised
        if rl_label and rl_conf > sup_conf:
            chosen_label  = rl_label
            chosen_conf   = rl_conf
            chosen_source = DecisionSource.AI_RL
        else:
            chosen_label  = sup_label
            chosen_conf   = sup_conf
            chosen_source = DecisionSource.AI_SUPERVISED

        # 5. Confidence gate → rule-based fallback
        if chosen_conf < self._confidence_threshold or not self._supervised._trained:
            return self._rule_fallback(circuit, features, preferences, chosen_conf)

        # 6. Parse label → backend + strategy
        # FIX BUG-001: removed dead decode_label call that always passed 0 due to `if False`
        parts        = chosen_label.split("|")
        backend_str  = parts[0] if parts else "statevector"
        strategy_str = parts[1] if len(parts) > 1 else "rule_only"

        backend  = self._to_backend_choice(backend_str)
        strategy = self._to_strategy(strategy_str, features)
        params   = self._build_params(backend, features, preferences)

        # 7. Safety override: statevector not feasible for large circuits
        backend, strategy, params = self._apply_safety_overrides(
            backend, strategy, params, features
        )

        # 8. Feature attribution (lightweight: top-5 by magnitude for RF)
        attribution = self._compute_attribution(arr, features)

        # 9. Build explanation
        reason = self._build_reason(
            backend, strategy, features, chosen_conf, attribution
        )

        return AIDecision(
            backend=backend,
            transpilation_strategy=strategy,
            params=params,
            confidence=chosen_conf,
            reason=reason,
            fallback_used=False,
            source=chosen_source,
            feature_attribution=attribution,
        )

    # ------------------------------------------------------------------
    # Rule-based fallback
    # ------------------------------------------------------------------

    def _rule_fallback(
        self, circuit, features: FeatureVector, preferences: dict, ai_conf: float
    ) -> AIDecision:
        """Delegate to the existing BackendSelector."""
        try:
            rule_decision = self._invoke_rule_selector(circuit, preferences)
            reason = (
                f"AI confidence {ai_conf:.2f} < threshold {self._confidence_threshold:.2f} "
                f"— rule-based selector chose {rule_decision['backend']}. "
                f"(model not yet trained)" if not self._supervised._trained
                else f"AI confidence {ai_conf:.2f} below threshold."
            )
            return AIDecision(
                backend=self._to_backend_choice(rule_decision.get("backend", "statevector")),
                transpilation_strategy=self._to_strategy(
                    rule_decision.get("strategy", "rule_only"), features
                ),
                params=rule_decision.get("params", {}),
                confidence=ai_conf,
                reason=reason,
                fallback_used=True,
                source=DecisionSource.RULE_FALLBACK,
            )
        except Exception as exc:
            _log.warning("Rule fallback also failed (%s) — emergency default", exc)
            return make_emergency_decision()

    def _invoke_rule_selector(self, circuit, preferences: dict) -> dict:
        """Call BackendSelector in a duck-typed way."""
        if self._rule_selector is None:
            # Minimal built-in rule
            n = int(getattr(circuit, "num_qubits", 0))
            if n <= 20:
                return {"backend": "statevector", "strategy": "rule_only", "params": {}}
            elif n <= 40:
                return {"backend": "tensornetwork", "strategy": "rule_only",
                        "params": {"max_bond_dim": 32}}
            else:
                return {"backend": "ibm", "strategy": "hardware_aware", "params": {}}

        # BackendSelector from zuntenium.adaptive
        try:
            sel = self._rule_selector
            # BackendSelector.select() returns a SelectionDecision namedtuple/dataclass
            result = sel.select(circuit)
            if hasattr(result, "backend"):
                return {
                    "backend":  str(result.backend).lower(),
                    "strategy": "rule_only",
                    "params":   getattr(result, "params", {}),
                }
        except Exception:
            pass
        return {"backend": "statevector", "strategy": "rule_only", "params": {}}

    # ------------------------------------------------------------------
    # Safety overrides
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_safety_overrides(
        backend: BackendChoice,
        strategy: TranspilationStrategy,
        params: dict,
        features: FeatureVector,
    ) -> tuple[BackendChoice, TranspilationStrategy, dict]:
        n = features.num_qubits

        # Statevector infeasible for > 28 qubits
        if backend == BackendChoice.STATEVECTOR and n > 28:
            _log.info("Safety override: statevector → tensornetwork (n=%d > 28)", int(n))
            backend = BackendChoice.TENSORNETWORK
            params.setdefault("max_bond_dim", _DEFAULT_BOND_DIM)

        # Stabilizer only valid for all-Clifford circuits
        if backend == BackendChoice.STABILIZER and features.clifford_fraction < 0.95:
            _log.info("Safety override: stabilizer → statevector (non-Clifford fraction %.2f)",
                      1 - features.clifford_fraction)
            backend = BackendChoice.STATEVECTOR

        # IBM backend requires hardware strategy
        if backend == BackendChoice.IBM and strategy == TranspilationStrategy.NONE:
            strategy = TranspilationStrategy.HARDWARE

        return backend, strategy, params

    # ------------------------------------------------------------------
    # Parameter builder
    # ------------------------------------------------------------------

    @staticmethod
    def _build_params(
        backend: BackendChoice,
        features: FeatureVector,
        preferences: dict,
    ) -> dict:
        params: dict[str, Any] = {}
        shots = int(10 ** features.shots_requested) if features.shots_requested > 0 else 1024
        params["shots"] = preferences.get("shots", shots)

        if backend == BackendChoice.TENSORNETWORK:
            # Bond dimension heuristic: scale with entanglement
            chi = int(np.clip(32 * (1 + 2 * features.entanglement_score), 16, 256))
            params["max_bond_dim"]   = preferences.get("max_bond_dim", chi)
            params["contraction_method"] = "auto"

        elif backend == BackendChoice.STATEVECTOR:
            params["precision"] = "complex128"

        elif backend == BackendChoice.IBM:
            params["optimization_level"] = 3 if features.t_gate_count > 0 else 2
            params["resilience_level"]   = 1
            if features.noise_model_requested:
                params["noise_model"] = "ibm_default"

        elif backend == BackendChoice.STABILIZER:
            params["method"] = "clifford"

        return params

    # ------------------------------------------------------------------
    # Explainability
    # ------------------------------------------------------------------

    def _compute_attribution(
        self, arr: np.ndarray, features: FeatureVector
    ) -> dict[str, float]:
        """
        Lightweight feature attribution: uses RF feature importances if available,
        otherwise returns top-5 by |value| scaled by index.
        """
        try:
            model = self._supervised._model
            if hasattr(model, "feature_importances_"):
                imps = model.feature_importances_
                idx  = np.argsort(imps)[::-1][:5]
                return {features.feature_names[i]: float(imps[i]) for i in idx}
        except Exception:
            pass
        # Fallback: return magnitude-scaled features
        abs_vals = np.abs(arr)
        top5     = np.argsort(abs_vals)[::-1][:5]
        total    = abs_vals.sum() + 1e-10
        return {
            features.feature_names[i]: float(abs_vals[i] / total)
            for i in top5
            if i < len(features.feature_names)
        }

    @staticmethod
    def _build_reason(
        backend: BackendChoice,
        strategy: TranspilationStrategy,
        features: FeatureVector,
        confidence: float,
        attribution: dict[str, float],
    ) -> str:
        """Generate a human-readable explanation of the decision."""
        n      = int(features.num_qubits)
        ent    = features.entanglement_score
        cliff  = features.clifford_fraction
        hw     = bool(features.hardware_available)
        queue  = features.queue_time_estimate_s

        lines = []

        # Backend rationale
        if backend == BackendChoice.STATEVECTOR:
            lines.append(f"Statevector simulation chosen: {n} qubits within feasible range (≤28).")
        elif backend == BackendChoice.TENSORNETWORK:
            chi = 32 * int(1 + 2 * ent)
            lines.append(
                f"TensorNetwork/MPS chosen: {n} qubits, entanglement={ent:.2f} "
                f"→ bond dim χ={chi}."
            )
        elif backend == BackendChoice.STABILIZER:
            lines.append(f"Stabilizer (Clifford) chosen: {cliff*100:.0f}% Clifford gates, low overhead.")
        elif backend == BackendChoice.IBM:
            if hw:
                lines.append(f"IBM hardware chosen: hardware available, queue={queue:.0f}s.")
            else:
                lines.append("IBM backend chosen despite hardware unavailability — check config.")

        # Strategy rationale
        strat_map = {
            TranspilationStrategy.NONE:          "No transpilation (pre-optimised circuit).",
            TranspilationStrategy.RULE_ONLY:     "Rule-based transpilation passes.",
            TranspilationStrategy.RL_REFINE:     "Rule-based + PPO RL refinement for gate reduction.",
            TranspilationStrategy.HARDWARE:      "Full hardware-aware stack (mapping + SWAP + decompose).",
        }
        lines.append(f"Strategy: {strat_map.get(strategy, strategy.value)}")

        # Top attribution
        if attribution:
            top = sorted(attribution.items(), key=lambda x: -x[1])[:2]
            top_str = ", ".join(f"{k}={v:.3f}" for k, v in top)
            lines.append(f"Key drivers: {top_str}.")

        lines.append(f"Model confidence: {confidence:.1%}.")
        return " | ".join(lines)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_prefs(prefs: dict) -> dict:
        """Map preferences dict to FeatureExtractor.extract() kwargs."""
        return {
            "hardware_available":      bool(prefs.get("hardware_available", False)),
            "queue_time_estimate_s":   float(prefs.get("queue_time_estimate_s", 0.0)),
            "current_system_load":     float(prefs.get("current_system_load", 0.0)),
            "noise_model_requested":   bool(prefs.get("noise_model_requested", False)),
            "shots_requested":         int(prefs.get("shots", 1024)),
            "real_hardware_requested": bool(prefs.get("real_hardware", False)),
        }

    @staticmethod
    def _to_backend_choice(s: str) -> BackendChoice:
        mapping = {
            "statevector":   BackendChoice.STATEVECTOR,
            "tensornetwork": BackendChoice.TENSORNETWORK,
            "mps":           BackendChoice.TENSORNETWORK,
            "stabilizer":    BackendChoice.STABILIZER,
            "clifford":      BackendChoice.STABILIZER,
            "ibm":           BackendChoice.IBM,
        }
        return mapping.get(s.lower(), BackendChoice.STATEVECTOR)

    @staticmethod
    def _to_strategy(s: str, features: FeatureVector | None = None) -> TranspilationStrategy:
        mapping = {
            "none":           TranspilationStrategy.NONE,
            "rule_only":      TranspilationStrategy.RULE_ONLY,
            "rl_refine":      TranspilationStrategy.RL_REFINE,
            "hardware_aware": TranspilationStrategy.HARDWARE,
        }
        return mapping.get(s.lower(), TranspilationStrategy.RULE_ONLY)

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_or_create_supervised(self, state_dim: int) -> SupervisedModel:
        if self._model_path:
            meta = self._model_path / "supervised.meta.json"
            if meta.exists():
                try:
                    m = SupervisedModel.load(self._model_path / "supervised")
                    _log.info("Loaded supervised model from %s", self._model_path)
                    return m
                except Exception as exc:
                    _log.warning("Failed to load supervised model (%s), creating fresh.", exc)
        return SupervisedModel(state_dim=state_dim)

    def _load_or_create_rl(self, state_dim: int) -> PPOAgent:
        if self._model_path:
            pt = self._model_path / "ppo.ppo.pt"
            if pt.exists():
                try:
                    agent = PPOAgent.load(self._model_path / "ppo", state_dim=state_dim)
                    _log.info("Loaded PPO agent from %s", self._model_path)
                    return agent
                except Exception as exc:
                    _log.warning("Failed to load PPO agent (%s), creating fresh.", exc)
        return PPOAgent(state_dim=state_dim)
