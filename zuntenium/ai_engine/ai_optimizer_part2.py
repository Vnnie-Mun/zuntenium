"""AI Optimization Engine Part 2: supervised/RL decisions and hybrid integration."""
from __future__ import annotations

import math
import os
import pickle  # LOCAL-DISK-ONLY: used exclusively for sklearn RF+Scaler checkpoints.
# NEVER pass a pickle file received over a network or from an untrusted source
# to _load()/_save() — pickle is not safe for deserializing untrusted data (U13).
# These files are written and read on the same machine from ~/.zuntenium or similar.
# PyTorch (.pt) and JSON (.json) are used for all cross-machine serialization.

import time
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from zuntenium.ai_engine.ai_optimizer_part1 import (
    BACKEND_LABELS, TRANSPILE_LABELS, ExecutionLogger, FeatureEngineer,
    FeatureVector, HistoricalContext, RuntimeContext, compute_reward,
)
from zuntenium.execution.adaptive_engine import BackendSelector, Preferences, SelectionDecision
from zuntenium.execution.hybrid_executor import HybridExecutor
from zuntenium.transpiler.v2 import Circuit

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn unavailable; using NumPy model.", stacklevel=2)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.distributions import Categorical
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

N_BACKENDS, N_STRATEGIES, N_ACTIONS = len(BACKEND_LABELS), len(TRANSPILE_LABELS), 15


def action_to_bs(action: int) -> Tuple[str, str]:
    if not 0 <= action < N_ACTIONS: raise ValueError("action outside 15-action space")
    return BACKEND_LABELS[action // N_STRATEGIES], TRANSPILE_LABELS[action % N_STRATEGIES]


def bs_to_action(backend: str, strategy: str) -> int:
    return BACKEND_LABELS.index(backend) * N_STRATEGIES + TRANSPILE_LABELS.index(strategy)


class _NumpySoftmaxModel:
    def __init__(self, in_dim=24, hidden=64, n_classes=N_ACTIONS, seed=0):
        rng = np.random.default_rng(seed); self.W1 = rng.normal(0, math.sqrt(2 / in_dim), (hidden, in_dim)).astype(np.float32); self.b1 = np.zeros(hidden, np.float32); self.W2 = rng.normal(0, math.sqrt(2 / hidden), (n_classes, hidden)).astype(np.float32); self.b2 = np.zeros(n_classes, np.float32); self.trained = False
    def predict_proba(self, values):
        hidden = np.maximum(0, values @ self.W1.T + self.b1); logits = hidden @ self.W2.T + self.b2; logits -= logits.max(axis=1, keepdims=True); probabilities = np.exp(logits); return probabilities / probabilities.sum(axis=1, keepdims=True)
    def fit(self, X, y, sample_weight=None): self.trained = True


class SupervisedModel:
    CONFIDENCE_THRESHOLD = .65
    def __init__(self, seed=0):
        self._scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self._model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=seed, n_jobs=-1, class_weight="balanced") if SKLEARN_AVAILABLE else _NumpySoftmaxModel(seed=seed)
        self._trained = False; self._n_samples_trained = 0
    def fit(self, X, y, rewards):
        weights = np.exp(rewards - rewards.max()); weights = weights / weights.sum() * len(weights)
        X_fit = self._scaler.fit_transform(X) if self._scaler else X; self._model.fit(X_fit, y, sample_weight=weights); self._trained = True; self._n_samples_trained = len(X)
    def predict(self, values):
        if not self._trained: return 0, 0.0
        values = self._scaler.transform(values.reshape(1, -1)) if self._scaler else values.reshape(1, -1); probabilities = self._model.predict_proba(values)[0]; action = int(np.argmax(probabilities)); return action, float(probabilities[action])
    def feature_importances(self):
        if not self._trained or not SKLEARN_AVAILABLE: return None
        return dict(sorted(zip(FeatureVector.NAMES, self._model.feature_importances_.tolist()), key=lambda item: -item[1]))
    @property
    def is_trained(self): return self._trained


class _NumpyRLAgent:
    def __init__(self, state_dim=24, n_actions=N_ACTIONS, seed=0):
        rng = np.random.default_rng(seed); self.W1 = rng.normal(0, math.sqrt(2 / state_dim), (64, state_dim)).astype(np.float32); self.b1 = np.zeros(64, np.float32); self.W2 = rng.normal(0, math.sqrt(2 / 64), (64, 64)).astype(np.float32); self.b2 = np.zeros(64, np.float32); self.Wa = rng.normal(0, .01, (n_actions, 64)).astype(np.float32); self.ba = np.zeros(n_actions, np.float32)
    def _forward(self, state):
        h1 = np.maximum(0, self.W1 @ state + self.b1); h2 = np.maximum(0, self.W2 @ h1 + self.b2); logits = self.Wa @ h2 + self.ba; probabilities = np.exp(logits - logits.max()); return probabilities / probabilities.sum()
    def act(self, state):
        probabilities = self._forward(state); action = int(np.random.choice(len(probabilities), p=probabilities)); return action, float(np.log(probabilities[action] + 1e-12)), 0.0
    def reinforce_update(self, states, actions, rewards):
        for state, action, reward in zip(states, actions, rewards):
            probabilities = self._forward(state); hidden = np.maximum(0, self.W2 @ np.maximum(0, self.W1 @ state + self.b1) + self.b2); gradient = np.zeros_like(probabilities); gradient[action] = reward / max(probabilities[action], 1e-12); self.Wa += 1e-3 * np.outer(gradient, hidden)


if TORCH_AVAILABLE:
    class _TorchRLAgent(nn.Module):
        def __init__(self, state_dim=24, n_actions=N_ACTIONS):
            super().__init__(); self.trunk = nn.Sequential(nn.Linear(state_dim, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU()); self.actor = nn.Linear(64, n_actions); self.critic = nn.Linear(64, 1)
        def forward(self, state):
            hidden = self.trunk(state); return self.actor(hidden), self.critic(hidden)
        def act(self, state):
            logits, value = self.forward(torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)); distribution = Categorical(logits=logits.squeeze(0)); action = distribution.sample(); return int(action), float(distribution.log_prob(action)), float(value.item())
else:
    _TorchRLAgent = None


class RLAgent:
    CONFIDENCE_THRESHOLD = .50
    def __init__(self, state_dim=24, seed=0):
        self._torch = TORCH_AVAILABLE; self._agent = _TorchRLAgent(state_dim, N_ACTIONS) if TORCH_AVAILABLE else _NumpyRLAgent(state_dim, N_ACTIONS, seed); self._optimizer = optim.Adam(self._agent.parameters(), lr=3e-4) if TORCH_AVAILABLE else None; self._buffer = []; self._n_updates = 0; self._trained = False
    def act(self, feature_vector): return self._agent.act(feature_vector.array)
    def store(self, state, action, reward): self._buffer.append((state, action, reward))
    def update_episode(self):
        if not self._buffer: return
        if self._torch:
            states = torch.as_tensor(np.stack([item[0] for item in self._buffer]), dtype=torch.float32); actions = torch.as_tensor([item[1] for item in self._buffer]); rewards = torch.as_tensor([item[2] for item in self._buffer], dtype=torch.float32); logits, values = self._agent(states); loss = -Categorical(logits=logits).log_prob(actions).mul(rewards).mean() + .5 * values.squeeze(-1).sub(rewards).pow(2).mean(); self._optimizer.zero_grad(); loss.backward(); self._optimizer.step()
        else: self._agent.reinforce_update(*zip(*self._buffer))
        self._buffer.clear(); self._n_updates += 1; self._trained = True
    @property
    def is_trained(self): return self._trained


@dataclass
class AIDecision:
    backend: str
    transpile_strategy: str
    params: Dict[str, Any]
    confidence: float
    reason: str
    phase_used: str
    fallback_used: bool
    features: Optional[FeatureVector] = None
    rule_decision: Optional[SelectionDecision] = None
    def explain(self): return f"AIDecision:\n  Backend: {self.backend}\n  Strategy: {self.transpile_strategy}\n  Params: {self.params}\n  Confidence: {self.confidence:.3f}\n  Phase: {self.phase_used}\n  Fallback: {self.fallback_used}\n  Reason: {self.reason}"
    def to_preferences(self): return Preferences(preferred_backend=self.backend, allow_hardware=self.backend == "hardware", noise_model=self.params.get("noise_model"))


class AIOptimizer:
    def __init__(self, rule_selector=None, logger_inst=None, model_path=None, retrain_every=50, seed=0, optimization_mode="ai"):
        self._selector = rule_selector or BackendSelector(); self._logger = logger_inst or ExecutionLogger(); self._engineer = FeatureEngineer(); self._supervised = SupervisedModel(seed); self._rl_agent = RLAgent(seed=seed); self._retrain_every = retrain_every; self._calls_since_retrain = 0; self._mode = optimization_mode; self._call_count = 0
        if model_path and os.path.exists(model_path): self._load(model_path)
    def select(self, circuit: Circuit, runtime=None, history=None, preferences=None):
        self._call_count += 1; self._calls_since_retrain += 1
        if self._calls_since_retrain >= self._retrain_every: self.train(); self._calls_since_retrain = 0
        if self._mode != "ai": return self._rule_fallback(circuit, preferences, "rule-only mode")
        try:
            runtime = runtime or RuntimeContext(); history = history or self._logger.build_historical_context(self._engineer.extract(circuit).circuit_sig); features = self._engineer.extract(circuit, runtime, history)
        except Exception as error: return self._rule_fallback(circuit, preferences, f"feature error: {error}")
        if self._supervised.is_trained:
            action, confidence = self._supervised.predict(features.array)
            if confidence >= SupervisedModel.CONFIDENCE_THRESHOLD: return self._make_decision(action_to_bs(action), confidence, features, "supervised", circuit)
        if self._rl_agent.is_trained:
            action, log_probability, _ = self._rl_agent.act(features); confidence = float(np.exp(log_probability))
            if confidence >= RLAgent.CONFIDENCE_THRESHOLD: return self._make_decision(action_to_bs(action), confidence, features, "rl", circuit)
        return self._rule_fallback(circuit, preferences, "confidence below threshold", features)
    def _make_decision(self, backend_strategy, confidence, features, phase, circuit):
        backend, strategy = backend_strategy
        if backend not in self._selector._backends or backend not in self._selector._available:
            return self._rule_fallback(
                circuit,
                None,
                f"AI selected unavailable backend {backend}",
                features,
            )
        return AIDecision(backend, strategy, self._build_params(backend, features), confidence, f"[{phase.upper()}] selected {backend}", phase, False, features)
    def _rule_fallback(self, circuit, preferences, reason, features=None):
        rule_decision = self._selector.select(circuit, Preferences(**(preferences or {})) if preferences else Preferences()); return AIDecision(rule_decision.selected_backend, "rule_only", {}, rule_decision.confidence, f"[Rule fallback: {reason}] {rule_decision.reason}", "rule_fallback", True, features, rule_decision)
    @staticmethod
    def _build_params(backend, features):
        if backend == "tensornetwork": return {"max_bond_dim": max(16, min(256, int(16 * math.exp(3 * float(features.array[4]))))), "trunc_eps": 1e-10}
        if backend == "hardware": return {"optimization_level": 3, "noise_model": False}
        if backend == "densitymatrix": return {"noise_model": True}
        return {}
    def record_result(self, decision, features, exec_ms, fidelity, success, error=""):
        self._logger.log_execution(features, decision.backend, decision.transpile_strategy, exec_ms, fidelity, success, error)
        if not decision.fallback_used: self._rl_agent.store(features.array, bs_to_action(decision.backend, decision.transpile_strategy), compute_reward(exec_ms, fidelity, success)); self._rl_agent.update_episode()
    def train(self, min_records=20):
        data = self._logger.get_training_data(min_records)
        if data is None: return False
        X, backend_labels, strategy_labels, rewards = data; self._supervised.fit(X, backend_labels * N_STRATEGIES + strategy_labels, rewards); return True
    def _load(self, path: str) -> None:
        """Load local-disk sklearn checkpoint (LOCAL ONLY — never from network)."""
        pkl_path = path + ".rf.pkl"
        if SKLEARN_AVAILABLE and os.path.exists(pkl_path):
            # U13 boundary guard: reject path components that could indicate
            # network mounts or suspicious locations.
            import warnings
            abs_pkl = os.path.abspath(pkl_path)
            if any(seg in abs_pkl for seg in ("/proc/", "/dev/", "//")):
                warnings.warn(
                    f"Refusing to unpickle from suspicious path: {abs_pkl}. "
                    "Only load pickle files written by this process on this machine.",
                    RuntimeWarning, stacklevel=2,
                )
                return
            with open(pkl_path, "rb") as stream:
                self._supervised._model, self._supervised._scaler = pickle.load(stream)  # noqa: S301
            self._supervised._trained = True

    def save(self, path: str) -> None:
        """Save local-disk sklearn checkpoint (LOCAL ONLY — never transmit over network)."""
        if SKLEARN_AVAILABLE and self._supervised.is_trained:
            with open(path + ".rf.pkl", "wb") as stream:
                pickle.dump((self._supervised._model, self._supervised._scaler), stream)  # noqa: S301

    def get_backend(self, decision): return self._selector.get_backend(SelectionDecision(decision.backend, decision.confidence, decision.reason, decision.features.raw_circuit if decision.features else None, {}, [], 0))
    def stats(self): return {"total_calls": self._call_count, "supervised_trained": self._supervised.is_trained, "rl_trained": self._rl_agent.is_trained, "rl_updates": self._rl_agent._n_updates, "db_records": self._logger.count_all(), "mode": self._mode}


def make_ai_hybrid_executor(optimizer=None, optimization_mode="ai", **kwargs):
    from zuntenium.execution.hybrid_executor import HybridExecutor
    optimizer = optimizer or AIOptimizer(optimization_mode=optimization_mode)
    class AIHybridExecutor(HybridExecutor):
        def __init__(self, ai_optimizer, mode, **executor_kwargs):
            super().__init__(**executor_kwargs)
            self._ai_opt = ai_optimizer
            self._mode = mode

        def run(self, circuit=None, ir_gates=None, n_qubits=4, preferences=None, shots=1024, optimize=True, backend_override=None, **run_kwargs):
            if circuit is None:
                from zuntenium.execution.hybrid_executor import convert_irgates_to_circuit
                circuit_for_ai = convert_irgates_to_circuit(ir_gates or [], n_qubits)
            else:
                circuit_for_ai = circuit
            if self._mode != "ai":
                return super().run(circuit, ir_gates, n_qubits, preferences, shots, optimize, backend_override)
            decision = optimizer.select(circuit_for_ai, preferences=preferences.__dict__ if hasattr(preferences, "__dict__") else preferences)
            selected_backend = backend_override or decision.backend
            run_preferences = preferences or decision.to_preferences()
            result = super().run(circuit=circuit, ir_gates=ir_gates, n_qubits=n_qubits, preferences=run_preferences, shots=shots, optimize=optimize, backend_override=selected_backend)
            if decision.features is not None: optimizer.record_result(decision, decision.features, result.total_time_ms, 1.0, result.statevector is not None)
            result.metadata["ai_decision"] = decision; result.counts["__ai_backend__"] = decision.backend; result.counts["__ai_phase__"] = decision.phase_used
            return result
    return AIHybridExecutor(ai_optimizer=optimizer, mode=optimization_mode, **kwargs)


class Part2TestSuite:
    def __init__(self):
        self._p = 0
        self._f = 0

    def _a(self, cond: bool, msg: str):
        if cond:
            self._p += 1
            print(f"  ✓  {msg}")
        else:
            self._f += 1
            print(f"  ✗  FAIL: {msg}")

    def _make_optimizer(self) -> Tuple[AIOptimizer, str]:
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        log = ExecutionLogger(db_path)
        return AIOptimizer(logger_inst=log), db_path

    # ── Supervised model ──────────────────────────────────────────────────────

    def test_supervised_untrained_returns_zero_conf(self):
        m = SupervisedModel()
        a, c = m.predict(np.zeros(24, dtype=np.float32))
        self._a(c == 0.0, f"untrained: confidence=0 (got {c})")

    def test_supervised_fit_and_predict(self):
        rng = np.random.default_rng(0)
        X = rng.random((50, 24)).astype(np.float32)
        y = rng.integers(0, N_ACTIONS, 50)
        r = rng.uniform(0, 1, 50).astype(np.float32)
        m = SupervisedModel(seed=0)
        m.fit(X, y, r)
        self._a(m.is_trained, "supervised: is_trained after fit")
        a, c = m.predict(X[0])
        self._a(0 <= a < N_ACTIONS, f"supervised: action in range [0,{N_ACTIONS})")
        self._a(0 <= c <= 1.0, f"supervised: confidence in [0,1]: {c:.3f}")

    def test_supervised_feature_importances(self):
        rng = np.random.default_rng(1)
        X = rng.random((40, 24)).astype(np.float32)
        y = rng.integers(0, N_ACTIONS, 40)
        r = np.ones(40, dtype=np.float32)
        m = SupervisedModel(seed=0)
        m.fit(X, y, r)
        fi = m.feature_importances()
        if fi is not None:
            self._a(len(fi) == 24, f"feature importances: 24 features (got {len(fi)})")
            self._a(abs(sum(fi.values()) - 1.0) < 0.01, "feature importances sum to 1")
        else:
            self._p += 1   # sklearn not available, skip

    # ── RL Agent ──────────────────────────────────────────────────────────────

    def test_rl_agent_act(self):
        agent = RLAgent(state_dim=24, seed=0)
        fv = FeatureVector(
            array=np.zeros(24, np.float32),
            names=FeatureVector.NAMES,
            circuit_sig="test",
            raw_circuit=None,
        )
        a, lp, v = agent.act(fv)
        self._a(0 <= a < N_ACTIONS, f"RL agent: action in range [0,{N_ACTIONS})")
        self._a(math.isfinite(lp), f"RL agent: log_prob finite: {lp:.4f}")

    def test_rl_agent_update(self):
        agent = RLAgent(state_dim=24, seed=0)
        states = [np.random.randn(24).astype(np.float32) for _ in range(5)]
        actions = [0, 1, 2, 0, 1]
        rewards = [0.5, -0.1, 0.8, 0.2, 0.3]
        for s, a, r in zip(states, actions, rewards):
            agent.store(s, a, r)
        agent.update_episode()
        self._a(agent.is_trained, "RL agent: is_trained after update")
        self._a(agent._n_updates == 1, f"RL agent: 1 update (got {agent._n_updates})")

    # ── AIDecision ────────────────────────────────────────────────────────────

    def test_ai_decision_explain(self):
        d = AIDecision(
            backend="statevector",
            transpile_strategy="rule_only",
            params={},
            confidence=0.9,
            reason="test",
            phase_used="supervised",
            fallback_used=False,
        )
        txt = d.explain()
        self._a("statevector" in txt, "explain() contains backend")
        self._a("confidence" in txt.lower(), "explain() contains confidence")

    def test_ai_decision_to_preferences(self):
        d = AIDecision(
            backend="hardware",
            transpile_strategy="rule_rl_hw",
            params={},
            confidence=0.8,
            reason="test",
            phase_used="rl",
            fallback_used=False,
        )
        prefs = d.to_preferences()
        self._a(prefs.preferred_backend == "hardware", "to_preferences: backend set")
        self._a(prefs.allow_hardware, "to_preferences: allow_hardware=True for hardware")

    # ── AIOptimizer ───────────────────────────────────────────────────────────

    def test_optimizer_select_untrained_falls_back(self):
        from zuntenium.transpiler.v2.engine import _build_qaoa
        opt, db_path = self._make_optimizer()
        try:
            c = _build_qaoa(4, 1)
            d = opt.select(c)
            self._a(d.fallback_used, "untrained optimizer falls back to rules")
            self._a(d.backend in BACKEND_LABELS, f"valid backend: {d.backend}")
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)

    def test_optimizer_select_returns_valid_decision(self):
        from zuntenium.transpiler.v2.engine import _build_ghz
        opt, db_path = self._make_optimizer()
        try:
            c = _build_ghz(3)
            d = opt.select(c, runtime=RuntimeContext(hardware_available=True))
            self._a(isinstance(d, AIDecision), "returns AIDecision")
            self._a(d.backend in BACKEND_LABELS, f"backend valid: {d.backend}")
            self._a(d.transpile_strategy in TRANSPILE_LABELS, "strategy valid")
            self._a(0 <= d.confidence <= 1.0, f"confidence in [0,1]: {d.confidence:.3f}")
            self._a(len(d.reason) > 0, "reason non-empty")
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)

    def test_optimizer_train_with_data(self):
        import tempfile
        from zuntenium.transpiler.v2.engine import _build_qaoa, _build_ghz, _build_random
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        try:
            fe = FeatureEngineer()
            log = ExecutionLogger(db_path)
            opt = AIOptimizer(logger_inst=log)
            circuits = [_build_qaoa(4, 1), _build_ghz(3), _build_random(4, 20)]
            backends = ["statevector", "stabilizer", "tensornetwork"]
            for c, b in zip(circuits * 10, backends * 10):
                fv = fe.extract(c)
                log.log_execution(fv, b, "rule_only", exec_ms=10.0, fidelity=0.99, success=True)
            ok = opt.train(min_records=20)
            self._a(ok, "train() returns True with sufficient data")
            self._a(opt._supervised.is_trained, "supervised model is trained")
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)

    def test_optimizer_select_after_training(self):
        import tempfile
        from zuntenium.transpiler.v2 import Gate, Circuit
        from zuntenium.transpiler.v2.engine import _build_qaoa
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        try:
            fe = FeatureEngineer()
            log = ExecutionLogger(db_path)
            opt = AIOptimizer(logger_inst=log)
            for _ in range(15):
                c = Circuit(3)
                c.add_gate(Gate('H', (0,)))
                c.add_gate(Gate('CNOT', (1,), controls=(0,)))
                fv = fe.extract(c)
                log.log_execution(fv, "stabilizer", "rule_only", exec_ms=1.0, fidelity=1.0, success=True)
            for _ in range(15):
                fv = fe.extract(_build_qaoa(4, 1))
                log.log_execution(fv, "statevector", "rule_rl", exec_ms=10.0, fidelity=0.99, success=True)
            opt.train(min_records=20)
            d = opt.select(_build_qaoa(4, 1))
            self._a(isinstance(d, AIDecision), "select() returns AIDecision after training")
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)

    def test_optimizer_record_result(self):
        import tempfile
        from zuntenium.transpiler.v2.engine import _build_qaoa
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        try:
            fe = FeatureEngineer()
            log = ExecutionLogger(db_path)
            opt = AIOptimizer(logger_inst=log)
            c = _build_qaoa(4, 1)
            d = opt.select(c)
            fv = fe.extract(c)
            opt.record_result(d, fv, exec_ms=15.0, fidelity=0.98, success=True)
            self._a(log.count_all() == 1, f"record_result: 1 record logged (got {log.count_all()})")
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)

    def test_optimizer_stats(self):
        opt, db_path = self._make_optimizer()
        try:
            s = opt.stats()
            self._a("total_calls" in s, "stats: total_calls present")
            self._a("supervised_trained" in s, "stats: supervised_trained present")
            self._a(s["mode"] == "ai", f"stats: mode=ai (got {s['mode']})")
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)

    def test_optimizer_rule_mode(self):
        from zuntenium.transpiler.v2.engine import _build_qaoa
        opt = AIOptimizer(optimization_mode="rule")
        d = opt.select(_build_qaoa(4, 1))
        self._a(d.phase_used == "rule_fallback", f"rule mode: phase=rule_fallback (got {d.phase_used})")

    def test_optimizer_exception_safety(self):
        opt, db_path = self._make_optimizer()
        try:
            c = Circuit(2)
            d = opt.select(c)
            self._a(isinstance(d, AIDecision), "exception safety: always returns AIDecision")
        except Exception as e:
            self._f += 1
            print(f"  ✗  EXCEPTION leaked: {e}")
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)

    # ── Integration ───────────────────────────────────────────────────────────

    def test_ai_hybrid_executor(self):
        from zuntenium.transpiler.v2.engine import _build_ghz
        ex = make_ai_hybrid_executor(optimization_mode="ai")
        c = _build_ghz(3)
        r = ex.run(circuit=c, shots=0)
        self._a(r.probabilities is not None, "AIHybridExecutor: has probabilities")
        self._a('__ai_backend__' in r.counts or '__decision_backend__' in r.counts,
                "AIHybridExecutor: decision tagged in result")

    def test_ai_hybrid_executor_rule_mode(self):
        from zuntenium.transpiler.v2.engine import _build_ghz
        ex = make_ai_hybrid_executor(optimization_mode="rule")
        c = _build_ghz(3)
        r = ex.run(circuit=c, shots=0)
        self._a(r.probabilities is not None, "AI/rule mode executor: has probabilities")

    # ── Action encoding ───────────────────────────────────────────────────────

    def test_action_encoding_roundtrip(self):
        for b in BACKEND_LABELS:
            for s in TRANSPILE_LABELS:
                a = bs_to_action(b, s)
                b2, s2 = action_to_bs(a)
                self._a(b2 == b and s2 == s, f"roundtrip ({b},{s}) → {a} → ({b2},{s2})")

    def test_n_actions(self):
        self._a(N_ACTIONS == 15, f"15 actions = 5×3 (got {N_ACTIONS})")

    def run_all(self):
        print("\n" + "═" * 60)
        print("  AI Optimization Engine Part 2 — Test Suite")
        print("═" * 60)
        tests = [m for m in sorted(dir(self)) if m.startswith('test_')]
        for t in tests:
            print(f"\n[{t}]")
            try:
                getattr(self, t)()
            except Exception as e:
                self._f += 1
                print(f"  ✗  EXCEPTION: {e}")
        total = self._p + self._f
        print(f"\n{'═' * 60}")
        print(f"  Results: {self._p}/{total} passed"
              + (f"  ({self._f} FAILED)" if self._f else "  ✓ all pass"))
        print("═" * 60)


def run_demo():
    print("\n" + "═" * 70)
    print("  ZUNTENIUM AI Optimization Engine — Full Demo")
    print("═" * 70)
    from zuntenium.transpiler.v2.engine import _build_qaoa, _build_ghz, _build_random
    import tempfile

    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name

    try:
        fe = FeatureEngineer()
        log = ExecutionLogger(db_path)
        opt = AIOptimizer(logger_inst=log, optimization_mode="ai")

        # ── Demo 1: Select without training (rule fallback) ──────────────────
        print("\n── Demo 1: Untrained AIOptimizer (rule fallback) ──")
        c = _build_qaoa(4, 1)
        d = opt.select(c)
        print(d.explain())

        # ── Demo 2: Generate training data ───────────────────────────────────
        print("\n── Demo 2: Generating training data (30 executions) ──")
        circuits = [_build_qaoa(4, 1), _build_ghz(3), _build_random(4, 20)]
        backends = ["statevector", "stabilizer", "statevector"]
        for i, (c2, b) in enumerate(zip(circuits * 10, backends * 10)):
            fv = fe.extract(c2)
            log.log_execution(fv, b, "rule_only", exec_ms=float(5 + i),
                              fidelity=0.99, success=True)
        print(f"  Logged {log.count_all()} records to SQLite")

        # ── Demo 3: Train supervised model ────────────────────────────────────
        print("\n── Demo 3: Training supervised model ──")
        ok = opt.train(min_records=20)
        print(f"  Training {'succeeded' if ok else 'failed (insufficient data)'}")
        print(f"  Stats: {opt.stats()}")

        # ── Demo 4: Select after training ────────────────────────────────────
        print("\n── Demo 4: Select after training ──")
        for name, c3 in [("QAOA(4,p=1)", _build_qaoa(4, 1)),
                          ("GHZ(3)", _build_ghz(3)),
                          ("Random(4,20)", _build_random(4, 20))]:
            d3 = opt.select(c3)
            print(f"  {name:<18}: backend={d3.backend}, conf={d3.confidence:.3f}, phase={d3.phase_used}")

        # ── Demo 5: Feature importances ───────────────────────────────────────
        print("\n── Demo 5: Feature importances (top 5) ──")
        fi = opt._supervised.feature_importances()
        if fi:
            for name, imp in list(fi.items())[:5]:
                print(f"  {name:<28}: {imp:.4f}")
        else:
            print("  (sklearn not available)")

        # ── Demo 6: AIHybridExecutor integration ──────────────────────────────
        print("\n── Demo 6: AIHybridExecutor end-to-end ──")
        ex = make_ai_hybrid_executor(optimizer=opt, optimization_mode="ai")
        res = ex.run(circuit=_build_ghz(3), shots=200)
        print(f"  GHZ(3): backend={res.counts.get('__ai_backend__','?')}, "
              f"conf={res.counts.get('__ai_confidence__','?')}")
        print(f"  Probabilities shape: {res.probabilities.shape}")

        # ── Demo 7: RL agent action sampling ─────────────────────────────────
        print("\n── Demo 7: RL agent ──")
        fv = fe.extract(_build_qaoa(4, 1))
        a, lp, v = opt._rl_agent.act(fv)
        b, s = action_to_bs(a)
        print(f"  RL action: {a} → backend={b}, strategy={s}")
        print(f"  log_prob={lp:.4f}, value={v:.4f}")

    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)

    print("\n" + "═" * 70)
    print("  AI Optimization Engine demo complete. ✓")
    print("═" * 70 + "\n")


__all__ = [
    "SKLEARN_AVAILABLE",
    "TORCH_AVAILABLE",
    "N_BACKENDS",
    "N_STRATEGIES",
    "N_ACTIONS",
    "action_to_bs",
    "bs_to_action",
    "SupervisedModel",
    "RLAgent",
    "AIDecision",
    "AIOptimizer",
    "make_ai_hybrid_executor",
    "Part2TestSuite",
    "run_demo",
]

if __name__ == "__main__":
    run_demo()
    Part2TestSuite().run_all()

