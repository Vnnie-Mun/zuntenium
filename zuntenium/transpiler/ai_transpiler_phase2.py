"""ZUNTENIUM AI Transpiler Phase 2: RL refinement over v2 Circuit IR.

This module is intentionally separate from the legacy list-based AI transpiler.
It operates on ``zuntenium.transpiler.v2.Circuit`` and is safe to use with the
execution integration layer.
"""
from __future__ import annotations

import copy
import math
import statistics
import time
import warnings
from collections import deque
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from zuntenium.transpiler.v2 import Circuit, Gate, Transpiler
from zuntenium.transpiler.v2.passes import (
    CommutationReorderingPass, DeadGateEliminationPass, DepthOptimizationPass,
    OptimizationPass, RedundantGateCancellationPass, RotationMergingPass,
    SingleQubitFusionPass,
)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.distributions import Categorical
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch unavailable; NumPy policy fallback active.", stacklevel=2)


class CircuitStateEncoder:
    GATE_TYPES = ["X", "Y", "Z", "H", "S", "T", "SDG", "TDG", "SX", "SXD",
                  "RX", "RY", "RZ", "PHASE", "U3", "CNOT", "CX", "CZ", "SWAP", "ISWAP"]
    N_GATE_TYPES = 20
    MAX_GATES, MAX_QUBITS, MAX_PARAMS = 50, 4, 3
    MAX_N, MAX_DEPTH = 20.0, 100.0
    _g2i = {name: index for index, name in enumerate(GATE_TYPES)}

    @classmethod
    def state_dim(cls) -> int:
        return cls.MAX_GATES * (cls.N_GATE_TYPES + cls.MAX_QUBITS + cls.MAX_PARAMS) + 4

    @classmethod
    def encode(cls, circuit: Circuit, fidelity: float = 1.0) -> np.ndarray:
        per = cls.N_GATE_TYPES + cls.MAX_QUBITS + cls.MAX_PARAMS
        state = np.zeros(cls.state_dim(), dtype=np.float32)
        for index, gate in enumerate(circuit.gates[:cls.MAX_GATES]):
            base = index * per
            state[base + cls._g2i.get(gate.name, cls.N_GATE_TYPES - 1)] = 1.0
            for offset, qubit in enumerate(gate.qubits[:cls.MAX_QUBITS]):
                state[base + cls.N_GATE_TYPES + offset] = qubit / cls.MAX_N
            for offset, parameter in enumerate(gate.params[:cls.MAX_PARAMS]):
                state[base + cls.N_GATE_TYPES + cls.MAX_QUBITS + offset] = np.tanh(parameter / math.pi)
        global_base = cls.MAX_GATES * per
        state[global_base:global_base + 4] = (
            min(circuit.depth(), cls.MAX_DEPTH) / cls.MAX_DEPTH,
            min(circuit.num_gates, cls.MAX_GATES) / cls.MAX_GATES,
            np.clip(fidelity, 0.0, 1.0),
            sum(g.num_qubits >= 2 for g in circuit.gates) / max(circuit.num_gates, 1),
        )
        return state


def estimate_fidelity(circuit: Circuit, p1q: float = 0.001, p2q: float = 0.01) -> float:
    fidelity = 1.0
    for gate in circuit.gates:
        if gate.is_measurement:
            continue
        error = gate.metadata.get("error_rate", p1q if gate.num_qubits == 1 else p2q)
        fidelity *= 1.0 - float(error)
    return max(fidelity, 1e-12)


class RewardFunction:
    def __init__(self, w1: float = 1.0, w2: float = 0.5, w3: float = 2.0, c: float = 0.3):
        self.w1, self.w2, self.w3, self.c = w1, w2, w3, c

    def compute(self, old: Circuit, new: Circuit, n_swaps: int = 0) -> float:
        reward = (self.w1 * (old.depth() - new.depth())
                  + self.w2 * (old.num_gates - new.num_gates)
                  + self.w3 * math.log(estimate_fidelity(new) / estimate_fidelity(old))
                  - self.c * n_swaps)
        return float(np.clip(reward, -5.0, 5.0))

    def no_change_reward(self) -> float:
        return -0.05


ActionFn = Callable[[Circuit], Tuple[Circuit, int]]


def _run(pass_type: type[OptimizationPass], circuit: Circuit) -> Tuple[Circuit, int]:
    return pass_type()._apply(circuit), 0


def _chain(circuit: Circuit, passes: List[type[OptimizationPass]]) -> Tuple[Circuit, int]:
    for pass_type in passes:
        circuit = pass_type()._apply(circuit)
    return circuit, 0


def _a_cancel(circuit): return _run(RedundantGateCancellationPass, circuit)
def _a_rotate(circuit): return _run(RotationMergingPass, circuit)
def _a_commute(circuit): return _run(CommutationReorderingPass, circuit)
def _a_fuse(circuit): return _run(SingleQubitFusionPass, circuit)
def _a_depth(circuit): return _run(DepthOptimizationPass, circuit)
def _a_dead(circuit): return _run(DeadGateEliminationPass, circuit)
def _a_cancel_rotate(circuit): return _chain(circuit, [RedundantGateCancellationPass, RotationMergingPass])
def _a_rotate_fuse(circuit): return _chain(circuit, [RotationMergingPass, SingleQubitFusionPass])
def _a_commute_cancel(circuit): return _chain(circuit, [CommutationReorderingPass, RedundantGateCancellationPass])
def _a_full(circuit): return _chain(circuit, [RedundantGateCancellationPass, RotationMergingPass,
                                               CommutationReorderingPass, SingleQubitFusionPass,
                                               DepthOptimizationPass])
def _a_rot_cancel_fuse(circuit): return _chain(circuit, [RotationMergingPass, RedundantGateCancellationPass,
                                                         SingleQubitFusionPass])
def _a_identity(circuit): return circuit.copy(), 0


ACTION_REGISTRY: List[Tuple[str, ActionFn]] = [
    ("CancelPauli", _a_cancel), ("MergeRotations", _a_rotate),
    ("CommutePast", _a_commute), ("FuseSingleQubit", _a_fuse),
    ("DepthSchedule", _a_depth), ("DeadEliminate", _a_dead),
    ("CancelRotate", _a_cancel_rotate), ("RotateFuse", _a_rotate_fuse),
    ("CommuteCancel", _a_commute_cancel), ("FullPipeline", _a_full),
    ("RotCancelFuse", _a_rot_cancel_fuse), ("Identity", _a_identity),
]
N_ACTIONS = len(ACTION_REGISTRY)


@dataclass
class StepInfo:
    action_name: str
    reward: float
    gates_before: int
    gates_after: int
    depth_before: int
    depth_after: int
    done: bool


class QuantumCircuitEnv:
    def __init__(self, reward_fn: Optional[RewardFunction] = None, max_steps: int = 30, patience: int = 10):
        self.reward_fn = reward_fn or RewardFunction()
        self.max_steps, self.patience = max_steps, patience
        self._cur: Optional[Circuit] = None
        self._step = self._no_imp = 0
        self._best = 0
        self._done = True
        self._fid = 1.0

    @property
    def state_dim(self) -> int: return CircuitStateEncoder.state_dim()
    @property
    def n_actions(self) -> int: return N_ACTIONS

    def reset(self, circuit: Circuit) -> np.ndarray:
        self._cur, self._step, self._no_imp = circuit.copy(), 0, 0
        self._best, self._done, self._fid = circuit.num_gates, False, estimate_fidelity(circuit)
        return self._state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, StepInfo]:
        if self._done:
            raise RuntimeError("Call reset() first.")
        if not 0 <= action < N_ACTIONS:
            raise ValueError(f"action must be in [0, {N_ACTIONS})")
        old = self._cur
        name, transform = ACTION_REGISTRY[action]
        try:
            new, swaps = transform(old)
        except Exception:
            new, swaps = old.copy(), 0
        changed = new.num_gates != old.num_gates or new.depth() != old.depth()
        reward = self.reward_fn.compute(old, new, swaps) if changed else self.reward_fn.no_change_reward()
        if not changed:
            new = old.copy()
        self._cur, self._step, self._fid = new, self._step + 1, estimate_fidelity(new)
        if new.num_gates < self._best:
            self._best, self._no_imp = new.num_gates, 0
        else:
            self._no_imp += 1
        self._done = self._step >= self.max_steps or self._no_imp >= self.patience or action == N_ACTIONS - 1
        info = StepInfo(name, reward, old.num_gates, new.num_gates, old.depth(), new.depth(), self._done)
        return self._state(), float(reward), self._done, info

    def current_circuit(self) -> Circuit:
        if self._cur is None:
            raise RuntimeError("Call reset() first.")
        return self._cur.copy()

    def _state(self) -> np.ndarray:
        return CircuitStateEncoder.encode(self._cur, self._fid)


if TORCH_AVAILABLE:
    class PPOTranspilerPolicy(nn.Module):
        def __init__(self, state_dim: int, n_actions: int = N_ACTIONS, hidden: int = 256):
            super().__init__()
            self.trunk = nn.Sequential(nn.Linear(state_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
            self.actor = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, n_actions))
            self.critic = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

        def forward(self, state):
            hidden = self.trunk(state)
            return self.actor(hidden), self.critic(hidden)

        def get_action_and_value(self, state, action=None):
            tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            logits, value = self.forward(tensor)
            distribution = Categorical(logits=logits.squeeze(0))
            selected = distribution.sample() if action is None else torch.tensor(action)
            return int(selected), float(distribution.log_prob(selected)), float(value.item()), float(distribution.entropy())

        def evaluate_actions(self, states, actions):
            logits, values = self.forward(states)
            distribution = Categorical(logits=logits)
            return distribution.log_prob(actions), values.squeeze(-1), distribution.entropy()
else:
    PPOTranspilerPolicy = None


class _NumpyPolicy:
    def __init__(self, state_dim: int, n_actions: int = N_ACTIONS, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, math.sqrt(2 / state_dim), (256, state_dim)).astype(np.float32)
        self.b1 = np.zeros(256, np.float32)
        self.W2 = rng.normal(0, math.sqrt(2 / 256), (256, 256)).astype(np.float32)
        self.b2 = np.zeros(256, np.float32)
        self.Wa = rng.normal(0, 0.01, (n_actions, 256)).astype(np.float32)
        self.ba = np.zeros(n_actions, np.float32)

    def _forward(self, state):
        hidden = np.maximum(0, self.W1 @ state + self.b1)
        hidden = np.maximum(0, self.W2 @ hidden + self.b2)
        logits = self.Wa @ hidden + self.ba
        probabilities = np.exp(logits - logits.max())
        return probabilities / probabilities.sum()

    def get_action_and_value(self, state, action=None):
        probabilities = self._forward(state)
        selected = int(np.random.choice(len(probabilities), p=probabilities)) if action is None else int(action)
        return selected, float(np.log(probabilities[selected] + 1e-12)), 0.0, float(-np.sum(probabilities * np.log(probabilities + 1e-12)))


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    value: float
    log_prob: float
    done: bool


class RolloutBuffer:
    def __init__(self, capacity: int = 2048): self._buf, self._cap = [], capacity
    def add(self, transition: Transition):
        if len(self._buf) < self._cap: self._buf.append(transition)
    def clear(self): self._buf.clear()
    def __len__(self): return len(self._buf)
    def compute_gae(self, last_val=0.0, gamma=0.99, lam=0.95):
        values = np.array([item.value for item in self._buf], dtype=np.float32)
        rewards = np.array([item.reward for item in self._buf], dtype=np.float32)
        done = np.array([item.done for item in self._buf], dtype=np.float32)
        advantages = np.zeros(len(self._buf), dtype=np.float32); running = 0.0
        for index in reversed(range(len(self._buf))):
            next_value = values[index + 1] if index + 1 < len(self._buf) else last_val
            delta = rewards[index] + gamma * next_value * (1 - done[index]) - values[index]
            running = delta + gamma * lam * (1 - done[index]) * running
            advantages[index] = running
        return advantages, advantages + values
    def as_numpy(self):
        return {"states": np.stack([x.state for x in self._buf]), "actions": np.array([x.action for x in self._buf]),
                "log_probs": np.array([x.log_prob for x in self._buf], dtype=np.float32),
                "values": np.array([x.value for x in self._buf], dtype=np.float32)}


if TORCH_AVAILABLE:
    class PPOTrainer:
        def __init__(self, policy, lr=3e-4, clip=0.2, vf=0.5, ent=0.01, max_grad=0.5, epochs=10, bs=64, gamma=0.99, lam=0.95):
            self.policy, self.opt = policy, optim.Adam(policy.parameters(), lr=lr)
            self.clip, self.vf, self.ent, self.max_grad = clip, vf, ent, max_grad
            self.epochs, self.bs, self.gamma, self.lam = epochs, bs, gamma, lam
        def update(self, buffer: RolloutBuffer) -> Dict[str, float]:
            data = buffer.as_numpy(); advantages, returns = buffer.compute_gae(gamma=self.gamma, lam=self.lam)
            states = torch.as_tensor(data["states"], dtype=torch.float32)
            actions = torch.as_tensor(data["actions"], dtype=torch.long)
            old_log_probs = torch.as_tensor(data["log_probs"], dtype=torch.float32)
            advantages = torch.as_tensor((advantages - advantages.mean()) / (advantages.std() + 1e-8))
            returns = torch.as_tensor(returns, dtype=torch.float32)
            logs = {"pl": [], "vl": [], "total": []}
            for _ in range(self.epochs):
                for start in range(0, len(states), self.bs):
                    batch = slice(start, start + self.bs)
                    log_probs, values, entropy = self.policy.evaluate_actions(states[batch], actions[batch])
                    ratio = torch.exp(log_probs - old_log_probs[batch])
                    clipped = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * advantages[batch]
                    policy_loss = -torch.min(ratio * advantages[batch], clipped).mean()
                    value_loss = nn.functional.mse_loss(values, returns[batch])
                    loss = policy_loss + self.vf * value_loss - self.ent * entropy.mean()
                    self.opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad); self.opt.step()
                    logs["pl"].append(float(policy_loss)); logs["vl"].append(float(value_loss)); logs["total"].append(float(loss))
            return {key: float(np.mean(value)) for key, value in logs.items()}
else:
    PPOTrainer = None


class OnlineLearningBuffer:
    def __init__(self, capacity=500): self._buf = deque(maxlen=capacity)
    def add(self, initial, final, reward): self._buf.append((initial.copy(), final.copy(), float(reward)))
    def sample(self, n):
        indices = np.random.choice(len(self._buf), min(n, len(self._buf)), replace=False) if self._buf else []
        return [self._buf[index] for index in indices]
    def __len__(self): return len(self._buf)


class RLRefinementPass(OptimizationPass):
    name, description = "RLRefinementPass", "PPO policy refinement (safety guarded)."
    def __init__(self, policy, env: QuantumCircuitEnv, k_steps=10, online_buffer=None, min_improvement=0):
        self._policy, self._env, self._k, self._buf, self._min_imp = policy, env, k_steps, online_buffer, min_improvement
    def _apply(self, circuit: Circuit) -> Circuit:
        state = self._env.reset(circuit); total_reward = 0.0
        for _ in range(self._k):
            if self._env._done: break
            action, _, _, _ = self._policy.get_action_and_value(state)
            state, reward, _, _ = self._env.step(action); total_reward += reward
        final = self._env.current_circuit()
        if final.num_gates > circuit.num_gates - self._min_imp: final = circuit.copy()
        if self._buf is not None: self._buf.add(circuit, final, total_reward)
        return final


@dataclass
class AITranspilerResult:
    circuit: Circuit
    original_gates: int
    original_depth: int
    phase1_gates: int
    phase1_depth: int
    final_gates: int
    final_depth: int
    phase2_extra_gates: int
    estimated_fidelity: float
    total_time_ms: float
    phase1_report: Any
    @property
    def total_gate_pct(self): return 100.0 * (1 - self.final_gates / max(self.original_gates, 1))
    @property
    def total_depth_pct(self): return 100.0 * (1 - self.final_depth / max(self.original_depth, 1))
    def summary(self): return (f"Gates: {self.original_gates}->{self.phase1_gates}(P1)->{self.final_gates}(P2) "
                               f"({self.total_gate_pct:.1f}%down P2+{self.phase2_extra_gates})  "
                               f"Depth: {self.original_depth}->{self.final_depth} ({self.total_depth_pct:.1f}%down)  "
                               f"Fidelity: {self.estimated_fidelity:.4f} {self.total_time_ms:.1f}ms")


class LearningCurve:
    def __init__(self): self.episodes, self.rewards = [], []
    def add(self, episode, reward): self.episodes.append(episode); self.rewards.append(float(reward))
    def smoothed(self, window=50):
        if len(self.rewards) < window: return self.episodes, self.rewards
        return self.episodes, [statistics.mean(self.rewards[max(0, i - window + 1):i + 1]) for i in range(len(self.rewards))]
    def summary(self):
        if not self.rewards: return "empty"
        return f"Episodes:{len(self.rewards)} EarlyAvg:{statistics.mean(self.rewards[:50]):.3f} LateAvg:{statistics.mean(self.rewards[-50:]):.3f} Max:{max(self.rewards):.3f} Min:{min(self.rewards):.3f}"


class AITranspiler:
    def __init__(self, use_ai=False, phase1_passes=None, k_steps=10, seed=0, max_steps=30, patience=10):
        self.use_ai = use_ai
        self._phase1 = Transpiler(passes=phase1_passes or ["cancel", "rotation", "commute", "fusion", "depth"])
        self._env = QuantumCircuitEnv(max_steps=max_steps, patience=patience)
        self._buffer, self._ep_rewards, self._train_logs = OnlineLearningBuffer(), [], []
        self._rf = RewardFunction()
        if TORCH_AVAILABLE:
            self._policy, self._trainer = PPOTranspilerPolicy(CircuitStateEncoder.state_dim()), None
            self._trainer = PPOTrainer(self._policy)
        else:
            self._policy, self._trainer = _NumpyPolicy(CircuitStateEncoder.state_dim(), seed=seed), None
        self._rl_pass = RLRefinementPass(self._policy, self._env, k_steps, self._buffer) if use_ai else None

    def transpile(self, circuit: Circuit, use_ai: Optional[bool] = None) -> AITranspilerResult:
        started = time.perf_counter(); effective_ai = self.use_ai if use_ai is None else use_ai
        phase1, report = self._phase1.transpile(circuit)
        final = self._rl_pass._apply(phase1) if effective_ai and self._rl_pass else phase1
        return AITranspilerResult(final, circuit.num_gates, circuit.depth(), phase1.num_gates, phase1.depth(),
                                  final.num_gates, final.depth(), phase1.num_gates - final.num_gates,
                                  estimate_fidelity(final), (time.perf_counter() - started) * 1000, report)

    def pretrain(self, n_episodes=500, n_qubits_range=(3, 10), depth_range=(10, 40), verbose=True, verbose_every=50):
        from zuntenium.transpiler.v2.engine import _build_adversarial, _build_ghz, _build_qaoa, _build_random
        rng, curve = np.random.default_rng(0), LearningCurve()
        rollout = RolloutBuffer(capacity=2048)
        for episode in range(n_episodes):
            n, depth = int(rng.integers(*n_qubits_range)), int(rng.integers(*depth_range))
            if episode % 5 == 0:
                raw = _build_adversarial(min(n, 5))
            elif episode % 5 == 1:
                raw = _build_qaoa(min(n, 6), 2)
            elif episode % 5 == 2:
                raw = _build_ghz(min(n, 8))
            else:
                raw = _build_random(n, depth)
            phase1, _ = self._phase1.transpile(raw)
            if not phase1.num_gates: curve.add(episode, 0.0); continue
            state, total = self._env.reset(phase1), 0.0
            while not self._env._done:
                action, log_prob, value, _ = self._policy.get_action_and_value(state)
                next_state, reward, done, _ = self._env.step(action)
                rollout.add(Transition(state, action, reward, value, log_prob, done))
                state, total = next_state, total + reward
            self._ep_rewards.append(total); curve.add(episode, total)
            if len(rollout) >= 256 and self._trainer is not None:
                self._train_logs.append(self._trainer.update(rollout))
                rollout.clear()
            if verbose and (episode + 1) % verbose_every == 0: print(f"[Pretrain] {episode + 1}/{n_episodes} avg={statistics.mean(curve.rewards[-verbose_every:]):.3f}")
        if len(rollout) > 32 and self._trainer is not None:
            self._train_logs.append(self._trainer.update(rollout))
        return curve

    def feedback(self, initial: Circuit, final: Circuit, real_fidelity: float):
        calibration = 2.0 * math.log(max(real_fidelity, 1e-9) / max(estimate_fidelity(final), 1e-9))
        self._buffer.add(initial, final, self._rf.compute(initial, final) + calibration)

    def save_policy(self, path):
        if TORCH_AVAILABLE: torch.save(self._policy.state_dict(), path)
        else: np.savez(path, **{key: getattr(self._policy, key) for key in ("W1", "b1", "W2", "b2", "Wa", "ba")})

    def load_policy(self, path):
        if TORCH_AVAILABLE: self._policy.load_state_dict(torch.load(path, map_location="cpu"))
        else:
            data = np.load(path if str(path).endswith(".npz") else str(path) + ".npz")
            for key in ("W1", "b1", "W2", "b2", "Wa", "ba"): setattr(self._policy, key, data[key])


def build_ai_executor(n_qubits, use_ai=True, seed=0):
    from zuntenium.execution.quantum_executor import QuantumExecutor
    ai = AITranspiler(use_ai=use_ai, seed=seed)
    base = QuantumExecutor.default(n_qubits, seed=seed, optimization_level=0)

    class QuantumExecutorAI:
        @property
        def num_qubits(self): return self._base.simulator.num_qubits
        @property
        def simulator(self): return self._base.simulator
        @property
        def ai_transpiler(self): return self._ai
        def __init__(self): self._base, self._ai = base, ai
        def execute(self, circuit, optimize=True, optimization_level=2, shots=1024, debug=False):
            if not optimize or optimization_level == 0: return self._base.execute(circuit, optimize=False, shots=shots, debug=debug)
            ai_result = self._ai.transpile(circuit, use_ai=optimization_level >= 2)
            result = self._base.execute(ai_result.circuit, optimize=False, shots=shots, debug=debug)
            result = replace(result, opt_report=ai_result.phase1_report)
            if optimization_level >= 2:
                real_fidelity = float(np.prod(result.probabilities[result.probabilities > 0]) **
                                      (1 / max(np.count_nonzero(result.probabilities), 1)))
                self._ai.feedback(circuit, ai_result.circuit, real_fidelity)
            if debug:
                print(f"[AIExecutor] {ai_result.summary()}")
            return result
    return QuantumExecutorAI()


def benchmark_phase1_vs_phase2(n_episodes_train=100, n_bench=20, verbose=True):
    from zuntenium.transpiler.v2.engine import _build_adversarial, _build_ghz, _build_qaoa, _build_random
    circuits = [(f"GHZ-{n}", _build_ghz(n)) for n in range(3, 8)] + [(f"QAOA-{n}p{p}", _build_qaoa(n, p)) for n, p in [(4, 1), (5, 1), (6, 2)]]
    circuits += [(f"Adv-{n}", _build_adversarial(n)) for n in range(3, 8)]
    ai = AITranspiler(use_ai=True); curve = ai.pretrain(n_episodes_train, verbose=verbose, verbose_every=max(20, n_episodes_train // 4))
    phase1, results = AITranspiler(use_ai=False), []
    for name, circuit in circuits[:n_bench]:
        first, second = phase1.transpile(circuit, False), ai.transpile(circuit, True)
        results.append({"name": name, "p1g": first.final_gates, "p2g": second.final_gates, "extra": 100 * max(first.final_gates - second.final_gates, 0) / max(first.final_gates, 1), "p1d": first.final_depth, "p2d": second.final_depth})
    return {"results": results, "mean_extra": statistics.mean([x["extra"] for x in results]) if results else 0.0, "curve": curve}


class AITranspilerTestSuite:
    def __init__(self): self._p = self._f = 0
    def _a(self, condition, message):
        if not condition: self._f += 1; raise AssertionError(message)
        self._p += 1
    def run_all(self):
        circuit = Circuit(3); circuit.add_gate(Gate("H", (0,))); circuit.add_gate(Gate("H", (0,)))
        self._a(CircuitStateEncoder.state_dim() == 1354, "state dimension")
        self._a(CircuitStateEncoder.encode(circuit).shape == (1354,), "state shape")
        self._a(len(ACTION_REGISTRY) == 12, "action count")
        env = QuantumCircuitEnv(max_steps=3); env.reset(circuit); self._a(env.step(0)[0].shape == (1354,), "environment state")
        self._a(RewardFunction().no_change_reward() < 0, "no-change reward")
        result = AITranspiler(use_ai=True).transpile(circuit)
        self._a(result.final_gates <= result.phase1_gates, "RL safety guard")
        print(f"AITranspilerTestSuite: {self._p} passed")


__all__ = ["TORCH_AVAILABLE", "CircuitStateEncoder", "estimate_fidelity", "RewardFunction", "ActionFn",
           "_a_cancel", "_a_rotate", "_a_commute", "_a_fuse", "_a_depth", "_a_dead",
           "_a_cancel_rotate", "_a_rotate_fuse", "_a_commute_cancel", "_a_full",
           "_a_rot_cancel_fuse", "_a_identity", "ACTION_REGISTRY", "N_ACTIONS", "StepInfo",
           "QuantumCircuitEnv", "PPOTranspilerPolicy", "_NumpyPolicy", "Transition", "RolloutBuffer",
           "PPOTrainer", "OnlineLearningBuffer", "RLRefinementPass", "AITranspilerResult",
           "LearningCurve", "AITranspiler", "build_ai_executor", "benchmark_phase1_vs_phase2",
           "AITranspilerTestSuite"]
