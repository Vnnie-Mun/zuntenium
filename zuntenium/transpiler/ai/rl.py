"""
Reinforcement Learning components (Phase 2) for the Zuntenium AI Transpiler.

Provides:
- ``CircuitStateEncoder`` — fixed-length 1354-dim float32 state vector
- ``RewardFunction`` — composite reward with depth, gate-count, fidelity terms
- ``TransformAction`` — 12 discrete circuit transformations
- ``CircuitOptimizationEnv`` — Gym-style MDP environment
- ``ReplayBuffer`` — circular buffer for PPO / online learning
- ``RLAgent`` — unified agent (PyTorch PPO or NumPy fallback)

Author:  ZUNTENIUM Core Team
License: Apache 2.0
"""
from __future__ import annotations

import math
import warnings
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from zuntenium.transpiler.ai.ir import (
    CircuitDAG,
    Gate,
    HardwareConfig,
    circuit_to_dag,
)
from zuntenium.transpiler.ai.passes import (
    PauliCancellationPass,
    RotationFusionPass,
    SingleQubitUnifyPass,
    SWAPRoutingPass,
    gates_commute,
)

# ── Optional PyTorch ─────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.distributions import Categorical

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn(
        "PyTorch not found. Phase 2 RL will use the NumPy fallback policy. "
        "Install torch for full PPO training.",
        stacklevel=2,
    )


# ═══════════════════════════════════════════════════════════════════════════
# TransformAction
# ═══════════════════════════════════════════════════════════════════════════

_ACTION_NAMES = [
    "FuseRotations", "CancelPauliPair", "CommutePastGate",
    "UnifySingleQubit", "InsertSWAP", "RemoveSWAP",
    "RZ_to_Phase", "CX_Decompose", "CZ_Decompose",
    "T_cancel", "GateReorder_Left", "GateReorder_Right",
]


@dataclass(frozen=True)
class TransformAction:
    """One of 12 discrete circuit transformations."""

    action_id: int
    NUM_ACTIONS: int = 12

    @property
    def name(self) -> str:
        return _ACTION_NAMES[self.action_id]


# ═══════════════════════════════════════════════════════════════════════════
# CircuitStateEncoder
# ═══════════════════════════════════════════════════════════════════════════

class CircuitStateEncoder:
    """Fixed-length circuit feature vector for the MLP policy input.

    Encoding (per gate, zero-padded to ``MAX_GATES``):
        - One-hot gate type  (20 bits)
        - Qubit indices      (4 floats, normalised / 20)
        - Parameters         (3 floats, tanh-normalised)

    Global features (4 floats):
        - depth / 100
        - gate_count / 50
        - estimated fidelity
        - 2Q gate ratio

    Total dimensionality: 50 × (20 + 4 + 3) + 4 = **1354**.
    """

    GATE_TYPES = [
        "X", "Y", "Z", "H", "S", "T", "SDG", "TDG",
        "RX", "RY", "RZ", "U3", "PHASE",
        "CNOT", "CX", "CZ", "SWAP", "ISWAP", "CCX", "CUSTOM",
    ]
    NUM_GATE_TYPES = len(GATE_TYPES)
    _g2i = {g: i for i, g in enumerate(GATE_TYPES)}
    MAX_GATES = 50
    MAX_QUBITS = 4
    MAX_PARAMS = 3
    MAX_DEPTH = 100
    MAX_N = 20

    @classmethod
    def state_dim(cls) -> int:
        """Total dimensionality of the state vector."""
        per = cls.NUM_GATE_TYPES + cls.MAX_QUBITS + cls.MAX_PARAMS
        return cls.MAX_GATES * per + 4

    @classmethod
    def encode(cls, gates: List[Gate], fidelity: float = 1.0) -> np.ndarray:
        """Encode a gate list into a fixed-size float32 vector."""
        per = cls.NUM_GATE_TYPES + cls.MAX_QUBITS + cls.MAX_PARAMS
        vec = np.zeros(cls.MAX_GATES * per + 4, dtype=np.float32)
        for i, g in enumerate(gates[: cls.MAX_GATES]):
            b = i * per
            idx = cls._g2i.get(g.name.upper(), cls.NUM_GATE_TYPES - 1)
            vec[b + idx] = 1.0
            for qi, q in enumerate(g.qubits[: cls.MAX_QUBITS]):
                vec[b + cls.NUM_GATE_TYPES + qi] = q / cls.MAX_N
            for pi, p in enumerate(g.params[: cls.MAX_PARAMS]):
                vec[b + cls.NUM_GATE_TYPES + cls.MAX_QUBITS + pi] = float(
                    np.tanh(p / math.pi)
                )
        gs = cls.MAX_GATES * per
        dag = circuit_to_dag(gates)
        vec[gs] = min(dag.depth(), cls.MAX_DEPTH) / cls.MAX_DEPTH
        vec[gs + 1] = min(len(gates), cls.MAX_GATES) / cls.MAX_GATES
        vec[gs + 2] = float(fidelity)
        n_gates = max(len(gates), 1)
        vec[gs + 3] = sum(1 for g in gates if g.num_qubits >= 2) / n_gates
        return vec


# ═══════════════════════════════════════════════════════════════════════════
# RewardFunction
# ═══════════════════════════════════════════════════════════════════════════

class RewardFunction:
    """Composite reward for RL circuit optimisation.

    ``r = w₁(d_old − d_new) + w₂(g_old − g_new) + w₃·log(f_new/f_old) − c·swaps``

    Fidelity model: ``f = ∏ (1 − p_gate)`` with ``p_1q = 0.001, p_2q = 0.01``.
    """

    def __init__(
        self,
        w1: float = 1.0,
        w2: float = 0.5,
        w3: float = 2.0,
        c: float = 1.5,
        p_1q: float = 0.001,
        p_2q: float = 0.01,
    ) -> None:
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.c = c
        self.p_1q = p_1q
        self.p_2q = p_2q

    def fidelity(self, gates: List[Gate]) -> float:
        """Estimate circuit fidelity from gate error model."""
        f = 1.0
        for g in gates:
            if g.error_rate > 0:
                f *= 1 - g.error_rate
            elif g.num_qubits == 1:
                f *= 1 - self.p_1q
            else:
                f *= 1 - self.p_2q
        return max(f, 1e-12)

    def compute(
        self,
        old: List[Gate],
        new: List[Gate],
        swaps: int = 0,
    ) -> float:
        """Compute scalar reward for a transition."""
        d_old = circuit_to_dag(old).depth()
        d_new = circuit_to_dag(new).depth()
        f_old = self.fidelity(old)
        f_new = self.fidelity(new)
        return (
            self.w1 * (d_old - d_new)
            + self.w2 * (len(old) - len(new))
            + self.w3 * math.log(f_new / f_old + 1e-12)
            - self.c * swaps
        )


# ═══════════════════════════════════════════════════════════════════════════
# CircuitOptimizationEnv
# ═══════════════════════════════════════════════════════════════════════════

class CircuitOptimizationEnv:
    """Gym-style MDP for circuit optimisation.

    State  : ``CircuitStateEncoder.encode(current_circuit)``
    Action : ``Discrete(12)``
    Reward : ``RewardFunction.compute(old, new)``
    Done   : ``max_steps`` reached **or** ``patience`` steps without improvement.
    """

    def __init__(
        self,
        initial: List[Gate],
        reward_fn: Optional[RewardFunction] = None,
        max_steps: int = 100,
        patience: int = 20,
        hw: Optional[HardwareConfig] = None,
    ) -> None:
        self._init = list(initial)
        self._rf = reward_fn or RewardFunction()
        self.max_steps = max_steps
        self.patience = patience
        self.hw = hw
        self._cur: List[Gate] = []
        self._step = 0
        self._no_imp = 0
        self._best = 0
        self._done = False
        self.reset()

    @property
    def state_dim(self) -> int:
        return CircuitStateEncoder.state_dim()

    @property
    def num_actions(self) -> int:
        return TransformAction.NUM_ACTIONS

    def reset(self) -> np.ndarray:
        """Reset environment and return initial state."""
        self._cur = list(self._init)
        self._step = 0
        self._no_imp = 0
        self._best = len(self._init)
        self._done = False
        return self._state()

    def step(self, action_id: int) -> Tuple[np.ndarray, float, bool]:
        """Apply *action_id* and return ``(next_state, reward, done)``."""
        if self._done:
            raise RuntimeError("Episode done. Call reset().")
        old = list(self._cur)
        new, swaps = self._apply(action_id, old)
        if new is None or new == old:
            reward = -0.1
            self._no_imp += 1
        else:
            reward = self._rf.compute(old, new, swaps)
            self._cur = new
            if len(new) < self._best:
                self._best = len(new)
                self._no_imp = 0
            else:
                self._no_imp += 1
        self._step += 1
        self._done = (
            self._step >= self.max_steps or self._no_imp >= self.patience
        )
        return self._state(), reward, self._done

    def _state(self) -> np.ndarray:
        return CircuitStateEncoder.encode(
            self._cur, self._rf.fidelity(self._cur)
        )

    def _apply(
        self, aid: int, gates: List[Gate]
    ) -> Tuple[Optional[List[Gate]], int]:
        try:
            if aid == 0:
                return RotationFusionPass().run(gates), 0
            if aid == 1:
                return PauliCancellationPass().run(gates), 0
            if aid == 2:
                return self._commute(gates), 0
            if aid == 3:
                return SingleQubitUnifyPass().run(gates), 0
            if aid == 4:
                if self.hw is None:
                    return None, 0
                r = SWAPRoutingPass().run(gates, self.hw)
                ns = sum(1 for g in r if g.name == "SWAP") - sum(
                    1 for g in gates if g.name == "SWAP"
                )
                return r, max(0, ns)
            if aid == 5:
                return PauliCancellationPass().run(gates), 0
            if aid == 6:
                return self._rz_special(gates), 0
            if aid == 7:
                return self._cx_decomp(gates), 0
            if aid == 8:
                return self._cz_decomp(gates), 0
            if aid == 9:
                return PauliCancellationPass().run(gates), 0
            if aid == 10:
                return self._reorder(gates, -1), 0
            if aid == 11:
                return self._reorder(gates, +1), 0
        except Exception:
            pass
        return None, 0

    # ── Helper transforms ─────────────────────────────────────────────────

    @staticmethod
    def _commute(g: List[Gate]) -> List[Gate]:
        r = list(g)
        for i in range(len(r) - 1):
            if gates_commute(r[i], r[i + 1]):
                r[i], r[i + 1] = r[i + 1], r[i]
                return r
        return g

    @staticmethod
    def _rz_special(gates: List[Gate]) -> List[Gate]:
        out: List[Gate] = []
        for g in gates:
            if g.name == "RZ" and g.params:
                a = g.params[0] % (2 * math.pi)
                if math.isclose(a, math.pi, abs_tol=1e-6):
                    out.append(Gate("Z", g.qubits))
                elif math.isclose(a, math.pi / 2, abs_tol=1e-6):
                    out.append(Gate("S", g.qubits))
                elif math.isclose(a, math.pi / 4, abs_tol=1e-6):
                    out.append(Gate("T", g.qubits))
                else:
                    out.append(g)
            else:
                out.append(g)
        return out

    @staticmethod
    def _cx_decomp(gates: List[Gate]) -> List[Gate]:
        out: List[Gate] = []
        for g in gates:
            if g.name in ("CNOT", "CX"):
                c, t = g.qubits
                out += [Gate("H", (t,)), Gate("CZ", (c, t)), Gate("H", (t,))]
            else:
                out.append(g)
        return out

    @staticmethod
    def _cz_decomp(gates: List[Gate]) -> List[Gate]:
        out: List[Gate] = []
        for g in gates:
            if g.name == "CZ":
                c, t = g.qubits
                out += [
                    Gate("H", (t,)), Gate("CNOT", (c, t)), Gate("H", (t,))
                ]
            else:
                out.append(g)
        return out

    @staticmethod
    def _reorder(gates: List[Gate], d: int) -> List[Gate]:
        r = list(gates)
        if len(r) < 2:
            return r
        i = int(np.random.randint(0, len(r) - 1))
        j = i + d
        if 0 <= j < len(r) and gates_commute(r[i], r[j]):
            r[i], r[j] = r[j], r[i]
        return r


# ═══════════════════════════════════════════════════════════════════════════
# ReplayBuffer
# ═══════════════════════════════════════════════════════════════════════════

class ReplayBuffer:
    """Fixed-capacity circular replay buffer for PPO / online learning."""

    def __init__(self, capacity: int = 2048) -> None:
        self.capacity = capacity
        self._s: List[Any] = []
        self._a: List[int] = []
        self._r: List[float] = []
        self._ns: List[Any] = []
        self._d: List[float] = []
        self._lp: List[float] = []
        self._v: List[float] = []
        self._ptr = 0

    def push(
        self,
        s: Any,
        a: int,
        r: float,
        ns: Any,
        d: bool,
        lp: float = 0.0,
        v: float = 0.0,
    ) -> None:
        """Store one transition."""
        vals = [s, a, r, ns, float(d), lp, v]
        lists = [self._s, self._a, self._r, self._ns, self._d, self._lp, self._v]
        for lst, val in zip(lists, vals):
            if len(lst) < self.capacity:
                lst.append(val)
            else:
                lst[self._ptr % self.capacity] = val
        self._ptr += 1

    def get_all(self) -> Dict[str, np.ndarray]:
        """Return all stored data as NumPy arrays."""
        return {
            "states": np.array(self._s, dtype=np.float32),
            "actions": np.array(self._a, dtype=np.int64),
            "rewards": np.array(self._r, dtype=np.float32),
            "next_states": np.array(self._ns, dtype=np.float32),
            "dones": np.array(self._d, dtype=np.float32),
            "log_probs": np.array(self._lp, dtype=np.float32),
            "values": np.array(self._v, dtype=np.float32),
        }

    def clear(self) -> None:
        """Drop all stored transitions."""
        for lst in [self._s, self._a, self._r, self._ns, self._d, self._lp, self._v]:
            lst.clear()
        self._ptr = 0

    def __len__(self) -> int:
        return len(self._s)


# ═══════════════════════════════════════════════════════════════════════════
# Policy networks
# ═══════════════════════════════════════════════════════════════════════════

if TORCH_AVAILABLE:

    class ActorCriticMLP(nn.Module):
        """Actor-Critic network for PPO circuit optimisation.

        Architecture:
            Shared trunk:  Linear(D,256)→ReLU→Linear(256,256)→ReLU
            Actor head:    Linear(256,256)→ReLU→Linear(256,12)  [logits]
            Critic head:   Linear(256,256)→ReLU→Linear(256,1)   [value]

        Initialised with orthogonal weights (Andrychowicz et al. 2021).
        """

        def __init__(
            self,
            state_dim: int,
            num_actions: int = 12,
            hidden: int = 256,
        ) -> None:
            super().__init__()
            self.trunk = nn.Sequential(
                nn.Linear(state_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
            )
            self.actor = nn.Sequential(
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, num_actions),
            )
            self.critic = nn.Sequential(
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, 1),
            )
            for m in self.trunk.modules():
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, math.sqrt(2))
                    nn.init.constant_(m.bias, 0)
            for m in self.actor.modules():
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, 0.01)
                    nn.init.constant_(m.bias, 0)
            for m in self.critic.modules():
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, 1.0)
                    nn.init.constant_(m.bias, 0)

        def forward(
            self, x: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            h = self.trunk(x)
            return self.actor(h), self.critic(h)

        def get_action(
            self, state: np.ndarray
        ) -> Tuple[int, float, float]:
            x = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                logits, val = self.forward(x)
            dist = Categorical(logits=logits.squeeze(0))
            a = dist.sample()
            return (
                int(a.item()),
                float(dist.log_prob(a).item()),
                float(val.item()),
            )

        def evaluate(
            self,
            states: torch.Tensor,
            actions: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            logits, vals = self.forward(states)
            dist = Categorical(logits=logits)
            return dist.log_prob(actions), vals.squeeze(-1), dist.entropy()

    class PPOTrainer:
        """PPO-Clip with GAE (Schulman et al. 2017).

        Hyperparameters:
            γ=0.99, λ=0.95, ε=0.2, c₁=0.5, c₂=0.01, lr=3e-4,
            epochs=4, minibatch=64, grad_clip=0.5
        """

        def __init__(
            self,
            model: ActorCriticMLP,
            lr: float = 3e-4,
            gamma: float = 0.99,
            lam: float = 0.95,
            eps: float = 0.2,
            vc: float = 0.5,
            ec: float = 0.01,
            epochs: int = 4,
            bs: int = 64,
        ) -> None:
            self.model = model
            self.opt = optim.Adam(model.parameters(), lr=lr)
            self.gamma = gamma
            self.lam = lam
            self.eps = eps
            self.vc = vc
            self.ec = ec
            self.epochs = epochs
            self.bs = bs

        def _gae(
            self,
            rewards: np.ndarray,
            values: np.ndarray,
            dones: np.ndarray,
            last_val: float = 0.0,
        ) -> Tuple[np.ndarray, np.ndarray]:
            T = len(rewards)
            adv = np.zeros(T, dtype=np.float32)
            last = 0.0
            for t in reversed(range(T)):
                nv = values[t + 1] if t + 1 < T else last_val
                delta = rewards[t] + self.gamma * nv * (1 - dones[t]) - values[t]
                last = delta + self.gamma * self.lam * (1 - dones[t]) * last
                adv[t] = last
            return adv, adv + values

        def update(self, buffer: ReplayBuffer) -> Dict[str, float]:
            d = buffer.get_all()
            S = torch.tensor(d["states"], dtype=torch.float32)
            A = torch.tensor(d["actions"], dtype=torch.long)
            old_lp = torch.tensor(d["log_probs"], dtype=torch.float32)
            adv, ret = self._gae(d["rewards"], d["values"], d["dones"])
            adv_t = torch.tensor(adv, dtype=torch.float32)
            ret_t = torch.tensor(ret, dtype=torch.float32)
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
            T = len(S)
            logs: Dict[str, list] = defaultdict(list)
            for _ in range(self.epochs):
                idx = np.random.permutation(T)
                for start in range(0, T, self.bs):
                    bi = torch.tensor(idx[start : start + self.bs])
                    lp, vals, ent = self.model.evaluate(S[bi], A[bi])
                    ratio = torch.exp(lp - old_lp[bi])
                    s1 = ratio * adv_t[bi]
                    s2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * adv_t[bi]
                    pl = -torch.min(s1, s2).mean()
                    vl = nn.functional.mse_loss(vals, ret_t[bi])
                    loss = pl + self.vc * vl - self.ec * ent.mean()
                    self.opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                    self.opt.step()
                    logs["policy_loss"].append(float(pl))
                    logs["value_loss"].append(float(vl))
            return {k: float(np.mean(v)) for k, v in logs.items()}


class _NumpyActorCritic:
    """NumPy fallback MLP policy (forward-only; no gradient training)."""

    def __init__(self, state_dim: int, num_actions: int) -> None:
        rng = np.random.default_rng(42)
        s1 = math.sqrt(2 / state_dim)
        s2 = math.sqrt(2 / 256)
        self.W1 = rng.normal(0, s1, (256, state_dim)).astype(np.float32)
        self.b1 = np.zeros(256, np.float32)
        self.W2 = rng.normal(0, s2, (num_actions, 256)).astype(np.float32) * 0.01
        self.b2 = np.zeros(num_actions, np.float32)

    def forward(self, s: np.ndarray) -> np.ndarray:
        h = np.maximum(0, self.W1 @ s + self.b1)
        logits = self.W2 @ h + self.b2
        logits -= logits.max()
        e = np.exp(logits)
        return e / e.sum()

    def get_action(self, state: np.ndarray) -> Tuple[int, float, float]:
        p = self.forward(state)
        a = int(np.random.choice(len(p), p=p))
        return a, float(np.log(p[a] + 1e-12)), 0.0


class RLAgent:
    """Unified RL agent: PyTorch PPO if available, else NumPy fallback."""

    def __init__(
        self,
        state_dim: int,
        num_actions: int = 12,
        hidden: int = 256,
        lr: float = 3e-4,
    ) -> None:
        self._torch = TORCH_AVAILABLE
        if self._torch:
            self._model = ActorCriticMLP(state_dim, num_actions, hidden)
            self._trainer = PPOTrainer(self._model, lr=lr)
        else:
            self._model = _NumpyActorCritic(state_dim, num_actions)
            self._trainer = None

    def get_action(self, state: np.ndarray) -> Tuple[int, float, float]:
        """Sample an action from the policy."""
        return self._model.get_action(state)

    def train_step(self, buf: ReplayBuffer) -> Dict[str, float]:
        """Run one PPO update (no-op for NumPy fallback)."""
        if self._torch and self._trainer is not None:
            return self._trainer.update(buf)
        return {}

    def save(self, path: str) -> None:
        """Save model checkpoint (PyTorch only)."""
        if self._torch:
            import torch as _torch
            _torch.save(self._model.state_dict(), path)

    def load(self, path: str) -> None:
        """Load model checkpoint (PyTorch only)."""
        if self._torch:
            import torch as _torch
            self._model.load_state_dict(
                _torch.load(path, map_location="cpu")
            )
