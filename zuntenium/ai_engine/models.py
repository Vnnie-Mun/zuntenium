# zuntenium/ai_engine/models.py
"""
AI Engine Models — Phase 1 (Supervised) and Phase 2 (RL / PPO).

Phase 1 — SupervisedModel
  RandomForestClassifier by default; optionally replaced by a small PyTorch MLP.
  Predicts (backend, transpilation_strategy) from FeatureVector.

Phase 2 — PPOAgent (Actor-Critic)
  State  : s_t ∈ ℝ^d  (FeatureVector)
  Action : a_t ∈ {0,…,K-1}  (BackendChoice × TranspilationStrategy)
  Reward : r_t  (from ExecutionLogger._compute_reward)

  Policy loss:  L^CLIP = E[min(r_t·Â_t, clip(r_t, 1-ε, 1+ε)·Â_t)]
  Value  loss:  L^VF   = E[(V(s_t) − R_t)^2]
  Entropy bonus: L^S   = E[-π·log π]
  Total: L = -L^CLIP + c₁·L^VF - c₂·L^S

All models expose a unified interface:
    predict(features: np.ndarray) → (label: str, confidence: float)
    train(X, y) / update(batch)
    save(path) / load(path)
"""

from __future__ import annotations

import json
import logging
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Label registry — shared by both models
# ---------------------------------------------------------------------------

BACKEND_LABELS = ["statevector", "tensornetwork", "stabilizer", "ibm"]
STRATEGY_LABELS = ["rule_only", "rl_refine", "hardware_aware", "none"]
JOINT_LABELS = [
    f"{b}|{s}"
    for b in BACKEND_LABELS
    for s in STRATEGY_LABELS
]
LABEL_TO_IDX = {label: idx for idx, label in enumerate(JOINT_LABELS)}
IDX_TO_LABEL = {idx: label for label, idx in LABEL_TO_IDX.items()}

NUM_ACTIONS = len(JOINT_LABELS)  # 16


def encode_label(backend: str, strategy: str) -> int:
    key = f"{backend}|{strategy}"
    return LABEL_TO_IDX.get(key, LABEL_TO_IDX["statevector|rule_only"])


def decode_label(idx: int) -> tuple[str, str]:
    label = IDX_TO_LABEL.get(idx, "statevector|rule_only")
    parts = label.split("|")
    return parts[0], parts[1]


# ---------------------------------------------------------------------------
# ReplayBuffer (shared by PPO and can be used offline)
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """
    Fixed-capacity ring buffer for (s, a, r, s', done) tuples.

    Parameters
    ----------
    capacity : int   — max stored transitions
    state_dim : int  — feature vector dimension
    """

    def __init__(self, capacity: int = 10_000, state_dim: int = 27) -> None:
        self.capacity  = capacity
        self.state_dim = state_dim
        self._ptr      = 0
        self._size     = 0

        self.states      = np.zeros((capacity, state_dim), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions     = np.zeros(capacity, dtype=np.int64)
        self.rewards     = np.zeros(capacity, dtype=np.float32)
        self.dones       = np.zeros(capacity, dtype=np.float32)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.states[self._ptr]      = state
        self.actions[self._ptr]     = action
        self.rewards[self._ptr]     = reward
        self.next_states[self._ptr] = next_state
        self.dones[self._ptr]       = float(done)
        self._ptr  = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        idx = np.random.choice(self._size, min(batch_size, self._size), replace=False)
        return {
            "states":      self.states[idx],
            "actions":     self.actions[idx],
            "rewards":     self.rewards[idx],
            "next_states": self.next_states[idx],
            "dones":       self.dones[idx],
        }

    def __len__(self) -> int:
        return self._size


# ---------------------------------------------------------------------------
# SupervisedModel (Phase 1)
# ---------------------------------------------------------------------------

class SupervisedModel:
    """
    Phase-1 model: predicts (backend, strategy) from FeatureVector.

    Uses scikit-learn RandomForestClassifier as default (always available).
    Optionally upgrades to a small PyTorch MLP when torch is available.

    Interface
    ---------
    predict(features: np.ndarray) → (label: str, confidence: float)
    train(X: np.ndarray, y: np.ndarray[int]) → metrics: dict
    save(path: str | Path)
    load(path: str | Path) [classmethod]
    """

    def __init__(self, use_torch: bool = False, state_dim: int = 27) -> None:
        self.state_dim  = state_dim
        self.use_torch  = use_torch and self._torch_available()
        self._model     = None
        self._trained   = False
        self._build()
        _log.info("SupervisedModel: backend=%s", "PyTorch MLP" if self.use_torch else "RandomForest")

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _build(self) -> None:
        if self.use_torch:
            self._model = self._build_mlp()
        else:
            self._model = self._build_rf()

    @staticmethod
    def _build_rf():
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )

    def _build_mlp(self):
        """Small 3-layer MLP: [d → 128 → 64 → K] with dropout."""
        import torch
        import torch.nn as nn

        class _MLP(nn.Module):
            def __init__(self, in_dim: int, n_actions: int):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(in_dim, 128), nn.LayerNorm(128), nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(128, 64),  nn.LayerNorm(64),  nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(64, n_actions),
                )
            def forward(self, x):
                return self.net(x)

        return _MLP(self.state_dim, NUM_ACTIONS)

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(self, features: np.ndarray) -> tuple[str, float]:
        """
        Returns
        -------
        label : str   — "backend|strategy"  e.g. "tensornetwork|rl_refine"
        confidence : float — [0, 1]
        """
        if not self._trained:
            # Cold-start: return rule-based default
            return "statevector|rule_only", 0.0

        features = features.astype(np.float32).reshape(1, -1)

        if self.use_torch:
            return self._predict_torch(features)
        else:
            return self._predict_rf(features)

    def _predict_rf(self, features: np.ndarray) -> tuple[str, float]:
        probs = self._model.predict_proba(features)[0]
        classes = self._model.classes_
        idx_in_classes = int(np.argmax(probs))
        action_idx = int(classes[idx_in_classes])
        confidence = float(probs[idx_in_classes])
        label = IDX_TO_LABEL.get(action_idx, "statevector|rule_only")
        return label, confidence

    def _predict_torch(self, features: np.ndarray) -> tuple[str, float]:
        import torch
        import torch.nn.functional as F
        self._model.eval()
        with torch.no_grad():
            x = torch.from_numpy(features)
            logits = self._model(x)
            probs  = F.softmax(logits, dim=-1).squeeze()
            idx    = int(probs.argmax().item())
            conf   = float(probs[idx].item())
        return IDX_TO_LABEL.get(idx, "statevector|rule_only"), conf

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------

    def train(self, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        """
        Train / retrain on labelled data.

        Parameters
        ----------
        X : (N, d) float32
        y : (N,)   int  — action indices from encode_label()

        Returns
        -------
        metrics : {"train_accuracy": float, "n_samples": int}
        """
        if len(X) < 4:
            _log.warning("Too few samples (%d) for training, skipping.", len(X))
            return {"train_accuracy": 0.0, "n_samples": len(X)}

        if self.use_torch:
            metrics = self._train_torch(X, y)
        else:
            metrics = self._train_rf(X, y)

        self._trained = True
        return metrics

    def _train_rf(self, X: np.ndarray, y: np.ndarray) -> dict:
        from sklearn.model_selection import cross_val_score
        self._model.fit(X, y)
        if len(X) >= 10:
            cv = cross_val_score(self._model, X, y, cv=min(3, len(X)//3), scoring="accuracy")
            acc = float(cv.mean())
        else:
            acc = float(np.mean(self._model.predict(X) == y))
        _log.info("RF trained: accuracy=%.3f n=%d", acc, len(X))
        return {"train_accuracy": acc, "n_samples": len(X)}

    def _train_torch(self, X: np.ndarray, y: np.ndarray) -> dict:
        import torch
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, TensorDataset

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(device)
        opt = torch.optim.AdamW(self._model.parameters(), lr=1e-3, weight_decay=1e-4)
        ds  = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long),
        )
        loader = DataLoader(ds, batch_size=64, shuffle=True)
        self._model.train()
        for epoch in range(20):
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                loss = F.cross_entropy(self._model(xb), yb)
                opt.zero_grad(); loss.backward(); opt.step()
        # Accuracy
        self._model.eval()
        with torch.no_grad():
            preds = self._model(torch.tensor(X, dtype=torch.float32).to(device)).argmax(-1)
            acc   = float((preds.cpu().numpy() == y).mean())
        _log.info("MLP trained: accuracy=%.3f n=%d", acc, len(X))
        return {"train_accuracy": acc, "n_samples": len(X)}

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {"use_torch": self.use_torch, "trained": self._trained, "state_dim": self.state_dim}
        if self.use_torch:
            import torch
            torch.save(self._model.state_dict(), str(path) + ".pt")
        else:
            with open(str(path) + ".pkl", "wb") as f:
                pickle.dump(self._model, f)
        with open(str(path) + ".meta.json", "w") as f:
            json.dump(meta, f)
        _log.info("SupervisedModel saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "SupervisedModel":
        path = Path(path)
        meta_path = str(path) + ".meta.json"
        if not Path(meta_path).exists():
            raise FileNotFoundError(f"No model meta at {meta_path}")
        with open(meta_path) as f:
            meta = json.load(f)
        obj = cls(use_torch=meta["use_torch"], state_dim=meta["state_dim"])
        if meta["use_torch"]:
            import torch
            obj._model.load_state_dict(torch.load(str(path) + ".pt", map_location="cpu"))
        else:
            with open(str(path) + ".pkl", "rb") as f:
                obj._model = pickle.load(f)
        obj._trained = meta["trained"]
        return obj

    @staticmethod
    def _torch_available() -> bool:
        try:
            import torch  # noqa: F401
            return True
        except ImportError:
            return False


# ---------------------------------------------------------------------------
# PPOAgent (Phase 2)
# ---------------------------------------------------------------------------

class PPOAgent:
    """
    Proximal Policy Optimisation agent for quantum backend selection.

    State space  : s ∈ ℝ^d        (FeatureVector)
    Action space : a ∈ {0,…,K-1}  (backend × strategy joint action)
    Reward       : r_t ∈ ℝ        (from ExecutionLogger._compute_reward)

    Network architecture (Actor-Critic shared trunk):
      trunk   : Linear(d,128) → LayerNorm → ReLU → Linear(128,64) → LayerNorm → ReLU
      actor   : Linear(64, K) → Softmax         (policy π(a|s))
      critic  : Linear(64, 1)                   (value V(s))

    Loss:
      L = -L^CLIP + c₁·L^VF - c₂·L^S
      L^CLIP = E[min(ρ·Â, clip(ρ, 1-ε, 1+ε)·Â)]   ρ = π(a|s)/π_old(a|s)
      L^VF   = MSE(V(s), R_t)
      L^S    = -E[Σ π·log π]
    """

    def __init__(
        self,
        state_dim: int = 27,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        c1: float = 0.5,
        c2: float = 0.01,
        update_epochs: int = 4,
        batch_size: int = 32,
    ) -> None:
        self.state_dim     = state_dim
        self.gamma         = gamma
        self.gae_lambda    = gae_lambda
        self.clip_eps      = clip_eps
        self.c1            = c1
        self.c2            = c2
        self.update_epochs = update_epochs
        self.batch_size    = batch_size
        self._trained      = False

        if not self._torch_available():
            _log.warning("PPOAgent: PyTorch not available, agent disabled.")
            self._enabled = False
            return

        self._enabled = True
        import torch
        import torch.nn as nn

        class _ActorCritic(nn.Module):
            def __init__(self, in_dim: int, n_actions: int):
                super().__init__()
                self.trunk = nn.Sequential(
                    nn.Linear(in_dim, 128), nn.LayerNorm(128), nn.ReLU(),
                    nn.Linear(128, 64),     nn.LayerNorm(64),  nn.ReLU(),
                )
                self.actor  = nn.Linear(64, n_actions)
                self.critic = nn.Linear(64, 1)

            def forward(self, x):
                h = self.trunk(x)
                return self.actor(h), self.critic(h)

            def get_action_and_value(self, x, action=None):
                import torch.distributions as dist
                logits, value = self.forward(x)
                pd = dist.Categorical(logits=logits)
                if action is None:
                    action = pd.sample()
                log_prob = pd.log_prob(action)
                entropy  = pd.entropy()
                return action, log_prob, entropy, value

        device_str = "cuda" if self._cuda_available() else "cpu"
        self._device = torch.device(device_str)
        self._net    = _ActorCritic(state_dim, NUM_ACTIONS).to(self._device)
        self._opt    = torch.optim.Adam(self._net.parameters(), lr=lr)
        _log.info("PPOAgent initialised on %s, actions=%d", device_str, NUM_ACTIONS)

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(self, features: np.ndarray) -> tuple[str, float]:
        """Greedy prediction (argmax of policy π)."""
        if not self._enabled or not self._trained:
            return "statevector|rule_only", 0.0
        import torch
        import torch.nn.functional as F
        self._net.eval()
        with torch.no_grad():
            x      = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self._device)
            logits, _ = self._net(x)
            probs  = F.softmax(logits, dim=-1).squeeze()
            idx    = int(probs.argmax().item())
            conf   = float(probs[idx].item())
        # FIX BUG-008: removed stray _trained=True here; flag set by update() only
        return IDX_TO_LABEL.get(idx, "statevector|rule_only"), conf

    # ------------------------------------------------------------------
    # PPO Update (from ReplayBuffer batch)
    # ------------------------------------------------------------------

    def update(self, buffer: "ReplayBuffer") -> dict[str, float]:
        """
        Run PPO update over collected trajectories.

        Returns
        -------
        dict with policy_loss, value_loss, entropy, approx_kl
        """
        if not self._enabled or len(buffer) < self.batch_size:
            return {}

        import torch
        import torch.nn.functional as F

        batch = buffer.sample(min(len(buffer), 512))
        states      = torch.tensor(batch["states"],      dtype=torch.float32).to(self._device)
        actions     = torch.tensor(batch["actions"],     dtype=torch.long   ).to(self._device)
        rewards     = torch.tensor(batch["rewards"],     dtype=torch.float32).to(self._device)
        next_states = torch.tensor(batch["next_states"], dtype=torch.float32).to(self._device)
        dones       = torch.tensor(batch["dones"],       dtype=torch.float32).to(self._device)

        # Compute advantages via GAE (single-step approximation for simplicity)
        with torch.no_grad():
            _, next_values = self._net(next_states)
            next_values    = next_values.squeeze(-1)
            _, curr_values = self._net(states)
            curr_values    = curr_values.squeeze(-1)
            td_target = rewards + self.gamma * next_values * (1 - dones)
            advantages = td_target - curr_values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Old log probs
            _, old_log_probs, _, _ = self._net.get_action_and_value(states, actions)

        metrics = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0}

        for _ in range(self.update_epochs):
            # Random mini-batches
            perm = torch.randperm(len(states))
            for start in range(0, len(states), self.batch_size):
                idx  = perm[start:start + self.batch_size]
                sb, ab, adv_b, td_b, old_lp_b = (
                    states[idx], actions[idx], advantages[idx],
                    td_target[idx], old_log_probs[idx]
                )

                _, new_lp, entropy, new_val = self._net.get_action_and_value(sb, ab)
                new_val = new_val.squeeze(-1)

                ratio        = torch.exp(new_lp - old_lp_b)
                clip_ratio   = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
                policy_loss  = -torch.min(ratio * adv_b, clip_ratio * adv_b).mean()
                value_loss   = F.mse_loss(new_val, td_b)
                entropy_loss = -entropy.mean()

                loss = policy_loss + self.c1 * value_loss + self.c2 * entropy_loss

                self._opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._net.parameters(), 0.5)
                self._opt.step()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - torch.log(ratio)).mean().item()

                metrics["policy_loss"] += policy_loss.item()
                metrics["value_loss"]  += value_loss.item()
                metrics["entropy"]     += (-entropy_loss.item())
                metrics["approx_kl"]   += approx_kl

        self._trained = True
        _log.debug("PPO update: %s", metrics)
        return metrics

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        if not self._enabled:
            return
        import torch
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self._net.state_dict(),
            "opt_state":  self._opt.state_dict(),
            "trained":    self._trained,
            "state_dim":  self.state_dim,
        }, str(path) + ".ppo.pt")
        _log.info("PPOAgent saved to %s", path)

    @classmethod
    def load(cls, path: str | Path, **kwargs) -> "PPOAgent":
        import torch
        data = torch.load(str(path) + ".ppo.pt", map_location="cpu")
        obj  = cls(state_dim=data["state_dim"], **kwargs)
        if obj._enabled:
            obj._net.load_state_dict(data["state_dict"])
            obj._opt.load_state_dict(data["opt_state"])
            obj._trained = data["trained"]
        return obj

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _torch_available() -> bool:
        try:
            import torch  # noqa: F401
            return True
        except ImportError:
            return False

    @staticmethod
    def _cuda_available() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
