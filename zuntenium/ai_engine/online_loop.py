# zuntenium/ai_engine/online_loop.py
"""
OnlineLearningLoop — Orchestrates periodic supervised retraining and RL updates.

Runs as a background daemon thread attached to an AIOptimizer instance.
Triggered by execution count thresholds or explicit flush() calls.

Schedule
--------
  Every RETRAIN_EVERY executions  → supervised retrain
  Every RL_UPDATE_EVERY executions → PPO update step
  Every CHECKPOINT_EVERY minutes  → save model checkpoint
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Callable

_log = logging.getLogger(__name__)


class OnlineLearningLoop:
    """
    Background learning loop for the AI Optimization Engine.

    Parameters
    ----------
    optimizer : AIOptimizer
        The optimizer whose models will be updated.
    retrain_every : int
        Retrain supervised model every N new executions.
    rl_update_every : int
        Update PPO agent every N new executions.
    checkpoint_every_min : float
        Save model checkpoints every N minutes.
    on_retrain : callable | None
        Optional callback fired after each successful retrain.
    """

    def __init__(
        self,
        optimizer,
        retrain_every: int = 50,
        rl_update_every: int = 10,
        checkpoint_every_min: float = 30.0,
        on_retrain: Callable | None = None,
    ) -> None:
        self._opt              = optimizer
        self._retrain_every    = retrain_every
        self._rl_update_every  = rl_update_every
        self._checkpoint_every = checkpoint_every_min * 60  # → seconds
        self._on_retrain       = on_retrain

        self._exec_count_at_last_retrain  = 0
        self._exec_count_at_last_rl       = 0
        self._last_checkpoint_time        = time.monotonic()

        self._lock   = threading.Lock()
        self._stop   = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="ZunteniumAILoop")

    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background loop thread."""
        if self._thread.is_alive():
            return
        self._stop.clear()
        self._thread.start()
        _log.info("OnlineLearningLoop started.")

    def stop(self) -> None:
        """Gracefully stop the background loop."""
        self._stop.set()
        self._thread.join(timeout=5.0)
        _log.info("OnlineLearningLoop stopped.")

    def flush(self) -> dict:
        """
        Force immediate retrain + RL update regardless of thresholds.
        Useful at shutdown or after a burst of new executions.
        """
        results = {}
        try:
            results["supervised"] = self._opt.retrain_supervised()
        except Exception as exc:
            _log.warning("flush() supervised retrain failed: %s", exc)
            results["supervised"] = {"error": str(exc)}

        try:
            results["rl"] = self._opt.update_rl()
        except Exception as exc:
            _log.warning("flush() RL update failed: %s", exc)
            results["rl"] = {"error": str(exc)}

        self._checkpoint()
        return results

    # ------------------------------------------------------------------

    def notify_execution(self) -> None:
        """
        Called by HybridExecutor after every execution.
        Checks thresholds and triggers updates non-blockingly.
        """
        n = self._opt._n_executions

        if n - self._exec_count_at_last_rl >= self._rl_update_every:
            self._exec_count_at_last_rl = n
            t = threading.Thread(target=self._safe_rl_update, daemon=True)
            t.start()

        if n - self._exec_count_at_last_retrain >= self._retrain_every:
            self._exec_count_at_last_retrain = n
            t = threading.Thread(target=self._safe_retrain, daemon=True)
            t.start()

    # ------------------------------------------------------------------
    # Background thread
    # ------------------------------------------------------------------

    def _run(self) -> None:
        """Periodic background loop: checkpoint only (updates triggered on notify)."""
        while not self._stop.wait(timeout=60.0):
            if time.monotonic() - self._last_checkpoint_time >= self._checkpoint_every:
                self._checkpoint()

    def _safe_retrain(self) -> None:
        try:
            metrics = self._opt.retrain_supervised()
            _log.info("Online retrain complete: %s", metrics)
            if self._on_retrain:
                self._on_retrain(metrics)
        except Exception as exc:
            _log.warning("Online retrain failed: %s", exc)

    def _safe_rl_update(self) -> None:
        try:
            metrics = self._opt.update_rl()
            if metrics:
                _log.debug("PPO update: %s", metrics)
        except Exception as exc:
            _log.warning("PPO update failed: %s", exc)

    def _checkpoint(self) -> None:
        try:
            if self._opt._model_path:
                self._opt._supervised.save(self._opt._model_path / "supervised")
                if self._opt._rl_agent:
                    self._opt._rl_agent.save(self._opt._model_path / "ppo")
                _log.info("Checkpoint saved to %s", self._opt._model_path)
        except Exception as exc:
            _log.warning("Checkpoint failed: %s", exc)
        finally:
            self._last_checkpoint_time = time.monotonic()
