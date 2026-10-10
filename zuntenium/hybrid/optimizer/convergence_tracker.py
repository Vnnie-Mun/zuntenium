"""
ConvergenceTracker — Loss history with early stopping.
"""
from __future__ import annotations


class ConvergenceTracker:
    """
    Tracks optimization convergence with optional early stopping.

    Args:
        patience: Stop if no improvement for this many iterations.
        min_delta: Minimum absolute improvement to count as progress.
    """

    def __init__(self, patience: int = 20, min_delta: float = 1e-8) -> None:
        self._patience = patience
        self._min_delta = min_delta
        self._history: list[float] = []
        self._best: float = float("inf")
        self._no_improve = 0

    def update(self, value: float) -> bool:
        """
        Record a new cost value.

        Returns:
            True if training should stop (early stopping triggered).
        """
        self._history.append(value)
        if self._best - value > self._min_delta:
            self._best = value
            self._no_improve = 0
        else:
            self._no_improve += 1
        return self._no_improve >= self._patience

    @property
    def history(self) -> list[float]:
        return list(self._history)

    @property
    def best(self) -> float:
        return self._best

    @property
    def converged(self) -> bool:
        return self._no_improve >= self._patience

    def reset(self) -> None:
        self._history.clear()
        self._best = float("inf")
        self._no_improve = 0
