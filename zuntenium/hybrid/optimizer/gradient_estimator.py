"""
GradientEstimator — parameter-shift rule and finite differences.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np


class GradientEstimator:
    """
    Estimates gradients of quantum cost functions.

    Supports:
    - Parameter-shift rule: exact gradient for Pauli rotation gates
    - Finite differences: approximate, single extra evaluation per param

    Args:
        method: "parameter_shift" | "finite_diff".
        eps: Step size for finite differences.
    """

    def __init__(
        self,
        method: str = "parameter_shift",
        eps: float = 1e-5,
    ) -> None:
        self.method = method
        self._eps = eps

    def gradient(
        self,
        cost_fn: Callable[[np.ndarray], float],
        params: np.ndarray,
    ) -> np.ndarray:
        """
        Compute gradient ∂E/∂θ at the given parameter point.

        Args:
            cost_fn: Scalar function of parameter vector.
            params: Current parameter values.

        Returns:
            Gradient vector of shape (len(params),).
        """
        if self.method == "parameter_shift":
            return self._parameter_shift(cost_fn, params)
        return self._finite_diff(cost_fn, params)

    def _parameter_shift(
        self,
        cost_fn: Callable[[np.ndarray], float],
        params: np.ndarray,
    ) -> np.ndarray:
        """
        Parameter shift rule: ∂E/∂θ_i = [E(θ + π/2·e_i) - E(θ - π/2·e_i)] / 2.

        Requires 2·M circuit evaluations for M parameters.
        Exact for Pauli rotation gates (Rx, Ry, Rz).
        """
        grad = np.zeros_like(params)
        shift = np.pi / 2
        for i in range(len(params)):
            delta = np.zeros_like(params)
            delta[i] = shift
            grad[i] = (cost_fn(params + delta) - cost_fn(params - delta)) / 2
        return grad

    def _finite_diff(
        self,
        cost_fn: Callable[[np.ndarray], float],
        params: np.ndarray,
    ) -> np.ndarray:
        """Forward finite differences: ∂E/∂θ_i ≈ [E(θ + ε·e_i) - E(θ)] / ε."""
        base = cost_fn(params)
        grad = np.zeros_like(params)
        for i in range(len(params)):
            delta = np.zeros_like(params)
            delta[i] = self._eps
            grad[i] = (cost_fn(params + delta) - base) / self._eps
        return grad
