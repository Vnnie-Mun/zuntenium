"""
ClassicalOptimizer — scipy.optimize wrapper for hybrid algorithms.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np


class ClassicalOptimizer:
    """
    Wraps scipy.optimize.minimize for use in hybrid quantum-classical loops.

    Args:
        method: scipy optimization method. Default "L-BFGS-B".
                Common options: "L-BFGS-B", "COBYLA", "SLSQP", "Nelder-Mead".
        options: Additional scipy.optimize options dict.
        tol: Convergence tolerance.
    """

    def __init__(
        self,
        method: str = "L-BFGS-B",
        options: dict[str, Any] | None = None,
        tol: float = 1e-6,
    ) -> None:
        self.method = method
        self._options = options or {}
        self._tol = tol

    def minimize(
        self,
        cost_fn: Callable[[np.ndarray], float],
        initial_params: np.ndarray,
        max_iter: int = 300,
    ) -> dict[str, Any]:
        """
        Minimize cost_fn starting from initial_params.

        Args:
            cost_fn: Scalar function of a parameter vector.
            initial_params: Starting parameter values.
            max_iter: Maximum iterations.

        Returns:
            Dict with keys: "x" (optimal params), "fun" (optimal value), "nit".
        """
        try:
            from scipy.optimize import minimize as scipy_minimize

            options = {"maxiter": max_iter, **self._options}
            result = scipy_minimize(
                cost_fn,
                initial_params,
                method=self.method,
                tol=self._tol,
                options=options,
            )
            return {
                "x": result.x,
                "fun": float(result.fun),
                "nit": getattr(result, "nit", getattr(result, "nfev", 0)),
                "success": result.success,
                "message": str(result.message),
            }
        except ImportError:
            # Naive gradient-free fallback if scipy not available
            return self._fallback_minimize(cost_fn, initial_params, max_iter)

    def _fallback_minimize(
        self,
        cost_fn: Callable[[np.ndarray], float],
        params: np.ndarray,
        max_iter: int,
    ) -> dict[str, Any]:
        """Finite-difference gradient descent fallback."""
        lr = 0.01
        eps = 1e-5
        for i in range(max_iter):
            current = cost_fn(params)
            grad = np.zeros_like(params)
            for j in range(len(params)):
                delta = np.zeros_like(params)
                delta[j] = eps
                grad[j] = (cost_fn(params + delta) - current) / eps
            params = params - lr * grad
            if np.linalg.norm(grad) < self._tol:
                break
        return {"x": params, "fun": float(cost_fn(params)), "nit": i + 1, "success": True}
