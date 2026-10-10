"""
energy_landscape.py — Energy landscape scanner for variational algorithms.

Scans the cost function surface over a 1D or 2D parameter grid,
useful for diagnosing barren plateaus and verifying VQE convergence basins.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np


class EnergyLandscape:
    """
    Scans the energy surface E(θ) over a parameter grid.

    Useful for diagnosing barren plateaus and verifying that the VQE
    cost landscape has a reachable minimum.

    Args:
        cost_fn: Scalar function of a parameter vector (e.g. VQE cost function).
        param_names: Optional names for each parameter.

    Example::

        landscape = EnergyLandscape(cost_fn=vqe_cost)
        grid_1d = landscape.scan_1d(param_idx=0, theta_range=(-π, π), steps=50)
        landscape.plot_1d(grid_1d)
    """

    def __init__(
        self,
        cost_fn: Callable[[np.ndarray], float],
        param_names: list[str] | None = None,
    ) -> None:
        self._cost_fn = cost_fn
        self._param_names = param_names

    def scan_1d(
        self,
        base_params: np.ndarray,
        param_idx: int,
        theta_range: tuple[float, float] = (-np.pi, np.pi),
        steps: int = 100,
    ) -> dict[str, Any]:
        """
        Scan energy along a single parameter axis.

        Args:
            base_params: Base parameter vector (all params except param_idx fixed).
            param_idx: Index of the parameter to vary.
            theta_range: (min, max) range for the scanned parameter.
            steps: Number of grid points.

        Returns:
            Dict with keys ``thetas`` and ``energies``.
        """
        thetas = np.linspace(theta_range[0], theta_range[1], steps)
        energies = np.zeros(steps)
        params = base_params.copy()
        for i, theta in enumerate(thetas):
            params[param_idx] = theta
            energies[i] = self._cost_fn(params)
        return {"thetas": thetas, "energies": energies, "param_idx": param_idx}

    def scan_2d(
        self,
        base_params: np.ndarray,
        param_idx_x: int,
        param_idx_y: int,
        x_range: tuple[float, float] = (-np.pi, np.pi),
        y_range: tuple[float, float] = (-np.pi, np.pi),
        steps: int = 30,
    ) -> dict[str, Any]:
        """
        Scan energy over a 2D grid of two parameter axes.

        Args:
            base_params: Base parameter vector (all others fixed).
            param_idx_x: Index of parameter for x-axis.
            param_idx_y: Index of parameter for y-axis.
            x_range: (min, max) for x-axis.
            y_range: (min, max) for y-axis.
            steps: Grid resolution (steps × steps evaluations).

        Returns:
            Dict with keys ``x_thetas``, ``y_thetas``, and ``energy_grid`` (2D array).
        """
        xs = np.linspace(x_range[0], x_range[1], steps)
        ys = np.linspace(y_range[0], y_range[1], steps)
        grid = np.zeros((steps, steps))
        params = base_params.copy()
        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                params[param_idx_x] = x
                params[param_idx_y] = y
                grid[i, j] = self._cost_fn(params)
        return {
            "x_thetas": xs,
            "y_thetas": ys,
            "energy_grid": grid,
            "param_idx_x": param_idx_x,
            "param_idx_y": param_idx_y,
        }

    def plot_1d(self, scan_result: dict[str, Any], **kwargs: Any) -> None:
        """Plot a 1D energy scan (requires matplotlib)."""
        try:
            import matplotlib.pyplot as plt
            thetas = scan_result["thetas"]
            energies = scan_result["energies"]
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.plot(thetas, energies, color="#6366f1", linewidth=2)
            ax.set_xlabel(f"θ[{scan_result['param_idx']}] (rad)", fontsize=12)
            ax.set_ylabel("Energy ⟨H⟩", fontsize=12)
            ax.set_title("Energy Landscape (1D scan)", fontsize=14)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
        except ImportError:
            print("Install matplotlib to use plot_1d(): pip install matplotlib")

    def plot_2d(self, scan_result: dict[str, Any], **kwargs: Any) -> None:
        """Plot a 2D energy heatmap (requires matplotlib)."""
        try:
            import matplotlib.pyplot as plt
            grid = scan_result["energy_grid"]
            fig, ax = plt.subplots(figsize=(7, 6))
            px = scan_result["param_idx_x"]
            py = scan_result["param_idx_y"]
            im = ax.imshow(
                grid.T, origin="lower", aspect="auto", cmap="plasma",
                extent=[
                    scan_result["x_thetas"][0], scan_result["x_thetas"][-1],
                    scan_result["y_thetas"][0], scan_result["y_thetas"][-1],
                ],
            )
            plt.colorbar(im, ax=ax, label="Energy ⟨H⟩")
            ax.set_xlabel(f"θ[{px}] (rad)", fontsize=12)
            ax.set_ylabel(f"θ[{py}] (rad)", fontsize=12)
            ax.set_title("Energy Landscape (2D scan)", fontsize=14)
            plt.tight_layout()
            plt.show()
        except ImportError:
            print("Install matplotlib to use plot_2d(): pip install matplotlib")
