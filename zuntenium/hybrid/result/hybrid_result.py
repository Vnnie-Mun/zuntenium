"""
HybridResult — Convergence data and final VQE/QAOA output.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class HybridResult:
    """
    Result of a hybrid quantum-classical optimization run (VQE, QAOA).

    Attributes:
        energy: Final optimal energy value (minimum expectation value found).
        optimal_params: Parameter vector at the optimal point.
        num_iterations: Number of optimizer iterations taken.
        convergence_history: Energy value at each iteration.
        metadata: Execution metadata (backend, shots, optimizer, etc.).
    """
    energy: float
    optimal_params: np.ndarray
    num_iterations: int
    convergence_history: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def converged(self) -> bool:
        """True if the optimizer reported convergence."""
        return bool(self.metadata.get("success", True))

    def plot_convergence(self, **kwargs: Any) -> None:
        """Plot the convergence history (requires matplotlib)."""
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(self.convergence_history, color="#6366f1", linewidth=2)
            ax.axhline(y=self.energy, color="#f43f5e", linestyle="--",
                       label=f"Final: {self.energy:.6f}")
            ax.set_xlabel("Iteration", fontsize=12)
            ax.set_ylabel("Energy", fontsize=12)
            ax.set_title("VQE/QAOA Convergence", fontsize=14)
            ax.legend()
            plt.tight_layout()
            plt.show()
        except ImportError:
            print("Install matplotlib to use plot_convergence().")

    def __repr__(self) -> str:
        return (
            f"HybridResult(energy={self.energy:.8f}, "
            f"num_iterations={self.num_iterations}, "
            f"converged={self.converged})"
        )
