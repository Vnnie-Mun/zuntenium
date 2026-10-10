"""
HybridSolver — Abstract base for variational hybrid algorithms.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from zuntenium.backend.base_backend import AbstractBackend
    from zuntenium.circuit.circuit import QuantumCircuit
    from zuntenium.hybrid.observable.hamiltonian import Hamiltonian
    from zuntenium.hybrid.optimizer.classical_optimizer import ClassicalOptimizer
    from zuntenium.hybrid.result.hybrid_result import HybridResult


class HybridSolver(ABC):
    """
    Base class for variational hybrid quantum-classical algorithms.

    Owns the classical optimization loop and delegates quantum evaluation
    to the backend. Subclasses implement ``_build_cost_function()``.

    Args:
        ansatz: Parameterized quantum circuit (the variational ansatz).
        hamiltonian: Observable to minimize/compute expectation of.
        optimizer: Classical optimization strategy (scipy wrapper).
        backend: Quantum execution backend.
        shots: Measurement shots per circuit evaluation.
    """

    def __init__(
        self,
        ansatz: QuantumCircuit,
        hamiltonian: Hamiltonian,
        optimizer: ClassicalOptimizer,
        backend: AbstractBackend,
        shots: int = 1024,
    ) -> None:
        self.ansatz = ansatz
        self.hamiltonian = hamiltonian
        self.optimizer = optimizer
        self.backend = backend
        self._shots = shots
        self._history: list[float] = []

    @abstractmethod
    def _build_cost_function(self) -> Callable[[np.ndarray], float]:
        """Return the cost function for the classical optimizer."""
        ...

    def run(
        self,
        initial_params: np.ndarray,
        max_iter: int = 300,
    ) -> HybridResult:
        """
        Execute the full variational optimization loop.

        Args:
            initial_params: Initial parameter vector.
            max_iter: Maximum optimization iterations.

        Returns:
            HybridResult with optimal parameters, energy, and convergence history.
        """
        from zuntenium.hybrid.result.hybrid_result import HybridResult

        self._history = []
        cost_fn = self._build_cost_function()

        opt_result = self.optimizer.minimize(
            cost_fn,
            initial_params,
            max_iter=max_iter,
        )

        return HybridResult(
            energy=opt_result["fun"],
            optimal_params=opt_result["x"],
            num_iterations=opt_result.get("nit", 0),
            convergence_history=list(self._history),
            metadata={
                "backend": self.backend.name,
                "shots": self._shots,
                "optimizer": self.optimizer.method,
            },
        )

    def _evaluate_expectation(self, params: np.ndarray) -> float:
        """Bind parameters, run circuit, return ⟨H⟩."""
        bound = self.ansatz.bind_parameters(params)
        result = self.backend.run(bound, shots=self._shots)
        energy = self.hamiltonian.expectation_value(result)
        self._history.append(float(energy))
        return float(energy)
