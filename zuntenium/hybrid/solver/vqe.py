"""
VQE — Variational Quantum Eigensolver.

Finds the ground state energy of a Hamiltonian by minimizing
⟨ψ(θ)|H|ψ(θ)⟩ over the variational parameter vector θ.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from zuntenium.hybrid.solver.hybrid_solver import HybridSolver

if TYPE_CHECKING:
    from zuntenium.backend.base_backend import AbstractBackend
    from zuntenium.circuit.circuit import QuantumCircuit
    from zuntenium.hybrid.observable.hamiltonian import Hamiltonian
    from zuntenium.hybrid.optimizer.classical_optimizer import ClassicalOptimizer
    from zuntenium.hybrid.result.hybrid_result import HybridResult


class VQE(HybridSolver):
    """
    Variational Quantum Eigensolver.

    Minimizes E(θ) = ⟨ψ(θ)|H|ψ(θ)⟩ to find the ground state energy.

    Args:
        ansatz: Parameterized quantum circuit.
        hamiltonian: The Hamiltonian to diagonalize.
        optimizer: Classical optimizer (default: L-BFGS-B).
        backend: Quantum execution backend.
        shots: Measurement shots per expectation value evaluation.
        gradient_method: "parameter_shift" | "finite_diff".

    Example::

        from zuntenium.hybrid import VQE
        from zuntenium.hybrid.observable import Hamiltonian, PauliString

        H = Hamiltonian([(0.5, PauliString("ZZ")), (-0.3, PauliString("XX"))])
        vqe = VQE(ansatz=qc, hamiltonian=H, backend=backend)
        result = vqe.run(initial_params=np.array([0.1, 0.2]))
        print(result.energy)
    """

    def __init__(
        self,
        ansatz: QuantumCircuit,
        hamiltonian: Hamiltonian,
        optimizer: ClassicalOptimizer | None = None,
        backend: AbstractBackend | None = None,
        shots: int = 1024,
        gradient_method: str = "parameter_shift",
    ) -> None:
        from zuntenium.backend.simulator.statevector_backend import StatevectorBackend
        from zuntenium.hybrid.optimizer.classical_optimizer import ClassicalOptimizer

        opt = optimizer or ClassicalOptimizer(method="L-BFGS-B")
        bkd = backend or StatevectorBackend()
        super().__init__(ansatz, hamiltonian, opt, bkd, shots)
        self._gradient_method = gradient_method

    def _build_cost_function(self) -> Callable[[np.ndarray], float]:
        def cost(params: np.ndarray) -> float:
            return self._evaluate_expectation(params)
        return cost

    def run(
        self,
        initial_params: np.ndarray | list[float],
        max_iter: int = 300,
    ) -> HybridResult:
        params = np.asarray(initial_params, dtype=np.float64)
        return super().run(params, max_iter=max_iter)
