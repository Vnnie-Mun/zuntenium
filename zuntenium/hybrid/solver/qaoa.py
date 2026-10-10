"""
QAOA — Quantum Approximate Optimization Algorithm.

Prepares an ansatz of alternating cost and mixer unitaries and minimizes
the expectation value of a problem Hamiltonian.
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


class QAOA(HybridSolver):
    """
    Quantum Approximate Optimization Algorithm.

    Builds the QAOA ansatz with p layers and minimizes the expectation
    value of the cost Hamiltonian.

    Args:
        hamiltonian: Cost Hamiltonian to optimize.
        p: Number of QAOA layers (depth). Default: 1.
        optimizer: Classical optimizer.
        backend: Quantum execution backend.
        shots: Measurement shots per iteration.
        mixer: Mixer Hamiltonian circuit (default: X⊗n).
    """

    def __init__(
        self,
        hamiltonian: Hamiltonian,
        p: int = 1,
        optimizer: ClassicalOptimizer | None = None,
        backend: AbstractBackend | None = None,
        shots: int = 1024,
    ) -> None:
        from zuntenium.backend.simulator.statevector_backend import StatevectorBackend
        from zuntenium.hybrid.optimizer.classical_optimizer import ClassicalOptimizer

        opt = optimizer or ClassicalOptimizer(method="COBYLA")
        bkd = backend or StatevectorBackend()

        # Build the QAOA ansatz
        ansatz = self._build_qaoa_ansatz(hamiltonian, p)
        super().__init__(ansatz, hamiltonian, opt, bkd, shots)
        self._p = p

    def _build_qaoa_ansatz(
        self,
        hamiltonian: Hamiltonian,
        p: int,
    ) -> QuantumCircuit:
        """Construct the QAOA ansatz circuit for p layers."""
        from zuntenium.circuit.circuit import QuantumCircuit
        n = hamiltonian.num_qubits
        qc = QuantumCircuit(n, name=f"qaoa_p{p}")
        # Initial state: uniform superposition
        for q in range(n):
            qc.h(q)
        # p layers of cost + mixer unitaries (parameterized)
        # In v0.1: simplified — users bind gamma/beta externally
        for _ in range(p):
            qc.barrier()
            # Cost unitary placeholder (Rz rotations)
            for q in range(n):
                qc.rz(0.0, q)   # gamma parameter
            # Mixer unitary (Rx rotations)
            for q in range(n):
                qc.rx(0.0, q)   # beta parameter
        return qc

    def _build_cost_function(self) -> Callable[[np.ndarray], float]:
        def cost(params: np.ndarray) -> float:
            return self._evaluate_expectation(params)
        return cost
