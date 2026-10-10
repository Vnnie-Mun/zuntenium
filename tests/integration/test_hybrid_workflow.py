"""
test_hybrid_workflow.py — Integration test: VQE on known Hamiltonians.

Tests the full hybrid workflow end-to-end: ansatz → VQE → converged energy.
"""
from __future__ import annotations

import math
import numpy as np
import pytest

from zuntenium import QuantumCircuit
from zuntenium.hybrid import VQE, QAOA, Hamiltonian, PauliString
from zuntenium.hybrid.result.hybrid_result import HybridResult


class TestVQEWorkflow:

    def test_vqe_single_z(self):
        """VQE on H = -Z should converge to energy = -1."""
        ansatz = QuantumCircuit(1, name="ry_ansatz").ry(0.1, 0)
        H = Hamiltonian([(-1.0, PauliString("Z"))])
        vqe = VQE(ansatz=ansatz, hamiltonian=H)
        result = vqe.run(initial_params=np.array([0.5]), max_iter=200)
        assert isinstance(result, HybridResult)
        assert result.energy < -0.95

    def test_vqe_returns_optimal_params(self):
        ansatz = QuantumCircuit(1).ry(0.1, 0)
        H = Hamiltonian([(1.0, PauliString("Z"))])
        vqe = VQE(ansatz=ansatz, hamiltonian=H)
        result = vqe.run([0.5])
        assert isinstance(result.optimal_params, np.ndarray)
        assert len(result.optimal_params) == 1

    def test_vqe_history_tracked(self):
        ansatz = QuantumCircuit(1).ry(0.1, 0)
        H = Hamiltonian([(-1.0, PauliString("Z"))])
        vqe = VQE(ansatz=ansatz, hamiltonian=H)
        result = vqe.run([0.1], max_iter=50)
        assert result.num_iterations >= 0


class TestQAOAWorkflow:

    def test_qaoa_zz_hamiltonian(self):
        """QAOA on H = ZZ should produce a valid result."""
        H = Hamiltonian([(1.0, PauliString("ZZ"))])
        qaoa = QAOA(hamiltonian=H, p=1)
        result = qaoa.run(initial_params=np.array([0.3, 0.5]))
        assert isinstance(result, HybridResult)
        assert isinstance(result.energy, float)

    def test_qaoa_multi_layer(self):
        H = Hamiltonian([(0.5, PauliString("ZZ"))])
        qaoa = QAOA(hamiltonian=H, p=2)
        result = qaoa.run(initial_params=np.zeros(4))
        assert result.num_iterations >= 0


class TestHamiltonianPhysics:

    def test_ground_state_energy_z(self):
        """H = Z has ground state energy -1."""
        H = Hamiltonian([(1.0, PauliString("Z"))])
        assert abs(H.ground_state_energy() - (-1.0)) < 1e-10

    def test_ground_state_energy_xx_zz(self):
        """H = 0.5·ZZ - 0.3·XX: check exact diag matches."""
        H = Hamiltonian([
            (0.5, PauliString("ZZ")),
            (-0.3, PauliString("XX")),
        ])
        E0 = H.ground_state_energy()
        # The matrix is 4×4; eigenvalues are computable
        mat = H.to_matrix()
        eigs = np.linalg.eigvalsh(mat)
        assert abs(E0 - eigs.min()) < 1e-10
