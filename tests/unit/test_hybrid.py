"""
tests/unit/test_hybrid.py — Unit tests for VQE, QAOA, Hamiltonian, PauliString.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from zuntenium.hybrid import QAOA, VQE
from zuntenium.hybrid.observable.hamiltonian import Hamiltonian
from zuntenium.hybrid.observable.pauli import PauliString
from zuntenium.hybrid.optimizer.classical_optimizer import ClassicalOptimizer
from zuntenium.hybrid.optimizer.gradient_estimator import GradientEstimator
from zuntenium.hybrid.result.energy_landscape import EnergyLandscape
from zuntenium.hybrid.result.hybrid_result import HybridResult

# ── PauliString ──────────────────────────────────────────────────────────────

class TestPauliString:

    def test_identity(self):
        """PauliString('II') = 4x4 identity."""
        ps = PauliString("II")
        mat = ps.to_matrix()
        assert mat.shape == (4, 4)
        np.testing.assert_allclose(mat, np.eye(4, dtype=np.complex128), atol=1e-12)

    def test_single_z(self):
        """PauliString('Z') has eigenvalues +1, -1."""
        ps = PauliString("Z")
        mat = ps.to_matrix()
        assert mat.shape == (2, 2)
        eigs = np.linalg.eigvalsh(mat)
        np.testing.assert_allclose(sorted(eigs), [-1.0, 1.0], atol=1e-12)

    def test_xx_expectation_bell(self):
        """⟨Φ+|XX|Φ+⟩ = 1 for the Bell state (1/√2)(|00⟩+|11⟩)."""
        ps = PauliString("XX")
        # Bell state |Φ+⟩ = (|00⟩ + |11⟩) / √2
        sv = np.array([1, 0, 0, 1], dtype=np.complex128) / math.sqrt(2)
        energy = ps.expectation_value(sv)
        assert abs(energy - 1.0) < 1e-10

    def test_zz_expectation_minus_one(self):
        """⟨ψ|ZZ|ψ⟩ = -1 for (|01⟩+|10⟩)/√2.
        ZZ|01⟩ = Z|0⟩⊗Z|1⟩ = (+1)(-1)|01⟩ = -|01⟩
        ZZ|10⟩ = Z|1⟩⊗Z|0⟩ = (-1)(+1)|10⟩ = -|10⟩
        → ⟨ψ|ZZ|ψ⟩ = ½(−1 − 1) = −1
        """
        ps = PauliString("ZZ")
        sv = np.array([0, 1, 1, 0], dtype=np.complex128) / math.sqrt(2)
        energy = ps.expectation_value(sv)
        assert abs(energy - (-1.0)) < 1e-10

    def test_num_qubits(self):
        assert PauliString("XYZII").num_qubits == 5

    def test_to_matrix_hermitian(self):
        """All Pauli tensor products are Hermitian."""
        for s in ["X", "Y", "Z", "XZ", "YY", "ZXY"]:
            ps = PauliString(s)
            mat = ps.to_matrix()
            np.testing.assert_allclose(mat, mat.conj().T, atol=1e-12,
                                       err_msg=f"PauliString('{s}') not Hermitian")


# ── Hamiltonian ──────────────────────────────────────────────────────────────

class TestHamiltonian:

    def _z_hamiltonian(self) -> Hamiltonian:
        """H = 0.5·Z (diagonal)."""
        return Hamiltonian([(0.5, PauliString("Z"))])

    def test_ground_state_z(self):
        """H = 0.5·Z has ground state energy -0.5."""
        H = self._z_hamiltonian()
        assert abs(H.ground_state_energy() - (-0.5)) < 1e-10

    def test_to_matrix_hermitian(self):
        H = self._z_hamiltonian()
        mat = H.to_matrix()
        np.testing.assert_allclose(mat, mat.conj().T, atol=1e-12)

    def test_add_hamiltonians(self):
        H1 = Hamiltonian([(1.0, PauliString("Z"))])
        H2 = Hamiltonian([(0.5, PauliString("X"))])
        H3 = H1 + H2
        assert len(H3.terms) == 2

    def test_expectation_from_statevector(self):
        """H = Z: ⟨0|Z|0⟩ = +1."""
        H = Hamiltonian([(1.0, PauliString("Z"))])
        from zuntenium.quantum_core.executor.result import ExecutionResult
        sv = np.array([1.0, 0.0], dtype=np.complex128)
        result = ExecutionResult(
            counts={"0": 1024},
            num_qubits=1,
            shots=1024,
            statevector=sv,
        )
        ev = H.expectation_value(result)
        assert abs(ev - 1.0) < 1e-10

    def test_mismatched_terms_raises(self):
        with pytest.raises(ValueError):
            Hamiltonian([(1.0, PauliString("Z")), (1.0, PauliString("ZZ"))])

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            Hamiltonian([])


# ── GradientEstimator ────────────────────────────────────────────────────────

class TestGradientEstimator:

    def _quadratic(self, params: np.ndarray) -> float:
        """Simple f(θ) = θ[0]² + θ[1]² — gradient should be [2θ₀, 2θ₁]."""
        return float(params[0]**2 + params[1]**2)

    def test_finite_diff_gradient(self):
        ge = GradientEstimator(method="finite_diff", eps=1e-5)
        params = np.array([1.0, 2.0])
        grad = ge.gradient(self._quadratic, params)
        np.testing.assert_allclose(grad, [2.0, 4.0], atol=1e-4)

    def test_parameter_shift_gradient(self):
        """For f = sin(θ), parameter-shift gives cos(θ)."""
        def f(p): return float(np.sin(p[0]))
        ge = GradientEstimator(method="parameter_shift")
        for theta in [0.1, 0.5, 1.0, math.pi / 4]:
            grad = ge.gradient(f, np.array([theta]))
            expected = math.cos(theta)
            assert abs(grad[0] - expected) < 1e-6, \
                f"theta={theta}: got {grad[0]}, expected {expected}"

    def test_gradient_shape(self):
        ge = GradientEstimator()
        params = np.zeros(5)
        grad = ge.gradient(lambda p: float(np.sum(p**2)), params)
        assert grad.shape == (5,)


# ── ClassicalOptimizer ────────────────────────────────────────────────────────

class TestClassicalOptimizer:

    def test_minimize_quadratic(self):
        """Minimize f(x,y) = x² + y² — minimum is at (0, 0)."""
        opt = ClassicalOptimizer(method="L-BFGS-B")
        result = opt.minimize(
            lambda p: float(p[0]**2 + p[1]**2),
            np.array([1.0, -1.0]),
        )
        assert "x" in result and "fun" in result
        np.testing.assert_allclose(result["x"], [0.0, 0.0], atol=1e-6)
        assert abs(result["fun"]) < 1e-10

    def test_minimize_cobyla(self):
        """COBYLA should also converge on a simple bowl."""
        opt = ClassicalOptimizer(method="COBYLA")
        result = opt.minimize(
            lambda p: float(p[0]**2 + 0.5),
            np.array([2.0]),
        )
        assert abs(result["fun"] - 0.5) < 1e-3

    def test_result_keys(self):
        opt = ClassicalOptimizer()
        result = opt.minimize(lambda p: float(p[0]**2), np.array([1.0]))
        required = {"x", "fun", "nit"}
        assert required.issubset(result.keys())


# ── HybridResult ──────────────────────────────────────────────────────────────

class TestHybridResult:

    def test_repr(self):
        r = HybridResult(
            energy=-1.234,
            optimal_params=np.array([0.1, 0.2]),
            num_iterations=42,
        )
        text = repr(r)
        assert "energy" in text.lower() or "-1.234" in text

    def test_converged_default(self):
        r = HybridResult(energy=0.0, optimal_params=np.array([]), num_iterations=0)
        assert r.converged is True


# ── EnergyLandscape ───────────────────────────────────────────────────────────

class TestEnergyLandscape:

    def _cost(self, params: np.ndarray) -> float:
        """Simple cos(θ₀) landscape."""
        return float(np.cos(params[0]))

    def test_scan_1d_shape(self):
        landscape = EnergyLandscape(self._cost)
        result = landscape.scan_1d(
            np.zeros(2), param_idx=0, theta_range=(-math.pi, math.pi), steps=50
        )
        assert len(result["thetas"]) == 50
        assert len(result["energies"]) == 50

    def test_scan_1d_minimum(self):
        """cos(θ) minimum is at θ=π, value=-1."""
        landscape = EnergyLandscape(self._cost)
        result = landscape.scan_1d(
            np.zeros(2), param_idx=0, theta_range=(0, 2*math.pi), steps=1000
        )
        min_idx = int(np.argmin(result["energies"]))
        assert abs(result["energies"][min_idx] - (-1.0)) < 1e-3

    def test_scan_2d_shape(self):
        def cost2d(p): return float(p[0]**2 + p[1]**2)
        landscape = EnergyLandscape(cost2d)
        result = landscape.scan_2d(
            np.zeros(2), 0, 1,
            x_range=(-1, 1), y_range=(-1, 1), steps=10
        )
        assert result["energy_grid"].shape == (10, 10)


# ── VQE Integration ───────────────────────────────────────────────────────────

class TestVQE:

    def test_vqe_h2_toy(self):
        """
        Toy VQE: minimize H = -Z on a 1-qubit Ry ansatz.
        Ground state energy = -1.0.
        """
        from zuntenium import QuantumCircuit
        from zuntenium.hybrid import VQE, Hamiltonian, PauliString

        # Ansatz: Ry(θ)|0⟩
        ansatz = QuantumCircuit(1, name="ry_ansatz")
        ansatz.ry(0.0, 0)

        H = Hamiltonian([(-1.0, PauliString("Z"))])
        vqe = VQE(ansatz=ansatz, hamiltonian=H)

        result = vqe.run(initial_params=np.array([0.1]), max_iter=200)
        assert result.energy < -0.95, \
            f"VQE did not converge: energy={result.energy}"

    def test_vqe_result_type(self):
        from zuntenium import QuantumCircuit
        ansatz = QuantumCircuit(1).ry(0.01, 0)
        H = Hamiltonian([(1.0, PauliString("Z"))])
        vqe = VQE(ansatz=ansatz, hamiltonian=H)
        result = vqe.run([0.1])
        assert isinstance(result, HybridResult)
        assert isinstance(result.optimal_params, np.ndarray)


# ── QAOA Integration ──────────────────────────────────────────────────────────

class TestQAOA:

    def test_qaoa_builds_ansatz(self):
        """QAOA should build a valid parameterized ansatz and run without error."""
        H = Hamiltonian([(1.0, PauliString("ZZ"))])
        qaoa = QAOA(hamiltonian=H, p=1)
        result = qaoa.run(initial_params=np.array([0.1, 0.1]))
        assert isinstance(result, HybridResult)

    def test_qaoa_multi_layer(self):
        H = Hamiltonian([(0.5, PauliString("ZZ"))])
        qaoa = QAOA(hamiltonian=H, p=2)
        result = qaoa.run(initial_params=np.zeros(4))
        assert result.num_iterations >= 0
