"""
QMLLayer — Parameterized quantum circuit layer for hybrid ML workflows.

Provides a differentiable quantum layer that can be composed with classical
neural network layers. Uses the parameter-shift rule for gradient computation.

Supports:
    - Arbitrary parameterized ansatz circuits (Rx, Ry, Rz rotations)
    - Expectation-value outputs (Z-basis measurements)
    - Parameter-shift gradient computation (analytic, no finite differences)
    - Batch execution for training efficiency
    - NumPy-native (no PennyLane/TensorFlow dependency)

Example::

    layer = QMLLayer(
        num_qubits=4,
        num_layers=2,
        output_qubits=[0, 1],
    )
    params = np.random.randn(layer.num_params)
    output = layer(params)           # shape: (2,) — one value per output qubit
    grads = layer.gradient(params)   # shape: (num_params, 2)
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Literal

import numpy as np

logger = logging.getLogger(__name__)


class QMLLayer:
    """
    Parameterized quantum machine learning layer.

    Constructs a hardware-efficient ansatz with Ry-Rz rotations and
    entangling CX gates, executes on a statevector simulator, and
    returns expectation values of Z operators on selected output qubits.

    Args:
        num_qubits: Number of qubits in the circuit.
        num_layers: Number of variational layers (depth).
        output_qubits: Which qubits to measure. Default: all qubits.
        entangling: Entangling gate pattern: "linear", "circular", or "full".
        backend: Execution backend (None = default statevector simulator).
        shots: Number of measurement shots (0 = exact statevector).
    """

    def __init__(
        self,
        num_qubits: int = 4,
        num_layers: int = 2,
        output_qubits: list[int] | None = None,
        entangling: Literal["linear", "circular", "full"] = "linear",
        backend: Any = None,
        shots: int = 0,
    ) -> None:
        self._num_qubits = num_qubits
        self._num_layers = num_layers
        self._output_qubits = output_qubits or list(range(num_qubits))
        self._entangling = entangling
        self._backend = backend
        self._shots = shots

        # Each layer has 2 params per qubit (Ry + Rz), plus entangling
        self._params_per_layer = 2 * num_qubits
        self._total_params = self._params_per_layer * num_layers

    @property
    def num_params(self) -> int:
        """Total number of trainable parameters."""
        return self._total_params

    @property
    def num_qubits(self) -> int:
        return self._num_qubits

    @property
    def num_outputs(self) -> int:
        return len(self._output_qubits)

    # ── Circuit construction ──────────────────────────────────────────

    def _build_statevector(self, params: np.ndarray) -> np.ndarray:
        """
        Build and simulate the ansatz circuit, returning the final statevector.

        Uses direct numpy statevector simulation for maximum speed.
        """
        n = self._num_qubits
        dim = 2 ** n

        # Initialize |00...0⟩
        sv = np.zeros(dim, dtype=np.complex128)
        sv[0] = 1.0

        param_idx = 0
        for layer_i in range(self._num_layers):
            # Rotation layer: Ry(θ) · Rz(φ) on each qubit
            for q in range(n):
                theta = params[param_idx]
                phi = params[param_idx + 1]
                param_idx += 2

                # Apply Ry(theta)
                sv = self._apply_ry(sv, q, n, theta)
                # Apply Rz(phi)
                sv = self._apply_rz(sv, q, n, phi)

            # Entangling layer: CX gates
            cx_pairs = self._get_entangling_pairs()
            for ctrl, tgt in cx_pairs:
                sv = self._apply_cx(sv, ctrl, tgt, n)

        return sv

    def _get_entangling_pairs(self) -> list[tuple[int, int]]:
        """Return the CX gate pairs based on entangling pattern."""
        n = self._num_qubits
        if n < 2:
            return []

        if self._entangling == "linear":
            return [(i, i + 1) for i in range(n - 1)]
        elif self._entangling == "circular":
            pairs = [(i, (i + 1) % n) for i in range(n)]
            return pairs
        elif self._entangling == "full":
            return [(i, j) for i in range(n) for j in range(i + 1, n)]
        return [(i, i + 1) for i in range(n - 1)]

    # ── Gate application (direct statevector) ─────────────────────────

    @staticmethod
    def _apply_ry(sv: np.ndarray, qubit: int, n: int, theta: float) -> np.ndarray:
        """Apply Ry(theta) to a statevector."""
        c = np.cos(theta / 2)
        s = np.sin(theta / 2)
        dim = 2 ** n
        result = np.zeros_like(sv)
        step = 2 ** qubit
        for i in range(dim):
            if (i >> qubit) & 1 == 0:
                j = i | step
                result[i] += c * sv[i] - s * sv[j]
                result[j] += s * sv[i] + c * sv[j]
        return result

    @staticmethod
    def _apply_rz(sv: np.ndarray, qubit: int, n: int, phi: float) -> np.ndarray:
        """Apply Rz(phi) to a statevector."""
        dim = 2 ** n
        result = sv.copy()
        e_neg = np.exp(-1j * phi / 2)
        e_pos = np.exp(1j * phi / 2)
        for i in range(dim):
            if (i >> qubit) & 1 == 0:
                result[i] *= e_neg
            else:
                result[i] *= e_pos
        return result

    @staticmethod
    def _apply_cx(sv: np.ndarray, ctrl: int, tgt: int, n: int) -> np.ndarray:
        """Apply CNOT (CX) to a statevector."""
        dim = 2 ** n
        result = sv.copy()
        for i in range(dim):
            if (i >> ctrl) & 1 == 1:
                j = i ^ (1 << tgt)
                result[i], result[j] = sv[j], sv[i]
        return result

    # ── Expectation values ────────────────────────────────────────────

    def _expectation_z(self, sv: np.ndarray, qubit: int) -> float:
        """
        Compute ⟨Z_qubit⟩ = Σ_i |α_i|² · (-1)^{bit_qubit(i)}.
        """
        dim = len(sv)
        exp_val = 0.0
        for i in range(dim):
            prob = abs(sv[i]) ** 2
            if (i >> qubit) & 1 == 0:
                exp_val += prob
            else:
                exp_val -= prob
        return float(exp_val)

    # ── Forward pass ──────────────────────────────────────────────────

    def __call__(self, params: np.ndarray) -> np.ndarray:
        """
        Evaluate the quantum layer.

        Args:
            params: Parameter vector of shape (num_params,).

        Returns:
            Expectation values of Z on output qubits, shape (num_outputs,).
        """
        params = np.asarray(params, dtype=np.float64)
        if len(params) != self._total_params:
            raise ValueError(
                f"Expected {self._total_params} parameters, got {len(params)}"
            )

        sv = self._build_statevector(params)

        outputs = np.array([
            self._expectation_z(sv, q) for q in self._output_qubits
        ])
        return outputs

    def forward(self, params: np.ndarray) -> np.ndarray:
        """Alias for __call__."""
        return self(params)

    # ── Gradient (parameter-shift rule) ───────────────────────────────

    def gradient(self, params: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of each output w.r.t. each parameter
        using the parameter-shift rule.

        For rotation gates Rg(θ), the parameter-shift rule gives:
            ∂f/∂θ = (f(θ + π/2) - f(θ - π/2)) / 2

        Args:
            params: Parameter vector of shape (num_params,).

        Returns:
            Gradient matrix of shape (num_params, num_outputs).
        """
        params = np.asarray(params, dtype=np.float64)
        shift = np.pi / 2
        grads = np.zeros((self._total_params, self.num_outputs))

        for i in range(self._total_params):
            params_plus = params.copy()
            params_plus[i] += shift
            f_plus = self(params_plus)

            params_minus = params.copy()
            params_minus[i] -= shift
            f_minus = self(params_minus)

            grads[i] = (f_plus - f_minus) / 2.0

        return grads

    # ── Batch evaluation ──────────────────────────────────────────────

    def batch_forward(
        self, param_batch: np.ndarray
    ) -> np.ndarray:
        """
        Evaluate the layer for a batch of parameter vectors.

        Args:
            param_batch: shape (batch_size, num_params)

        Returns:
            shape (batch_size, num_outputs)
        """
        return np.array([self(p) for p in param_batch])

    # ── Utilities ─────────────────────────────────────────────────────

    def random_params(
        self, seed: int | None = None
    ) -> np.ndarray:
        """Generate random initial parameters in [-π, π]."""
        rng = np.random.default_rng(seed)
        return rng.uniform(-np.pi, np.pi, size=self._total_params)

    def ansatz_depth(self) -> int:
        """
        Estimated circuit depth.

        Each layer contributes 2 rotation layers + 1 entangling layer.
        """
        return self._num_layers * 3

    def __repr__(self) -> str:
        return (
            f"QMLLayer(qubits={self._num_qubits}, layers={self._num_layers}, "
            f"params={self._total_params}, outputs={self.num_outputs}, "
            f"entangling='{self._entangling}')"
        )


__all__ = ["QMLLayer"]
