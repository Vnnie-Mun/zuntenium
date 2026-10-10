"""
NoiseModel — Noise channel simulation for quantum circuits.

Provides common noise channels as Kraus operators:
- Depolarizing channel
- Amplitude damping
- Phase damping (dephasing)
- Readout error
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class NoiseModel:
    """
    Noise model container. Attach to a backend to enable noisy simulation.

    Gate errors are stored as per-gate depolarizing probabilities.
    Readout errors are stored as per-qubit confusion matrices.

    Attributes:
        gate_errors:         {gate_name: error_probability}
        readout_errors:      {qubit_idx: [p(0|1), p(1|0)]}
        thermal_relaxation:  {qubit_idx: {'T1': ..., 'T2': ...}}
        kraus_channels:      {gate_name: list of Kraus matrices}
    """
    gate_errors: dict[str, float] = field(default_factory=dict)
    readout_errors: dict[int, list[float]] = field(default_factory=dict)
    thermal_relaxation: dict[int, dict[str, float]] = field(default_factory=dict)
    kraus_channels: dict[str, list[np.ndarray]] = field(default_factory=dict)

    # ── Factory methods ──────────────────────────────────────────────────────

    @classmethod
    def depolarizing(cls, error_rate: float, gates: list[str] | None = None) -> NoiseModel:
        """
        Create a uniform depolarizing noise model.

        For each gate G, applies the depolarizing channel:
          ε(ρ) = (1−p)ρ + (p/3)(XρX + YρY + ZρZ)   [single qubit]

        Args:
            error_rate: Depolarizing probability p ∈ [0, 1].
            gates: Gate names to apply noise to. None = common set.

        Returns:
            NoiseModel with depolarizing Kraus operators.
        """
        if not 0.0 <= error_rate <= 1.0:
            raise ValueError(f"error_rate must be in [0, 1], got {error_rate}")
        gates = gates or ["H", "X", "Y", "Z", "Rx", "Ry", "Rz", "CX", "CZ"]
        gate_errors = dict.fromkeys(gates, error_rate)

        # Build Kraus operators for depolarizing channel
        p = error_rate
        kraus = _depolarizing_kraus(p)
        kraus_channels = dict.fromkeys(gates, kraus)

        return cls(gate_errors=gate_errors, kraus_channels=kraus_channels)

    @classmethod
    def amplitude_damping(cls, gamma: float, gates: list[str] | None = None) -> NoiseModel:
        """
        Create an amplitude damping noise model (T1 relaxation).

        Kraus operators: K0 = [[1,0],[0,√(1-γ)]], K1 = [[0,√γ],[0,0]]

        Args:
            gamma: Damping parameter γ ∈ [0, 1]. γ=0 → no damping.
            gates: Gate names to apply noise to. None = common set.

        Returns:
            NoiseModel with amplitude damping Kraus operators.
        """
        if not 0.0 <= gamma <= 1.0:
            raise ValueError(f"gamma must be in [0, 1], got {gamma}")
        gates = gates or ["H", "X", "Y", "Z", "Rx", "Ry", "Rz"]
        gate_errors = dict.fromkeys(gates, gamma)
        kraus = _amplitude_damping_kraus(gamma)
        kraus_channels = dict.fromkeys(gates, kraus)
        return cls(gate_errors=gate_errors, kraus_channels=kraus_channels)

    @classmethod
    def phase_damping(cls, lambda_: float, gates: list[str] | None = None) -> NoiseModel:
        """
        Create a phase damping noise model (T2 dephasing).

        Kraus operators: K0 = [[1,0],[0,√(1-λ)]], K1 = [[0,0],[0,√λ]]

        Args:
            lambda_: Dephasing parameter λ ∈ [0, 1].
            gates: Gate names to apply noise to. None = common set.
        """
        if not 0.0 <= lambda_ <= 1.0:
            raise ValueError(f"lambda_ must be in [0, 1], got {lambda_}")
        gates = gates or ["H", "X", "Y", "Z", "Rx", "Ry", "Rz"]
        gate_errors = dict.fromkeys(gates, lambda_)
        kraus: list[np.ndarray] = [
            np.array([[1, 0], [0, np.sqrt(1 - lambda_)]], dtype=np.complex128),
            np.array([[0, 0], [0, np.sqrt(lambda_)]], dtype=np.complex128),
        ]
        return cls(gate_errors=gate_errors, kraus_channels=dict.fromkeys(gates, kraus))

    @classmethod
    def readout_error(cls, p0given1: float, p1given0: float,
                      qubits: list[int] | None = None,
                      num_qubits: int = 1) -> NoiseModel:
        """
        Create a readout confusion model.

        Args:
            p0given1: P(measuring 0 | state is 1) — mismeasurement rate.
            p1given0: P(measuring 1 | state is 0) — mismeasurement rate.
            qubits: Qubit indices to apply to. None = all (up to num_qubits).
            num_qubits: Total qubits if qubits=None.
        """
        target_qubits = qubits or list(range(num_qubits))
        errors = {q: [p0given1, p1given0] for q in target_qubits}
        return cls(readout_errors=errors)

    # ── Query methods ────────────────────────────────────────────────────────

    def is_empty(self) -> bool:
        """Return True if no noise is configured."""
        return (
            not self.gate_errors
            and not self.readout_errors
            and not self.thermal_relaxation
            and not self.kraus_channels
        )

    def get_kraus(self, gate_name: str) -> list[np.ndarray] | None:
        """Return Kraus operators for the given gate, or None."""
        return self.kraus_channels.get(gate_name)

    def add_gate_error(self, gate_name: str, error_rate: float) -> None:
        """Add a depolarizing error on a specific gate."""
        self.gate_errors[gate_name] = error_rate
        self.kraus_channels[gate_name] = _depolarizing_kraus(error_rate)

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate_errors": self.gate_errors,
            "readout_errors": self.readout_errors,
            "thermal_relaxation": self.thermal_relaxation,
        }

    def __repr__(self) -> str:
        return (
            f"NoiseModel(gates={list(self.gate_errors.keys())}, "
            f"readout_qubits={list(self.readout_errors.keys())})"
        )


# ── Kraus operator builders ──────────────────────────────────────────────────

def _depolarizing_kraus(p: float) -> list[np.ndarray]:
    """Single-qubit depolarizing Kraus operators."""
    # ε(ρ) = (1−p)ρ + (p/3)(XρX + YρY + ZρZ)
    c = np.sqrt(1 - p)
    s = np.sqrt(p / 3)
    eye = np.eye(2, dtype=np.complex128)
    pauli_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    pauli_y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    pauli_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    return [c * eye, s * pauli_x, s * pauli_y, s * pauli_z]


def _amplitude_damping_kraus(gamma: float) -> list[np.ndarray]:
    """Single-qubit amplitude damping Kraus operators."""
    K0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=np.complex128)
    K1 = np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=np.complex128)
    return [K0, K1]
