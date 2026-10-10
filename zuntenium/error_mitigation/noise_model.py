# zuntenium/error_mitigation/noise_model.py
"""
Noise Model — Mathematical Foundation
======================================

All noise channels are expressed as completely-positive trace-preserving (CPTP)
maps via their Kraus operator representation:

    ε(ρ) = Σ_k  K_k ρ K_k†      subject to  Σ_k K_k† K_k = I

────────────────────────────────────────────────────────────────
1.  Depolarizing Channel  (single qubit, error probability p)

    ε_dep(ρ) = (1-p)ρ + (p/3)(XρX + YρY + ZρZ)

    Kraus operators:
        K_0 = √(1-p)·I,  K_1 = √(p/3)·X,  K_2 = √(p/3)·Y,  K_3 = √(p/3)·Z

    Two-qubit generalization (p per gate):
        16-term sum over all Pauli tensor products {I,X,Y,Z}⊗{I,X,Y,Z}

2.  Amplitude Damping  (energy relaxation, T1 process, γ = 1 - e^{-t/T1})

    K_0 = [[1, 0], [0, √(1-γ)]],  K_1 = [[0, √γ], [0, 0]]

    Physical meaning: |1⟩ → |0⟩ with probability γ

3.  Phase Damping  (dephasing, T2 process, λ = 1 - e^{-t/T2})

    K_0 = [[1, 0], [0, √(1-λ)]],  K_1 = [[0, 0], [0, √λ]]

    Superoperator effect: off-diagonal elements ρ_{01} → √(1-λ)·ρ_{01}

4.  Thermal Relaxation  (combined T1/T2, excited-state population p_e)

    γ_1 = 1 - e^{-t/T1}
    γ_φ = max(0, 1 - e^{-t/T_φ})   where  1/T2 = 1/(2T1) + 1/T_φ
    K_0 = [[1,        0         ], [0, √(1-γ_1)       ]]
    K_1 = [[0,        √γ_1      ], [0, 0               ]]
    K_2 = [[0,        0         ], [0, √(γ_φ(1-γ_1))  ]]  (pure dephasing part)

5.  Readout Error  (SPAM, classical assignment matrix A)

    P(m=0|true=0) = 1 - ε_0,   P(m=1|true=0) = ε_0
    P(m=0|true=1) = ε_1,       P(m=1|true=1) = 1 - ε_1

    A = [[1-ε_0,  ε_1  ],
         [ε_0,    1-ε_1]]

    p_meas = A · p_true     →     p_corrected = A^{-1} · p_meas

6.  Per-Gate Error  (gate-specific depolarizing probability)

    Applied after each specified gate type.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

_log = logging.getLogger(__name__)

# ── Pauli matrices ──────────────────────────────────────────────────────────
_I  = np.eye(2,  dtype=complex)
_X  = np.array([[0,1],[1,0]],   dtype=complex)
_Y  = np.array([[0,-1j],[1j,0]],dtype=complex)
_Z  = np.array([[1,0],[0,-1]],  dtype=complex)
PAULIS = (_I, _X, _Y, _Z)


# ============================================================================
# Kraus operator helpers
# ============================================================================

def depolarizing_kraus(p: float, n_qubits: int = 1) -> list[np.ndarray]:
    """
    Return Kraus operators for n-qubit depolarizing channel with error rate p.
    Channel: E(ρ) = (1 - p)ρ + p (I / 2^n).
    At p=1, produces the maximally mixed state I / 2^n.
    """
    p = float(np.clip(p, 0.0, 1.0))
    if n_qubits == 1:
        if p == 0.0:
            return [_I.copy()]
        scale_id = np.sqrt(1.0 - 0.75 * p)
        scale_err = np.sqrt(0.25 * p)
        return [scale_id * _I, scale_err * _X, scale_err * _Y, scale_err * _Z]
    elif n_qubits == 2:
        ops = []
        paulis_1q = [_I, _X, _Y, _Z]
        scale_identity = np.sqrt(1.0 - (15.0 / 16.0) * p)
        scale_error = np.sqrt(p / 16.0)
        for i, P1 in enumerate(paulis_1q):
            for j, P2 in enumerate(paulis_1q):
                kron = np.kron(P1, P2)
                scale = scale_identity if (i == 0 and j == 0) else scale_error
                ops.append(scale * kron)
        return ops
    else:
        raise ValueError(f"depolarizing_kraus: n_qubits must be 1 or 2, got {n_qubits}")


def amplitude_damping_kraus(gamma: float) -> list[np.ndarray]:
    """Kraus operators for amplitude damping with decay probability γ."""
    gamma = float(np.clip(gamma, 0.0, 1.0))
    K0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=complex)
    K1 = np.array([[0, np.sqrt(gamma)], [0, 0]],    dtype=complex)
    return [K0, K1]


def phase_damping_kraus(lam: float) -> list[np.ndarray]:
    """Kraus operators for phase damping (dephasing) with rate λ."""
    lam = float(np.clip(lam, 0.0, 1.0))
    K0 = np.array([[1, 0], [0, np.sqrt(1 - lam)]], dtype=complex)
    K1 = np.array([[0, 0], [0, np.sqrt(lam)]],     dtype=complex)
    return [K0, K1]


def thermal_relaxation_kraus(
    t_gate: float, T1: float, T2: float, excited_state_pop: float = 0.0
) -> list[np.ndarray]:
    """
    Kraus operators for combined T1/T2 thermal relaxation.

    Parameters
    ----------
    t_gate : float  — gate duration (ns)
    T1     : float  — energy relaxation time (ns)
    T2     : float  — total dephasing time (ns), T2 ≤ 2·T1
    excited_state_pop : float — steady-state excited population (default 0 = ground)
    """
    T1 = max(T1, 1e-9); T2 = max(T2, 1e-9)
    T2 = min(T2, 2 * T1)
    t_gate = max(t_gate, 0.0)

    gamma1 = 1 - np.exp(-t_gate / T1)
    gamma1 = float(np.clip(gamma1, 0.0, 1.0))

    # Pure dephasing rate: 1/T_phi = 1/T2 - 1/(2*T1)
    t_phi_inv = max(1 / T2 - 1 / (2 * T1), 0.0)
    t_phi = 1 / t_phi_inv if t_phi_inv > 0 else np.inf
    gamma_phi = 0.0 if np.isinf(t_phi) else float(np.clip(1 - np.exp(-t_gate / t_phi), 0.0, 1.0))

    p1 = excited_state_pop
    p0 = 1 - p1

    kraus_ops = []
    # K0: no-jump
    K0 = np.array([
        [np.sqrt(1 - p1 * gamma1),                 0                           ],
        [0,                          np.sqrt(1 - p0 * gamma1) * np.sqrt(1 - gamma_phi)],
    ], dtype=complex)
    kraus_ops.append(K0)

    # K1: decay from |1⟩ → |0⟩
    K1 = np.array([
        [0, np.sqrt(p0 * gamma1)],
        [0, 0],
    ], dtype=complex)
    kraus_ops.append(K1)

    # K2: excitation from |0⟩ → |1⟩ (if thermal equilibrium)
    K2 = np.array([
        [0, 0],
        [np.sqrt(p1 * gamma1), 0],
    ], dtype=complex)
    kraus_ops.append(K2)

    # K3: pure dephasing
    K3 = np.array([
        [0,                                                  0],
        [0, np.sqrt((1 - p0 * gamma1) * gamma_phi)],
    ], dtype=complex)
    kraus_ops.append(K3)

    # Validate CPTP: Σ K†K ≈ I
    check = sum(K.conj().T @ K for K in kraus_ops)
    if not np.allclose(check, np.eye(2), atol=1e-6):
        _log.debug("ThermalRelaxation CPTP check deviation: max=%.2e", np.max(np.abs(check - np.eye(2))))

    return kraus_ops


def apply_kraus(rho: np.ndarray, kraus_ops: list[np.ndarray]) -> np.ndarray:
    """Apply CPTP map ε(ρ) = Σ_k K_k ρ K_k† to density matrix ρ."""
    result = np.zeros_like(rho, dtype=complex)
    for K in kraus_ops:
        result += K @ rho @ K.conj().T
    # Enforce hermiticity and trace normalization
    result = (result + result.conj().T) / 2
    tr = np.trace(result).real
    if tr > 1e-10:
        result /= tr
    return result


# ============================================================================
# Noise channel dataclasses
# ============================================================================

@dataclass
class DepolarizingNoise:
    """Single- or two-qubit depolarizing noise with error probability p."""
    p: float          # error probability ∈ [0, 1]
    n_qubits: int = 1

    def kraus_ops(self) -> list[np.ndarray]:
        return depolarizing_kraus(self.p, self.n_qubits)

    def __post_init__(self):
        if not 0 <= self.p <= 1:
            raise ValueError(f"Depolarizing p must be in [0,1], got {self.p}")


@dataclass
class AmplitudeDampingNoise:
    """Amplitude damping (T1 relaxation) with decay probability γ."""
    gamma: float      # decay probability ∈ [0, 1]

    def kraus_ops(self) -> list[np.ndarray]:
        return amplitude_damping_kraus(self.gamma)


@dataclass
class PhaseDampingNoise:
    """Phase damping (T2 dephasing) with rate λ."""
    lam: float        # dephasing rate ∈ [0, 1]

    def kraus_ops(self) -> list[np.ndarray]:
        return phase_damping_kraus(self.lam)


@dataclass
class ThermalRelaxationNoise:
    """Combined T1/T2 thermal relaxation."""
    t_gate: float     # gate duration (ns)
    T1: float         # relaxation time (ns)
    T2: float         # dephasing time (ns)
    excited_state_pop: float = 0.0

    def kraus_ops(self) -> list[np.ndarray]:
        return thermal_relaxation_kraus(
            self.t_gate, self.T1, self.T2, self.excited_state_pop
        )


@dataclass
class ReadoutNoise:
    """
    SPAM readout assignment error per qubit.

    Assignment matrix A[i, j] = P(measure i | true state j)
    """
    epsilon_0: float = 0.01   # P(measure 1 | true 0)
    epsilon_1: float = 0.01   # P(measure 0 | true 1)

    def assignment_matrix(self) -> np.ndarray:
        """2×2 column-stochastic assignment matrix."""
        e0, e1 = float(np.clip(self.epsilon_0, 0, 1)), float(np.clip(self.epsilon_1, 0, 1))
        return np.array([
            [1 - e0,  e1    ],
            [e0,      1 - e1],
        ], dtype=float)

    def apply(self, p_true: np.ndarray) -> np.ndarray:
        """Apply readout noise: p_meas = A · p_true."""
        return self.assignment_matrix() @ p_true


@dataclass
class GateNoise:
    """Per-gate-type depolarizing error rates."""
    gate_errors: dict[str, float] = field(default_factory=dict)
    # e.g. {"cx": 0.01, "h": 0.001, "rx": 0.002}

    def get_rate(self, gate_name: str) -> float:
        return self.gate_errors.get(gate_name.lower(), 0.0)


# ============================================================================
# NoiseModel — unified container
# ============================================================================

@dataclass
class NoiseModel:
    """
    Unified noise model for the Zuntenium EME.

    Stores per-qubit and global noise channels, readout errors,
    and gate-specific error rates. All fields are optional;
    unset fields mean zero noise for that channel.

    Parameters
    ----------
    depolarizing         : global depolarizing noise (applied to all qubits)
    amplitude_damping    : global amplitude damping
    phase_damping        : global phase damping
    thermal_relaxation   : global thermal relaxation
    readout              : global readout noise
    gate_noise           : per-gate-type error rates
    qubit_depolarizing   : per-qubit overrides {qubit_idx: DepolarizingNoise}
    qubit_readout        : per-qubit readout overrides {qubit_idx: ReadoutNoise}
    time_dependent       : if True, scale noise by circuit depth
    name                 : human-readable label
    metadata             : arbitrary dict for hardware calibration data
    """
    depolarizing:       DepolarizingNoise | None  = None
    amplitude_damping:  AmplitudeDampingNoise | None = None
    phase_damping:      PhaseDampingNoise | None   = None
    thermal_relaxation: ThermalRelaxationNoise | None = None
    readout:            ReadoutNoise | None          = None
    gate_noise:         GateNoise | None             = None
    qubit_depolarizing: dict[int, DepolarizingNoise] = field(default_factory=dict)
    qubit_readout:      dict[int, ReadoutNoise]       = field(default_factory=dict)
    time_dependent:     bool                          = False
    name:               str                           = "default"
    metadata:           dict[str, Any]                = field(default_factory=dict)

    @property
    def is_noiseless(self) -> bool:
        """True if all channels are None/empty (zero noise)."""
        return (
            self.depolarizing is None
            and self.amplitude_damping is None
            and self.phase_damping is None
            and self.thermal_relaxation is None
            and self.readout is None
            and not self.qubit_depolarizing
            and not self.qubit_readout
        )

    def get_qubit_kraus(self, qubit: int) -> list[list[np.ndarray]]:
        """
        Return list of Kraus sets to apply to ``qubit`` (in application order).
        Each element is one CPTP channel's Kraus operators.
        """
        channels: list[list[np.ndarray]] = []

        # Per-qubit override takes priority over global
        dep = self.qubit_depolarizing.get(qubit, self.depolarizing)
        if dep is not None:
            channels.append(dep.kraus_ops())

        if self.amplitude_damping is not None:
            channels.append(self.amplitude_damping.kraus_ops())

        if self.phase_damping is not None:
            channels.append(self.phase_damping.kraus_ops())

        if self.thermal_relaxation is not None:
            channels.append(self.thermal_relaxation.kraus_ops())

        return channels

    def get_readout_for_qubit(self, qubit: int) -> ReadoutNoise | None:
        return self.qubit_readout.get(qubit, self.readout)

    def get_gate_error(self, gate_name: str) -> float:
        if self.gate_noise is None:
            return 0.0
        return self.gate_noise.get_rate(gate_name)

    def summary(self) -> str:
        lines = [f"NoiseModel '{self.name}':"]
        if self.depolarizing:
            lines.append(f"  Depolarizing p={self.depolarizing.p:.4f}")
        if self.amplitude_damping:
            lines.append(f"  AmplitudeDamping γ={self.amplitude_damping.gamma:.4f}")
        if self.phase_damping:
            lines.append(f"  PhaseDamping λ={self.phase_damping.lam:.4f}")
        if self.thermal_relaxation:
            tr = self.thermal_relaxation
            lines.append(f"  ThermalRelax T1={tr.T1}ns T2={tr.T2}ns t_gate={tr.t_gate}ns")
        if self.readout:
            lines.append(f"  Readout ε0={self.readout.epsilon_0:.4f} ε1={self.readout.epsilon_1:.4f}")
        if self.gate_noise:
            lines.append(f"  GateNoise: {self.gate_noise.gate_errors}")
        return "\n".join(lines)


# ============================================================================
# NoiseModelBuilder — fluent API
# ============================================================================

class NoiseModelBuilder:
    """
    Fluent builder for constructing NoiseModel objects.

    Example
    -------
    model = (NoiseModelBuilder()
             .with_depolarizing(p=0.005)
             .with_readout(epsilon_0=0.01, epsilon_1=0.02)
             .with_thermal_relaxation(t_gate=50, T1=50_000, T2=70_000)
             .build())
    """

    def __init__(self) -> None:
        self._kwargs: dict[str, Any] = {}

    def with_depolarizing(self, p: float, n_qubits: int = 1) -> "NoiseModelBuilder":
        self._kwargs["depolarizing"] = DepolarizingNoise(p=p, n_qubits=n_qubits)
        return self

    def with_amplitude_damping(self, gamma: float) -> "NoiseModelBuilder":
        self._kwargs["amplitude_damping"] = AmplitudeDampingNoise(gamma=gamma)
        return self

    def with_phase_damping(self, lam: float) -> "NoiseModelBuilder":
        self._kwargs["phase_damping"] = PhaseDampingNoise(lam=lam)
        return self

    def with_thermal_relaxation(
        self, t_gate: float, T1: float, T2: float, excited_state_pop: float = 0.0
    ) -> "NoiseModelBuilder":
        self._kwargs["thermal_relaxation"] = ThermalRelaxationNoise(
            t_gate=t_gate, T1=T1, T2=T2, excited_state_pop=excited_state_pop
        )
        return self

    def with_readout(self, epsilon_0: float = 0.01, epsilon_1: float = 0.01) -> "NoiseModelBuilder":
        self._kwargs["readout"] = ReadoutNoise(epsilon_0=epsilon_0, epsilon_1=epsilon_1)
        return self

    def with_gate_noise(self, **gate_errors: float) -> "NoiseModelBuilder":
        self._kwargs["gate_noise"] = GateNoise(gate_errors=dict(gate_errors))
        return self

    def with_qubit_depolarizing(self, qubit: int, p: float) -> "NoiseModelBuilder":
        qd = self._kwargs.setdefault("qubit_depolarizing", {})
        qd[qubit] = DepolarizingNoise(p=p)
        return self

    def with_qubit_readout(self, qubit: int, epsilon_0: float, epsilon_1: float) -> "NoiseModelBuilder":
        qr = self._kwargs.setdefault("qubit_readout", {})
        qr[qubit] = ReadoutNoise(epsilon_0=epsilon_0, epsilon_1=epsilon_1)
        return self

    def with_time_dependent(self, enabled: bool = True) -> "NoiseModelBuilder":
        self._kwargs["time_dependent"] = enabled
        return self

    def with_name(self, name: str) -> "NoiseModelBuilder":
        self._kwargs["name"] = name
        return self

    def build(self) -> NoiseModel:
        return NoiseModel(**self._kwargs)


# ── Preset noise models ─────────────────────────────────────────────────────

def low_noise_model() -> NoiseModel:
    """Typical near-term device: ~0.1% gate error, ~1% readout."""
    return (NoiseModelBuilder()
            .with_depolarizing(p=0.001)
            .with_readout(epsilon_0=0.005, epsilon_1=0.01)
            .with_gate_noise(cx=0.005, h=0.0005, t=0.001)
            .with_name("low_noise")
            .build())


def medium_noise_model() -> NoiseModel:
    """Current NISQ devices: ~0.5% gate error, ~3% readout."""
    return (NoiseModelBuilder()
            .with_depolarizing(p=0.005)
            .with_amplitude_damping(gamma=0.003)
            .with_phase_damping(lam=0.004)
            .with_readout(epsilon_0=0.02, epsilon_1=0.03)
            .with_gate_noise(cx=0.01, h=0.001, t=0.002, rx=0.001)
            .with_name("medium_noise")
            .build())


def high_noise_model() -> NoiseModel:
    """Early/poor-quality NISQ: ~2% gate error, ~10% readout."""
    return (NoiseModelBuilder()
            .with_depolarizing(p=0.02)
            .with_amplitude_damping(gamma=0.02)
            .with_phase_damping(lam=0.03)
            .with_thermal_relaxation(t_gate=100, T1=10_000, T2=8_000)
            .with_readout(epsilon_0=0.08, epsilon_1=0.12)
            .with_gate_noise(cx=0.05, h=0.005, t=0.01)
            .with_name("high_noise")
            .build())
