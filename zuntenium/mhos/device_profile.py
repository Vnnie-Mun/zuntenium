# zuntenium/mhos/device_profile.py
"""
DeviceProfile — Exhaustive hardware capability model.

Every quantum device registered with MHOS has exactly one DeviceProfile.
Profiles are updated dynamically via background refresh (queue times,
calibration data, error rates) and used by HardwareSelector for scoring.

Mathematical conventions:
  T1, T2         : nanoseconds
  error_rate     : probability ∈ [0, 1]
  cost_per_shot  : USD (micro-dollars internally for precision)
  fidelity_est   : mean gate fidelity = 1 - mean(single-qubit errors)
"""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field
from typing import Any


class DeviceStatus(str, enum.Enum):
    ONLINE      = "online"
    DEGRADED    = "degraded"    # available but calibration is stale / error rates high
    MAINTENANCE = "maintenance"
    OFFLINE     = "offline"
    UNKNOWN     = "unknown"


@dataclass
class CouplingEdge:
    """Directed edge in the coupling map (control, target)."""
    control: int
    target:  int
    cx_error_rate: float = 0.0   # 2-qubit gate error on this edge


@dataclass
class QubitProperties:
    """Per-qubit calibration data."""
    index:        int
    T1_ns:        float = 0.0    # energy relaxation time
    T2_ns:        float = 0.0    # dephasing time
    readout_err:  float = 0.0    # P(0|1) + P(1|0) average
    frequency_GHz: float = 0.0


# GateErrorMap: gate_name → error_rate (per-qubit errors averaged)
GateErrorMap = dict[str, float]


@dataclass
class DeviceProfile:
    """
    Complete hardware capability descriptor for one quantum device.

    Attributes mirror IBM Quantum's V2 backend properties, extended to cover
    AWS Braket, IonQ, and Rigetti device APIs.

    All error rates are probabilities ∈ [0,1]. Cost in USD/shot.
    """
    # ── Identity ────────────────────────────────────────────────────────────
    device_id:      str                        # unique: "ibm_kyoto", "ionq_aria-1"
    provider:       str                        # "ibm" | "braket" | "ionq" | "rigetti"
    display_name:   str                        # human label
    backend_type:   str = "gate_based"         # "gate_based" | "annealing" | "photonic"

    # ── Qubit topology ───────────────────────────────────────────────────────
    num_qubits:     int = 0
    coupling_map:   list[CouplingEdge] = field(default_factory=list)
    qubit_props:    list[QubitProperties] = field(default_factory=list)
    all_to_all:     bool = False               # True for trapped-ion (IonQ)

    # ── Native gate set ──────────────────────────────────────────────────────
    native_gates:   list[str] = field(default_factory=list)
    # e.g. ["cx","id","rz","sx","x"] for IBM, ["ms","gpi","gpi2"] for IonQ

    # ── Aggregate error rates ────────────────────────────────────────────────
    gate_errors:      GateErrorMap = field(default_factory=dict)
    mean_1q_error:    float = 0.0     # averaged across qubits and 1q gates
    mean_2q_error:    float = 0.0     # averaged across edges and 2q gates
    mean_readout_err: float = 0.0
    fidelity_estimate: float = 1.0   # 1 - mean_1q_error (quick proxy)

    # ── Coherence ────────────────────────────────────────────────────────────
    mean_T1_ns:     float = 0.0
    mean_T2_ns:     float = 0.0

    # ── Operational status ───────────────────────────────────────────────────
    status:          DeviceStatus = DeviceStatus.UNKNOWN
    queue_depth:     int   = 0          # jobs currently in queue
    queue_time_s:    float = 0.0        # estimated wait in seconds
    avg_gate_time_ns: float = 0.0       # average gate duration

    # ── Economics ────────────────────────────────────────────────────────────
    cost_per_shot:   float = 0.0        # USD per shot
    free_tier_shots: int   = 0          # shots available free per month
    priority_premium: float = 0.0      # multiplier for priority queue

    # ── Features ─────────────────────────────────────────────────────────────
    max_shots:       int = 100_000
    max_circuits_per_job: int = 1
    supports_parametric: bool = True
    supports_mid_circuit_measure: bool = False
    supports_reset: bool = False
    supports_delay: bool = False
    supported_memory: bool = False

    # ── Refresh tracking ─────────────────────────────────────────────────────
    last_calibrated_epoch: float = 0.0
    last_refreshed_epoch:  float = 0.0
    refresh_interval_s:    float = 300.0   # refresh every 5 min

    # ── Raw metadata ─────────────────────────────────────────────────────────
    raw_properties: dict[str, Any] = field(default_factory=dict)

    # ── Computed properties ───────────────────────────────────────────────────

    @property
    def is_available(self) -> bool:
        return self.status in (DeviceStatus.ONLINE, DeviceStatus.DEGRADED)

    @property
    def is_stale(self) -> bool:
        """True if calibration data is more than 24 hours old."""
        return time.time() - self.last_calibrated_epoch > 86_400

    @property
    def needs_refresh(self) -> bool:
        return time.time() - self.last_refreshed_epoch > self.refresh_interval_s

    def estimated_fidelity_for_circuit(
        self, n_1q_gates: int, n_2q_gates: int, depth: int
    ) -> float:
        """
        Estimate circuit execution fidelity using gate error propagation.

        F_circuit ≈ (1 - ε_1q)^{n_1q} · (1 - ε_2q)^{n_2q} · (1 - ε_r)^{n_qubits}

        where ε_1q, ε_2q, ε_r are mean single-qubit, two-qubit, and readout errors.
        This is a lower bound (ignores correlated errors).
        """
        import math
        e1 = self.mean_1q_error
        e2 = self.mean_2q_error
        er = self.mean_readout_err

        f = ((1 - e1) ** max(n_1q_gates, 0)
             * (1 - e2) ** max(n_2q_gates, 0)
             * (1 - er) ** max(self.num_qubits, 1))
        return float(max(0.0, min(1.0, f)))

    def topology_match_score(self, circuit_edges: set[tuple[int, int]]) -> float:
        """
        Score ∈ [0, 1] indicating how well the circuit's 2-qubit connectivity
        matches the device coupling map. 1.0 = all circuit edges are native.

        For all-to-all devices (IonQ): always 1.0.
        """
        if self.all_to_all:
            return 1.0
        if not circuit_edges:
            return 1.0

        native_edges = {(e.control, e.target) for e in self.coupling_map}
        native_edges |= {(e.target, e.control) for e in self.coupling_map}

        matched = sum(1 for e in circuit_edges if e in native_edges)
        return float(matched) / len(circuit_edges)

    def estimated_cost(self, shots: int) -> float:
        """Estimated USD cost for ``shots`` shots on this device."""
        billable = max(0, shots - self.free_tier_shots)
        return billable * self.cost_per_shot

    def update_from_refresh(self, data: dict) -> None:
        """Update dynamic fields from a provider refresh response."""
        if "queue_depth"  in data: self.queue_depth  = int(data["queue_depth"])
        if "queue_time_s" in data: self.queue_time_s = float(data["queue_time_s"])
        if "status"       in data:
            try:
                self.status = DeviceStatus(data["status"])
            except ValueError:
                self.status = DeviceStatus.UNKNOWN
        if "mean_1q_error"     in data: self.mean_1q_error = float(data["mean_1q_error"])
        if "mean_2q_error"     in data: self.mean_2q_error = float(data["mean_2q_error"])
        if "mean_readout_err"  in data: self.mean_readout_err = float(data["mean_readout_err"])
        if "mean_T1_ns"        in data: self.mean_T1_ns = float(data["mean_T1_ns"])
        if "mean_T2_ns"        in data: self.mean_T2_ns = float(data["mean_T2_ns"])
        self.fidelity_estimate = max(0.0, 1.0 - self.mean_1q_error)
        self.last_refreshed_epoch = time.time()

    def __repr__(self) -> str:
        return (f"DeviceProfile({self.device_id} [{self.provider}] "
                f"n={self.num_qubits} status={self.status.value} "
                f"F≈{self.fidelity_estimate:.3f} q={self.queue_depth})")
