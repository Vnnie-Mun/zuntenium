# zuntenium/ai_engine/features.py
"""
Feature Engineering for the Zuntenium AI Optimization Engine.

Extends CircuitAnalyzer output with runtime context and historical features
into a unified, typed FeatureVector suitable for ML inference.

Mathematical notation:
  s_t = [s_circuit | s_context | s_history] ∈ ℝ^d
  where d = len(FeatureVector.to_array())
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FeatureVector
# ---------------------------------------------------------------------------

@dataclass
class FeatureVector:
    """
    Fully typed feature vector used as input state s_t for both supervised
    model and RL policy.

    Sections
    --------
    Circuit features   — intrinsic properties of the quantum circuit
    Context features   — runtime / hardware environment at decision time
    Historical features — rolling statistics from ExecutionLogger

    All float64 for numerical stability in downstream models.
    """

    # ── Circuit features ───────────────────────────────────────────────────
    num_qubits: float = 0.0
    depth: float = 0.0
    gate_count: float = 0.0
    entanglement_score: float = 0.0          # fraction of 2-qubit gates
    clifford_fraction: float = 0.0           # fraction that are Clifford
    parametric_fraction: float = 0.0         # fraction that are parametric
    connectivity_density: float = 0.0        # edges / max_edges in qubit graph
    estimated_memory_mb: float = 0.0         # 2^n * 16 bytes (complex128)
    two_qubit_gate_count: float = 0.0
    single_qubit_gate_count: float = 0.0
    t_gate_count: float = 0.0               # T / Tdag gates (non-Clifford)
    max_gate_arity: float = 0.0
    circuit_volume: float = 0.0             # depth * num_qubits
    expected_statevector_feasible: float = 0.0  # binary: 1 if n<=28

    # ── Context features ───────────────────────────────────────────────────
    hardware_available: float = 0.0          # binary: IBM backend reachable
    queue_time_estimate_s: float = 0.0       # estimated IBM queue wait (s)
    current_system_load: float = 0.0         # normalised [0,1]
    noise_model_requested: float = 0.0       # binary
    shots_requested: float = 0.0            # normalised log10(shots)
    real_hardware_requested: float = 0.0     # binary: user wants physical QPU

    # ── Historical features ────────────────────────────────────────────────
    avg_exec_time_last_10: float = 0.0       # seconds
    success_rate_last_20: float = 1.0        # [0,1]
    prev_backend_statevector: float = 0.0    # one-hot
    prev_backend_tensornetwork: float = 0.0
    prev_backend_ibm: float = 0.0
    prev_backend_stabilizer: float = 0.0
    total_executions: float = 0.0           # log10(1 + n)

    # ── Metadata (NOT part of the array, used for explainability) ──────────
    feature_names: list[str] = field(default_factory=list, repr=False)
    raw_circuit_info: dict[str, Any] = field(default_factory=dict, repr=False)

    # ── Dimension constant ─────────────────────────────────────────────────
    DIM: int = field(default=27, init=False, repr=False)  # FIX BUG-006: was 21, actual is 27

    def __post_init__(self) -> None:
        # Populate feature_names if empty (used for SHAP / attribution)
        if not self.feature_names:
            self.feature_names = [
                "num_qubits", "depth", "gate_count", "entanglement_score",
                "clifford_fraction", "parametric_fraction", "connectivity_density",
                "estimated_memory_mb", "two_qubit_gate_count", "single_qubit_gate_count",
                "t_gate_count", "max_gate_arity", "circuit_volume",
                "expected_statevector_feasible",
                "hardware_available", "queue_time_estimate_s", "current_system_load",
                "noise_model_requested", "shots_requested", "real_hardware_requested",
                "avg_exec_time_last_10", "success_rate_last_20",
                "prev_backend_statevector", "prev_backend_tensornetwork",
                "prev_backend_ibm", "prev_backend_stabilizer",
                "total_executions",
            ]
        # Update DIM dynamically
        object.__setattr__(self, "DIM", len(self.feature_names))

    def to_array(self) -> np.ndarray:
        """Return ordered float64 numpy array (shape [DIM])."""
        return np.array([
            self.num_qubits,
            self.depth,
            self.gate_count,
            self.entanglement_score,
            self.clifford_fraction,
            self.parametric_fraction,
            self.connectivity_density,
            self.estimated_memory_mb,
            self.two_qubit_gate_count,
            self.single_qubit_gate_count,
            self.t_gate_count,
            self.max_gate_arity,
            self.circuit_volume,
            self.expected_statevector_feasible,
            self.hardware_available,
            self.queue_time_estimate_s,
            self.current_system_load,
            self.noise_model_requested,
            self.shots_requested,
            self.real_hardware_requested,
            self.avg_exec_time_last_10,
            self.success_rate_last_20,
            self.prev_backend_statevector,
            self.prev_backend_tensornetwork,
            self.prev_backend_ibm,
            self.prev_backend_stabilizer,
            self.total_executions,
        ], dtype=np.float64)

    def to_dict(self) -> dict[str, float]:
        """Serialisable dict for SQLite persistence."""
        return {name: float(val) for name, val in
                zip(self.feature_names, self.to_array())}

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "FeatureVector":
        """Reconstruct from a flat numpy array (for model inference)."""
        fv = cls()
        names = fv.feature_names
        if len(arr) != len(names):
            raise ValueError(
                f"Array length {len(arr)} != FeatureVector DIM {len(names)}"
            )
        for name, val in zip(names, arr):
            if hasattr(fv, name):
                object.__setattr__(fv, name, float(val))
        return fv

    @classmethod
    def from_dict(cls, d: dict[str, float]) -> "FeatureVector":
        fv = cls()
        for name, val in d.items():
            if hasattr(fv, name):
                object.__setattr__(fv, name, float(val))
        return fv


# ---------------------------------------------------------------------------
# FeatureExtractor
# ---------------------------------------------------------------------------

_CLIFFORD_GATES = frozenset({
    "h", "s", "sdg", "x", "y", "z", "cx", "cy", "cz", "swap",
    "sx", "sxdg", "id", "cnot",
})
_PARAMETRIC_GATES = frozenset({
    "rx", "ry", "rz", "u", "u1", "u2", "u3", "p", "cp", "crx", "cry", "crz",
    "rxx", "ryy", "rzz",
})
_T_GATES = frozenset({"t", "tdg"})
_TWO_QUBIT = frozenset({
    "cx", "cy", "cz", "cnot", "swap", "iswap", "cp", "crx", "cry", "crz",
    "rxx", "ryy", "rzz", "ecr", "dcx",
})


class FeatureExtractor:
    """
    Extracts a FeatureVector from a Circuit + runtime context + historical stats.

    Designed to be dependency-injected with an optional CircuitAnalyzer from
    zuntenium.adaptive. Falls back to internal lightweight analysis if that
    module is unavailable.

    Parameters
    ----------
    circuit_analyzer : optional CircuitAnalyzer instance
    execution_logger  : optional ExecutionLogger for historical stats
    """

    def __init__(
        self,
        circuit_analyzer=None,
        execution_logger=None,
    ) -> None:
        self._analyzer = circuit_analyzer
        self._logger = execution_logger
        self._log = logging.getLogger(self.__class__.__name__)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(
        self,
        circuit,
        *,
        hardware_available: bool = False,
        queue_time_estimate_s: float = 0.0,
        current_system_load: float = 0.0,
        noise_model_requested: bool = False,
        shots_requested: int = 1024,
        real_hardware_requested: bool = False,
        preferences: dict | None = None,
    ) -> FeatureVector:
        """
        Build a complete FeatureVector for ``circuit`` given runtime context.

        Parameters
        ----------
        circuit                 : zuntenium.quantum_core.Circuit
        hardware_available      : IBM backend currently reachable
        queue_time_estimate_s   : estimated IBM queue wait in seconds
        current_system_load     : normalised CPU/GPU load [0, 1]
        noise_model_requested   : user or policy requested noise simulation
        shots_requested         : number of measurement shots
        real_hardware_requested : user explicitly wants physical QPU
        preferences             : free-form dict passed by executor
        """
        prefs = preferences or {}

        # 1. Circuit features
        cf = self._circuit_features(circuit)

        # 2. Context features
        ctx = self._context_features(
            hardware_available=hardware_available or prefs.get("hardware_available", False),
            queue_time_estimate_s=queue_time_estimate_s or prefs.get("queue_time_estimate_s", 0.0),
            current_system_load=current_system_load,
            noise_model_requested=noise_model_requested or prefs.get("noise_model_requested", False),
            shots_requested=shots_requested or prefs.get("shots", 1024),
            real_hardware_requested=real_hardware_requested or prefs.get("real_hardware", False),
        )

        # 3. Historical features
        hist = self._historical_features(circuit)

        fv = FeatureVector(
            **cf,
            **ctx,
            **hist,
            raw_circuit_info={
                "name": getattr(circuit, "name", "unknown"),
                "num_qubits": cf["num_qubits"],
                "depth": cf["depth"],
            },
        )
        self._log.debug("Extracted features: %s", fv.to_dict())
        return fv

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _circuit_features(self, circuit) -> dict:
        """Extract circuit-intrinsic features. Works with or without CircuitAnalyzer."""
        try:
            return self._extract_direct(circuit)
        except Exception as exc:
            self._log.warning("Direct feature extraction failed: %s", exc)
            return self._zero_circuit_features()

    def _extract_direct(self, circuit) -> dict:
        gates = list(getattr(circuit, "gates", []))
        n = int(getattr(circuit, "num_qubits", 0))

        total = len(gates)
        two_q = sum(1 for g in gates if getattr(g, "name", "").lower() in _TWO_QUBIT)
        single_q = sum(
            1 for g in gates
            if len(getattr(g, "qubits", [])) == 1
        )
        clifford = sum(1 for g in gates if getattr(g, "name", "").lower() in _CLIFFORD_GATES)
        parametric = sum(1 for g in gates if getattr(g, "name", "").lower() in _PARAMETRIC_GATES)
        t_count = sum(1 for g in gates if getattr(g, "name", "").lower() in _T_GATES)
        max_arity = max(
            (len(getattr(g, "qubits", [])) for g in gates), default=0
        )

        # Depth estimation (topological layers)
        depth = self._estimate_depth(gates, n)

        # Connectivity density: unique qubit pairs in 2q gates / C(n,2)
        pairs = set()
        for g in gates:
            qs = getattr(g, "qubits", [])
            if len(qs) == 2:
                pairs.add(tuple(sorted(qs)))
        max_pairs = max(n * (n - 1) // 2, 1)
        conn_density = len(pairs) / max_pairs

        # Memory: complex128 statevector = 2^n * 16 bytes
        mem_mb = (2 ** n * 16) / (1024 ** 2) if n <= 30 else float("inf")

        entanglement = two_q / max(total, 1)
        clifford_frac = clifford / max(total, 1)
        parametric_frac = parametric / max(total, 1)

        return dict(
            num_qubits=float(n),
            depth=float(depth),
            gate_count=float(total),
            entanglement_score=float(entanglement),
            clifford_fraction=float(clifford_frac),
            parametric_fraction=float(parametric_frac),
            connectivity_density=float(conn_density),
            estimated_memory_mb=float(min(mem_mb, 1e6)),
            two_qubit_gate_count=float(two_q),
            single_qubit_gate_count=float(single_q),
            t_gate_count=float(t_count),
            max_gate_arity=float(max_arity),
            circuit_volume=float(depth * n),
            expected_statevector_feasible=float(n <= 28),
        )

    @staticmethod
    def _estimate_depth(gates, n_qubits: int) -> int:
        """Greedy layer-by-layer depth estimate in O(|gates|)."""
        qubit_layer = [0] * max(n_qubits, 1)
        for g in gates:
            qs = getattr(g, "qubits", [])
            if not qs:
                continue
            layer = max(qubit_layer[q] for q in qs if q < len(qubit_layer)) + 1
            for q in qs:
                if q < len(qubit_layer):
                    qubit_layer[q] = layer
        return max(qubit_layer) if qubit_layer else 0

    @staticmethod
    def _zero_circuit_features() -> dict:
        return dict(
            num_qubits=0.0, depth=0.0, gate_count=0.0, entanglement_score=0.0,
            clifford_fraction=0.0, parametric_fraction=0.0, connectivity_density=0.0,
            estimated_memory_mb=0.0, two_qubit_gate_count=0.0, single_qubit_gate_count=0.0,
            t_gate_count=0.0, max_gate_arity=0.0, circuit_volume=0.0,
            expected_statevector_feasible=0.0,
        )

    @staticmethod
    def _context_features(
        *,
        hardware_available: bool,
        queue_time_estimate_s: float,
        current_system_load: float,
        noise_model_requested: bool,
        shots_requested: int,
        real_hardware_requested: bool,
    ) -> dict:
        import math
        return dict(
            hardware_available=float(hardware_available),
            queue_time_estimate_s=float(queue_time_estimate_s),
            current_system_load=float(np.clip(current_system_load, 0.0, 1.0)),
            noise_model_requested=float(noise_model_requested),
            shots_requested=float(math.log10(max(shots_requested, 1))),
            real_hardware_requested=float(real_hardware_requested),
        )

    def _historical_features(self, circuit) -> dict:
        import math
        if self._logger is None:
            return dict(
                avg_exec_time_last_10=0.0, success_rate_last_20=1.0,
                prev_backend_statevector=0.0, prev_backend_tensornetwork=0.0,
                prev_backend_ibm=0.0, prev_backend_stabilizer=0.0,
                total_executions=0.0,
            )
        try:
            stats = self._logger.query_stats(
                num_qubits=int(getattr(circuit, "num_qubits", 0))
            )
            pb = stats.get("prev_backend", "statevector").lower()
            return dict(
                avg_exec_time_last_10=float(stats.get("avg_exec_time_last_10", 0.0)),
                success_rate_last_20=float(stats.get("success_rate_last_20", 1.0)),
                prev_backend_statevector=float("statevector" in pb),
                prev_backend_tensornetwork=float("tensor" in pb or "mps" in pb),
                prev_backend_ibm=float("ibm" in pb),
                prev_backend_stabilizer=float("stabilizer" in pb or "clifford" in pb),
                total_executions=float(math.log10(1 + stats.get("total_executions", 0))),
            )
        except Exception as exc:
            self._log.warning("Historical feature query failed: %s", exc)
            return dict(
                avg_exec_time_last_10=0.0, success_rate_last_20=1.0,
                prev_backend_statevector=0.0, prev_backend_tensornetwork=0.0,
                prev_backend_ibm=0.0, prev_backend_stabilizer=0.0,
                total_executions=0.0,
            )
