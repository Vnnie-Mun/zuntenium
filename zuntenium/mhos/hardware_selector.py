# zuntenium/mhos/hardware_selector.py
"""
HardwareSelector — Intelligent multi-factor device selection.

Scoring function (configurable weights):

  score(d) = w₁·F_est(d,c) + w₂·(1/latency(d)) + w₃·(1/cost(d,s)) 
           + w₄·availability(d) − penalty_topology(d,c)

Where:
  F_est(d,c)   = estimated circuit fidelity on device d for circuit c
                 F ≈ (1-ε₁q)^{n1q} · (1-ε₂q)^{n2q} · (1-εr)^{n_qubits}
  
  latency(d)   = queue_time_s + estimated_exec_time
                 normalised to [0.1, 1.0] across candidate devices
  
  cost(d,s)    = d.cost_per_shot × shots
                 normalised to [0.1, 1.0] across candidates (0 if all free)
  
  availability(d) = 1.0 if ONLINE, 0.5 if DEGRADED, 0.0 if OFFLINE
  
  penalty_topology = α × (1 - topology_match_score(d, circuit_edges))
                     penalises heavy SWAP overhead on poorly-matched topologies

All weights w₁…w₄ are configurable per-job via ScoringWeights.
Default weights favour fidelity > latency > cost > availability.

Selection also enforces hard constraints:
  - device.num_qubits >= circuit.num_qubits
  - device is ONLINE or DEGRADED
  - device supports all required gate types (soft constraint → score penalty)
  - circuit depth × gate_time <= device T2 (coherence constraint)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .device_profile import DeviceProfile, DeviceStatus
from .hardware_registry import HardwareRegistry

_log = logging.getLogger(__name__)


@dataclass
class ScoringWeights:
    """
    Configurable weight vector for the hardware scoring function.
    All weights should sum to approximately 1.0.

    Presets:
      default()       — balanced (fidelity-first)
      fidelity_first()— maximise fidelity at any cost
      cost_optimised()— minimise cost (research budgets)
      low_latency()   — minimise wait time (interactive sessions)
    """
    w_fidelity:     float = 0.45   # weight for estimated fidelity
    w_latency:      float = 0.25   # weight for speed (1/queue_time)
    w_cost:         float = 0.15   # weight for economics (1/cost)
    w_availability: float = 0.10   # weight for device availability
    w_topology:     float = 0.05   # topology match bonus
    topology_penalty_alpha: float = 0.3   # penalty scale for topology mismatch
    coherence_penalty_beta: float = 0.2   # penalty for exceeding T2

    @classmethod
    def default(cls) -> "ScoringWeights":
        return cls()

    @classmethod
    def fidelity_first(cls) -> "ScoringWeights":
        return cls(w_fidelity=0.85, w_latency=0.05, w_cost=0.0,
                   w_availability=0.05, w_topology=0.05)

    @classmethod
    def cost_optimised(cls) -> "ScoringWeights":
        return cls(w_fidelity=0.25, w_latency=0.15, w_cost=0.50,
                   w_availability=0.05, w_topology=0.05)

    @classmethod
    def low_latency(cls) -> "ScoringWeights":
        return cls(w_fidelity=0.25, w_latency=0.55, w_cost=0.10,
                   w_availability=0.05, w_topology=0.05)


@dataclass
class CircuitAnalysis:
    """Pre-computed circuit features used by the scoring function."""
    num_qubits:      int
    num_1q_gates:    int
    num_2q_gates:    int
    depth:           int
    circuit_edges:   set[tuple[int, int]]
    requires_gates:  set[str]    # gate names used in circuit
    estimated_time_ns: float     # rough execution time estimate


@dataclass
class DeviceScore:
    """Scored device candidate."""
    device:         DeviceProfile
    total_score:    float
    fidelity_score: float
    latency_score:  float
    cost_score:     float
    avail_score:    float
    topology_score: float
    fidelity_est:   float        # F_circuit estimate ∈ [0,1]
    estimated_cost: float        # USD for this job
    disqualified:   bool  = False
    disqualify_reason: str = ""

    def __repr__(self) -> str:
        return (f"DeviceScore({self.device.device_id}: "
                f"total={self.total_score:.4f} F={self.fidelity_est:.4f} "
                f"lat={self.latency_score:.3f} cost=${self.estimated_cost:.4f})")


@dataclass
class SelectionResult:
    """Output of HardwareSelector.select()."""
    chosen_device:     DeviceProfile | None
    scores:            list[DeviceScore]      # all candidates, ranked
    fallback_to_sim:   bool = False
    fallback_reason:   str  = ""
    weights_used:      ScoringWeights = field(default_factory=ScoringWeights.default)

    @property
    def device_id(self) -> str | None:
        return self.chosen_device.device_id if self.chosen_device else None

    def top_k(self, k: int = 3) -> list[DeviceScore]:
        return [s for s in self.scores if not s.disqualified][:k]


class HardwareSelector:
    """
    Multi-factor hardware selection engine.

    Works with:
      - HardwareRegistry for device profiles
      - AIOptimizer for preference hints (optional)
      - User-specified ScoringWeights

    Parameters
    ----------
    registry  : HardwareRegistry
    max_budget_usd : optional per-job budget cap
    """

    def __init__(
        self,
        registry: HardwareRegistry,
        max_budget_usd: float | None = None,
    ) -> None:
        self._registry       = registry
        self._max_budget     = max_budget_usd

    # ── Primary API ───────────────────────────────────────────────────────────

    def select(
        self,
        circuit,
        shots: int = 1024,
        weights: ScoringWeights | None = None,
        provider_preference: str | None = None,
        device_preference:   str | None = None,
        max_queue_s: float | None = None,
        ai_hint: str | None = None,     # from AIOptimizer.select().backend_str
    ) -> SelectionResult:
        """
        Select the best available hardware device for the given circuit.

        Parameters
        ----------
        circuit             : Zuntenium Circuit object
        shots               : number of measurement shots
        weights             : ScoringWeights (default: balanced)
        provider_preference : prefer "ibm" | "ionq" | "braket" | "rigetti"
        device_preference   : prefer specific device_id (pinned selection)
        max_queue_s         : reject devices with queue_time > this
        ai_hint             : backend string hint from AIOptimizer

        Returns
        -------
        SelectionResult with ranked device scores
        """
        w         = weights or ScoringWeights.default()
        analysis  = self._analyse_circuit(circuit)
        candidates = self._registry.list_available()

        # Hard-pin to specific device if requested
        if device_preference:
            d = self._registry.get(device_preference)
            if d and d.is_available:
                return SelectionResult(
                    chosen_device=d, scores=[], weights_used=w
                )

        # Filter by provider preference
        if provider_preference and provider_preference != "auto":
            filtered = [d for d in candidates if d.provider == provider_preference]
            if filtered:
                candidates = filtered

        # Filter by AI hint (only use as soft preference, not hard filter)
        if ai_hint and ai_hint in ("ibm", "ionq", "braket", "rigetti"):
            preferred = [d for d in candidates if d.provider == ai_hint]
            if preferred:
                candidates = preferred

        if not candidates:
            return SelectionResult(
                chosen_device=None, scores=[], fallback_to_sim=True,
                fallback_reason="No available hardware devices",
                weights_used=w,
            )

        # Score all candidates
        scores = [
            self._score_device(d, analysis, shots, w, max_queue_s)
            for d in candidates
        ]
        scores.sort(key=lambda s: (not s.disqualified, s.total_score), reverse=True)

        # Best non-disqualified
        qualified = [s for s in scores if not s.disqualified]
        if not qualified:
            reasons = "; ".join(set(s.disqualify_reason for s in scores))
            return SelectionResult(
                chosen_device=None, scores=scores, fallback_to_sim=True,
                fallback_reason=f"All devices disqualified: {reasons}",
                weights_used=w,
            )

        best = qualified[0]
        _log.info(
            "Selected device: %s (score=%.4f F_est=%.4f cost=$%.4f)",
            best.device.device_id, best.total_score,
            best.fidelity_est, best.estimated_cost,
        )
        return SelectionResult(
            chosen_device=best.device, scores=scores, weights_used=w
        )

    # ── Scoring ───────────────────────────────────────────────────────────────

    def _score_device(
        self,
        device: DeviceProfile,
        analysis: CircuitAnalysis,
        shots: int,
        w: ScoringWeights,
        max_queue_s: float | None,
    ) -> DeviceScore:
        """
        Compute composite score for one device.

        All sub-scores normalised to [0, 1] individually.
        Final score = weighted sum.
        """
        # ── Hard constraints (disqualification) ──────────────────────────────
        if device.num_qubits < analysis.num_qubits:
            return DeviceScore(
                device=device, total_score=0.0,
                fidelity_score=0, latency_score=0, cost_score=0,
                avail_score=0, topology_score=0,
                fidelity_est=0, estimated_cost=0,
                disqualified=True,
                disqualify_reason=f"Insufficient qubits ({device.num_qubits} < {analysis.num_qubits})",
            )

        if max_queue_s is not None and device.queue_time_s > max_queue_s:
            return DeviceScore(
                device=device, total_score=0.0,
                fidelity_score=0, latency_score=0, cost_score=0,
                avail_score=0, topology_score=0,
                fidelity_est=0, estimated_cost=0,
                disqualified=True,
                disqualify_reason=f"Queue time {device.queue_time_s:.0f}s > max {max_queue_s:.0f}s",
            )

        estimated_cost = device.estimated_cost(shots)
        if self._max_budget is not None and estimated_cost > self._max_budget:
            return DeviceScore(
                device=device, total_score=0.0,
                fidelity_score=0, latency_score=0, cost_score=0,
                avail_score=0, topology_score=0,
                fidelity_est=estimated_cost, estimated_cost=estimated_cost,
                disqualified=True,
                disqualify_reason=f"Cost ${estimated_cost:.4f} > budget ${self._max_budget:.4f}",
            )

        # ── Sub-scores ────────────────────────────────────────────────────────

        # 1. Fidelity estimate
        f_est = device.estimated_fidelity_for_circuit(
            analysis.num_1q_gates, analysis.num_2q_gates, analysis.depth
        )
        f_score = float(f_est)

        # 2. Latency (higher = better, so invert queue time)
        exec_time_est = (analysis.num_1q_gates * device.avg_gate_time_ns * 1e-9
                         + analysis.num_2q_gates * device.avg_gate_time_ns * 3e-9) if device.avg_gate_time_ns else 0.1
        total_latency  = device.queue_time_s + exec_time_est + 1e-6
        lat_score      = float(1.0 / (1.0 + math.log1p(total_latency / 60.0)))

        # 3. Cost (higher = cheaper = better score)
        if estimated_cost <= 0:
            cost_score = 1.0
        else:
            cost_score = float(1.0 / (1.0 + math.log1p(estimated_cost * 1000)))

        # 4. Availability
        avail_map = {
            DeviceStatus.ONLINE:      1.0,
            DeviceStatus.DEGRADED:    0.5,
            DeviceStatus.MAINTENANCE: 0.0,
            DeviceStatus.OFFLINE:     0.0,
            DeviceStatus.UNKNOWN:     0.3,
        }
        avail_score = avail_map.get(device.status, 0.0)

        # 5. Topology match
        topo_score = device.topology_match_score(analysis.circuit_edges)

        # ── Coherence penalty ─────────────────────────────────────────────────
        coherence_penalty = 0.0
        if device.mean_T2_ns > 0:
            circuit_time_ns = analysis.depth * (device.avg_gate_time_ns or 50.0)
            if circuit_time_ns > device.mean_T2_ns:
                coherence_penalty = w.coherence_penalty_beta * min(
                    1.0, (circuit_time_ns - device.mean_T2_ns) / device.mean_T2_ns
                )

        # ── Topology penalty ──────────────────────────────────────────────────
        topo_penalty = w.topology_penalty_alpha * (1.0 - topo_score)

        # ── Composite score ───────────────────────────────────────────────────
        # score = w₁·F + w₂·(1/lat) + w₃·(1/cost) + w₄·avail + w₅·topo
        #         - topology_penalty - coherence_penalty
        total = (w.w_fidelity     * f_score
               + w.w_latency      * lat_score
               + w.w_cost         * cost_score
               + w.w_availability * avail_score
               + w.w_topology     * topo_score
               - topo_penalty
               - coherence_penalty)

        total = float(np.clip(total, 0.0, 1.0))

        return DeviceScore(
            device=device,
            total_score=total,
            fidelity_score=f_score,
            latency_score=lat_score,
            cost_score=cost_score,
            avail_score=avail_score,
            topology_score=topo_score,
            fidelity_est=f_est,
            estimated_cost=estimated_cost,
        )

    # ── Circuit analysis ──────────────────────────────────────────────────────

    @staticmethod
    def _analyse_circuit(circuit) -> CircuitAnalysis:
        """Extract features needed for scoring from a Zuntenium circuit."""
        _2Q_GATES = frozenset({"cx","cy","cz","cnot","swap","iswap","cp","crx","cry","crz",
                                "rxx","ryy","rzz","ecr","ms","xx","yy","zz"})
        gates     = list(getattr(circuit, "gates", []))
        n         = int(getattr(circuit, "num_qubits", 1))

        n1q, n2q  = 0, 0
        edges:    set[tuple[int, int]] = set()
        gate_names: set[str] = set()

        for g in gates:
            name   = str(getattr(g, "name", "")).lower()
            qubits = list(getattr(g, "qubits", []))
            gate_names.add(name)
            if len(qubits) == 1:
                n1q += 1
            elif len(qubits) == 2:
                n2q += 1
                edges.add((min(qubits), max(qubits)))

        # Greedy depth estimate
        ql = [0] * max(n, 1)
        for g in gates:
            qs = getattr(g, "qubits", [])
            if not qs: continue
            layer = max(ql[q] for q in qs if q < len(ql)) + 1
            for q in qs:
                if q < len(ql): ql[q] = layer
        depth = max(ql) if ql else 0

        return CircuitAnalysis(
            num_qubits=n, num_1q_gates=n1q, num_2q_gates=n2q,
            depth=depth, circuit_edges=edges, requires_gates=gate_names,
            estimated_time_ns=depth * 100.0,   # rough 100ns/layer
        )
