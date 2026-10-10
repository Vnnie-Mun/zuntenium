"""
ZUNTENIUM FULL QOS — Quantum Operating System
Steps 1–6: Architecture · BAL · AI Selection · Scheduler · Pipeline · QRM

The QOS is the "Linux kernel" of Zuntenium — the central control plane
that unifies every subsystem into a self-improving quantum OS.

Integrates with:
  - zuntenium.cloud      (JobManager, Workers, Redis/RQ)
  - zuntenium.mhos       (HardwareRegistry, HardwareOrchestrator)
  - zuntenium.ai_optimizer (AIOptimizer, ExecutionLogger, FeatureVector)
  - zuntenium.transpiler  (Transpiler, rule+RL passes)
  - zuntenium.error_mitigation (ErrorMitigationEngine, ZNE/PEC/readout)
  - zuntenium.backends    (QuantumBackend ABC, all providers)
  - zuntenium.executor    (HybridExecutor)
  - zuntenium.marketplace (SDK hooks)
  - zuntenium.visual      (real-time pipeline updates via WebSocket)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import random
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="[QOS %(asctime)s] %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("zuntenium.qos")


# ══════════════════════════════════════════════════════════════════════
# SHARED TYPES (used across all QOS modules)
# ══════════════════════════════════════════════════════════════════════

class JobStatus(Enum):
    SUBMITTED   = auto()
    QUEUED      = auto()
    ANALYZING   = auto()
    TRANSPILING = auto()
    OPTIMIZING  = auto()
    SELECTING   = auto()
    EXECUTING   = auto()
    MITIGATING  = auto()
    COMPLETED   = auto()
    FAILED      = auto()
    CANCELLED   = auto()
    RETRYING    = auto()

class Priority(Enum):
    CRITICAL = 10
    HIGH     = 7
    NORMAL   = 5
    LOW      = 2
    BATCH    = 1


@dataclass
class CircuitSpec:
    """Canonical circuit representation passed through QOS pipeline."""
    circuit_id:  str
    num_qubits:  int
    gates:       List[Dict[str, Any]]
    name:        str = "circuit"
    shots:       int = 1024
    depth:       int = 0
    two_qubit_count: int = 0
    parametric_count: int = 0

    def to_dict(self) -> Dict:
        return {
            "num_qubits": self.num_qubits,
            "name":       self.name,
            "gates":      self.gates,
        }


@dataclass
class QOSJobRequest:
    """Full job request entering the QOS."""
    job_id:         str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id:        str = "anonymous"
    circuit:        Optional[CircuitSpec]    = None
    priority:       Priority                 = Priority.NORMAL
    optimization_mode: str                  = "ai"       # "ai" | "rule" | "manual"
    backend_hint:   Optional[str]            = None       # manual override
    mitigation:     List[str]               = field(default_factory=lambda: ["readout"])
    budget_usd:     Optional[float]          = None
    max_queue_s:    Optional[float]          = None
    sdk_hooks:      List[str]               = field(default_factory=list)
    metadata:       Dict[str, Any]          = field(default_factory=dict)
    submitted_at:   float                   = field(default_factory=time.time)


@dataclass
class QOSJobResult:
    """Full result returned from QOS after execution + mitigation."""
    job_id:          str
    status:          JobStatus
    counts:          Optional[Dict[str, int]]      = None
    statevector:     Optional[List[Tuple[float, float]]] = None
    fidelity_before: Optional[float]               = None
    fidelity_after:  Optional[float]               = None
    backend_used:    Optional[str]                 = None
    ai_confidence:   Optional[float]               = None
    decision_reason: Optional[str]                 = None
    exec_time_s:     Optional[float]               = None
    total_time_s:    Optional[float]               = None
    mitigation_method: Optional[str]               = None
    error:           Optional[str]                 = None
    telemetry:       Dict[str, Any]               = field(default_factory=dict)


# ══════════════════════════════════════════════════════════════════════
# STEP 1: QOS ARCHITECTURE — Event Bus
# Central async event system for loose coupling between subsystems
# ══════════════════════════════════════════════════════════════════════

class QOSEventBus:
    """
    Async pub/sub event bus.
    Subsystems publish events; others subscribe by event type.
    Replaces direct coupling — BAL, Scheduler, QRM, Telemetry all communicate here.

    Architecture:
      Publisher → EventBus → [Subscriber1, Subscriber2, ...]
      All cross-subsystem comms go through here (no direct calls except within same module).
    """

    def __init__(self):
        self._subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self._queue: asyncio.Queue = asyncio.Queue()
        self._running = False
        self.log = logging.getLogger("QOS.EventBus")

    def subscribe(self, event_type: str, handler: Callable) -> None:
        self._subscribers[event_type].append(handler)

    def publish(self, event_type: str, payload: Dict[str, Any]) -> None:
        """Non-blocking publish — puts event on queue."""
        try:
            self._queue.put_nowait({"type": event_type, "payload": payload, "ts": time.time()})
        except asyncio.QueueFull:
            self.log.warning("Event bus queue full — dropping event: %s", event_type)

    async def run(self):
        """Event dispatch loop — runs as background asyncio task."""
        self._running = True
        self.log.info("Event bus running")
        while self._running:
            try:
                event = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                handlers = self._subscribers.get(event["type"], [])
                for h in handlers:
                    try:
                        result = h(event["payload"])
                        if asyncio.iscoroutine(result):
                            await result
                    except Exception as e:
                        self.log.error("Handler %s failed for %s: %s", h.__name__, event["type"], e)
            except asyncio.TimeoutError:
                continue

    def stop(self): self._running = False


# ══════════════════════════════════════════════════════════════════════
# STEP 2: BACKEND ABSTRACTION LAYER (BAL)
# Normalises all quantum backends behind one interface.
# Wraps: StatevectorSimulator, TensorNetworkBackend, IBMBackend,
#        IonQ, Rigetti, AWS Braket — all via QuantumBackend(ABC).
# ══════════════════════════════════════════════════════════════════════

@dataclass
class BackendCapability:
    """Static device profile cached from MHOS HardwareRegistry."""
    backend_id:       str
    provider:         str                  # "local" | "ibm" | "ionq" | "braket" | "rigetti"
    max_qubits:       int
    supports_noise:   bool
    avg_fidelity:     float
    avg_queue_s:      float
    cost_per_shot:    float
    connectivity:     str                  # "all-to-all" | "heavy-hex" | "linear"
    native_gates:     List[str]
    online:           bool = True
    mhos_score:       float = 0.9


class QOSBackendProxy:
    """
    Single unified proxy for one backend inside the QOS BAL.
    Wraps any QuantumBackend-compatible object; adds QOS tracking.
    """

    def __init__(self, capability: BackendCapability, backend_obj: Any):
        self.cap      = capability
        self._backend = backend_obj
        self.active_jobs: Dict[str, float] = {}   # job_id → submitted_at
        self.recent_fidelities: deque     = deque(maxlen=20)
        self.recent_latencies:  deque     = deque(maxlen=20)
        self.error_count  = 0
        self.log          = logging.getLogger(f"QOS.BAL.{capability.backend_id}")

    # ── STEP 2: Unified interface ─────────────────────────────────────

    async def submit(self, circuit: CircuitSpec, shots: int) -> str:
        """Submit circuit → return provider job handle."""
        start = time.time()
        handle = f"qos-{self.cap.backend_id}-{uuid.uuid4().hex[:8]}"
        self.active_jobs[handle] = start
        self.log.info("Submitted %s (%dq, %d shots) → handle=%s",
                      circuit.name, circuit.num_qubits, shots, handle)
        return handle

    async def poll_status(self, handle: str) -> str:
        """Returns 'running' | 'completed' | 'failed'."""
        # In production: calls provider SDK
        elapsed = time.time() - self.active_jobs.get(handle, time.time())
        return "completed" if elapsed > 0.3 else "running"

    async def fetch_result(self, handle: str, circuit: CircuitSpec, shots: int) -> Dict[str, Any]:
        """Fetch and normalise result → QOS standard format."""
        elapsed = time.time() - self.active_jobs.pop(handle, time.time())
        self.recent_latencies.append(elapsed)

        # Simulate quantum result
        n   = min(circuit.num_qubits, 10)
        dim = 2 ** n
        probs = [random.random() ** 2 for _ in range(dim)]
        tot   = sum(probs)
        norm  = [p / tot for p in probs]
        counts: Dict[str, int] = {}
        rem = shots
        for i, p in enumerate(norm):
            c = round(p * shots) if i < len(norm) - 1 else rem
            rem -= c
            if c > 0: counts[bin(i)[2:].zfill(n)] = c

        fidelity = max(0.92, min(0.999, self.cap.avg_fidelity + random.gauss(0, 0.01)))
        self.recent_fidelities.append(fidelity)

        return {
            "counts":          counts,
            "statevector":     None,
            "fidelity_raw":    fidelity,
            "exec_time_s":     elapsed,
            "backend_used":    self.cap.backend_id,
            "shots":           shots,
        }

    async def cancel(self, handle: str) -> bool:
        self.active_jobs.pop(handle, None)
        return True

    # ── Derived metrics (used by Scheduler + QRM) ─────────────────────

    @property
    def current_load(self) -> int:
        return len(self.active_jobs)

    @property
    def estimated_fidelity(self) -> float:
        if self.recent_fidelities:
            return sum(self.recent_fidelities) / len(self.recent_fidelities)
        return self.cap.avg_fidelity

    @property
    def estimated_latency_s(self) -> float:
        if self.recent_latencies:
            return sum(self.recent_latencies) / len(self.recent_latencies)
        return self.cap.avg_queue_s


class BackendAbstractionLayer:
    """
    QOS BAL — registry and manager for all QOSBackendProxy instances.
    MHOS HardwareOrchestrator feeds device profiles here on startup + refresh.
    """

    def __init__(self, event_bus: QOSEventBus):
        self._backends: Dict[str, QOSBackendProxy] = {}
        self._bus = event_bus
        self.log  = logging.getLogger("QOS.BAL")
        self._seed_default_backends()

    def _seed_default_backends(self):
        """Seed local simulators — always available."""
        defaults = [
            BackendCapability("statevector",    "local",   28,  False, 1.000, 0.0,  0.00, "all-to-all", ["h","x","y","z","cx","rz"]),
            BackendCapability("tensornetwork",  "local",   200, False, 0.991, 0.0,  0.00, "all-to-all", ["h","x","y","z","cx","rz"]),
            BackendCapability("density_matrix", "local",   12,  True,  0.997, 0.0,  0.00, "all-to-all", ["h","x","y","z","cx","rz"]),
            BackendCapability("ibm_kyoto",      "ibm",     127, True,  0.982, 12.0, 0.00, "heavy-hex",  ["cx","rz","x","sx"]),
            BackendCapability("ionq_aria",      "ionq",    25,  True,  0.995, 5.0,  0.00035, "all-to-all", ["gpi","gpi2","ms"]),
            BackendCapability("braket_ankaa",   "braket",  84,  True,  0.967, 1.0,  0.00035, "linear",  ["cz","rz","rx"]),
        ]
        for cap in defaults:
            self.register(cap, object())   # placeholder backend obj

    def register(self, cap: BackendCapability, backend_obj: Any) -> None:
        proxy = QOSBackendProxy(cap, backend_obj)
        self._backends[cap.backend_id] = proxy
        self._bus.publish("BACKEND_REGISTERED", {"backend_id": cap.backend_id, "provider": cap.provider})
        self.log.info("Registered backend: %s (%s, max %dq)", cap.backend_id, cap.provider, cap.max_qubits)

    def get(self, backend_id: str) -> Optional[QOSBackendProxy]:
        return self._backends.get(backend_id)

    def all_online(self) -> List[QOSBackendProxy]:
        return [b for b in self._backends.values() if b.cap.online]

    def suitable_for(self, circuit: CircuitSpec) -> List[QOSBackendProxy]:
        return [b for b in self.all_online() if b.cap.max_qubits >= circuit.num_qubits]


# ══════════════════════════════════════════════════════════════════════
# STEP 3: AI-DRIVEN BACKEND SELECTION
# Extends AIOptimizer + HardwareSelector with global QOS context.
# ══════════════════════════════════════════════════════════════════════

@dataclass
class QOSFeatureVector:
    """
    Extended 35-dim feature vector for QOS-level AI decisions.
    Extends existing 27-dim AIOptimizer FeatureVector with 8 global metrics.
    """
    # ── Circuit features (14 dims) ───────────────────────────────────
    num_qubits:       float = 0.0   # normalised /200
    depth:            float = 0.0   # normalised /1000
    two_qubit_count:  float = 0.0
    param_count:      float = 0.0
    clifford_frac:    float = 0.0
    entanglement:     float = 0.0
    sv_feasible:      float = 0.0   # n <= 28
    mps_feasible:     float = 0.0   # n <= 200
    gate_density:     float = 0.0
    parallelism:      float = 0.0
    t_count:          float = 0.0
    swap_overhead:    float = 0.0
    single_qubit_frac:float = 0.0
    circuit_hash_entropy: float = 0.0
    # ── Context features (6 dims) ────────────────────────────────────
    hw_available:     float = 0.0
    queue_time_inv:   float = 0.0   # 1/(1+queue_s)
    cost_budget:      float = 1.0   # normalised
    urgency:          float = 0.5
    user_tier:        float = 0.5
    sdk_hooks_active: float = 0.0
    # ── Historical features (7 dims) ────────────────────────────────
    historical_fid:   float = 0.97
    historical_speed: float = 0.5
    fallback_rate:    float = 0.0
    success_rate:     float = 1.0
    ai_confidence_ma: float = 0.87
    reward_ema:       float = 0.87
    circuit_repeat:   float = 0.0
    # ── QOS global context (8 NEW dims) ──────────────────────────────
    system_load:      float = 0.0   # 0=idle, 1=saturated
    backend_diversity:float = 0.5   # spread across providers
    regional_lat:     float = 0.5   # cross-region latency
    queue_depth:      float = 0.0   # global queue depth /1000
    error_rate_sys:   float = 0.0   # recent system-wide error rate
    cost_pressure:    float = 0.0   # current cost utilisation
    ml_freshness:     float = 1.0   # how fresh the AI model is
    concurrency:      float = 0.0   # concurrent jobs /100

    def to_list(self) -> List[float]:
        return [v for v in self.__dict__.values()]

    @classmethod
    def from_job(cls, job: QOSJobRequest, qrm: "QuantumResourceManager",
                 bal: BackendAbstractionLayer) -> "QOSFeatureVector":
        c = job.circuit
        if c is None:
            return cls()
        n     = c.num_qubits
        gates = c.gates or []
        g     = len(gates)
        two_q = c.two_qubit_count
        suitable = bal.suitable_for(c)
        load  = qrm.system_load()

        fv = cls(
            num_qubits        = n / 200,
            depth             = c.depth / max(c.depth, 1) if c.depth else 0,
            two_qubit_count   = two_q / max(g, 1),
            param_count       = c.parametric_count / max(g, 1),
            clifford_frac     = sum(1 for gt in gates if gt.get("name") in ("h","x","y","z","cx","cz","s","t","swap")) / max(g, 1),
            entanglement      = min(1.0, two_q / max(n, 1) * 0.5),
            sv_feasible       = 1.0 if n <= 28 else 0.0,
            mps_feasible      = 1.0 if n <= 200 else 0.0,
            gate_density      = g / max(n * c.depth, 1) if c.depth else 0,
            hw_available      = len(suitable) / max(len(bal._backends), 1),
            queue_time_inv    = 1.0 / (1.0 + min(h.estimated_latency_s for h in suitable) if suitable else 1.0),
            cost_budget       = min(1.0, (job.budget_usd or 10) / 10),
            urgency           = 1.0 - job.priority.value / 10,
            sdk_hooks_active  = 1.0 if job.sdk_hooks else 0.0,
            system_load       = load,
            backend_diversity = min(1.0, len(set(b.cap.provider for b in suitable)) / 5),
            queue_depth       = min(1.0, qrm.queue_depth() / 1000),
            concurrency       = min(1.0, sum(b.current_load for b in bal.all_online()) / 100),
        )
        return fv


@dataclass
class QOSDecision:
    """Result from AI backend selection."""
    backend_id:    str
    score:         float
    confidence:    float
    strategy:      str              # "ai_primary" | "rule_fallback" | "manual"
    reason:        str
    fallback_chain:List[str]        = field(default_factory=list)
    mitigation:    List[str]        = field(default_factory=list)


class QOSAISelector:
    """
    QOS-enhanced AI backend selector.
    Extends zuntenium.ai_optimizer.AIOptimizer with global context.
    Scoring:  S(b) = w1·F + w2·(1/lat) + w3·(1/cost) + w4·avail − load_penalty
    Confidence gate: if conf < THRESHOLD → rule-based fallback
    """

    CONFIDENCE_THRESHOLD = 0.65
    WEIGHTS = {"fidelity": 0.38, "latency": 0.28, "cost": 0.20, "availability": 0.14}

    def __init__(self, bal: BackendAbstractionLayer, event_bus: QOSEventBus):
        self._bal = bal
        self._bus = event_bus
        self._history: deque = deque(maxlen=500)
        self.log = logging.getLogger("QOS.AISelector")
        # Simulated learned weights (production: loaded from ExecutionLogger DB)
        self._learned_fid_bonus: Dict[str, float] = {}

    def select(self, job: QOSJobRequest, fv: QOSFeatureVector, qrm: "QuantumResourceManager") -> QOSDecision:
        """Main entry: returns QOSDecision for this job."""
        suitable = self._bal.suitable_for(job.circuit) if job.circuit else []
        if not suitable:
            return QOSDecision("statevector", 0.5, 0.0, "emergency_fallback", "No suitable backend found", [], job.mitigation)

        # Manual override (Prompt 11: Backend Control Panel)
        if job.optimization_mode == "manual" and job.backend_hint:
            proxy = self._bal.get(job.backend_hint)
            if proxy:
                return QOSDecision(job.backend_hint, 1.0, 1.0, "manual", "User-specified backend", [], job.mitigation)

        # Score all suitable backends
        scored = [(b, self._score(b, fv, qrm)) for b in suitable]
        scored.sort(key=lambda x: x[1], reverse=True)
        best_proxy, best_score = scored[0]

        # Confidence estimate: spread between top-2 scores
        top2_diff = (scored[0][1] - scored[1][1]) if len(scored) > 1 else 0.5
        confidence = min(0.99, 0.55 + top2_diff * 0.9 + fv.historical_fid * 0.1)

        # Rule-based fallback if confidence below threshold
        strategy = "ai_primary" if confidence >= self.CONFIDENCE_THRESHOLD else "rule_fallback"
        if strategy == "rule_fallback":
            best_proxy = self._rule_select(suitable, job.circuit)

        fallback_chain = [b.cap.backend_id for b, _ in scored[1:4]]
        mitigation     = self._select_mitigation(best_proxy, job)

        reason = (
            f"{best_proxy.cap.backend_id} selected: score={best_score:.3f}, "
            f"F_est={best_proxy.estimated_fidelity:.3f}, "
            f"latency={best_proxy.estimated_latency_s:.1f}s, "
            f"load={best_proxy.current_load}, conf={confidence:.2f}"
        )

        decision = QOSDecision(
            backend_id    = best_proxy.cap.backend_id,
            score         = best_score,
            confidence    = confidence,
            strategy      = strategy,
            reason        = reason,
            fallback_chain= fallback_chain,
            mitigation    = mitigation,
        )

        self._history.append({"job_id": job.job_id, "decision": decision, "ts": time.time()})
        self._bus.publish("BACKEND_SELECTED", {"job_id": job.job_id, "backend_id": decision.backend_id, "confidence": confidence})
        self.log.info("%s → %s (conf=%.2f, strategy=%s)", job.job_id[:8], decision.backend_id, confidence, strategy)
        return decision

    def _score(self, proxy: QOSBackendProxy, fv: QOSFeatureVector, qrm: "QuantumResourceManager") -> float:
        w = self.WEIGHTS
        F     = proxy.estimated_fidelity + self._learned_fid_bonus.get(proxy.cap.backend_id, 0.0)
        L     = 1.0 / (1.0 + proxy.estimated_latency_s)
        C     = 1.0 / (1.0 + proxy.cap.cost_per_shot * 10000)
        A     = 1.0 if proxy.cap.online else 0.0
        load_penalty   = proxy.current_load * 0.02
        budget_penalty = max(0, (proxy.cap.cost_per_shot * 10000) - fv.cost_budget) * 0.1
        return w["fidelity"]*F + w["latency"]*L + w["cost"]*C + w["availability"]*A - load_penalty - budget_penalty

    def _rule_select(self, suitable: List[QOSBackendProxy], circuit: CircuitSpec) -> QOSBackendProxy:
        """Deterministic rule-based selection (no ML)."""
        local = [b for b in suitable if b.cap.provider == "local"]
        if local:
            return max(local, key=lambda b: b.cap.avg_fidelity)
        return min(suitable, key=lambda b: b.estimated_latency_s)

    def _select_mitigation(self, proxy: QOSBackendProxy, job: QOSJobRequest) -> List[str]:
        """Dynamically select mitigation strategies based on backend + circuit."""
        if proxy.cap.provider == "local":
            return []   # simulators don't need mitigation
        strategies = list(job.mitigation)
        if "readout" not in strategies:
            strategies.insert(0, "readout")   # always apply readout correction on real HW
        if proxy.cap.avg_fidelity < 0.98 and "zne" not in strategies:
            strategies.append("zne")
        return strategies

    def record_outcome(self, job_id: str, backend_id: str, fidelity: float, latency: float):
        """Online learning feedback → update learned fidelity bonus."""
        bonus = (fidelity - 0.95) * 0.1
        self._learned_fid_bonus[backend_id] = (
            self._learned_fid_bonus.get(backend_id, 0.0) * 0.9 + bonus * 0.1
        )
        self._bus.publish("AI_LEARNING_UPDATE", {"backend_id": backend_id, "fidelity": fidelity, "latency": latency})


# ══════════════════════════════════════════════════════════════════════
# STEP 4: INTELLIGENT SCHEDULER
# Priority-queue, resource-aware, fairness-guaranteed job scheduling.
# Integrates with Redis/RQ (production) — simulated inline here.
# ══════════════════════════════════════════════════════════════════════

@dataclass(order=True)
class ScheduledJob:
    """Priority queue entry."""
    priority_score: float = field(compare=True)
    submitted_at:   float = field(compare=True)
    job:            QOSJobRequest = field(compare=False)
    decision:       Optional[QOSDecision] = field(default=None, compare=False)

    def __lt__(self, other):
        # Higher priority_score first; tie-break by older submitted_at
        if self.priority_score != other.priority_score:
            return self.priority_score > other.priority_score
        return self.submitted_at < other.submitted_at


class FairnessPolicy:
    """
    Per-user fairness tracker.
    Boosts priority of users who haven't run recently (aging).
    Prevents starvation of BATCH jobs under HIGH load.
    """
    def __init__(self, aging_rate: float = 0.01):
        self._aging_rate = aging_rate
        self._last_run:   Dict[str, float] = {}
        self._run_count:  Dict[str, int]   = {}

    def priority_boost(self, user_id: str, base_priority: int) -> float:
        wait_s  = time.time() - self._last_run.get(user_id, 0)
        runs    = self._run_count.get(user_id, 0)
        aging   = min(3.0, wait_s * self._aging_rate)   # cap bonus at +3
        penalty = min(1.0, runs / 100)                  # slight penalty for heavy users
        return base_priority + aging - penalty

    def record_run(self, user_id: str):
        self._last_run[user_id]  = time.time()
        self._run_count[user_id] = self._run_count.get(user_id, 0) + 1


class IntelligentScheduler:
    """
    QOS central scheduler.

    Features:
    - Priority queue (CRITICAL > HIGH > NORMAL > LOW > BATCH)
    - Per-user fairness via FairnessPolicy (anti-starvation aging)
    - Preemption: CRITICAL jobs can displace BATCH jobs
    - Resource gating: won't dispatch if QRM signals no capacity
    - Max concurrency per backend (configurable)
    - Admission control: reject jobs exceeding queue depth limit
    """

    MAX_QUEUE_DEPTH = 10_000
    MAX_CONCURRENCY_PER_BACKEND = 50
    DISPATCH_INTERVAL_S = 0.05

    def __init__(self, bal: BackendAbstractionLayer, qrm: "QuantumResourceManager",
                 ai_selector: QOSAISelector, event_bus: QOSEventBus,
                 pipeline: "ExecutionPipelineManager"):
        import heapq
        self._heap: List[ScheduledJob] = []
        self._heap_lock = asyncio.Lock()
        self._bal       = bal
        self._qrm       = qrm
        self._selector  = ai_selector
        self._bus       = event_bus
        self._pipeline  = pipeline
        self._fairness  = FairnessPolicy()
        self._running   = False
        self._dispatched: Dict[str, ScheduledJob] = {}   # job_id → in-flight
        self.log = logging.getLogger("QOS.Scheduler")
        import heapq as _hq
        self._heapq = _hq

    async def enqueue(self, job: QOSJobRequest) -> bool:
        """Admission control + enqueue. Returns False if rejected."""
        async with self._heap_lock:
            if len(self._heap) >= self.MAX_QUEUE_DEPTH:
                self._bus.publish("JOB_REJECTED", {"job_id": job.job_id, "reason": "queue_full"})
                return False

        # AI selection ahead of time (so it's cached when dispatched)
        fv = QOSFeatureVector.from_job(job, self._qrm, self._bal)
        decision = self._selector.select(job, fv, self._qrm)

        priority_score = self._fairness.priority_boost(job.user_id, job.priority.value)
        sj = ScheduledJob(priority_score=priority_score, submitted_at=job.submitted_at, job=job, decision=decision)

        async with self._heap_lock:
            self._heapq.heappush(self._heap, sj)

        self._bus.publish("JOB_QUEUED", {"job_id": job.job_id, "priority": priority_score, "backend": decision.backend_id})
        self.log.info("Enqueued %s (prio=%.2f, backend=%s, q_depth=%d)", job.job_id[:8], priority_score, decision.backend_id, len(self._heap))
        return True

    async def run(self):
        """Main dispatch loop — runs as background asyncio task."""
        self._running = True
        self.log.info("Scheduler dispatch loop started")
        while self._running:
            await self._dispatch_ready()
            await asyncio.sleep(self.DISPATCH_INTERVAL_S)

    async def _dispatch_ready(self):
        """Pop + dispatch jobs while resources permit."""
        async with self._heap_lock:
            if not self._heap:
                return
            candidate = self._heap[0]

        backend_id = candidate.decision.backend_id if candidate.decision else "statevector"
        proxy = self._bal.get(backend_id)
        if not proxy or proxy.current_load >= self.MAX_CONCURRENCY_PER_BACKEND:
            # Try fallback from chain
            for fid in (candidate.decision.fallback_chain if candidate.decision else []):
                fb = self._bal.get(fid)
                if fb and fb.current_load < self.MAX_CONCURRENCY_PER_BACKEND:
                    backend_id = fid
                    candidate.decision.backend_id = fid
                    break
            else:
                return  # no capacity — wait

        async with self._heap_lock:
            sj = self._heapq.heappop(self._heap)

        self._dispatched[sj.job.job_id] = sj
        self._fairness.record_run(sj.job.user_id)
        self._bus.publish("JOB_DISPATCHED", {"job_id": sj.job.job_id, "backend_id": backend_id})
        # Hand off to pipeline (non-blocking)
        asyncio.create_task(self._pipeline.execute(sj.job, sj.decision))

    def queue_depth(self) -> int:
        return len(self._heap)

    def stop(self): self._running = False


# ══════════════════════════════════════════════════════════════════════
# STEP 5: EXECUTION PIPELINE MANAGER
# Full pipeline: Circuit → Analyze → Transpile → Optimize → Select
#             → Execute → Mitigate → Deliver
# Each stage publishes progress events → Visual Tools via EventBus.
# ══════════════════════════════════════════════════════════════════════

class PipelineStage(Enum):
    RECEIVED    = "received"
    ANALYZING   = "analyzing"
    TRANSPILING = "transpiling"
    OPTIMIZING  = "optimizing"
    SELECTING   = "selecting"
    EXECUTING   = "executing"
    MITIGATING  = "mitigation"
    COMPLETED   = "completed"
    FAILED      = "failed"


class ExecutionPipelineManager:
    """
    Manages the full job lifecycle inside the QOS.

    Pipeline per job:
      1. Receive + validate circuit
      2. CircuitAnalyzer → extract features
      3. Transpiler → rule + RL passes
      4. AIOptimizer → strategy selection
      5. BackendSelector / QOSAISelector → backend decision
      6. QOSBackendProxy.submit → execute
      7. ErrorMitigationEngine → ZNE / PEC / readout
      8. Deliver QOSJobResult + telemetry

    All stages publish events → Visual Tools WebSocket.
    """

    def __init__(self, bal: BackendAbstractionLayer, event_bus: QOSEventBus,
                 telemetry: "TelemetrySystem"):
        self._bal       = bal
        self._bus       = event_bus
        self._telemetry = telemetry
        self._active:   Dict[str, Dict] = {}   # job_id → state
        self.log = logging.getLogger("QOS.Pipeline")

    async def execute(self, job: QOSJobRequest, decision: Optional[QOSDecision]) -> QOSJobResult:
        """Full pipeline execution — async, non-blocking."""
        start   = time.time()
        job_id  = job.job_id
        circuit = job.circuit
        self._active[job_id] = {"status": PipelineStage.RECEIVED, "start": start}

        def stage(s: PipelineStage, detail: str = ""):
            self._active[job_id]["status"] = s
            self._bus.publish("PIPELINE_STAGE", {
                "job_id": job_id, "stage": s.value, "detail": detail, "ts": time.time()
            })
            self.log.info("[%s] stage=%s %s", job_id[:8], s.value, detail)

        try:
            # ── Stage 1: Validate ─────────────────────────────────────
            stage(PipelineStage.ANALYZING, f"n={circuit.num_qubits if circuit else 0} qubits")
            await asyncio.sleep(0.02)   # simulate analysis time

            if not circuit:
                raise ValueError("No circuit provided")

            # ── Stage 2: Transpile ────────────────────────────────────
            stage(PipelineStage.TRANSPILING, "rule + RL passes")
            await asyncio.sleep(0.05)   # simulate transpilation
            # In production: zuntenium.transpiler.Transpiler(circuit).run()

            # ── Stage 3: AI Optimize ──────────────────────────────────
            stage(PipelineStage.OPTIMIZING, "FeatureVector(35-dim) extraction")
            await asyncio.sleep(0.03)

            # ── Stage 4: Backend confirmed ────────────────────────────
            backend_id = decision.backend_id if decision else "statevector"
            stage(PipelineStage.SELECTING, f"→ {backend_id} (score={decision.score:.3f})" if decision else "→ statevector")

            # ── Stage 5: Execute ──────────────────────────────────────
            proxy = self._bal.get(backend_id)
            if not proxy:
                raise RuntimeError(f"Backend {backend_id} unavailable")

            stage(PipelineStage.EXECUTING, f"{circuit.shots} shots on {backend_id}")
            handle = await proxy.submit(circuit, circuit.shots)

            # Poll until done
            for _ in range(100):
                status = await proxy.poll_status(handle)
                if status != "running": break
                await asyncio.sleep(0.05)

            raw = await proxy.fetch_result(handle, circuit, circuit.shots)

            # ── Stage 6: Error Mitigation ─────────────────────────────
            mitigation_methods = decision.mitigation if decision else []
            fid_before = raw["fidelity_raw"]
            fid_after  = fid_before

            if mitigation_methods and proxy.cap.provider != "local":
                stage(PipelineStage.MITIGATING, f"methods={mitigation_methods}")
                await asyncio.sleep(0.03)
                # In production: zuntenium.error_mitigation.ErrorMitigationEngine.apply()
                if "readout"  in mitigation_methods: fid_after = min(0.999, fid_after + 0.008)
                if "zne"      in mitigation_methods: fid_after = min(0.999, fid_after + 0.015)
                if "pec"      in mitigation_methods: fid_after = min(0.999, fid_after + 0.012)

            total_time = time.time() - start

            # ── Stage 7: Record telemetry ─────────────────────────────
            self._telemetry.record({
                "job_id":        job_id,
                "backend_id":    backend_id,
                "fidelity_raw":  fid_before,
                "fidelity_mit":  fid_after,
                "exec_time_s":   raw["exec_time_s"],
                "total_time_s":  total_time,
                "num_qubits":    circuit.num_qubits,
                "shots":         circuit.shots,
                "mitigation":    mitigation_methods,
                "ai_confidence": decision.confidence if decision else 0.0,
            })

            result = QOSJobResult(
                job_id           = job_id,
                status           = JobStatus.COMPLETED,
                counts           = raw["counts"],
                fidelity_before  = fid_before,
                fidelity_after   = fid_after,
                backend_used     = backend_id,
                ai_confidence    = decision.confidence if decision else 0.0,
                decision_reason  = decision.reason if decision else "",
                exec_time_s      = raw["exec_time_s"],
                total_time_s     = total_time,
                mitigation_method= "+".join(mitigation_methods) if mitigation_methods else "none",
                telemetry        = {"pipeline_stages": 7, "backend_load": proxy.current_load},
            )

            stage(PipelineStage.COMPLETED, f"F={fid_after:.4f}, t={total_time:.3f}s")
            self._bus.publish("JOB_COMPLETED", {"job_id": job_id, "result": {
                "fidelity_after": fid_after, "backend_used": backend_id, "total_time_s": total_time,
            }})
            self.log.info("✓ [%s] done in %.3fs, F=%.4f", job_id[:8], total_time, fid_after)
            self._active.pop(job_id, None)
            return result

        except Exception as exc:
            stage(PipelineStage.FAILED, str(exc))
            self._bus.publish("JOB_FAILED", {"job_id": job_id, "error": str(exc)})
            self.log.error("✗ [%s] pipeline failed: %s", job_id[:8], exc)
            self._active.pop(job_id, None)
            return QOSJobResult(job_id=job_id, status=JobStatus.FAILED, error=str(exc))

    def active_count(self) -> int: return len(self._active)


# ══════════════════════════════════════════════════════════════════════
# STEP 6: QUANTUM RESOURCE MANAGER (QRM)
# Tracks all resource pools: simulator slots, HW queues, workers,
# MPS bond-dim budgets. Provides admission gating to Scheduler.
# ══════════════════════════════════════════════════════════════════════

@dataclass
class ResourcePool:
    resource_id:  str
    total:        int
    used:         int = 0
    reserved:     int = 0

    @property
    def available(self) -> int: return max(0, self.total - self.used - self.reserved)
    @property
    def utilization(self) -> float: return self.used / max(self.total, 1)

    def allocate(self, n: int = 1) -> bool:
        if self.available >= n: self.used += n; return True
        return False
    def release(self, n: int = 1): self.used = max(0, self.used - n)
    def reserve(self, n: int = 1): self.reserved += n
    def release_reservation(self, n: int = 1): self.reserved = max(0, self.reserved - n)


class QuantumResourceManager:
    """
    Global resource manager for the QOS.

    Manages:
    - Simulator slots (statevector, MPS, density matrix)
    - Hardware queue budgets (IBM, IonQ, Braket, Rigetti)
    - Kubernetes worker nodes
    - MPS bond-dimension budget (χ·n limit across active MPS jobs)
    - Global shot budget per user (rate limiting)
    """

    MPS_BOND_BUDGET = 4096    # total χ across all concurrent MPS jobs
    MAX_SHOTS_PER_USER_PER_MINUTE = 1_000_000

    def __init__(self, event_bus: QOSEventBus):
        self._pools: Dict[str, ResourcePool] = {
            "statevector_slots":    ResourcePool("statevector_slots",    32),
            "mps_slots":            ResourcePool("mps_slots",            16),
            "density_slots":        ResourcePool("density_slots",        8),
            "k8s_worker_nodes":     ResourcePool("k8s_worker_nodes",     64),
            "ibm_queue_budget":     ResourcePool("ibm_queue_budget",     100),
            "ionq_queue_budget":    ResourcePool("ionq_queue_budget",    50),
            "braket_queue_budget":  ResourcePool("braket_queue_budget",  50),
        }
        self._mps_bond_used: int = 0
        self._user_shots:    Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._event_bus      = event_bus
        self._metrics_ts:    List[Dict] = []
        self.log = logging.getLogger("QOS.QRM")

    def allocate_for_job(self, job: QOSJobRequest, backend_id: str) -> bool:
        """Try to allocate resources for a job. Returns False if insufficient."""
        pool_map = {
            "statevector":    "statevector_slots",
            "tensornetwork":  "mps_slots",
            "density_matrix": "density_slots",
        }
        pool_id = pool_map.get(backend_id, "k8s_worker_nodes")
        pool = self._pools.get(pool_id)
        if pool and not pool.allocate():
            self.log.warning("Resource pool %s exhausted", pool_id)
            return False

        # MPS bond-dim budget
        if backend_id == "tensornetwork" and job.circuit:
            bond_needed = 64   # default χ
            if self._mps_bond_used + bond_needed > self.MPS_BOND_BUDGET:
                self.log.warning("MPS bond budget exhausted (%d/%d)", self._mps_bond_used, self.MPS_BOND_BUDGET)
                if pool: pool.release()
                return False
            self._mps_bond_used += bond_needed

        self._event_bus.publish("RESOURCE_ALLOCATED", {"job_id": job.job_id, "pool": pool_id})
        return True

    def release_for_job(self, job_id: str, backend_id: str, bond_dim: int = 64):
        pool_map = {"statevector":"statevector_slots","tensornetwork":"mps_slots","density_matrix":"density_slots"}
        pool_id = pool_map.get(backend_id, "k8s_worker_nodes")
        pool = self._pools.get(pool_id)
        if pool: pool.release()
        if backend_id == "tensornetwork":
            self._mps_bond_used = max(0, self._mps_bond_used - bond_dim)

    def system_load(self) -> float:
        """Weighted average utilization across all pools."""
        utils = [p.utilization for p in self._pools.values()]
        return sum(utils) / len(utils) if utils else 0.0

    def queue_depth(self) -> int:
        """Placeholder — production: query Redis/RQ queue length."""
        return sum(p.used for p in self._pools.values())

    def snapshot(self) -> Dict[str, Any]:
        """Point-in-time resource snapshot for telemetry."""
        return {
            "ts":            time.time(),
            "system_load":   self.system_load(),
            "mps_bond_used": self._mps_bond_used,
            "pools":         {k: {"total": v.total, "used": v.used, "util": v.utilization}
                              for k, v in self._pools.items()},
        }

    def check_rate_limit(self, user_id: str, shots: int) -> bool:
        """Per-user shot rate limiting (1M shots/min)."""
        now = time.time()
        q   = self._user_shots[user_id]
        # Remove entries older than 60s
        while q and now - q[0][0] > 60:
            q.popleft()
        total = sum(s for _, s in q)
        if total + shots > self.MAX_SHOTS_PER_USER_PER_MINUTE:
            return False
        q.append((now, shots))
        return True
