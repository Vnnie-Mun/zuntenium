# zuntenium_cloud/worker.py
"""
Execution Worker — stateless RQ task that runs the full Zuntenium pipeline.

Called by RQ workers on every dequeued job.  Each invocation:
  1. Loads the job from PostgreSQL
  2. Reconstructs the Circuit object from stored JSON
  3. Runs the full Zuntenium pipeline:
       CircuitAnalyzer → Transpiler → AIOptimizer → BackendSelector
       → Backend execution → ErrorMitigationEngine
  4. Stores the result in PostgreSQL
  5. Updates job status

Worker is completely stateless — all state is in DB + Redis.
Each Docker container runs N workers (RQ burst mode or forking).

Environment variables required:
  ZUNTENIUM_DATABASE_URL
  ZUNTENIUM_REDIS_URL
  ZUNTENIUM_WORKER_ID   (set by Kubernetes downward API)
  ZUNTENIUM_IBM_TOKEN   (optional, for IBM backend)
"""

from __future__ import annotations

import logging
import os
import socket
import time
from datetime import datetime, timezone
from typing import Any

_log = logging.getLogger(__name__)
_WORKER_ID = os.environ.get("ZUNTENIUM_WORKER_ID", socket.gethostname())


# ── Circuit reconstruction ────────────────────────────────────────────────────

def _build_circuit(circuit_json: dict):
    """
    Reconstruct a Zuntenium Circuit object from its JSON representation.
    Produces a canonical transpiler v2 Circuit with proper metadata, depth,
    and control/target qubit specifications.
    """
    num_qubits = circuit_json.get("num_qubits", 0)
    name = circuit_json.get("name", "cloud_job")
    raw_gates = circuit_json.get("gates", [])

    try:
        from zuntenium.transpiler.v2.core import Circuit, Gate
        circ = Circuit(num_qubits, name=name)
        for g in raw_gates:
            gname = str(g.get("name", "")).upper()
            qubits = tuple(int(q) for q in g.get("qubits", []))
            params = tuple(float(p) for p in g.get("params", []))
            if len(qubits) == 2 and gname in ("CX", "CNOT", "CZ", "CPHASE", "CRX", "CRY", "CRZ"):
                circ.add_gate(Gate(name=gname, targets=(qubits[1],), controls=(qubits[0],), params=params))
            else:
                circ.add_gate(Gate(name=gname, targets=qubits, params=params))
        return circ
    except Exception:
        # Fallback duck-typed dataclass
        from dataclasses import dataclass, field as dc_field

        @dataclass
        class _Gate:
            name: str
            qubits: list[int]
            params: list[float] = dc_field(default_factory=list)

            @property
            def is_measurement(self) -> bool:
                return self.name.upper() in ("M", "MEASURE", "MEAS")

            @property
            def is_single_qubit(self) -> bool:
                return len(self.qubits) == 1

            @property
            def targets(self) -> tuple[int, ...]:
                return tuple(self.qubits)

            @property
            def controls(self) -> tuple[int, ...]:
                return ()

        @dataclass
        class _Circuit:
            num_qubits: int
            gates: list
            name: str = "cloud_job"

            def depth(self) -> int:
                return len(self.gates)

            @property
            def num_gates(self) -> int:
                return len(self.gates)

        gates = [
            _Gate(
                name=g["name"],
                qubits=g["qubits"],
                params=g.get("params", []),
            )
            for g in raw_gates
        ]
        return _Circuit(num_qubits=num_qubits, gates=gates, name=name)


# ── Backend connector ─────────────────────────────────────────────────────────

def _get_backend_registry() -> dict:
    """
    Build the backend registry using existing Zuntenium QuantumBackend classes.
    Returns dict[str, QuantumBackend].
    """
    registry = {}

    # Statevector
    try:
        from zuntenium.backends import SimulatorBackend
        registry["statevector"] = SimulatorBackend(n_qubits=28)
    except Exception:
        try:
            from zuntenium.quantum_core.statevector import StatevectorSimulator
            registry["statevector"] = StatevectorSimulator(28)
        except Exception:
            pass

    # TensorNetworkBackend (MPS)
    try:
        from zuntenium.backends import TensorNetworkBackend
        registry["tensornetwork"] = TensorNetworkBackend(max_bond_dim=64)
    except Exception:
        try:
            from zuntenium.backends.tensor_network_backend import TensorNetworkBackend
            registry["tensornetwork"] = TensorNetworkBackend(max_bond_dim=64)
        except Exception:
            pass

    # Stabilizer (Gottesman-Knill)
    try:
        from zuntenium.backends.stabilizer import StabilizerSimulator
        registry["stabilizer"] = StabilizerSimulator(max_qubits=10000)
    except Exception:
        pass

    # GPU-accelerated statevector
    try:
        from zuntenium.backends.gpu_backend import GPUStatevectorSimulator
        registry["gpu_statevector"] = GPUStatevectorSimulator()
    except Exception:
        pass

    # Density matrix (noisy simulation)
    try:
        from zuntenium.backends.density_matrix_simulator import DensityMatrixSimulator
        registry["densitymatrix"] = DensityMatrixSimulator(num_qubits=1)  # placeholder; re-inited per job
    except Exception:
        pass

    # IBM
    try:
        token = os.environ.get("ZUNTENIUM_IBM_TOKEN")
        if token:
            from zuntenium.backends import IBMBackend
            registry["ibm"] = IBMBackend(token=token)
    except Exception:
        pass

    # IonQ
    try:
        ionq_key = os.environ.get("ZUNTENIUM_IONQ_API_KEY")
        if ionq_key:
            from zuntenium.backend.hardware.ionq.ionq_backend import IonQBackend
            registry["ionq"] = IonQBackend(api_key=ionq_key)
    except Exception:
        pass

    return registry


# ── Main execution function ───────────────────────────────────────────────────

def execute_job(job_id: str) -> dict:
    """
    Main RQ task function — executes a quantum job end-to-end.

    This function runs synchronously inside an RQ worker process.
    It creates its own DB session (sync SQLAlchemy) since RQ is not async.

    Returns the result dict (also stored in DB by mark_completed).
    """
    import asyncio
    import traceback

    _log.info("[Worker %s] Starting job %s", _WORKER_ID, job_id)
    t_total_start = time.perf_counter()

    # Run the async execution logic in a new event loop
    # (RQ workers are synchronous processes)
    loop = asyncio.new_event_loop()
    try:
        result = loop.run_until_complete(_execute_job_async(job_id, t_total_start))
        return result
    except Exception as exc:
        _log.exception("[Worker %s] Job %s raised unhandled exception", _WORKER_ID, job_id)
        loop.run_until_complete(_fail_job(job_id, traceback.format_exc()))
        raise
    finally:
        loop.close()


async def _execute_job_async(job_id: str, t_total_start: float) -> dict:
    """Async implementation of job execution."""
    from sqlalchemy import select
    from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

    from .models import Job, JobStatus
    from .job_manager import JobManager

    db_url   = os.environ.get("ZUNTENIUM_DATABASE_URL",
                              "postgresql+asyncpg://zuntenium:zuntenium@localhost:5432/zuntenium")
    redis_url = os.environ.get("ZUNTENIUM_REDIS_URL", "redis://localhost:6379/0")

    engine = create_async_engine(db_url, pool_size=5, max_overflow=2)
    Session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    import redis
    import redis.asyncio as aioredis

    redis_sync  = redis.Redis.from_url(redis_url)
    redis_async = aioredis.from_url(redis_url, decode_responses=True)

    async with Session() as db:
        manager = JobManager(db, redis_sync, redis_async)

        # 1. Load job
        import uuid as _uuid
        job_uuid = _uuid.UUID(str(job_id)) if not isinstance(job_id, _uuid.UUID) else job_id
        result = await db.execute(select(Job).where(Job.id == job_uuid))
        job = result.scalar_one_or_none()
        if not job:
            raise ValueError(f"Job {job_id} not found in database")

        if job.status == JobStatus.CANCELLED:
            _log.info("Job %s was cancelled before execution", job_id)
            return {}

        # 2. Mark as running
        await manager.mark_running(job_id, _WORKER_ID)
        t_exec_start = time.perf_counter()
        queue_time = t_exec_start - t_total_start

        try:
            # 3. Execute the Zuntenium pipeline
            result_data = await _run_zuntenium_pipeline(job, queue_time)

            # 4. Store result
            await manager.mark_completed(job_id, result_data)
            _log.info(
                "[Worker %s] Job %s completed in %.2fs (backend=%s)",
                _WORKER_ID, job_id,
                result_data.get("execution_time_s", 0),
                result_data.get("backend_used", "unknown"),
            )
            return result_data

        except Exception as exc:
            error_msg = str(exc)[:2000]
            _log.error("[Worker %s] Job %s failed: %s", _WORKER_ID, job_id, error_msg)
            await manager.mark_failed(job_id, error_msg, retry=True)
            raise

    await engine.dispose()


async def _run_zuntenium_pipeline(job, queue_time: float) -> dict:
    """
    Run the complete Zuntenium execution pipeline for a single job.

    Pipeline:
      Circuit → AIOptimizer.select() → Transpiler → Backend.run()
              → ErrorMitigationEngine.process() → Result
    """
    import time
    t0 = time.perf_counter()

    # ── Reconstruct circuit ────────────────────────────────────────────────
    if not job.circuit_json and not job.hybrid_config:
        raise ValueError("Job has neither circuit_json nor hybrid_config")

    circuit = _build_circuit(job.circuit_json) if job.circuit_json else None

    # ── Backend registry ───────────────────────────────────────────────────
    backends = _get_backend_registry()

    # ── AI Optimizer (optional, falls back to rule-based) ─────────────────
    ai_decision = None
    decision_reason = "rule-based selection"
    ai_confidence   = None
    chosen_backend  = job.backend_pref.value if job.backend_pref else "auto"

    if job.optimization_mode == "ai" and circuit:
        try:
            from zuntenium.ai_engine import AIOptimizer, ExecutionLogger
            logger    = ExecutionLogger()
            optimizer = AIOptimizer(logger=logger, use_rl=False)
            ai_decision = optimizer.select(circuit)
            chosen_backend  = ai_decision.backend.value
            decision_reason = ai_decision.reason
            ai_confidence   = ai_decision.confidence
            _log.debug("AI decision: %s", ai_decision)
        except Exception as exc:
            _log.warning("AI optimizer failed (%s), falling back to rule-based", exc)

    # ── Backend selection ──────────────────────────────────────────────────
    if chosen_backend == "auto":
        n = circuit.num_qubits if circuit else 0
        if n <= 20 and "statevector" in backends:
            chosen_backend = "statevector"
        elif "tensornetwork" in backends:
            chosen_backend = "tensornetwork"
        else:
            chosen_backend = "statevector"

    # User override for backend preference
    if job.backend_pref and job.backend_pref.value != "auto":
        chosen_backend = job.backend_pref.value

    # ── Transpiler ─────────────────────────────────────────────────────────
    transpiled_circuit = circuit
    if circuit:
        try:
            from zuntenium.transpiler import Transpiler
            transpiler = Transpiler()
            strategy = (ai_decision.strategy_str
                        if ai_decision else "rule_only")
            transpiled_circuit = transpiler.transpile(circuit, mode=strategy)
        except Exception as exc:
            _log.debug("Transpiler unavailable (%s), using original circuit", exc)

    # ── Execute on chosen backend ──────────────────────────────────────────
    backend_kwargs: dict[str, Any] = {"shots": job.shots}
    if job.max_bond_dim and chosen_backend == "tensornetwork":
        backend_kwargs["max_bond_dim"] = job.max_bond_dim

    if ai_decision and ai_decision.params:
        backend_kwargs.update(ai_decision.params)

    raw_result: dict = {}
    if job.job_type.value == "hybrid" and job.hybrid_config:
        raw_result = await _run_hybrid_job(job.hybrid_config, circuit, backend_kwargs)
    else:
        raw_result = _run_circuit_on_backend(
            transpiled_circuit or circuit,
            chosen_backend,
            backends,
            backend_kwargs,
        )

    t_exec = time.perf_counter() - t0

    # ── Error Mitigation ──────────────────────────────────────────────────
    mitigation_method = None
    fidelity_before   = None
    fidelity_after    = None

    if job.mitigation and circuit:
        try:
            from zuntenium.error_mitigation.integration import (
                MitigationPipeline, ExecutionResult,
            )
            from zuntenium.error_mitigation.noise_model import medium_noise_model

            noise_model = medium_noise_model()
            pipeline    = MitigationPipeline(
                noise_model=noise_model,
                strategy="auto",
            )
            er = ExecutionResult(
                counts=raw_result.get("counts", {}),
                statevector=raw_result.get("statevector"),
                shots=job.shots,
            )
            class _MC:
                num_qubits = circuit.num_qubits
                depth      = 5
                gates      = []
                name       = "m"

            mitigated = pipeline.process(er, circuit=_MC())
            raw_result["counts"] = mitigated.counts or raw_result.get("counts")
            mitigation_method = mitigated.metadata.get("mitigation_method")
            fidelity_before   = mitigated.metadata.get("fidelity_before")
            fidelity_after    = mitigated.metadata.get("fidelity_after")
        except Exception as exc:
            _log.warning("Error mitigation failed (%s), returning raw result", exc)

    # ── Package result ─────────────────────────────────────────────────────
    total_time = time.perf_counter() - t0 + queue_time

    # Serialise statevector if present (complex → [[real, imag], ...])
    sv = raw_result.get("statevector")
    sv_serialised = None
    if sv is not None:
        try:
            import numpy as np
            arr = np.asarray(sv)
            sv_serialised = [[float(x.real), float(x.imag)] for x in arr.ravel()]
        except Exception:
            pass

    backend_used = (
        raw_result.get("provider_used")
        or raw_result.get("backend_used")
        or chosen_backend
    )
    if raw_result.get("fallback_used"):
        decision_reason = (
            f"{decision_reason}; hardware fallback: "
            f"{raw_result.get('fallback_reason') or 'provider unavailable'}"
        )

    return {
        "counts":             raw_result.get("counts"),
        "statevector":        sv_serialised,
        "expectation_values": raw_result.get("expectation_values"),
        "energy":             raw_result.get("energy"),
        "backend_used":       backend_used,
        "decision_reason":    decision_reason,
        "ai_confidence":      ai_confidence,
        "mitigation_method":  mitigation_method,
        "fidelity_before":    fidelity_before,
        "fidelity_after":     fidelity_after,
        "truncation_error":   raw_result.get("metadata", {}).get("total_truncation_error"),
        "execution_time_s":   round(t_exec, 4),
        "queue_time_s":       round(queue_time, 4),
        "total_time_s":       round(total_time, 4),
    }


def _run_circuit_on_backend(
    circuit, backend_name: str, backends: dict, kwargs: dict
) -> dict:
    """
    Execute circuit on the named backend.
    Falls back to statevector if backend unavailable.
    """
    if not circuit:
        return {"counts": {}, "statevector": None}

    backend = backends.get(backend_name)

    # Graceful fallback chain: requested → tensornetwork → statevector
    if backend is None:
        _log.warning(
            "Backend '%s' not available, falling back. Available: %s",
            backend_name, list(backends.keys()),
        )
        for fallback in ("tensornetwork", "statevector"):
            backend = backends.get(fallback)
            if backend:
                backend_name = fallback
                break

    if backend is None:
        # Last resort: use TensorNetworkBackend directly
        try:
            from zuntenium.backends import TensorNetworkBackend
            backend = TensorNetworkBackend(max_bond_dim=kwargs.get("max_bond_dim", 32))
            backend_name = "tensornetwork"
        except Exception:
            return {"counts": {"error": 1}, "statevector": None}

    shots = kwargs.get("shots", 1024)

    # Normalise backend call interface (duck-typed)
    try:
        if hasattr(backend, "run"):
            result = backend.run(circuit, shots=shots,
                                  return_statevector=(circuit.num_qubits <= 20))
        elif hasattr(backend, "execute"):
            result = backend.execute(circuit, shots=shots)
        else:
            raise AttributeError(f"Backend {type(backend)} has no run() or execute()")

        if isinstance(result, dict):
            return result

        # Unpack ExecutionResult or duck-typed result object
        counts = getattr(result, "counts", {})
        if not counts and hasattr(result, "get_counts"):
            counts = result.get_counts()
        sv = getattr(result, "statevector", getattr(result, "state", None))
        probs = getattr(result, "probabilities", None)
        meta = getattr(result, "metadata", {}) or {}
        return {
            "counts": counts,
            "statevector": sv,
            "probabilities": probs,
            "metadata": meta,
        }

    except Exception as exc:
        _log.error("Backend execution failed (%s): %s", backend_name, exc)
        raise


async def _run_hybrid_job(
    hybrid_config: dict, circuit, backend_kwargs: dict
) -> dict:
    """Run a HybridSolver job (variational, QAOA, VQE)."""
    try:
        from zuntenium.hybrid import HybridSolver
        solver = HybridSolver(**hybrid_config)
        result = solver.run(circuit, **backend_kwargs)
        return result if isinstance(result, dict) else {"energy": float(result)}
    except Exception as exc:
        _log.error("HybridSolver failed: %s", exc)
        raise


async def _fail_job(job_id: str, error: str) -> None:
    """Mark a job as failed (best-effort, called from exception handler)."""
    try:
        from sqlalchemy import update
        from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
        from .models import Job, JobStatus

        db_url = os.environ.get("ZUNTENIUM_DATABASE_URL",
                                "postgresql+asyncpg://zuntenium:zuntenium@localhost:5432/zuntenium")
        engine = create_async_engine(db_url, pool_size=2, max_overflow=0)
        Session = async_sessionmaker(engine, class_=AsyncSession)
        import uuid as _uuid
        job_uuid = _uuid.UUID(str(job_id)) if not isinstance(job_id, _uuid.UUID) else job_id
        async with Session() as db:
            await db.execute(
                update(Job).where(Job.id == job_uuid).values(
                    status=JobStatus.FAILED,
                    completed_at=datetime.now(timezone.utc),
                    last_error=error[:2000],
                )
            )
            await db.commit()
        await engine.dispose()
    except Exception as exc:
        _log.error("Could not persist failure for job %s: %s", job_id, exc)


try:
    from zuntenium.mhos.integration import MHOSIntegration, patch_cloud_worker

    patch_cloud_worker(MHOSIntegration.from_env())
except Exception as exc:
    _log.warning("MHOS hardware routing is unavailable: %s", exc)
