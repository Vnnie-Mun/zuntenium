# zuntenium_cloud/job_manager.py
"""
JobManager — central lifecycle controller for quantum jobs.

Responsibilities:
  - Validate incoming job requests against circuit constraints
  - Persist jobs to PostgreSQL
  - Dispatch to the appropriate Redis RQ queue (based on priority + backend)
  - Poll and update job status from RQ
  - Cancel, retry, and expire jobs
  - Aggregate metrics for quota enforcement

Queue strategy:
  priority 8–10  → "high"    queue  (dedicated workers, SLA 5s)
  priority 4–7   → "default" queue  (standard workers, SLA 60s)
  priority 1–3   → "low"     queue  (batch workers, SLA 10m)
  IBM hardware   → "hardware" queue (separate worker with IBM credentials)
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

import redis.asyncio as aioredis
from rq import Queue
from rq.job import Job as RQJob
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from .models import Job, JobResult, JobStatus, JobType, BackendPreference
from .schemas import JobSubmitRequest, CircuitSchema

_log = logging.getLogger(__name__)


def _to_uuid(val: Any) -> uuid.UUID | None:
    if isinstance(val, uuid.UUID):
        return val
    try:
        return uuid.UUID(str(val))
    except (ValueError, TypeError):
        return None


# ── Queue names ───────────────────────────────────────────────────────────────

def _queue_for(priority: int, backend: str) -> str:
    if backend in {"ibm", "ionq", "braket", "rigetti"}:
        return "hardware"
    if priority >= 8:
        return "high"
    if priority >= 4:
        return "default"
    return "low"


# ── Circuit validation ────────────────────────────────────────────────────────

def validate_circuit(spec: CircuitSchema) -> None:
    """
    Validate circuit structure before accepting a job.
    Raises ValueError with a descriptive message on failure.
    """
    if spec.num_qubits < 1 or spec.num_qubits > 200:
        raise ValueError(f"num_qubits must be 1–200, got {spec.num_qubits}")

    seen_targets: set[int] = set()
    for i, gate in enumerate(spec.gates):
        # Qubit index bounds
        for q in gate.qubits:
            if q < 0 or q >= spec.num_qubits:
                raise ValueError(
                    f"Gate[{i}] '{gate.name}' references qubit {q} "
                    f"but circuit only has {spec.num_qubits} qubits"
                )
        # Duplicate qubit targets in a single gate
        if len(gate.qubits) != len(set(gate.qubits)):
            raise ValueError(
                f"Gate[{i}] '{gate.name}' references the same qubit twice"
            )
        # Gate arity sanity
        single_q = {"h","x","y","z","s","t","sdg","tdg","id","sx","sxdg"}
        two_q    = {"cx","cy","cz","cnot","swap","iswap","ecr","dcx"}
        param_1q = {"rx","ry","rz","p","u1","u2","u3"}
        param_2q = {"cp","crx","cry","crz","rxx","ryy","rzz"}

        name = gate.name.lower()
        if name in single_q and len(gate.qubits) != 1:
            raise ValueError(f"Gate '{gate.name}' expects 1 qubit, got {len(gate.qubits)}")
        if name in two_q and len(gate.qubits) != 2:
            raise ValueError(f"Gate '{gate.name}' expects 2 qubits, got {len(gate.qubits)}")


# ── JobManager class ──────────────────────────────────────────────────────────

class JobManager:
    """
    Central job lifecycle manager.

    Parameters
    ----------
    db         : async SQLAlchemy session
    redis_sync : synchronous redis.Redis instance (for RQ)
    redis_async: asynchronous redis.Redis instance (for status cache)
    """

    def __init__(
        self,
        db: AsyncSession,
        redis_sync,         # redis.Redis (sync) — required by RQ
        redis_async,        # redis.asyncio.Redis
    ) -> None:
        self._db     = db
        self._rsync  = redis_sync
        self._rasync = redis_async
        self._queues = {
            "high":     Queue("high",     connection=redis_sync),
            "default":  Queue("default",  connection=redis_sync),
            "low":      Queue("low",      connection=redis_sync),
            "hardware": Queue("hardware", connection=redis_sync),
        }

    # ── Submit ─────────────────────────────────────────────────────────────

    async def submit(
        self,
        request: JobSubmitRequest,
        user_id: str,
    ) -> Job:
        """
        Create and enqueue a new job.
        Returns the persisted Job ORM object.
        """
        # Validate circuit
        if request.circuit:
            try:
                validate_circuit(request.circuit)
            except ValueError as exc:
                from fastapi import HTTPException
                raise HTTPException(status_code=422, detail=str(exc))

        # Determine queue
        queue_name = _queue_for(request.priority, request.backend)

        # Create DB record
        job = Job(
            id               = uuid.uuid4(),
            user_id          = user_id,
            job_type         = JobType(request.job_type),
            status           = JobStatus.PENDING,
            circuit_json     = request.circuit.model_dump() if request.circuit else None,
            hybrid_config    = request.hybrid_config,
            parameters       = request.parameters,
            backend_pref     = BackendPreference(request.backend),
            shots            = request.shots,
            mitigation       = request.mitigation_enabled,
            optimization_mode = request.optimization_mode,
            max_bond_dim     = request.max_bond_dim,
            queue_name       = queue_name,
            priority         = request.priority,
            created_at       = datetime.now(timezone.utc),
        )
        self._db.add(job)
        await self._db.flush()  # get the ID

        # Dispatch to RQ queue
        rq_job = self._queues[queue_name].enqueue(
            "zuntenium_cloud.worker.execute_job",
            kwargs={"job_id": str(job.id)},
            job_id=str(job.id),
            job_timeout=600,    # 10-minute hard limit
            result_ttl=86400,   # keep result 24h
            failure_ttl=86400,
            meta={"user_id": str(user_id), "priority": request.priority},
        )

        # Update job with RQ info
        job.rq_job_id  = rq_job.id
        job.status     = JobStatus.QUEUED
        job.queued_at  = datetime.now(timezone.utc)

        await self._db.commit()

        # Cache job status in Redis (5 min TTL for fast polling)
        await self._cache_status(str(job.id), "queued")

        _log.info("Job %s queued → %s (priority=%d)", job.id, queue_name, request.priority)
        return job

    # ── Status queries ─────────────────────────────────────────────────────

    async def get_job(self, job_id: str, user_id: str) -> Job | None:
        """Fetch job, enforcing user ownership."""
        jid, uid = _to_uuid(job_id), _to_uuid(user_id)
        if not jid or not uid:
            return None
        result = await self._db.execute(
            select(Job).where(Job.id == jid, Job.user_id == uid)
        )
        return result.scalar_one_or_none()

    async def get_result(self, job_id: str, user_id: str) -> JobResult | None:
        job = await self.get_job(job_id, user_id)
        if not job or job.status != JobStatus.COMPLETED:
            return None
        jid = _to_uuid(job_id)
        result = await self._db.execute(
            select(JobResult).where(JobResult.job_id == jid)
        )
        return result.scalar_one_or_none()

    async def list_jobs(
        self,
        user_id: str,
        status: str | None = None,
        job_type: str | None = None,
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[list[Job], int]:
        """Returns (jobs, total_count)."""
        from sqlalchemy import func
        uid = _to_uuid(user_id)
        if not uid:
            return [], 0
        q = select(Job).where(Job.user_id == uid)
        if status:
            q = q.where(Job.status == JobStatus(status))
        if job_type:
            q = q.where(Job.job_type == JobType(job_type))

        total_q  = select(func.count()).select_from(q.subquery())
        total    = (await self._db.execute(total_q)).scalar() or 0

        q = q.order_by(Job.created_at.desc())
        q = q.offset((page - 1) * page_size).limit(page_size)
        jobs = (await self._db.execute(q)).scalars().all()
        return list(jobs), total

    async def queue_position(self, job_id: str) -> int | None:
        """Return position in queue (1 = next to run), or None if not queued."""
        try:
            rq_job = RQJob.fetch(job_id, connection=self._rsync)
            q = self._queues.get(rq_job.origin, self._queues["default"])
            jobs_ids = q.job_ids
            if job_id in jobs_ids:
                return jobs_ids.index(job_id) + 1
        except Exception:
            pass
        return None

    # ── Cancel ──────────────────────────────────────────────────────────────

    async def cancel_job(self, job_id: str, user_id: str) -> bool:
        """
        Cancel a PENDING or QUEUED job.
        Running jobs cannot be cancelled (use timeout instead).
        """
        job = await self.get_job(job_id, user_id)
        if not job:
            return False
        if job.status not in (JobStatus.PENDING, JobStatus.QUEUED, JobStatus.RETRYING):
            return False

        # Remove from RQ if possible
        try:
            rq_job = RQJob.fetch(job_id, connection=self._rsync)
            rq_job.cancel()
        except Exception as exc:
            _log.debug("RQ cancel for %s failed (may already be processing): %s", job_id, exc)

        await self._db.execute(
            update(Job)
            .where(Job.id == job_id)
            .values(status=JobStatus.CANCELLED, completed_at=datetime.now(timezone.utc))
        )
        await self._db.commit()
        await self._cache_status(job_id, "cancelled")
        _log.info("Job %s cancelled by user %s", job_id, user_id)
        return True

    # ── Status update (called by worker) ────────────────────────────────────

    async def mark_running(self, job_id: str, worker_id: str) -> None:
        jid = _to_uuid(job_id) or job_id
        await self._db.execute(
            update(Job).where(Job.id == jid).values(
                status=JobStatus.RUNNING,
                started_at=datetime.now(timezone.utc),
                worker_id=worker_id,
            )
        )
        await self._db.commit()
        await self._cache_status(job_id, "running")

    async def mark_completed(self, job_id: str, result_data: dict) -> None:
        now = datetime.now(timezone.utc)
        jid = _to_uuid(job_id) or job_id
        await self._db.execute(
            update(Job).where(Job.id == jid).values(
                status=JobStatus.COMPLETED,
                completed_at=now,
            )
        )
        # Persist result
        job_result = JobResult(
            job_id            = jid,
            counts            = result_data.get("counts"),
            statevector       = result_data.get("statevector"),
            expectation_values = result_data.get("expectation_values"),
            energy            = result_data.get("energy"),
            backend_used      = result_data.get("backend_used"),
            decision_reason   = result_data.get("decision_reason"),
            ai_confidence     = result_data.get("ai_confidence"),
            mitigation_method = result_data.get("mitigation_method"),
            fidelity_before   = result_data.get("fidelity_before"),
            fidelity_after    = result_data.get("fidelity_after"),
            truncation_error  = result_data.get("truncation_error"),
            execution_time_s  = result_data.get("execution_time_s"),
            queue_time_s      = result_data.get("queue_time_s"),
            total_time_s      = result_data.get("total_time_s"),
            raw_result        = result_data,
            created_at        = now,
        )
        self._db.add(job_result)
        await self._db.commit()
        await self._cache_status(job_id, "completed")

    async def mark_failed(
        self, job_id: str, error: str, retry: bool = True
    ) -> None:
        jid = _to_uuid(job_id) or job_id
        job_result = await self._db.execute(select(Job).where(Job.id == jid))
        job = job_result.scalar_one_or_none()
        if not job:
            return

        if retry and job.retry_count < job.max_retries:
            # Schedule retry with exponential backoff
            import time
            delay = 2 ** job.retry_count * 5   # 5s, 10s, 20s
            queue = self._queues.get(job.queue_name, self._queues["default"])
            queue.enqueue_in(
                timedelta=__import__("datetime").timedelta(seconds=delay),
                f="zuntenium_cloud.worker.execute_job",
                kwargs={"job_id": job_id},
                job_timeout=600,
            )
            await self._db.execute(
                update(Job).where(Job.id == job_id).values(
                    status=JobStatus.RETRYING,
                    retry_count=job.retry_count + 1,
                    last_error=error,
                )
            )
            _log.warning(
                "Job %s failed (%s), retry %d/%d in %ds",
                job_id, error[:80], job.retry_count + 1, job.max_retries, delay,
            )
        else:
            await self._db.execute(
                update(Job).where(Job.id == job_id).values(
                    status=JobStatus.FAILED,
                    completed_at=datetime.now(timezone.utc),
                    last_error=error,
                )
            )
            _log.error("Job %s permanently failed: %s", job_id, error[:200])

        await self._db.commit()
        await self._cache_status(job_id, "failed")

    # ── Redis cache helpers ──────────────────────────────────────────────────

    async def _cache_status(self, job_id: str, status: str) -> None:
        try:
            await self._rasync.setex(f"job:status:{job_id}", 300, status)
        except Exception as exc:
            _log.debug("Redis cache write failed: %s", exc)

    async def get_cached_status(self, job_id: str) -> str | None:
        try:
            val = await self._rasync.get(f"job:status:{job_id}")
            return val.decode() if val else None
        except Exception:
            return None
