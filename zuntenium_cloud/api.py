# zuntenium_cloud/api.py
"""
FastAPI REST API — Zuntenium Cloud Platform

Endpoints:
  POST   /auth/register           — create account
  POST   /auth/token              — login → JWT
  POST   /auth/refresh            — refresh access token
  GET    /auth/me                 — current user info
  POST   /auth/api-keys           — generate API key
  GET    /auth/api-keys           — list API keys
  DELETE /auth/api-keys/{id}      — revoke key

  POST   /jobs                    — submit job
  GET    /jobs                    — list jobs (paginated + filtered)
  GET    /jobs/{job_id}           — job details
  GET    /jobs/{job_id}/status    — lightweight status poll
  GET    /jobs/{job_id}/result    — execution result
  POST   /jobs/{job_id}/cancel    — cancel pending job

  GET    /backends                — available backends + queue depth
  GET    /health                  — liveness probe
  GET    /metrics                 — Prometheus text format (internal)
  GET    /admin/stats             — platform statistics (admin only)
"""

import logging
import os
import time
import uuid
from importlib.util import find_spec
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Annotated, Any

import redis
import redis.asyncio as aioredis
from fastapi import (
    Depends, FastAPI, HTTPException, Query, Request, status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import (
    Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST,
)
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from .auth import (
    TokenResponse, RegisterRequest, LoginRequest,
    APIKeyCreateRequest, APIKeyResponse,
    check_and_consume_quota, create_access_token, create_refresh_token,
    decode_token, generate_api_key, get_current_user, get_optional_user, hash_password,
    require_admin, verify_password,
)
from .database import get_db, init_db
from .job_manager import JobManager
from .models import APIKey, Job, JobStatus, User, UserQuota
from .schemas import (
    BackendInfo, BackendListResponse,
    ErrorResponse,
    JobListResponse, JobResponse, JobResultResponse,
    JobStatusResponse, JobSubmitRequest,
)

_log = logging.getLogger(__name__)

# ── Prometheus metrics ────────────────────────────────────────────────────────

JOBS_SUBMITTED  = Counter("zuntenium_jobs_submitted_total", "Total jobs submitted",
                           ["job_type", "backend"])
JOBS_COMPLETED  = Counter("zuntenium_jobs_completed_total", "Jobs completed",
                           ["backend", "status"])
JOB_DURATION    = Histogram("zuntenium_job_duration_seconds", "Job execution time",
                             ["backend"], buckets=[.1,.5,1,5,10,30,60,120,300,600])
ACTIVE_JOBS     = Gauge("zuntenium_active_jobs", "Jobs currently running")
QUEUE_DEPTH     = Gauge("zuntenium_queue_depth", "Jobs in queue", ["queue"])
API_REQUESTS    = Counter("zuntenium_api_requests_total", "API requests", ["method", "endpoint"])
API_ERRORS      = Counter("zuntenium_api_errors_total", "API errors", ["endpoint", "code"])

# ── Rate limiter ──────────────────────────────────────────────────────────────

limiter = Limiter(key_func=get_remote_address)

# ── Redis connections (module-level singletons) ───────────────────────────────

_REDIS_URL = os.environ.get("ZUNTENIUM_REDIS_URL", "redis://localhost:6379/0")
_redis_sync:  redis.Redis | None  = None
_redis_async: aioredis.Redis | None = None


def get_redis_sync() -> redis.Redis:
    global _redis_sync
    if _redis_sync is None:
        _redis_sync = redis.Redis.from_url(_REDIS_URL)
    return _redis_sync


def get_redis_async() -> aioredis.Redis:
    global _redis_async
    if _redis_async is None:
        _redis_async = aioredis.from_url(_REDIS_URL, decode_responses=True)
    return _redis_async


# ── App factory ───────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    _log.info("Zuntenium Cloud API starting up…")
    await init_db()
    yield
    _log.info("Zuntenium Cloud API shutting down…")
    if _redis_async:
        await _redis_async.aclose()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Zuntenium Cloud API",
        description="Quantum computing platform — Phase 4",
        version="4.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=os.environ.get("ALLOWED_ORIGINS", "*").split(","),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Rate limiter
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # Request ID middleware
    @app.middleware("http")
    async def add_request_id(request: Request, call_next):
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        t0 = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Request-Id"] = request_id
        response.headers["X-Response-Time"] = f"{(time.perf_counter()-t0)*1000:.1f}ms"
        API_REQUESTS.labels(method=request.method, endpoint=request.url.path).inc()
        return response

    # Register routers
    app.include_router(auth_router,    prefix="/auth",     tags=["Authentication"])
    app.include_router(jobs_router,    prefix="/jobs",     tags=["Jobs"])
    app.include_router(backend_router, prefix="/backends", tags=["Backends"])
    app.include_router(admin_router,   prefix="/admin",    tags=["Admin"])

    # Health + metrics
    @app.get("/health", tags=["System"])
    async def health():
        return {"status": "ok", "version": "4.0.0",
                "timestamp": datetime.now(timezone.utc).isoformat()}

    @app.get("/metrics", tags=["System"])
    async def metrics():
        return PlainTextResponse(
            generate_latest().decode(), media_type=CONTENT_TYPE_LATEST
        )

    return app


# ── Auth router ───────────────────────────────────────────────────────────────

from fastapi import APIRouter

auth_router = APIRouter()

@auth_router.post("/register", response_model=dict)
@limiter.limit("10/minute")
async def register(
    request: Request,
    body: RegisterRequest,
    db: AsyncSession = Depends(get_db),
):
    # Check uniqueness
    result = await db.execute(
        select(User).where(
            (User.email == body.email) | (User.username == body.username)
        )
    )
    if result.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Email or username already taken")

    user = User(
        email=body.email,
        username=body.username,
        hashed_pw=hash_password(body.password),
    )
    db.add(user)
    await db.flush()
    quota = UserQuota(user_id=user.id)
    db.add(quota)
    await db.commit()
    return {"message": "Account created", "user_id": str(user.id)}


@auth_router.post("/token", response_model=TokenResponse)
@limiter.limit("30/minute")
async def login(
    request: Request,
    body: LoginRequest,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(User).where(User.username == body.username, User.is_active == True)
    )
    user = result.scalar_one_or_none()
    if not user or not verify_password(body.password, user.hashed_pw):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    user.last_login = datetime.now(timezone.utc)
    await db.commit()
    return TokenResponse(
        access_token=create_access_token(str(user.id), user.role.value),
        refresh_token=create_refresh_token(str(user.id)),
    )


@auth_router.post("/refresh", response_model=TokenResponse)
async def refresh(token: str, db: AsyncSession = Depends(get_db)):
    payload = decode_token(token)
    if payload.get("type") != "refresh":
        raise HTTPException(status_code=400, detail="Not a refresh token")
    sub = payload.get("sub")
    try:
        user_uuid = uuid.UUID(str(sub)) if not isinstance(sub, uuid.UUID) else sub
    except (ValueError, TypeError):
        raise HTTPException(status_code=401, detail="User not found or inactive")
    result = await db.execute(select(User).where(User.id == user_uuid))
    user   = result.scalar_one_or_none()
    if not user or not user.is_active:
        raise HTTPException(status_code=401, detail="User not found or inactive")
    return TokenResponse(
        access_token=create_access_token(str(user.id), user.role.value),
        refresh_token=create_refresh_token(str(user.id)),
    )


@auth_router.get("/me")
async def me(current_user: User = Depends(get_current_user)):
    return {
        "id":         str(current_user.id),
        "username":   current_user.username,
        "email":      current_user.email,
        "role":       current_user.role.value,
        "created_at": current_user.created_at.isoformat(),
    }


@auth_router.post("/api-keys", response_model=APIKeyResponse)
async def create_api_key(
    body: APIKeyCreateRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    from datetime import timedelta
    raw, prefix, key_hash = generate_api_key()
    expires_at = (
        datetime.now(timezone.utc) + timedelta(days=body.expires_days)
        if body.expires_days else None
    )
    api_key = APIKey(
        user_id=current_user.id,
        key_prefix=prefix,
        key_hash=key_hash,
        name=body.name,
        expires_at=expires_at,
    )
    db.add(api_key)
    await db.commit()
    return APIKeyResponse(
        id=str(api_key.id),
        prefix=prefix,
        name=body.name,
        created_at=api_key.created_at,
        expires_at=expires_at,
        raw_key=raw,    # shown ONCE
    )


@auth_router.get("/api-keys", response_model=list[APIKeyResponse])
async def list_api_keys(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(APIKey).where(APIKey.user_id == current_user.id, APIKey.is_active == True)
    )
    keys = result.scalars().all()
    return [
        APIKeyResponse(id=str(k.id), prefix=k.key_prefix, name=k.name,
                       created_at=k.created_at, expires_at=k.expires_at)
        for k in keys
    ]


@auth_router.delete("/api-keys/{key_id}")
async def revoke_api_key(
    key_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    try:
        key_uuid = uuid.UUID(str(key_id)) if not isinstance(key_id, uuid.UUID) else key_id
    except (ValueError, TypeError):
        raise HTTPException(status_code=404, detail="API key not found")
    result = await db.execute(
        select(APIKey).where(APIKey.id == key_uuid, APIKey.user_id == current_user.id)
    )
    key = result.scalar_one_or_none()
    if not key:
        raise HTTPException(status_code=404, detail="API key not found")
    key.is_active = False
    await db.commit()
    return {"message": "API key revoked"}


# ── Jobs router ───────────────────────────────────────────────────────────────

jobs_router = APIRouter()

def _get_job_manager(
    db: AsyncSession = Depends(get_db),
) -> JobManager:
    return JobManager(db, get_redis_sync(), get_redis_async())


@jobs_router.post("", response_model=JobResponse, status_code=status.HTTP_202_ACCEPTED)
@limiter.limit("60/minute")
async def submit_job(
    request: Request,
    body: JobSubmitRequest,
    current_user: User = Depends(get_current_user),
    manager: JobManager = Depends(_get_job_manager),
    db: AsyncSession = Depends(get_db),
):
    # Quota check
    n_qubits = body.circuit.num_qubits if body.circuit else 0
    await check_and_consume_quota(current_user, db, n_qubits=n_qubits, shots=body.shots)

    job = await manager.submit(body, user_id=str(current_user.id))

    JOBS_SUBMITTED.labels(
        job_type=body.job_type, backend=body.backend
    ).inc()

    return _job_to_response(job)


@jobs_router.get("", response_model=JobListResponse)
async def list_jobs(
    current_user: User = Depends(get_current_user),
    manager: JobManager = Depends(_get_job_manager),
    status_filter: str | None = Query(default=None, alias="status"),
    job_type:      str | None = Query(default=None),
    page:      int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
):
    jobs, total = await manager.list_jobs(
        user_id=str(current_user.id),
        status=status_filter,
        job_type=job_type,
        page=page,
        page_size=page_size,
    )
    pages = max(1, (total + page_size - 1) // page_size)
    return JobListResponse(
        jobs=[_job_to_response(j) for j in jobs],
        total=total, page=page, page_size=page_size, pages=pages,
    )


@jobs_router.get("/{job_id}", response_model=JobResponse)
async def get_job(
    job_id: str,
    current_user: User = Depends(get_current_user),
    manager: JobManager = Depends(_get_job_manager),
):
    job = await manager.get_job(job_id, str(current_user.id))
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return _job_to_response(job)


@jobs_router.get("/{job_id}/status", response_model=JobStatusResponse)
async def get_job_status(
    job_id: str,
    current_user: User = Depends(get_current_user),
    manager: JobManager = Depends(_get_job_manager),
):
    # Fast path: check Redis cache first
    cached = await manager.get_cached_status(job_id)
    if cached:
        return JobStatusResponse(id=job_id, status=cached)

    job = await manager.get_job(job_id, str(current_user.id))
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    position = None
    if job.status == JobStatus.QUEUED:
        position = await manager.queue_position(job_id)

    return JobStatusResponse(
        id=str(job.id),
        status=job.status.value,
        position_in_queue=position,
    )


@jobs_router.get("/{job_id}/result", response_model=JobResultResponse)
async def get_result(
    job_id: str,
    current_user: User = Depends(get_current_user),
    manager: JobManager = Depends(_get_job_manager),
):
    job = await manager.get_job(job_id, str(current_user.id))
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status not in (JobStatus.COMPLETED, JobStatus.FAILED):
        raise HTTPException(
            status_code=202,
            detail=f"Job is {job.status.value}, result not yet available",
        )
    result = await manager.get_result(job_id, str(current_user.id))
    if not result:
        return JobResultResponse(job_id=job_id, status=job.status.value)

    if result.execution_time_s:
        JOB_DURATION.labels(backend=result.backend_used or "unknown").observe(
            result.execution_time_s
        )

    return JobResultResponse(
        job_id=str(result.job_id),
        status=job.status.value,
        counts=result.counts,
        statevector=result.statevector,
        expectation_values=result.expectation_values,
        energy=result.energy,
        backend_used=result.backend_used,
        decision_reason=result.decision_reason,
        ai_confidence=result.ai_confidence,
        mitigation_method=result.mitigation_method,
        fidelity_before=result.fidelity_before,
        fidelity_after=result.fidelity_after,
        truncation_error=result.truncation_error,
        execution_time_s=result.execution_time_s,
        queue_time_s=result.queue_time_s,
        total_time_s=result.total_time_s,
    )


@jobs_router.post("/{job_id}/cancel")
async def cancel_job(
    job_id: str,
    current_user: User = Depends(get_current_user),
    manager: JobManager = Depends(_get_job_manager),
):
    cancelled = await manager.cancel_job(job_id, str(current_user.id))
    if not cancelled:
        raise HTTPException(
            status_code=409,
            detail="Job cannot be cancelled (not pending/queued, or not found)",
        )
    return {"message": "Job cancelled", "job_id": job_id}


# ── Backends router ───────────────────────────────────────────────────────────

backend_router = APIRouter()

@backend_router.get("", response_model=BackendListResponse)
async def list_backends(_: User | None = Depends(get_optional_user)):
    backends = []

    # Statevector
    backends.append(BackendInfo(
        name="statevector", available=True,
        description="Exact statevector simulation (complex128, up to 28 qubits)",
        max_qubits=28,
    ))

    # TensorNetwork
    backends.append(BackendInfo(
        name="tensornetwork", available=True,
        description="MPS-based simulation (50–200+ qubits with bond dimension control)",
        max_qubits=200,
    ))

    # Stabilizer (Gottesman-Knill)
    backends.append(BackendInfo(
        name="stabilizer", available=True,
        description="Clifford-circuit simulator using Gottesman-Knill tableau (up to 10,000 qubits)",
        max_qubits=10000,
    ))

    # GPU Statevector
    backends.append(BackendInfo(
        name="gpu_statevector", available=True,
        description="GPU-accelerated statevector simulation (CuPy/JAX with NumPy fallback)",
        max_qubits=30,
    ))

    # Density Matrix
    backends.append(BackendInfo(
        name="densitymatrix", available=True,
        description="Density matrix simulation for noisy circuits (ρ→UρU†, Kraus channels, up to 15 qubits)",
        max_qubits=15,
    ))

    # IBM
    try:
        ibm_sdk_available = find_spec("qiskit_ibm_runtime") is not None
    except (ImportError, ValueError):
        ibm_sdk_available = False
    ibm_available = bool(os.environ.get("ZUNTENIUM_IBM_TOKEN")) and ibm_sdk_available
    backends.append(BackendInfo(
        name="ibm", available=ibm_available,
        description="IBM Quantum real hardware (requires IBM credentials)",
        max_qubits=127,
    ))

    hardware_providers = (
        ("ionq", ("ZUNTENIUM_IONQ_API_KEY",), ("aiohttp",), "IonQ Quantum Cloud", 36),
        ("braket", ("ZUNTENIUM_AWS_ACCESS_KEY_ID", "ZUNTENIUM_AWS_SECRET"), ("braket", "boto3"), "AWS Braket QPUs", None),
        ("rigetti", ("ZUNTENIUM_RIGETTI_USER_ID",), ("qcs_sdk",), "Rigetti QCS", None),
    )
    for name, credentials, modules, description, max_qubits in hardware_providers:
        try:
            sdk_available = all(find_spec(module) is not None for module in modules)
        except (ImportError, ValueError):
            sdk_available = False
        backends.append(BackendInfo(
            name=name,
            available=all(os.environ.get(credential) for credential in credentials) and sdk_available,
            description=f"{description} (requires credentials and provider SDK)",
            max_qubits=max_qubits,
        ))

    # Queue depths from Redis
    try:
        rsync = get_redis_sync()
        for b in backends:
            q_name = "hardware" if b.name == "ibm" else "default"
            depth = rsync.llen(f"rq:queue:{q_name}")
            b.queue_depth = int(depth)
    except Exception:
        pass

    return BackendListResponse(backends=backends)


# ── Admin router ──────────────────────────────────────────────────────────────

admin_router = APIRouter()

@admin_router.get("/stats")
async def admin_stats(
    _: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    total_users = (await db.execute(select(func.count()).select_from(User))).scalar()
    total_jobs  = (await db.execute(select(func.count()).select_from(Job))).scalar()
    running     = (await db.execute(
        select(func.count()).where(Job.status == JobStatus.RUNNING)
    )).scalar()
    queued      = (await db.execute(
        select(func.count()).where(Job.status == JobStatus.QUEUED)
    )).scalar()

    return {
        "users":           total_users,
        "jobs_total":      total_jobs,
        "jobs_running":    running,
        "jobs_queued":     queued,
        "worker_id":       os.environ.get("ZUNTENIUM_WORKER_ID", "api"),
        "timestamp":       datetime.now(timezone.utc).isoformat(),
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _job_to_response(job: Job) -> JobResponse:
    return JobResponse(
        id=str(job.id),
        status=job.status.value,
        job_type=job.job_type.value,
        created_at=job.created_at,
        queued_at=job.queued_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        retry_count=job.retry_count,
        last_error=job.last_error,
        user_id=str(job.user_id),
        queue_name=job.queue_name,
        priority=job.priority,
    )


# ── App instance ──────────────────────────────────────────────────────────────

app = create_app()
