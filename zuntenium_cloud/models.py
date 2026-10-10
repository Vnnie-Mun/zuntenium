# zuntenium_cloud/models.py
"""
Database models — SQLAlchemy ORM.

Tables:
  users       — accounts with quota tracking
  api_keys    — hashed API keys per user
  jobs        — full lifecycle from PENDING to COMPLETED/FAILED
  job_results — stored execution output (compressed JSON)
  quotas      — per-user resource limits and usage counters
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    BigInteger, Boolean, Column, DateTime, Enum as SAEnum,
    Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, relationship


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


# ── Enumerations ─────────────────────────────────────────────────────────────

class UserRole(str, enum.Enum):
    USER  = "user"
    ADMIN = "admin"


class JobStatus(str, enum.Enum):
    PENDING    = "pending"
    QUEUED     = "queued"
    RUNNING    = "running"
    COMPLETED  = "completed"
    FAILED     = "failed"
    CANCELLED  = "cancelled"
    RETRYING   = "retrying"


class JobType(str, enum.Enum):
    CIRCUIT      = "circuit"       # raw quantum circuit
    HYBRID       = "hybrid"        # HybridSolver job
    VARIATIONAL  = "variational"   # VQE / QAOA with parameter sweep


class BackendPreference(str, enum.Enum):
    AUTO          = "auto"
    STATEVECTOR   = "statevector"
    TENSORNETWORK = "tensornetwork"
    IBM           = "ibm"
    IONQ          = "ionq"
    BRAKET        = "braket"
    RIGETTI       = "rigetti"
    SIMULATOR     = "simulator"    # any simulator (prefer local)


# ── Models ───────────────────────────────────────────────────────────────────

class User(Base):
    __tablename__ = "users"

    id           = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email        = Column(String(255), unique=True, nullable=False, index=True)
    username     = Column(String(64),  unique=True, nullable=False, index=True)
    hashed_pw    = Column(String(128), nullable=False)
    role         = Column(SAEnum(UserRole), default=UserRole.USER, nullable=False)
    is_active    = Column(Boolean, default=True, nullable=False)
    created_at   = Column(DateTime(timezone=True), default=_utcnow, nullable=False)
    last_login   = Column(DateTime(timezone=True), nullable=True)

    # Relationships
    jobs     = relationship("Job",    back_populates="user", cascade="all, delete-orphan")
    api_keys = relationship("APIKey", back_populates="user", cascade="all, delete-orphan")
    quota    = relationship("UserQuota", back_populates="user", uselist=False,
                            cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<User {self.username} role={self.role}>"


class APIKey(Base):
    __tablename__ = "api_keys"

    id          = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id     = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"),
                         nullable=False, index=True)
    key_prefix  = Column(String(8),   nullable=False)       # first 8 chars (for display)
    key_hash    = Column(String(128),  nullable=False, unique=True)  # SHA-256
    name        = Column(String(100),  nullable=True)        # user-given label
    is_active   = Column(Boolean, default=True, nullable=False)
    created_at  = Column(DateTime(timezone=True), default=_utcnow, nullable=False)
    last_used   = Column(DateTime(timezone=True), nullable=True)
    expires_at  = Column(DateTime(timezone=True), nullable=True)

    user = relationship("User", back_populates="api_keys")

    __table_args__ = (
        Index("idx_api_keys_user_active", "user_id", "is_active"),
    )


class UserQuota(Base):
    __tablename__ = "user_quotas"

    id                 = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id            = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"),
                                nullable=False, unique=True)
    max_jobs_per_hour  = Column(Integer, default=100, nullable=False)
    max_qubits         = Column(Integer, default=50,  nullable=False)
    max_shots          = Column(Integer, default=100_000, nullable=False)
    max_concurrent     = Column(Integer, default=10, nullable=False)
    # Running counters (reset hourly by background task)
    jobs_this_hour     = Column(Integer, default=0, nullable=False)
    total_jobs         = Column(BigInteger, default=0, nullable=False)
    total_shot_seconds = Column(Float, default=0.0, nullable=False)
    quota_reset_at     = Column(DateTime(timezone=True), default=_utcnow, nullable=False)

    user = relationship("User", back_populates="quota")


class Job(Base):
    __tablename__ = "jobs"

    id              = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id         = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"),
                             nullable=False, index=True)
    job_type        = Column(SAEnum(JobType), nullable=False)
    status          = Column(SAEnum(JobStatus), default=JobStatus.PENDING, nullable=False)

    # Job configuration (serialised circuit / hybrid spec)
    circuit_json    = Column(JSONB, nullable=True)       # Circuit as JSON
    hybrid_config   = Column(JSONB, nullable=True)       # HybridSolver config
    parameters      = Column(JSONB, nullable=True)       # variational params

    # Execution preferences
    backend_pref    = Column(SAEnum(BackendPreference), default=BackendPreference.AUTO)
    shots           = Column(Integer, default=1024, nullable=False)
    mitigation      = Column(Boolean, default=False, nullable=False)
    optimization_mode = Column(String(16), default="rule", nullable=False)
    max_bond_dim    = Column(Integer, nullable=True)     # for MPS backend

    # Lifecycle tracking
    created_at      = Column(DateTime(timezone=True), default=_utcnow, nullable=False)
    queued_at       = Column(DateTime(timezone=True), nullable=True)
    started_at      = Column(DateTime(timezone=True), nullable=True)
    completed_at    = Column(DateTime(timezone=True), nullable=True)

    # Worker assignment
    worker_id       = Column(String(64), nullable=True)
    queue_name      = Column(String(64), default="default", nullable=False)
    rq_job_id       = Column(String(128), nullable=True, index=True)

    # Retry tracking
    retry_count     = Column(Integer, default=0, nullable=False)
    max_retries     = Column(Integer, default=3,  nullable=False)
    last_error      = Column(Text, nullable=True)

    # Priority (higher = run sooner)
    priority        = Column(Integer, default=5, nullable=False)

    # Relationships
    user   = relationship("User",      back_populates="jobs")
    result = relationship("JobResult", back_populates="job", uselist=False,
                          cascade="all, delete-orphan")

    __table_args__ = (
        Index("idx_jobs_user_status",    "user_id", "status"),
        Index("idx_jobs_status_created", "status",  "created_at"),
        Index("idx_jobs_queue",          "queue_name", "status"),
    )

    @property
    def execution_time_s(self) -> float | None:
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None

    def __repr__(self) -> str:
        return f"<Job {str(self.id)[:8]} type={self.job_type} status={self.status}>"


class JobResult(Base):
    __tablename__ = "job_results"

    id             = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_id         = Column(UUID(as_uuid=True), ForeignKey("jobs.id", ondelete="CASCADE"),
                            nullable=False, unique=True, index=True)

    # Execution outputs
    counts         = Column(JSONB, nullable=True)         # {"00": 512, "11": 512}
    statevector    = Column(JSONB, nullable=True)         # serialised complex array
    expectation_values = Column(JSONB, nullable=True)    # for variational jobs
    energy         = Column(Float, nullable=True)         # VQE ground state energy

    # AI + mitigation metadata
    backend_used   = Column(String(64), nullable=True)
    decision_reason = Column(Text, nullable=True)
    ai_confidence  = Column(Float, nullable=True)
    mitigation_method = Column(String(64), nullable=True)
    fidelity_before = Column(Float, nullable=True)
    fidelity_after  = Column(Float, nullable=True)
    truncation_error = Column(Float, nullable=True)       # MPS only

    # Performance
    execution_time_s = Column(Float, nullable=True)
    queue_time_s     = Column(Float, nullable=True)
    total_time_s     = Column(Float, nullable=True)

    # Full raw result blob (for debugging / detailed access)
    raw_result     = Column(JSONB, nullable=True)
    created_at     = Column(DateTime(timezone=True), default=_utcnow, nullable=False)

    job = relationship("Job", back_populates="result")
