# zuntenium_cloud/schemas.py
"""
API request/response schemas — Pydantic v2.

All schemas are versioned. Breaking changes get a new schema class.
JSON serialisation uses camelCase aliases for API surface; internal
Python code uses snake_case.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class _Base(BaseModel):
    model_config = ConfigDict(populate_by_name=True, from_attributes=True)


# ── Circuit representation ────────────────────────────────────────────────────

class GateSchema(_Base):
    """Serialised quantum gate."""
    name:   str              = Field(description="Gate name: 'h', 'cx', 'rx', etc.")
    qubits: list[int]        = Field(description="Target qubit indices (0-based)")
    params: list[float]      = Field(default_factory=list,
                                     description="Gate parameters (angles, etc.)")

    @field_validator("qubits")
    @classmethod
    def qubits_nonneg(cls, v: list[int]) -> list[int]:
        if any(q < 0 for q in v):
            raise ValueError("Qubit indices must be non-negative")
        return v


class CircuitSchema(_Base):
    """Serialised quantum circuit."""
    num_qubits: int              = Field(ge=1, le=200, description="Number of qubits")
    gates:      list[GateSchema] = Field(description="Ordered list of gates")
    name:       str | None       = Field(default=None, description="Optional circuit label")

    @field_validator("gates")
    @classmethod
    def validate_qubit_refs(cls, gates: list[GateSchema]) -> list[GateSchema]:
        return gates


# ── Job submission ─────────────────────────────────────────────────────────────

class JobSubmitRequest(_Base):
    """POST /jobs — submit a new quantum job."""
    job_type:   str = Field(default="circuit", description="'circuit', 'hybrid', 'variational'")
    circuit:    CircuitSchema | None = None
    hybrid_config: dict[str, Any] | None = Field(
        default=None,
        description="HybridSolver configuration dict",
    )
    parameters: dict[str, float] | None = Field(
        default=None,
        description="Variational parameters (name → value)",
    )
    # Execution preferences
    shots:             int  = Field(default=1024, ge=1, le=1_000_000)
    backend:           str  = Field(default="auto",
                                    description="'auto', local simulators, or configured IBM/IonQ/Braket/Rigetti hardware")
    mitigation_enabled: bool = Field(default=False)
    optimization_mode: str   = Field(default="rule",
                                     description="'rule' or 'ai'")
    max_bond_dim:      int | None = Field(default=None, ge=1, le=1024,
                                          description="MPS bond dimension (tensornetwork only)")
    priority:          int   = Field(default=5, ge=1, le=10,
                                     description="Job priority (higher = faster)")

    @field_validator("job_type")
    @classmethod
    def valid_job_type(cls, v: str) -> str:
        allowed = {"circuit", "hybrid", "variational"}
        if v not in allowed:
            raise ValueError(f"job_type must be one of {allowed}")
        return v

    @field_validator("backend")
    @classmethod
    def valid_backend(cls, v: str) -> str:
        allowed = {"auto", "statevector", "tensornetwork", "stabilizer", "gpu_statevector", "ibm", "ionq", "braket", "rigetti", "simulator"}
        if v not in allowed:
            raise ValueError(f"backend must be one of {allowed}")
        return v


class JobResponse(_Base):
    """Response for GET /jobs/{id} and POST /jobs."""
    id:           str
    status:       str
    job_type:     str
    created_at:   datetime
    queued_at:    datetime | None = None
    started_at:   datetime | None = None
    completed_at: datetime | None = None
    retry_count:  int = 0
    last_error:   str | None = None
    user_id:      str
    queue_name:   str
    priority:     int


class JobStatusResponse(_Base):
    """Lightweight status poll response."""
    id:          str
    status:      str
    message:     str | None = None
    progress:    float | None = Field(default=None, description="0.0–1.0 for variational jobs")
    position_in_queue: int | None = None


class JobResultResponse(_Base):
    """GET /jobs/{id}/result — full execution result."""
    job_id:         str
    status:         str
    counts:         dict[str, int] | None = None
    statevector:    list[list[float]] | None = None   # [[real, imag], ...]
    expectation_values: dict[str, float] | None = None
    energy:         float | None = None

    # Execution metadata
    backend_used:        str | None = None
    decision_reason:     str | None = None
    ai_confidence:       float | None = None
    mitigation_method:   str | None = None
    fidelity_before:     float | None = None
    fidelity_after:      float | None = None
    truncation_error:    float | None = None
    execution_time_s:    float | None = None
    queue_time_s:        float | None = None
    total_time_s:        float | None = None


class JobListResponse(_Base):
    """Paginated list of jobs."""
    jobs:       list[JobResponse]
    total:      int
    page:       int
    page_size:  int
    pages:      int


# ── Backend info ──────────────────────────────────────────────────────────────

class BackendInfo(_Base):
    name:         str
    available:    bool
    description:  str
    max_qubits:   int | None = None
    queue_depth:  int | None = None   # jobs currently queued for this backend
    avg_wait_s:   float | None = None


class BackendListResponse(_Base):
    backends: list[BackendInfo]


# ── Error responses ───────────────────────────────────────────────────────────

class ErrorDetail(_Base):
    code:    str
    message: str
    field:   str | None = None


class ErrorResponse(_Base):
    error:   str
    details: list[ErrorDetail] = Field(default_factory=list)
    request_id: str | None = None
