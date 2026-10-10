"""QuantumJob — Async execution handle."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class QuantumJob:
    """
    Handle for a submitted quantum circuit execution job.

    For hardware backends, jobs execute asynchronously.
    For simulator backends, jobs complete synchronously.

    Attributes:
        job_id: Unique job identifier.
        backend_name: Name of the backend that is running this job.
        status: "QUEUED" | "RUNNING" | "DONE" | "FAILED" | "CANCELLED".
        result: ExecutionResult (set when status is "DONE").
        metadata: Optional extra job metadata from the provider.
    """
    job_id: str
    backend_name: str
    status: str = "QUEUED"
    result: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_done(self) -> bool:
        return self.status == "DONE"

    def is_failed(self) -> bool:
        return self.status == "FAILED"

    def __repr__(self) -> str:
        return (
            f"QuantumJob(id='{self.job_id[:8]}...', "
            f"backend='{self.backend_name}', status='{self.status}')"
        )
