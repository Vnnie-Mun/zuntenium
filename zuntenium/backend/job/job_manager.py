"""
JobManager — Local job queue and status tracking.
"""
from __future__ import annotations

import threading
from typing import Any

from zuntenium.backend.job.job import QuantumJob


class JobManager:
    """
    Thread-safe local job queue with status tracking.

    Manages a dict of QuantumJob objects. For v0.1 (simulator-only),
    all jobs complete synchronously but are tracked here for monitoring.
    """

    def __init__(self) -> None:
        self._jobs: dict[str, QuantumJob] = {}
        self._lock = threading.Lock()

    def register(self, job: QuantumJob) -> None:
        with self._lock:
            self._jobs[job.job_id] = job

    def get(self, job_id: str) -> QuantumJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def update_status(self, job_id: str, status: str, result: Any = None) -> None:
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].status = status
                if result is not None:
                    self._jobs[job_id].result = result

    def list_jobs(self, status: str | None = None) -> list[QuantumJob]:
        with self._lock:
            jobs = list(self._jobs.values())
        if status:
            jobs = [j for j in jobs if j.status == status]
        return jobs

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            if job_id in self._jobs:
                job = self._jobs[job_id]
                if job.status in ("QUEUED", "RUNNING"):
                    job.status = "CANCELLED"
                    return True
        return False

    @property
    def num_jobs(self) -> int:
        return len(self._jobs)
