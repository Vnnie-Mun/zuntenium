# zuntenium/mhos/connectors/base.py
"""
BaseConnector — Abstract base for all hardware provider connectors.

Extends QuantumBackend(ABC) with hardware-specific lifecycle methods:
  submit()      — async non-blocking job submission
  poll_status() — check job state without blocking
  fetch_result()— retrieve completed result
  cancel()      — cancel a running job
  list_devices()— enumerate available devices on this provider

All connectors share identical async interfaces so HardwareOrchestrator
can treat IBM, Braket, IonQ, and Rigetti identically.

Error handling:
  ConnectorError    — all provider errors wrapped in this
  RetryableError    — transient (HTTP 429, queue full) — trigger retry
  FatalError        — unrecoverable (bad circuit, auth failed) — fail job
"""

from __future__ import annotations

import abc
import asyncio
import enum
import logging
import time
from dataclasses import dataclass, field
from typing import Any

_log = logging.getLogger(__name__)


# ── Error hierarchy ───────────────────────────────────────────────────────────

class ConnectorError(Exception):
    """Base for all connector errors."""
    def __init__(self, message: str, provider: str = "unknown", retryable: bool = False):
        super().__init__(message)
        self.provider  = provider
        self.retryable = retryable


class RetryableError(ConnectorError):
    """Transient error — safe to retry after backoff."""
    def __init__(self, message: str, provider: str = "unknown", retry_after_s: float = 5.0):
        super().__init__(message, provider, retryable=True)
        self.retry_after_s = retry_after_s


class FatalError(ConnectorError):
    """Unrecoverable error — fail the job."""
    def __init__(self, message: str, provider: str = "unknown"):
        super().__init__(message, provider, retryable=False)


# ── Job state ─────────────────────────────────────────────────────────────────

class JobState(str, enum.Enum):
    QUEUED     = "queued"
    RUNNING    = "running"
    COMPLETED  = "completed"
    FAILED     = "failed"
    CANCELLED  = "cancelled"
    UNKNOWN    = "unknown"


@dataclass
class JobHandle:
    """
    Reference to a submitted hardware job.
    Provider-specific ``native_id`` is the job ID on the provider's system.
    """
    native_id:   str
    provider:    str
    device_id:   str
    submitted_at: float = field(default_factory=time.time)
    state:       JobState = JobState.QUEUED
    metadata:    dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"JobHandle({self.provider}:{self.native_id} [{self.state.value}])"


@dataclass
class ProviderResult:
    """
    Normalised execution result from any provider.
    Always in terms of Zuntenium-standard bitstring counts.
    """
    job_handle:      JobHandle
    counts:          dict[str, int]
    shots:           int
    execution_time_s: float = 0.0
    queue_time_s:    float = 0.0

    # Optional detailed data (provider-dependent)
    statevector:     Any | None = None
    density_matrix:  Any | None = None
    probabilities:   dict[str, float] | None = None

    # Provider metadata
    device_id:       str = ""
    backend_version: str = ""
    metadata:        dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return bool(self.counts) or self.statevector is not None


# ── BaseConnector ─────────────────────────────────────────────────────────────

class BaseConnector(abc.ABC):
    """
    Abstract base for all hardware provider connectors.

    Subclass for each provider (IBM, Braket, IonQ, Rigetti).
    All I/O operations are async to allow concurrent polling across providers.

    Parameters
    ----------
    credentials : dict  — provider-specific auth (tokens, keys, arns, etc.)
    region      : str   — cloud region (for Braket/IBM)
    timeout_s   : float — maximum wait time for a submitted job
    """

    PROVIDER_NAME: str = "base"  # Override in subclass

    def __init__(
        self,
        credentials: dict[str, str],
        region: str = "us-east-1",
        timeout_s: float = 3600.0,
        poll_interval_s: float = 5.0,
        max_retries: int = 5,
    ) -> None:
        self._creds          = credentials
        self._region         = region
        self._timeout        = timeout_s
        self._poll_interval  = poll_interval_s
        self._max_retries    = max_retries
        self._log            = logging.getLogger(
            f"{__name__}.{self.__class__.__name__}"
        )
        self._authenticated  = False

    # ── Abstract interface ────────────────────────────────────────────────────

    @abc.abstractmethod
    async def authenticate(self) -> None:
        """
        Authenticate with the provider. Called once before any other method.
        Must set self._authenticated = True on success.
        """

    @abc.abstractmethod
    async def list_devices(self) -> list[dict]:
        """
        Return list of available devices as raw provider dicts.
        The HardwareRegistry converts these to DeviceProfile objects.
        """

    @abc.abstractmethod
    def circuit_to_native(self, circuit) -> Any:
        """
        Translate a Zuntenium Circuit object to this provider's native format.
        E.g. IBM → Qiskit QuantumCircuit, Braket → braket.Circuit, etc.
        """

    @abc.abstractmethod
    async def submit(
        self,
        circuit,
        device_id: str,
        shots: int = 1024,
        **kwargs,
    ) -> JobHandle:
        """
        Submit a circuit to the specified device.
        Returns a JobHandle immediately (non-blocking).
        """

    @abc.abstractmethod
    async def poll_status(self, handle: JobHandle) -> JobState:
        """Check the current state of a submitted job."""

    @abc.abstractmethod
    async def fetch_result(self, handle: JobHandle) -> ProviderResult:
        """
        Retrieve the result of a completed job.
        Raises ConnectorError if job is not COMPLETED.
        """

    @abc.abstractmethod
    async def cancel(self, handle: JobHandle) -> bool:
        """Cancel a submitted job. Returns True if cancelled."""

    @abc.abstractmethod
    async def get_device_status(self, device_id: str) -> dict:
        """Return real-time status/calibration for one device."""

    # ── Concrete helpers ──────────────────────────────────────────────────────

    async def run(
        self,
        circuit,
        device_id: str,
        shots: int = 1024,
        **kwargs,
    ) -> ProviderResult:
        """
        High-level blocking run: submit → poll → fetch result.
        Integrates retry logic with exponential backoff.
        """
        if not self._authenticated:
            await self.authenticate()

        handle = await self._submit_with_retry(circuit, device_id, shots, **kwargs)
        result = await self._wait_for_result(handle)
        return result

    async def _submit_with_retry(
        self, circuit, device_id: str, shots: int, **kwargs
    ) -> JobHandle:
        """Submit with exponential backoff on RetryableError."""
        last_exc = None
        for attempt in range(self._max_retries):
            try:
                handle = await self.submit(circuit, device_id, shots, **kwargs)
                self._log.info(
                    "[%s] Job submitted: %s (attempt %d)",
                    self.PROVIDER_NAME, handle.native_id, attempt + 1,
                )
                return handle
            except RetryableError as exc:
                last_exc  = exc
                wait_time = min(exc.retry_after_s * (2 ** attempt), 60)
                self._log.warning(
                    "[%s] Submit retryable error (attempt %d/%d): %s — waiting %.0fs",
                    self.PROVIDER_NAME, attempt + 1, self._max_retries, exc, wait_time,
                )
                await asyncio.sleep(wait_time)
            except FatalError:
                raise
            except Exception as exc:
                raise ConnectorError(
                    f"Unexpected submit error: {exc}", self.PROVIDER_NAME
                ) from exc
        raise ConnectorError(
            f"Submit failed after {self._max_retries} attempts: {last_exc}",
            self.PROVIDER_NAME,
        )

    async def _wait_for_result(self, handle: JobHandle) -> ProviderResult:
        """
        Poll until job is COMPLETED or FAILED, then fetch result.
        Raises ConnectorError on timeout or failure.
        """
        t_start   = time.monotonic()
        interval  = self._poll_interval

        while True:
            elapsed = time.monotonic() - t_start
            if elapsed > self._timeout:
                await self.cancel(handle)
                raise ConnectorError(
                    f"Job {handle.native_id} timed out after {elapsed:.0f}s",
                    self.PROVIDER_NAME, retryable=True,
                )

            state = await self.poll_status(handle)
            handle.state = state

            self._log.debug(
                "[%s] Job %s: %s (%.0fs elapsed)",
                self.PROVIDER_NAME, handle.native_id, state.value, elapsed,
            )

            if state == JobState.COMPLETED:
                return await self.fetch_result(handle)

            if state == JobState.FAILED:
                raise ConnectorError(
                    f"Job {handle.native_id} failed on provider",
                    self.PROVIDER_NAME, retryable=False,
                )

            if state == JobState.CANCELLED:
                raise ConnectorError(
                    f"Job {handle.native_id} was cancelled",
                    self.PROVIDER_NAME, retryable=True,
                )

            # Adaptive polling: start fast, slow down after 30s
            if elapsed > 30:
                interval = min(interval * 1.5, 60.0)
            await asyncio.sleep(interval)

    # ── Bitstring normalisation ───────────────────────────────────────────────

    @staticmethod
    def _normalise_counts(
        raw_counts: dict[str, int],
        n_qubits: int,
        reverse_bits: bool = False,
    ) -> dict[str, int]:
        """
        Normalise provider-specific count dicts to Zuntenium little-endian format.

        Some providers return "0x0" hex, some return "01" with MSB first,
        some return reversed bitstrings. This handles all cases.
        """
        normalised: dict[str, int] = {}
        for key, count in raw_counts.items():
            k = str(key).strip()
            # Hex format (Braket): "0x0f"
            if k.startswith("0x") or k.startswith("0X"):
                idx = int(k, 16)
                k   = format(idx, f"0{n_qubits}b")
            # Integer format
            elif k.isdigit():
                idx = int(k)
                k   = format(idx, f"0{n_qubits}b")
            # Ensure correct length
            k = k.zfill(n_qubits)[-n_qubits:]
            # Reverse if provider is MSB-first (IBM returns MSB-first by default)
            if reverse_bits:
                k = k[::-1]
            normalised[k] = normalised.get(k, 0) + count
        return normalised
