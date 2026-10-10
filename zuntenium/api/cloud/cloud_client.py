"""
CloudClient — REST client for ZUNTENIUM Cloud platform.

Provides a full client SDK for interacting with the Zuntenium Cloud API:
    - Job submission (circuit, hybrid, variational)
    - Job status polling and result retrieval
    - Job cancellation
    - Backend discovery
    - Authentication (JWT + API key)

The client connects to the FastAPI-based cloud platform defined in
zuntenium_cloud/api.py.

Example::

    from zuntenium.api.cloud.cloud_client import CloudClient

    client = CloudClient(
        base_url="https://api.zuntenium.io/v1",
        api_key="your-api-key",
    )

    # Submit a circuit job
    job = client.submit(
        circuit=my_circuit,
        shots=4096,
        backend="auto",
    )

    # Wait for result
    result = client.wait_for_result(job["job_id"], timeout=120)
    print(result["counts"])
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any

logger = logging.getLogger(__name__)


class CloudClient:
    """
    REST client for ZUNTENIUM Cloud job execution.

    Supports both API key and JWT token authentication.

    Args:
        base_url: API base URL (default: "https://api.zuntenium.io/v1").
        api_key: API key for authentication.
        token: JWT access token (alternative to api_key).
        timeout_s: Default HTTP request timeout in seconds.
    """

    def __init__(
        self,
        base_url: str = "https://api.zuntenium.io/v1",
        api_key: str = "",
        token: str = "",
        timeout_s: float = 30.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._token = token
        self._timeout = timeout_s
        self._session: Any = None

    # ── Session management ────────────────────────────────────────────

    def _ensure_session(self) -> Any:
        """Lazy-init the HTTP session with auth headers."""
        if self._session is not None:
            return self._session

        try:
            import requests
            self._session = requests.Session()
            self._session.headers.update({
                "Content-Type": "application/json",
                "Accept": "application/json",
            })

            if self._api_key:
                self._session.headers["X-API-Key"] = self._api_key
            elif self._token:
                self._session.headers["Authorization"] = f"Bearer {self._token}"

            return self._session
        except ImportError:
            raise ImportError(
                "The 'requests' library is required for CloudClient. "
                "Install it with: pip install requests"
            )

    def _url(self, path: str) -> str:
        """Build full URL from path."""
        return f"{self._base_url}{path}"

    def _handle_response(self, response: Any) -> dict[str, Any]:
        """Parse response and raise on errors."""
        if response.status_code in (200, 201, 202):
            return response.json()

        error_detail = ""
        try:
            error_body = response.json()
            error_detail = error_body.get("detail", str(error_body))
        except Exception:
            error_detail = response.text[:500]

        raise CloudAPIError(
            f"API request failed with status {response.status_code}: "
            f"{error_detail}",
            status_code=response.status_code,
        )

    # ── Authentication ────────────────────────────────────────────────

    def login(self, email: str, password: str) -> dict[str, str]:
        """
        Authenticate with email/password and store the JWT token.

        Returns:
            Dict with "access_token" and "refresh_token".
        """
        session = self._ensure_session()
        response = session.post(
            self._url("/auth/token"),
            json={"email": email, "password": password},
            timeout=self._timeout,
        )
        data = self._handle_response(response)
        self._token = data.get("access_token", "")
        session.headers["Authorization"] = f"Bearer {self._token}"
        return data

    def refresh_token(self, refresh_token: str) -> dict[str, str]:
        """Refresh the access token using a refresh token."""
        session = self._ensure_session()
        response = session.post(
            self._url("/auth/refresh"),
            json={"refresh_token": refresh_token},
            timeout=self._timeout,
        )
        data = self._handle_response(response)
        self._token = data.get("access_token", "")
        session.headers["Authorization"] = f"Bearer {self._token}"
        return data

    # ── Job submission ────────────────────────────────────────────────

    def submit(
        self,
        circuit: Any = None,
        shots: int = 1024,
        backend: str = "auto",
        job_type: str = "circuit",
        optimization_mode: str = "rule",
        mitigation_enabled: bool = False,
        priority: int = 5,
        parameters: dict[str, float] | None = None,
        hybrid_config: dict[str, Any] | None = None,
        max_bond_dim: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Submit a quantum job to the cloud platform.

        Args:
            circuit: QuantumCircuit, QuantumIR, or circuit dict.
            shots: Number of measurement shots.
            backend: Backend name ("auto", "statevector", "tensornetwork",
                     "ibm", "ionq", etc.).
            job_type: "circuit", "hybrid", or "variational".
            optimization_mode: "rule" or "ai".
            mitigation_enabled: Enable error mitigation.
            priority: Job priority (1-10, higher = faster).
            parameters: Variational parameters (for variational jobs).
            hybrid_config: Hybrid solver configuration.
            max_bond_dim: MPS bond dimension (tensornetwork backend).

        Returns:
            Dict with job details including "job_id" and "status".
        """
        session = self._ensure_session()

        # Serialize the circuit
        circuit_payload = self._serialize_circuit(circuit)

        payload: dict[str, Any] = {
            "job_type": job_type,
            "shots": shots,
            "backend": backend,
            "optimization_mode": optimization_mode,
            "mitigation_enabled": mitigation_enabled,
            "priority": priority,
        }

        if circuit_payload:
            payload["circuit"] = circuit_payload
        if parameters:
            payload["parameters"] = parameters
        if hybrid_config:
            payload["hybrid_config"] = hybrid_config
        if max_bond_dim is not None:
            payload["max_bond_dim"] = max_bond_dim

        response = session.post(
            self._url("/jobs"),
            json=payload,
            timeout=self._timeout,
        )
        return self._handle_response(response)

    # ── Job status and results ────────────────────────────────────────

    def get_job(self, job_id: str) -> dict[str, Any]:
        """Get full job details."""
        session = self._ensure_session()
        response = session.get(
            self._url(f"/jobs/{job_id}"),
            timeout=self._timeout,
        )
        return self._handle_response(response)

    def get_status(self, job_id: str) -> dict[str, Any]:
        """Lightweight status poll."""
        session = self._ensure_session()
        response = session.get(
            self._url(f"/jobs/{job_id}/status"),
            timeout=self._timeout,
        )
        return self._handle_response(response)

    def get_result(self, job_id: str) -> dict[str, Any]:
        """
        Get the execution result for a completed job.

        Returns:
            Dict with "counts", "statevector" (if available),
            "probabilities", and execution metadata.
        """
        session = self._ensure_session()
        response = session.get(
            self._url(f"/jobs/{job_id}/result"),
            timeout=self._timeout,
        )
        return self._handle_response(response)

    def cancel(self, job_id: str) -> dict[str, Any]:
        """Cancel a pending or running job."""
        session = self._ensure_session()
        response = session.post(
            self._url(f"/jobs/{job_id}/cancel"),
            timeout=self._timeout,
        )
        return self._handle_response(response)

    def list_jobs(
        self,
        status: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> dict[str, Any]:
        """
        List jobs with optional filtering.

        Args:
            status: Filter by status ("QUEUED", "RUNNING", "DONE", "FAILED").
            limit: Maximum number of results.
            offset: Pagination offset.

        Returns:
            Dict with "jobs" list and "total" count.
        """
        session = self._ensure_session()
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status

        response = session.get(
            self._url("/jobs"),
            params=params,
            timeout=self._timeout,
        )
        return self._handle_response(response)

    # ── Polling helper ────────────────────────────────────────────────

    def wait_for_result(
        self,
        job_id: str,
        timeout: float = 600.0,
        poll_interval: float = 2.0,
    ) -> dict[str, Any]:
        """
        Poll until the job completes and return the result.

        Args:
            job_id: Job identifier.
            timeout: Maximum wait time in seconds.
            poll_interval: Seconds between polls.

        Returns:
            Execution result dict.

        Raises:
            TimeoutError: If the job doesn't complete within timeout.
            CloudAPIError: If the job fails.
        """
        start = time.time()
        while time.time() - start < timeout:
            status_data = self.get_status(job_id)
            status = status_data.get("status", "UNKNOWN")

            if status == "DONE":
                return self.get_result(job_id)

            if status in ("FAILED", "CANCELLED"):
                raise CloudAPIError(
                    f"Job {job_id} {status}: "
                    f"{status_data.get('error', 'Unknown error')}",
                    status_code=500,
                )

            time.sleep(poll_interval)

        raise TimeoutError(
            f"Job {job_id} did not complete within {timeout}s."
        )

    # ── Backend discovery ─────────────────────────────────────────────

    def list_backends(self) -> dict[str, Any]:
        """
        List available backends and their queue depths.

        Returns:
            Dict with "backends" list, each containing name, status,
            queue_depth, max_qubits, etc.
        """
        session = self._ensure_session()
        response = session.get(
            self._url("/backends"),
            timeout=self._timeout,
        )
        return self._handle_response(response)

    # ── Health check ──────────────────────────────────────────────────

    def health(self) -> dict[str, Any]:
        """Check platform health."""
        session = self._ensure_session()
        response = session.get(
            self._url("/health"),
            timeout=self._timeout,
        )
        return self._handle_response(response)

    # ── Circuit serialization ─────────────────────────────────────────

    def _serialize_circuit(self, circuit: Any) -> dict[str, Any] | None:
        """Convert a circuit object to the API's JSON schema."""
        if circuit is None:
            return None

        if isinstance(circuit, dict):
            return circuit

        # QuantumIR
        if hasattr(circuit, "instructions") and hasattr(circuit, "num_qubits"):
            gates: list[dict[str, Any]] = []
            for node in circuit.instructions:
                if hasattr(node, "is_gate") and node.is_gate():
                    gate_name = node.gate.name if node.gate else "unknown"
                    gates.append({
                        "name": gate_name.lower(),
                        "qubits": list(node.qubits),
                        "params": [float(p) for p in node.params],
                    })
            return {
                "num_qubits": circuit.num_qubits,
                "gates": gates,
            }

        # QuantumCircuit
        if hasattr(circuit, "to_ir"):
            ir = circuit.to_ir()
            return self._serialize_circuit(ir)

        # Transpiler v2 Circuit
        if hasattr(circuit, "gates") and hasattr(circuit, "num_qubits"):
            gates_list: list[dict[str, Any]] = []
            for gate in circuit.gates:
                if hasattr(gate, "is_measurement") and gate.is_measurement:
                    continue
                gates_list.append({
                    "name": gate.name.lower(),
                    "qubits": list(gate.qubits),
                    "params": [float(p) for p in (gate.params or [])],
                })
            return {
                "num_qubits": circuit.num_qubits,
                "gates": gates_list,
            }

        logger.warning("Unknown circuit type: %s", type(circuit))
        return None

    # ── Context manager ───────────────────────────────────────────────

    def __enter__(self) -> CloudClient:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def close(self) -> None:
        """Close the HTTP session."""
        if self._session is not None:
            self._session.close()
            self._session = None

    def __repr__(self) -> str:
        auth = "api_key" if self._api_key else ("jwt" if self._token else "none")
        return f"CloudClient(url='{self._base_url}', auth={auth})"


# ── Custom exception ──────────────────────────────────────────────────

class CloudAPIError(Exception):
    """Raised when a ZUNTENIUM Cloud API request fails."""

    def __init__(self, message: str, status_code: int = 0) -> None:
        super().__init__(message)
        self.status_code = status_code


__all__ = ["CloudClient", "CloudAPIError"]
