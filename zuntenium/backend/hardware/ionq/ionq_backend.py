"""
IonQ hardware backend adapter.

Connects to the IonQ Cloud REST API for quantum circuit execution
on trapped-ion hardware (Harmony, Aria, Forte).

API reference: https://docs.ionq.com/api-reference

Features:
    - Circuit translation to IonQ native JSON format
    - Async job submission and polling
    - Result retrieval with probability → counts conversion
    - Job cancellation
    - Graceful fallback when API key is not configured
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from typing import TYPE_CHECKING, Any

from zuntenium.backend.hardware.hardware_backend import AbstractHardwareBackend

if TYPE_CHECKING:
    from zuntenium.backend.job.job import QuantumJob
    from zuntenium.circuit.circuit import QuantumCircuit
    from zuntenium.quantum_core.executor.result import ExecutionResult

logger = logging.getLogger(__name__)

# ── Gate translation tables ───────────────────────────────────────────

# IonQ native gate set uses lowercase names and a specific JSON schema.
_ZUNTENIUM_TO_IONQ: dict[str, str] = {
    "H": "h",
    "X": "x",
    "Y": "y",
    "Z": "z",
    "S": "s",
    "T": "t",
    "SDG": "si",
    "TDG": "ti",
    "Rx": "rx",
    "RX": "rx",
    "Ry": "ry",
    "RY": "ry",
    "Rz": "rz",
    "RZ": "rz",
    "CX": "cnot",
    "CNOT": "cnot",
    "CZ": "cz",
    "SWAP": "swap",
    "CRz": "crz",
    "CRZ": "crz",
    "CCX": "ccx",
}


def _circuit_to_ionq_json(
    circuit: Any,
    num_qubits: int | None = None,
) -> dict[str, Any]:
    """
    Translate a Zuntenium circuit to IonQ's JSON circuit format.

    IonQ format:
        {
            "qubits": N,
            "circuit": [
                {"gate": "h", "target": 0},
                {"gate": "cnot", "control": 0, "target": 1},
                {"gate": "rx", "rotation": 0.5, "target": 0},
                ...
            ]
        }
    """
    gates_json: list[dict[str, Any]] = []

    # Handle both QuantumIR and QuantumCircuit
    instructions: list[Any] = []
    n_qubits = num_qubits or 1

    if hasattr(circuit, "instructions"):
        # QuantumIR
        instructions = circuit.instructions
        n_qubits = num_qubits or circuit.num_qubits
    elif hasattr(circuit, "to_ir"):
        # QuantumCircuit
        ir = circuit.to_ir()
        instructions = ir.instructions
        n_qubits = num_qubits or ir.num_qubits
    elif hasattr(circuit, "gates"):
        # Transpiler v2 Circuit
        n_qubits = num_qubits or circuit.num_qubits
        for gate in circuit.gates:
            if gate.is_measurement:
                continue
            ionq_name = _ZUNTENIUM_TO_IONQ.get(gate.name)
            if ionq_name is None:
                logger.warning(
                    "Gate '%s' has no IonQ mapping — skipping.", gate.name
                )
                continue

            entry: dict[str, Any] = {"gate": ionq_name}

            qubits = gate.qubits
            if hasattr(gate, "controls") and gate.controls:
                entry["control"] = gate.controls[0]
                entry["target"] = qubits[0]
            elif len(qubits) == 2:
                entry["control"] = qubits[0]
                entry["target"] = qubits[1]
            elif len(qubits) == 3:
                entry["controls"] = [qubits[0], qubits[1]]
                entry["target"] = qubits[2]
            else:
                entry["target"] = qubits[0]

            if gate.params:
                entry["rotation"] = float(gate.params[0])

            gates_json.append(entry)

        return {"qubits": n_qubits, "circuit": gates_json}

    # Process QuantumIR instructions
    for node in instructions:
        if not hasattr(node, "is_gate") or not node.is_gate():
            continue

        gate_name = node.gate.name if node.gate else node.operation
        ionq_name = _ZUNTENIUM_TO_IONQ.get(gate_name)
        if ionq_name is None:
            logger.warning(
                "Gate '%s' has no IonQ mapping — skipping.", gate_name
            )
            continue

        entry = {"gate": ionq_name}
        qubits = node.qubits

        if len(qubits) == 2:
            entry["control"] = qubits[0]
            entry["target"] = qubits[1]
        elif len(qubits) == 3:
            entry["controls"] = [qubits[0], qubits[1]]
            entry["target"] = qubits[2]
        else:
            entry["target"] = qubits[0]

        if node.params:
            entry["rotation"] = float(node.params[0])

        gates_json.append(entry)

    return {"qubits": n_qubits, "circuit": gates_json}


class IonQBackend(AbstractHardwareBackend):
    """
    IonQ hardware backend.

    Submits circuits to IonQ Cloud via the REST API.
    Supports Harmony (11 qubits), Aria (25 qubits), and Forte (32 qubits).

    If no API key is configured, falls back to IonQ's cloud simulator.

    Args:
        device: IonQ device name: "ionq_harmony", "ionq_aria", "ionq_forte",
                or "simulator" for IonQ's cloud simulator.
        credentials: IonQCredentials object (or loaded from env).
        poll_interval_s: Seconds between job status polls (default 2.0).
        max_poll_time_s: Maximum time to wait for job completion (default 600).
    """

    name = "ionq_quantum"
    max_qubits = 29      # IonQ Aria
    supported_gates = [
        "H", "X", "Y", "Z", "S", "T", "SDG", "TDG",
        "Rx", "Ry", "Rz", "CX", "CZ", "SWAP", "CCX",
    ]
    is_simulator = False

    # Device qubit limits
    _DEVICE_LIMITS: dict[str, int] = {
        "ionq_harmony": 11,
        "ionq_aria": 25,
        "ionq_forte": 32,
        "simulator": 29,
    }

    def __init__(
        self,
        device: str = "simulator",
        credentials: object | None = None,
        poll_interval_s: float = 2.0,
        max_poll_time_s: float = 600.0,
    ) -> None:
        self._device = device
        self._poll_interval = poll_interval_s
        self._max_poll_time = max_poll_time_s

        # Resolve credentials
        if credentials is None:
            from zuntenium.backend.hardware.ionq.ionq_credentials import (
                IonQCredentials,
            )
            credentials = IonQCredentials.from_env()

        self._credentials = credentials
        self._api_key = getattr(credentials, "api_key", "")
        self._base_url = getattr(
            credentials, "base_url", "https://api.ionq.co/v0.3"
        )
        self.max_qubits = self._DEVICE_LIMITS.get(device, 29)

        # Import requests lazily
        self._session: Any = None

    def _ensure_session(self) -> Any:
        """Lazy-init the HTTP session."""
        if self._session is not None:
            return self._session
        try:
            import requests
            self._session = requests.Session()
            self._session.headers.update({
                "Authorization": f"apiKey {self._api_key}",
                "Content-Type": "application/json",
            })
            return self._session
        except ImportError:
            raise ImportError(
                "The 'requests' library is required for the IonQ backend. "
                "Install it with: pip install requests"
            )

    def _api_available(self) -> bool:
        """Check if the IonQ API is reachable and key is configured."""
        return bool(self._api_key)

    # ── Job submission ────────────────────────────────────────────────

    def submit_job(
        self,
        circuit: Any,
        shots: int = 1024,
        **options: Any,
    ) -> QuantumJob:
        """
        Submit a circuit to IonQ Cloud.

        Returns a QuantumJob handle for polling.
        """
        from zuntenium.backend.job.job import QuantumJob

        if not self._api_available():
            # Offline simulation fallback
            return self._simulate_locally(circuit, shots, **options)

        session = self._ensure_session()

        # Build the IonQ job payload
        circuit_json = _circuit_to_ionq_json(circuit)
        payload = {
            "target": self._device,
            "shots": shots,
            "input": circuit_json,
        }

        # Optional: name, metadata
        name = options.get("name", f"zuntenium-{uuid.uuid4().hex[:8]}")
        payload["name"] = name

        response = session.post(
            f"{self._base_url}/jobs",
            data=json.dumps(payload),
        )

        if response.status_code not in (200, 201):
            error_msg = response.text[:500]
            logger.error("IonQ job submission failed: %s", error_msg)
            return QuantumJob(
                job_id=f"failed-{uuid.uuid4().hex[:8]}",
                backend_name=self.name,
                status="FAILED",
                metadata={"error": error_msg},
            )

        result = response.json()
        job_id = result.get("id", f"ionq-{uuid.uuid4().hex[:8]}")

        return QuantumJob(
            job_id=job_id,
            backend_name=self.name,
            status="QUEUED",
            metadata={
                "device": self._device,
                "shots": shots,
                "ionq_response": result,
            },
        )

    def fetch_result(self, job_id: str) -> Any:
        """
        Fetch the result of a completed IonQ job.

        Returns a dict with keys: status, probabilities, counts, metadata.
        """
        if not self._api_available():
            return {"status": "UNAVAILABLE", "error": "No API key configured"}

        session = self._ensure_session()
        response = session.get(f"{self._base_url}/jobs/{job_id}")

        if response.status_code != 200:
            return {
                "status": "ERROR",
                "error": response.text[:500],
            }

        data = response.json()
        status = data.get("status", "unknown")

        result: dict[str, Any] = {
            "status": status.upper(),
            "metadata": data.get("metadata", {}),
        }

        if status == "completed":
            result["status"] = "DONE"
            # IonQ returns probabilities, not counts
            probs = data.get("data", {}).get("probabilities", {})
            shots = data.get("shots", 1024)

            # Convert probability keys (integers) to bitstrings
            n_qubits = data.get("qubits", 1)
            counts: dict[str, int] = {}
            for state_key, prob in probs.items():
                idx = int(state_key)
                bitstring = format(idx, f"0{n_qubits}b")
                counts[bitstring] = int(round(prob * shots))

            result["probabilities"] = probs
            result["counts"] = counts
            result["shots"] = shots

        return result

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a pending or running IonQ job."""
        if not self._api_available():
            logger.warning("Cannot cancel job — no API key configured.")
            return False

        session = self._ensure_session()
        response = session.put(
            f"{self._base_url}/jobs/{job_id}/status/cancel",
        )
        return response.status_code in (200, 204)

    # ── Synchronous run ───────────────────────────────────────────────

    def run(
        self,
        circuit: Any,
        shots: int = 1024,
        **options: Any,
    ) -> ExecutionResult:
        """
        Submit a circuit and wait for the result.

        If no API key is configured, uses a local simulator fallback.
        """
        from zuntenium.quantum_core.executor.result import ExecutionResult

        if not self._api_available():
            return self._run_local_fallback(circuit, shots, **options)

        # Submit
        job = self.submit_job(circuit, shots, **options)

        if job.status == "FAILED":
            return ExecutionResult(
                counts={},
                num_qubits=getattr(circuit, "num_qubits", 1),
                shots=shots,
                metadata={"error": job.metadata.get("error", "Unknown")},
            )

        # Poll until done
        start = time.time()
        while time.time() - start < self._max_poll_time:
            result = self.fetch_result(job.job_id)
            status = result.get("status", "UNKNOWN")

            if status == "DONE":
                return ExecutionResult(
                    counts=result.get("counts", {}),
                    num_qubits=getattr(circuit, "num_qubits", 1),
                    shots=result.get("shots", shots),
                    metadata={
                        "backend": self.name,
                        "device": self._device,
                        "job_id": job.job_id,
                        **result.get("metadata", {}),
                    },
                )

            if status in ("FAILED", "CANCELLED", "ERROR"):
                return ExecutionResult(
                    counts={},
                    num_qubits=getattr(circuit, "num_qubits", 1),
                    shots=shots,
                    metadata={
                        "error": result.get("error", f"Job {status}"),
                        "job_id": job.job_id,
                    },
                )

            time.sleep(self._poll_interval)

        # Timeout
        return ExecutionResult(
            counts={},
            num_qubits=getattr(circuit, "num_qubits", 1),
            shots=shots,
            metadata={"error": "Job timed out", "job_id": job.job_id},
        )

    # ── Local fallback ────────────────────────────────────────────────

    def _run_local_fallback(
        self, circuit: Any, shots: int, **options: Any
    ) -> ExecutionResult:
        """
        Use the local statevector simulator as a fallback when no
        IonQ API key is available.
        """
        from zuntenium.quantum_core.executor.result import ExecutionResult

        logger.info(
            "IonQ API key not configured — using local simulator fallback."
        )

        try:
            from zuntenium.backends.ibm_backend import SimulatorBackend
            sim = SimulatorBackend(
                n_qubits=getattr(circuit, "num_qubits", 10)
            )
            result = sim.execute(circuit, shots=shots)
            # Wrap in ExecutionResult if needed
            if isinstance(result, ExecutionResult):
                return result
            return ExecutionResult(
                counts=getattr(result, "counts", {}),
                num_qubits=getattr(circuit, "num_qubits", 1),
                shots=shots,
                statevector=getattr(result, "statevector", None),
                metadata={
                    "backend": "ionq_simulator_fallback",
                    "device": "local",
                },
            )
        except Exception as exc:
            logger.error("Local fallback failed: %s", exc)
            return ExecutionResult(
                counts={},
                num_qubits=getattr(circuit, "num_qubits", 1),
                shots=shots,
                metadata={"error": str(exc)},
            )

    def _simulate_locally(
        self, circuit: Any, shots: int, **options: Any
    ) -> QuantumJob:
        """Create a completed QuantumJob using local simulation."""
        from zuntenium.backend.job.job import QuantumJob

        result = self._run_local_fallback(circuit, shots, **options)
        return QuantumJob(
            job_id=f"local-{uuid.uuid4().hex[:8]}",
            backend_name="ionq_simulator_fallback",
            status="DONE",
            result=result,
            metadata={"fallback": True},
        )
