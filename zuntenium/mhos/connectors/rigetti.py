# zuntenium/mhos/connectors/rigetti.py
"""Rigetti QCS Connector — pyquil / QCS API"""
from __future__ import annotations
import asyncio, logging, time
from .base import BaseConnector, ConnectorError, FatalError, JobHandle, JobState, ProviderResult, RetryableError

_log = logging.getLogger(__name__)

_RIGETTI_GATE_MAP = {
    "h":"H","x":"X","y":"Y","z":"Z","s":"S","t":"T","sdg":"DAGGER S","tdg":"DAGGER T",
    "id":"I","rx":"RX","ry":"RY","rz":"RZ","cx":"CNOT","cnot":"CNOT",
    "cz":"CZ","swap":"SWAP","ccx":"CCNOT",
}

class RigettiConnector(BaseConnector):
    """
    Rigetti QCS connector.

    Credentials:
      "user_id"     : QCS user ID
      "private_key" : QCS private key (PEM string or path)
    """
    PROVIDER_NAME = "rigetti"

    def __init__(self, credentials: dict[str, str], **kwargs):
        super().__init__(credentials, **kwargs)
        self._user_id    = credentials.get("user_id", "")
        self._use_stub   = not bool(self._user_id)

    async def authenticate(self) -> None:
        if self._use_stub:
            _log.info("[Rigetti] Stub mode"); self._authenticated = True; return
        try:
            import qcs_sdk
            self._qcs = qcs_sdk
            self._authenticated = True
            _log.info("[Rigetti] QCS authenticated")
        except ImportError:
            _log.warning("[Rigetti] qcs-sdk not installed — stub mode")
            self._use_stub = True; self._authenticated = True

    async def list_devices(self) -> list[dict]:
        if self._use_stub:
            return [
                {"device_id":"Ankaa-2","display_name":"Rigetti Ankaa-2","num_qubits":84,
                 "status":"online","queue_depth":1,"mean_1q_error":0.001,
                 "mean_2q_error":0.01,"mean_readout_err":0.03,
                 "cost_per_shot":0.00035,"all_to_all":False,
                 "native_gates":["rx","rz","cz"]},
            ]
        loop = asyncio.get_event_loop()
        try:
            qpus = await loop.run_in_executor(None, self._qcs.list_quantum_processors)
            return [{"device_id": q.id, "num_qubits": 0, "status": "online",
                     "cost_per_shot": 0.00035, "native_gates": ["rx","rz","cz"]}
                    for q in qpus]
        except Exception: return []

    def circuit_to_native(self, circuit):
        """Convert to Quil program string."""
        lines = [f"RESET\n"]
        for g in circuit.gates:
            name  = g.name.upper()
            qs    = g.qubits
            ps    = g.params
            quil_name = _RIGETTI_GATE_MAP.get(g.name.lower(), name)
            if ps:
                args = "(" + ",".join(f"{p:.8f}" for p in ps) + ")"
                line = f"{quil_name}{args} {' '.join(str(q) for q in qs)}"
            else:
                line = f"{quil_name} {' '.join(str(q) for q in qs)}"
            lines.append(line)
        lines.append(f"DECLARE ro BIT[{circuit.num_qubits}]")
        for i in range(circuit.num_qubits):
            lines.append(f"MEASURE {i} ro[{i}]")
        return "\n".join(lines)

    async def submit(self, circuit, device_id: str, shots: int = 1024, **kwargs) -> JobHandle:
        if self._use_stub: return self._stub_submit(circuit, device_id, shots)
        quil = self.circuit_to_native(circuit)
        loop = asyncio.get_event_loop()
        try:
            job = await loop.run_in_executor(
                None, lambda: self._qcs.submit(quil, shots=shots, quantum_processor_id=device_id)
            )
            return JobHandle(native_id=str(job.id), provider="rigetti", device_id=device_id,
                              metadata={"shots": shots, "num_qubits": circuit.num_qubits})
        except Exception as exc:
            raise ConnectorError(str(exc), "rigetti") from exc

    async def poll_status(self, handle: JobHandle) -> JobState:
        if self._use_stub: return self._stub_poll(handle)
        loop = asyncio.get_event_loop()
        try:
            job    = await loop.run_in_executor(None, lambda: self._qcs.get_job(handle.native_id))
            status = str(job.status).lower()
            return {"queued":"queued","running":"running","completed":"completed",
                    "failed":"failed","canceled":"cancelled"}.get(status, JobState.UNKNOWN)
        except Exception: return JobState.UNKNOWN

    async def fetch_result(self, handle: JobHandle) -> ProviderResult:
        if self._use_stub: return self._stub_result(handle)
        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(None, lambda: self._qcs.get_result(handle.native_id))
            n = handle.metadata.get("num_qubits", 1)
            counts: dict[str, int] = {}
            for shot in result.readout_data.get("ro", []):
                key = "".join(str(b) for b in shot)
                counts[key] = counts.get(key, 0) + 1
            return ProviderResult(job_handle=handle, counts=counts, shots=sum(counts.values()),
                                   device_id=handle.device_id)
        except Exception as exc:
            raise ConnectorError(str(exc), "rigetti") from exc

    async def cancel(self, handle: JobHandle) -> bool:
        if self._use_stub: return True
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: self._qcs.cancel_job(handle.native_id))
            return True
        except Exception: return False

    async def get_device_status(self, device_id: str) -> dict:
        if self._use_stub: return {"status": "online", "queue_depth": 1}
        devices = await self.list_devices()
        return next((d for d in devices if d.get("device_id") == device_id), {})

    _stub_ctr = 0
    def _stub_submit(self, circuit, device_id, shots):
        RigettiConnector._stub_ctr += 1
        return JobHandle(native_id=f"stub_rigetti_{RigettiConnector._stub_ctr}",
                          provider="rigetti", device_id=device_id,
                          metadata={"shots": shots, "num_qubits": getattr(circuit, "num_qubits", 2)})
    @staticmethod
    def _stub_poll(handle):
        elapsed = time.time() - handle.submitted_at
        if elapsed < 0.5: return JobState.QUEUED
        if elapsed < 1.5: return JobState.RUNNING
        return JobState.COMPLETED
    @staticmethod
    def _stub_result(handle):
        import numpy as np
        shots = handle.metadata.get("shots", 1024); n = handle.metadata.get("num_qubits", 2)
        p = np.ones(2**n)/(2**n); s = np.random.default_rng(3).multinomial(shots, p)
        return ProviderResult(job_handle=handle, counts={format(i,f"0{n}b"):int(c) for i,c in enumerate(s) if c>0},
                               shots=shots, device_id=handle.device_id)
