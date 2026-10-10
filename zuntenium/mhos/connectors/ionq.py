# zuntenium/mhos/connectors/ionq.py
"""IonQ Cloud Connector — REST API"""
from __future__ import annotations
import asyncio, json, logging, time
from .base import BaseConnector, ConnectorError, FatalError, JobHandle, JobState, ProviderResult, RetryableError

_log = logging.getLogger(__name__)

_IONQ_GATE_MAP = {
    "h":"h","x":"x","y":"y","z":"z","rx":"rx","ry":"ry","rz":"rz",
    "cx":"cnot","cnot":"cnot","swap":"swap","t":"t","tdg":"ti","s":"s","sdg":"si",
    "toffoli":"ccnot","ccx":"ccnot",
    "xx":"xx","yy":"yy","zz":"zz",
}

class IonQConnector(BaseConnector):
    """
    IonQ Cloud REST API connector.
    All-to-all connectivity: no SWAP insertion required.

    Credentials:
      "api_key"  : IonQ API key
    """
    PROVIDER_NAME = "ionq"
    _API_BASE     = "https://api.ionq.co/v0.3"

    def __init__(self, credentials: dict[str, str], **kwargs):
        super().__init__(credentials, **kwargs)
        self._api_key = credentials.get("api_key", "")
        self._use_stub = not bool(self._api_key)
        self._session  = None

    async def authenticate(self) -> None:
        if self._use_stub:
            _log.info("[IonQ] Stub mode"); self._authenticated = True; return
        try:
            import aiohttp
            self._session = aiohttp.ClientSession(
                headers={"Authorization": f"apiKey {self._api_key}",
                         "Content-Type": "application/json"}
            )
            # Verify by listing jobs
            async with self._session.get(f"{self._API_BASE}/jobs?limit=1") as resp:
                if resp.status == 401:
                    raise FatalError("IonQ API key invalid", "ionq")
                self._authenticated = True
            _log.info("[IonQ] Authenticated")
        except ImportError:
            _log.warning("[IonQ] aiohttp not installed — stub mode")
            self._use_stub = True; self._authenticated = True

    async def list_devices(self) -> list[dict]:
        if self._use_stub:
            return [
                {"device_id":"ionq_aria-1","display_name":"IonQ Aria 1","num_qubits":25,
                 "status":"online","queue_depth":5,"mean_1q_error":0.0003,
                 "mean_2q_error":0.003,"mean_readout_err":0.005,"cost_per_shot":0.00035,
                 "all_to_all":True,"native_gates":["ms","gpi","gpi2"]},
                {"device_id":"ionq_forte-1","display_name":"IonQ Forte 1","num_qubits":36,
                 "status":"online","queue_depth":2,"mean_1q_error":0.0001,
                 "mean_2q_error":0.001,"mean_readout_err":0.003,"cost_per_shot":0.00097,
                 "all_to_all":True,"native_gates":["ms","gpi","gpi2"]},
            ]
        async with self._session.get(f"{self._API_BASE}/backends") as resp:
            if resp.status != 200:
                raise ConnectorError(f"IonQ list_devices failed: {resp.status}", "ionq")
            data = await resp.json()
            return [self._backend_to_dict(b) for b in data.get("backends", [])]

    def circuit_to_native(self, circuit) -> dict:
        """Return IonQ JSON circuit format."""
        ops = []
        for g in circuit.gates:
            name   = g.name.lower()
            qubits = g.qubits
            params = g.params
            ionq_name = _IONQ_GATE_MAP.get(name)
            if ionq_name is None:
                _log.warning("[IonQ] Unknown gate '%s' — skipping", name)
                continue
            op: dict = {"gate": ionq_name}
            if len(qubits) == 1:
                op["target"] = qubits[0]
            elif len(qubits) == 2:
                op["control"], op["target"] = qubits[0], qubits[1]
            elif len(qubits) == 3:
                op["controls"], op["target"] = [qubits[0], qubits[1]], qubits[2]
            if params:
                op["rotation"] = params[0]
            ops.append(op)
        return {
            "qubits": circuit.num_qubits,
            "circuit": ops,
        }

    async def submit(self, circuit, device_id: str, shots: int = 1024, **kwargs) -> JobHandle:
        if self._use_stub: return self._stub_submit(circuit, device_id, shots)
        native = self.circuit_to_native(circuit)
        payload = {
            "target": device_id,
            "shots":  shots,
            "input":  native,
            "name":   getattr(circuit, "name", "zuntenium_job"),
        }
        async with self._session.post(f"{self._API_BASE}/jobs", json=payload) as resp:
            if resp.status == 429:
                raise RetryableError("IonQ rate limited", "ionq", retry_after_s=60)
            if resp.status not in (200, 201):
                body = await resp.text()
                raise ConnectorError(f"IonQ submit failed {resp.status}: {body}", "ionq")
            data = await resp.json()
            return JobHandle(native_id=data["id"], provider="ionq", device_id=device_id,
                              metadata={"shots": shots, "num_qubits": circuit.num_qubits})

    async def poll_status(self, handle: JobHandle) -> JobState:
        if self._use_stub: return self._stub_poll(handle)
        async with self._session.get(f"{self._API_BASE}/jobs/{handle.native_id}") as resp:
            if resp.status != 200: return JobState.UNKNOWN
            data   = await resp.json()
            status = data.get("status", "").lower()
            return {"submitted":"queued","ready":"queued","running":"running",
                    "shot":"running","completed":"completed","failed":"failed",
                    "canceled":"cancelled"}.get(status, JobState.UNKNOWN)

    async def fetch_result(self, handle: JobHandle) -> ProviderResult:
        if self._use_stub: return self._stub_result(handle)
        async with self._session.get(f"{self._API_BASE}/jobs/{handle.native_id}") as resp:
            data = await resp.json()
        histogram = data.get("data", {}).get("histogram", {})
        n     = handle.metadata.get("num_qubits", 1)
        shots = handle.metadata.get("shots", 1024)
        counts: dict[str, int] = {}
        for bitstr, prob in histogram.items():
            k = format(int(bitstr, 10) if bitstr.isdigit() else int(bitstr, 2), f"0{n}b")
            counts[k] = int(round(float(prob) * shots))
        return ProviderResult(job_handle=handle, counts=counts, shots=sum(counts.values()),
                               device_id=handle.device_id)

    async def cancel(self, handle: JobHandle) -> bool:
        if self._use_stub: return True
        try:
            async with self._session.delete(f"{self._API_BASE}/jobs/{handle.native_id}") as r:
                return r.status in (200, 204)
        except Exception: return False

    async def get_device_status(self, device_id: str) -> dict:
        if self._use_stub: return {"status": "online", "queue_depth": 3}
        devices = await self.list_devices()
        return next((d for d in devices if d.get("device_id") == device_id), {})

    @staticmethod
    def _backend_to_dict(b: dict) -> dict:
        return {"device_id": b.get("backend",""), "display_name": b.get("backend",""),
                "num_qubits": b.get("qubits", 0), "status": "online" if b.get("available") else "offline",
                "cost_per_shot": 0.00035, "all_to_all": True,
                "native_gates": ["ms","gpi","gpi2"],
                "mean_1q_error": 0.0003, "mean_2q_error": 0.003, "mean_readout_err": 0.005}

    _stub_ctr = 0
    def _stub_submit(self, circuit, device_id, shots):
        IonQConnector._stub_ctr += 1
        return JobHandle(native_id=f"stub_ionq_{IonQConnector._stub_ctr}",
                          provider="ionq", device_id=device_id,
                          metadata={"shots": shots, "num_qubits": getattr(circuit, "num_qubits", 2)})
    @staticmethod
    def _stub_poll(handle):
        elapsed = time.time() - handle.submitted_at
        if elapsed < 0.5: return JobState.QUEUED
        if elapsed < 1:   return JobState.RUNNING
        return JobState.COMPLETED
    @staticmethod
    def _stub_result(handle):
        import numpy as np
        shots = handle.metadata.get("shots", 1024); n = handle.metadata.get("num_qubits", 2)
        p = np.ones(2**n)/(2**n); s = np.random.default_rng(2).multinomial(shots, p)
        return ProviderResult(job_handle=handle, counts={format(i,f"0{n}b"):int(c) for i,c in enumerate(s) if c>0},
                               shots=shots, device_id=handle.device_id)
