# zuntenium/mhos/connectors/braket.py
"""AWS Braket Connector"""
from __future__ import annotations
import asyncio, logging, time
from .base import BaseConnector, ConnectorError, FatalError, JobHandle, JobState, ProviderResult, RetryableError

_log = logging.getLogger(__name__)

_BRAKET_GATE_MAP = {
    "h":"H","x":"X","y":"Y","z":"Z","s":"S","t":"T","sdg":"Si","tdg":"Ti",
    "id":"I","rx":"Rx","ry":"Ry","rz":"Rz","cnot":"CNOT","cx":"CNOT",
    "cy":"CY","cz":"CZ","swap":"SWAP","ccx":"CCNot","xx":"XX","yy":"YY","zz":"ZZ",
}

class BraketConnector(BaseConnector):
    """
    AWS Braket connector.

    Credentials:
      "aws_access_key_id"     : AWS access key
      "aws_secret_access_key" : AWS secret
      "s3_bucket"             : S3 bucket for results
      "s3_prefix"             : S3 prefix
    """
    PROVIDER_NAME = "braket"

    def __init__(self, credentials: dict[str, str], region: str = "us-east-1", **kwargs):
        super().__init__(credentials, region=region, **kwargs)
        self._s3_bucket = credentials.get("s3_bucket", "zuntenium-results")
        self._s3_prefix = credentials.get("s3_prefix", "jobs")
        self._use_stub  = not bool(credentials.get("aws_access_key_id"))

    async def authenticate(self) -> None:
        if self._use_stub:
            _log.info("[Braket] Using stub mode")
            self._authenticated = True
            return
        try:
            import boto3
            self._session = boto3.Session(
                aws_access_key_id=self._creds.get("aws_access_key_id"),
                aws_secret_access_key=self._creds.get("aws_secret_access_key"),
                region_name=self._region,
            )
            self._authenticated = True
        except ImportError:
            _log.warning("[Braket] boto3 not installed — stub mode")
            self._use_stub = True; self._authenticated = True
        except Exception as exc:
            raise FatalError(f"Braket auth failed: {exc}", "braket") from exc

    async def list_devices(self) -> list[dict]:
        if self._use_stub:
            return [
                {"device_id":"arn:aws:braket:us-east-1::device/qpu/ionq/Aria-1",
                 "display_name":"IonQ Aria (via Braket)","num_qubits":25,
                 "status":"online","queue_depth":3,"mean_1q_error":0.0003,
                 "mean_2q_error":0.003,"mean_readout_err":0.005,
                 "cost_per_shot":0.00035,"all_to_all":True,
                 "native_gates":["ms","gpi","gpi2"]},
                {"device_id":"arn:aws:braket:us-west-1::device/qpu/rigetti/Ankaa-2",
                 "display_name":"Rigetti Ankaa-2 (via Braket)","num_qubits":84,
                 "status":"online","queue_depth":1,"mean_1q_error":0.001,
                 "mean_2q_error":0.01,"mean_readout_err":0.03,
                 "cost_per_shot":0.00035,"all_to_all":False,
                 "native_gates":["rx","rz","cz"]},
            ]
        loop = asyncio.get_event_loop()
        try:
            from braket.aws import AwsDevice
            devices = await loop.run_in_executor(None, AwsDevice.get_devices, ["QPU"])
            return [self._device_to_dict(d) for d in devices]
        except ImportError:
            return []

    def circuit_to_native(self, circuit):
        try:
            from braket.circuits import Circuit as BCircuit, Gate
            bc = BCircuit()
            for g in circuit.gates:
                name   = g.name.lower()
                qubits = g.qubits
                params = g.params
                braket_name = _BRAKET_GATE_MAP.get(name)
                if braket_name is None:
                    _log.warning("[Braket] Unknown gate '%s' — skipping", name)
                    continue
                gate_cls = getattr(Gate, braket_name, None)
                if gate_cls is None: continue
                try:
                    if params:
                        bc.add_instruction(gate_cls(*params)(*qubits))
                    else:
                        bc.add_instruction(gate_cls()(*qubits))
                except Exception as e:
                    _log.debug("[Braket] gate '%s' add error: %s", name, e)
            return bc
        except ImportError:
            return circuit

    async def submit(self, circuit, device_id: str, shots: int = 1024, **kwargs) -> JobHandle:
        if self._use_stub:
            return self._stub_submit(circuit, device_id, shots)
        loop   = asyncio.get_event_loop()
        native = self.circuit_to_native(circuit)
        try:
            from braket.aws import AwsDevice, AwsQuantumTask
            device = AwsDevice(device_id)
            task   = await loop.run_in_executor(
                None, lambda: device.run(native, shots=shots,
                                          s3_destination_folder=(self._s3_bucket, self._s3_prefix))
            )
            return JobHandle(native_id=task.id, provider="braket", device_id=device_id,
                              metadata={"shots": shots, "num_qubits": circuit.num_qubits})
        except Exception as exc:
            if "ThrottlingException" in str(exc):
                raise RetryableError(str(exc), "braket", retry_after_s=10)
            raise ConnectorError(str(exc), "braket") from exc

    async def poll_status(self, handle: JobHandle) -> JobState:
        if self._use_stub: return self._stub_poll(handle)
        loop = asyncio.get_event_loop()
        try:
            from braket.aws import AwsQuantumTask
            task   = AwsQuantumTask(handle.native_id)
            status = await loop.run_in_executor(None, lambda: task.state())
            return {"CREATED":"queued","QUEUED":"queued","RUNNING":"running",
                    "COMPLETED":"completed","FAILED":"failed","CANCELLING":"running",
                    "CANCELLED":"cancelled"}.get(status.upper(), JobState.UNKNOWN)
        except Exception: return JobState.UNKNOWN

    async def fetch_result(self, handle: JobHandle) -> ProviderResult:
        if self._use_stub: return self._stub_result(handle)
        loop = asyncio.get_event_loop()
        from braket.aws import AwsQuantumTask
        task   = AwsQuantumTask(handle.native_id)
        result = await loop.run_in_executor(None, task.result)
        n      = handle.metadata.get("num_qubits", 1)
        raw    = dict(result.measurement_counts)
        counts = self._normalise_counts({str(k): v for k, v in raw.items()}, n)
        return ProviderResult(job_handle=handle, counts=counts,
                               shots=sum(counts.values()), device_id=handle.device_id)

    async def cancel(self, handle: JobHandle) -> bool:
        if self._use_stub: return True
        try:
            from braket.aws import AwsQuantumTask
            loop = asyncio.get_event_loop()
            task = AwsQuantumTask(handle.native_id)
            await loop.run_in_executor(None, task.cancel)
            return True
        except Exception: return False

    async def get_device_status(self, device_id: str) -> dict:
        if self._use_stub: return {"status": "online", "queue_depth": 2}
        devices = await self.list_devices()
        return next((d for d in devices if d.get("device_id") == device_id), {})

    @staticmethod
    def _device_to_dict(d) -> dict:
        try:
            return {"device_id": d.arn, "display_name": d.name,
                    "num_qubits": d.qubit_count or 0, "status": d.status.lower(),
                    "queue_depth": getattr(d, "queue_depth", 0),
                    "cost_per_shot": 0.00035,
                    "native_gates": [g.lower() for g in (d.supported_quantum_operations or [])]}
        except Exception:
            return {"device_id": str(d), "status": "unknown"}

    _stub_ctr = 0
    def _stub_submit(self, circuit, device_id, shots):
        BraketConnector._stub_ctr += 1
        return JobHandle(native_id=f"stub_braket_{BraketConnector._stub_ctr}",
                          provider="braket", device_id=device_id,
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
        p = np.ones(2**n)/(2**n); s = np.random.default_rng(1).multinomial(shots, p)
        return ProviderResult(job_handle=handle, counts={format(i,f"0{n}b"):int(c) for i,c in enumerate(s) if c>0},
                               shots=shots, device_id=handle.device_id)
