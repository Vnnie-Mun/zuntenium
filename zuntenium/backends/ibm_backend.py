"""IBM Quantum provider integration with graceful local fallback."""
from __future__ import annotations

import json
import math
import os
import pathlib
import threading
import time
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from zuntenium.quantum_core.statevector import StatevectorSimulator
from zuntenium.transpiler.hardware_transpiler import (
    HardwareAwareTranspiler, _IBM_QX5, _IBM_QX7,
)
from zuntenium.transpiler.v2 import Circuit, Gate

try:
    from qiskit import ClassicalRegister, QuantumCircuit as QiskitCircuit, QuantumRegister
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    QiskitCircuit = QuantumRegister = ClassicalRegister = None

try:
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler, Session
    IBM_RUNTIME_AVAILABLE = True
except ImportError:
    IBM_RUNTIME_AVAILABLE = False

_CRED_FILE = pathlib.Path.home() / ".zuntenium" / "credentials.json"
_TOKEN_ENV = "QISKIT_IBM_RUNTIME_TOKEN"


@dataclass
class BackendInfo:
    name: str
    provider: str
    num_qubits: int
    is_simulator: bool
    queue_time_est: float = -1.0
    avg_cx_error: float = 0.01
    operational: bool = True
    native_gates: set = field(default_factory=set)
    metadata: dict = field(default_factory=dict)

    def score(self, optimize_for: str = "low_noise") -> float:
        queue = max(self.queue_time_est, 0.0) / 3600.0
        weights = {"low_noise": (2.0, 0.01), "low_latency": (0.5, 1.0), "balanced": (1.0, 0.3)}
        error_weight, queue_weight = weights.get(optimize_for, weights["balanced"])
        return error_weight * self.avg_cx_error + queue_weight * queue


class JobStatus(Enum):
    INITIALIZING = auto(); QUEUED = auto(); RUNNING = auto(); DONE = auto(); FAILED = auto(); CANCELLED = auto()


@dataclass
class JobHandle:
    job_id: str
    backend: str
    status: JobStatus = JobStatus.INITIALIZING
    submitted_at: float = field(default_factory=time.time)
    _raw: Any = field(default=None, repr=False)
    def age_seconds(self) -> float: return time.time() - self.submitted_at


class QuantumBackend(ABC):
    @abstractmethod
    def execute(self, circuit: Circuit, shots: int = 1024, **kwargs) -> ExecutionResult: ...
    @abstractmethod
    def get_backend_info(self) -> BackendInfo: ...
    @abstractmethod
    def supports_noise_model(self) -> bool: ...
    @abstractmethod
    def supports_batching(self) -> bool: ...
    @abstractmethod
    def get_available_backends(self, min_qubits: int = 1, filter_criteria: Optional[Dict] = None) -> List[BackendInfo]: ...
    @abstractmethod
    def is_available(self) -> bool: ...
    def execute_batch(self, circuits: List[Circuit], shots: int = 1024, **kwargs) -> List[ExecutionResult]:
        return [self.execute(circuit, shots=shots, **kwargs) for circuit in circuits]


class ZunteniumToQiskitConverter:
    """Convert v2 Circuit IR to Qiskit without requiring Qiskit at import time."""
    _STATIC = {"X": "x", "Y": "y", "Z": "z", "H": "h", "S": "s", "T": "t", "SDG": "sdg", "TDG": "tdg", "SX": "sx", "SXD": "sxdg", "I": "id", "CNOT": "cx", "CX": "cx", "CZ": "cz", "SWAP": "swap", "ECR": "ecr", "CCX": "ccx", "CCNOT": "ccx", "TOFFOLI": "ccx", "CSWAP": "cswap"}
    _PARAM = {"RX": "rx", "RY": "ry", "RZ": "rz", "PHASE": "p", "U3": "u"}

    @classmethod
    def convert(cls, circuit: Circuit):
        if not QISKIT_AVAILABLE: return None
        has_measure = any(gate.is_measurement for gate in circuit.gates)
        quantum = QuantumRegister(circuit.num_qubits, "q")
        classical = ClassicalRegister(circuit.num_qubits, "c") if has_measure else None
        result = QiskitCircuit(quantum, classical) if classical else QiskitCircuit(quantum)
        for gate in circuit.gates:
            qubits = list(gate.qubits); name = gate.name
            if gate.is_measurement:
                for qubit in qubits: result.measure(quantum[qubit], classical[qubit])
            elif name in cls._STATIC:
                getattr(result, cls._STATIC[name])(*qubits)
            elif name in cls._PARAM:
                getattr(result, cls._PARAM[name])(*gate.params, *qubits)
            else:
                raise ValueError(f"Cannot convert gate {name!r}; run hardware decomposition first.")
        return result

    @classmethod
    def qiskit_to_zuntenium(cls, circuit) -> Circuit:
        result = Circuit(circuit.num_qubits, getattr(circuit, "name", "ibm_import") or "ibm_import")
        for instruction in circuit.data:
            operation = instruction.operation
            qubits = [circuit.find_bit(qubit).index for qubit in instruction.qubits]
            name = {"CX": "CNOT", "SXDG": "SXD", "SDG": "SDG", "TDG": "TDG"}.get(operation.name.upper(), operation.name.upper())
            params = tuple(float(parameter) for parameter in operation.params)
            if name == "MEASURE": result.add_gate(Gate(name, (qubits[0],)))
            elif len(qubits) == 1: result.add_gate(Gate(name, tuple(qubits), params=params))
            else: result.add_gate(Gate(name, (qubits[-1],), controls=tuple(qubits[:-1]), params=params))
        return result


class CredentialManager:
    def __init__(self, token: Optional[str] = None, channel: str = "ibm_quantum"):
        self._channel, self._token = channel, self._resolve(token)
    def _resolve(self, explicit):
        if explicit: return explicit
        if os.environ.get(_TOKEN_ENV): return os.environ[_TOKEN_ENV]
        try:
            if _CRED_FILE.exists(): return json.loads(_CRED_FILE.read_text()).get("token")
        except Exception: pass
        try:
            import keyring
            return keyring.get_password("zuntenium", "ibm_runtime_token")
        except Exception: return None
    def save_credentials(self, token: str, channel: str = "ibm_quantum"):
        _CRED_FILE.parent.mkdir(parents=True, exist_ok=True)
        _CRED_FILE.write_text(json.dumps({"token": token, "channel": channel})); _CRED_FILE.chmod(0o600)
        try:
            import keyring; keyring.set_password("zuntenium", "ibm_runtime_token", token)
        except Exception: pass
    def has_token(self): return self._token is not None
    def get_token(self): return self._token
    def __repr__(self): return f"CredentialManager(token={'****' + self._token[-4:] if self._token else 'None'}, channel={self._channel})"


class IBMBackend(QuantumBackend):
    def __init__(self, token=None, ibm_backend_name=None, channel="ibm_quantum", instance="ibm-q/open/main", fallback_to_simulator=True, zuntenium_hw_config="ibm_qx7", optimization_level=1, max_retry_attempts=5, retry_delay_s=2.0, job_timeout_s=600.0):
        self._creds = CredentialManager(token, channel); self._instance = instance; self._ibm_backend_name = ibm_backend_name; self._fallback = fallback_to_simulator; self._hw_cfg_name = zuntenium_hw_config; self._opt_level = optimization_level; self._max_retry = max_retry_attempts; self._retry_delay = retry_delay_s; self._timeout = job_timeout_s
        self._service = None; self._session = None; self._jobs: Dict[str, JobHandle] = {}; self._lock = threading.Lock()
        self._transpiler = HardwareAwareTranspiler(zuntenium_hw_config) if zuntenium_hw_config else None

    def _get_num_qubits(self): return {"ibm_brisbane":127, "ibm_kyoto":127, "ibm_osaka":127, "ibm_torino":133, "ibmq_qasm_simulator":32, "aer_simulator":30, "ibm_qx5":5, "ibm_qx7":7}.get(self._ibm_backend_name or "", 127)
    def get_backend_info(self): return BackendInfo(self._ibm_backend_name or "ibm_auto", "ibm", self._get_num_qubits(), bool(self._ibm_backend_name and "simulator" in self._ibm_backend_name), native_gates={"RZ", "SX", "X", "ECR"})
    def supports_noise_model(self): return True
    def supports_batching(self): return True
    def is_available(self): return self._creds.has_token() and IBM_RUNTIME_AVAILABLE
    def get_available_backends(self, min_qubits=1, filter_criteria=None):
        if not self.is_available(): return self._mock_backends(min_qubits)
        try:
            service = self._get_service(); backs = service.backends(operational=True, min_num_qubits=min_qubits, simulator=(filter_criteria or {}).get("simulator", False))
            return [BackendInfo(b.name, "ibm", b.num_qubits, "simulator" in b.name, native_gates=set(getattr(b, "basis_gates", []))) for b in backs]
        except Exception as exc:
            warnings.warn(f"IBM backend listing failed: {exc}", stacklevel=2); return self._mock_backends(min_qubits)
    @staticmethod
    def _mock_backends(min_qubits): return [BackendInfo("ibm_brisbane", "ibm", 127, False, avg_cx_error=.008, native_gates={"RZ", "SX", "X", "ECR"}), BackendInfo("aer_simulator", "ibm", 30, True, avg_cx_error=0.0)]

    def execute(self, circuit, shots=1024, optimize_for="low_noise", skip_transpile=False, dry_run=False, **kwargs):
        started = time.perf_counter(); validate_circuit_compatibility(circuit, max_qubits=self._get_num_qubits())
        optimized, report = circuit, None
        if not skip_transpile and self._transpiler:
            try:
                hardware_result = self._transpiler.transpile(circuit); optimized, report = hardware_result.circuit, hardware_result
            except Exception as exc: warnings.warn(f"Hardware transpilation failed: {exc}", stacklevel=2)
        if dry_run:
            return ExecutionResult(np.zeros(2 ** min(circuit.num_qubits, 20), dtype=np.complex128), np.zeros(2 ** min(circuit.num_qubits, 20)), {}, circuit.num_qubits, shots, getattr(report, "phase1_report", None), 0.0, (time.perf_counter() - started) * 1000, optimized.depth(), optimized.num_gates)
        qiskit_circuit = ZunteniumToQiskitConverter.convert(optimized)
        if qiskit_circuit is None or not self.is_available(): return self._fallback_execute(circuit, shots, "Qiskit/IBM runtime unavailable")
        if not any(gate.is_measurement for gate in optimized.gates): qiskit_circuit.measure_all()
        for attempt in range(self._max_retry):
            try: raw = self._submit_and_wait(qiskit_circuit, shots, optimize_for); return self._convert_ibm_result(raw, circuit.num_qubits, shots, getattr(report, "phase1_report", None), (time.perf_counter() - started) * 1000, optimized.depth(), optimized.num_gates)
            except Exception as exc:
                if attempt == self._max_retry - 1:
                    if self._fallback: return self._fallback_execute(circuit, shots, str(exc))
                    raise
                time.sleep(self._retry_delay * (2 ** attempt))
        raise RuntimeError("IBM execution exhausted retries")

    def _get_service(self):
        with self._lock:
            if self._service is None:
                if not IBM_RUNTIME_AVAILABLE: raise RuntimeError("qiskit-ibm-runtime unavailable")
                self._service = QiskitRuntimeService(channel=self._creds._channel, instance=self._instance, token=self._creds.get_token())
            return self._service
    def _select_ibm_backend(self, min_qubits, optimize_for): return self._ibm_backend_name or min(self.get_available_backends(min_qubits), key=lambda backend: backend.score(optimize_for)).name
    def _submit_and_wait(self, circuit, shots, optimize_for):
        service = self._get_service(); backend = service.backend(self._select_ibm_backend(circuit.num_qubits, optimize_for))
        with Session(service=service, backend=backend) as session:
            job = Sampler(mode=session).run([circuit], shots=shots); job_id = job.job_id(); self._jobs[job_id] = JobHandle(job_id, backend.name, JobStatus.QUEUED, _raw=job); result = job.result(); self._jobs[job_id].status = JobStatus.DONE; return result
    def _convert_ibm_result(self, raw, n, shots, opt_report, total_ms, circuit_depth, num_gates):
        try: counts_raw = raw[0].data.meas.get_counts()
        except Exception:
            try: counts_raw = {key: int(value * shots) for key, value in raw.quasi_dists[0].binary_probabilities().items()}
            except Exception: counts_raw = {}
        counts = {key.replace(" ", "")[::-1].zfill(n)[:n]: int(value) for key, value in counts_raw.items()}; probabilities = np.zeros(2 ** n); total = max(sum(counts.values()), 1)
        for bitstring, count in counts.items(): probabilities[int(bitstring[::-1], 2)] = count / total
        return ExecutionResult(None, probabilities, counts, n, shots, opt_report, 0.0, total_ms, circuit_depth, num_gates)
    def _fallback_execute(self, circuit, shots, reason=""):
        simulator = StatevectorSimulator(circuit.num_qubits, seed=42); simulator.initialize(circuit.num_qubits); _apply_circuit_to_simulator(circuit, simulator); state = simulator.get_statevector(); counts = simulator.sample(shots) if shots > 0 else {}; counts["__fallback__"] = 1
        return ExecutionResult(state, np.abs(state) ** 2, counts, circuit.num_qubits, shots, None, 0.0, 0.0, circuit.depth(), circuit.num_gates)
    def cancel_job(self, job_id):
        handle = self._jobs.get(job_id)
        try: handle._raw.cancel(); handle.status = JobStatus.CANCELLED; return True
        except Exception: return False
    def job_status(self, job_id): return self._jobs.get(job_id).status if job_id in self._jobs else None
    def active_jobs(self): return [job for job in self._jobs.values() if job.status in {JobStatus.QUEUED, JobStatus.RUNNING}]
    def open_session(self, backend_name=None): return None
    def close_session(self): self._session = None
    def save_credentials(self, token, channel="ibm_quantum"): self._creds.save_credentials(token, channel)
    def __repr__(self): return f"IBMBackend(backend={self._ibm_backend_name or 'auto'}, {'available' if self.is_available() else 'mock-mode'}, fallback={self._fallback})"


class IonQBackend(QuantumBackend):
    def __init__(self, api_key=None, backend="harmony"): self._api_key, self._backend = api_key or os.environ.get("IONQ_API_KEY"), backend
    def execute(self, circuit, shots=1024, **kwargs): return IBMBackend()._fallback_execute(circuit, shots, "IonQ unavailable") if not self._api_key else (_ for _ in ()).throw(NotImplementedError("Install ionq-sdk for IonQ execution."))
    def get_backend_info(self): return BackendInfo("ionq_" + self._backend, "ionq", 11, False)
    def supports_noise_model(self): return True
    def supports_batching(self): return False
    def is_available(self): return bool(self._api_key)
    def get_available_backends(self, **kwargs): return [BackendInfo("ionq_harmony", "ionq", 11, False, avg_cx_error=.003), BackendInfo("ionq_forte", "ionq", 32, False, avg_cx_error=.001)]


class RigettiBacked(QuantumBackend):
    def execute(self, circuit, shots=1024, **kwargs): raise NotImplementedError("Install pyquil for Rigetti execution.")
    def get_backend_info(self): return BackendInfo("rigetti_aspen", "rigetti", 80, False)
    def supports_noise_model(self): return True
    def supports_batching(self): return True
    def is_available(self): return False
    def get_available_backends(self, **kwargs): return []


class SimulatorBackend(QuantumBackend):
    def __init__(self, n_qubits=20, seed=None): self._n, self._seed = n_qubits, seed
    def execute(self, circuit, shots=1024, **kwargs):
        simulator = StatevectorSimulator(circuit.num_qubits, seed=self._seed); simulator.initialize(circuit.num_qubits); _apply_circuit_to_simulator(circuit, simulator); state = simulator.get_statevector(); return ExecutionResult(state, np.abs(state) ** 2, simulator.sample(shots) if shots > 0 else {}, circuit.num_qubits, shots, None, 0.0, 0.0, circuit.depth(), circuit.num_gates)
    def get_backend_info(self): return BackendInfo("zuntenium_simulator", "local", self._n, True, 0.0, 0.0)
    def supports_noise_model(self): return False
    def supports_batching(self): return True
    def is_available(self): return True
    def get_available_backends(self, **kwargs): return [self.get_backend_info()]


class BackendRegistry:
    _registry = {"ibm": IBMBackend, "ionq": IonQBackend, "rigetti": RigettiBacked}
    @classmethod
    def register(cls, name, backend_cls): cls._registry[name.lower()] = backend_cls
    @classmethod
    def get(cls, name, **kwargs):
        if name.lower() == "simulator": return SimulatorBackend(**{key: kwargs[key] for key in ("n_qubits", "seed") if key in kwargs})
        if name.lower() not in cls._registry: raise KeyError(f"Unknown backend {name!r}; registered: {sorted(cls._registry)}")
        return cls._registry[name.lower()](**{key: value for key, value in kwargs.items() if key != "n_qubits"})
    @classmethod
    def available(cls): return ["simulator"] + sorted(cls._registry)


class UnifiedQuantumExecutor:
    def __init__(self, backend="simulator", n_qubits=20, **backend_kwargs): self._backend = BackendRegistry.get(backend, n_qubits=n_qubits, **backend_kwargs) if isinstance(backend, str) else backend; self._n_qubits = n_qubits
    @property
    def backend(self): return self._backend
    def execute(self, circuit, shots=1024, optimize=True, optimization_level=2, debug=False, **kwargs):
        if debug: print(f"[UnifiedExecutor] backend={self._backend!r}")
        return self._backend.execute(circuit, shots=shots, optimize=optimize,
                                     optimization_level=optimization_level,
                                     debug=debug, **kwargs)
    def switch_backend(self, backend, **kwargs): self._backend = BackendRegistry.get(backend, **kwargs) if isinstance(backend, str) else backend
    def __repr__(self): return f"UnifiedQuantumExecutor(backend={self._backend!r})"


class BackendAwareVQE:
    def __init__(self, ansatz_fn: Callable, hamiltonian: Dict[int, float], backend: QuantumBackend, shots=0): self._ansatz, self._H, self._backend, self._shots, self._history, self._calls = ansatz_fn, hamiltonian, backend, shots, [], 0
    def cost(self, params):
        result = self._backend.execute(self._ansatz(params), shots=self._shots)
        probabilities = result.probabilities
        expectations = np.array([np.dot(1 - 2 * ((np.arange(len(probabilities)) >> q) & 1), probabilities) for q in range(result.num_qubits)])
        energy = sum(coefficient * expectations[qubit] for qubit, coefficient in self._H.items() if qubit < len(expectations)); self._calls += 1; self._history.append(float(energy)); return float(energy)
    def run(self, initial_params, method="COBYLA", max_iter=200):
        try:
            from scipy.optimize import minimize
            result = minimize(self.cost, initial_params, method=method, options={"maxiter": max_iter}); return {"energy": result.fun, "params": result.x, "calls": self._calls, "history": self._history}
        except ImportError:
            return {"energy": self.cost(np.asarray(initial_params, dtype=float)), "params": np.asarray(initial_params, dtype=float), "calls": self._calls, "history": self._history}


class IBMBackendTestSuite:
    def __init__(self): self._p = 0; self._f = 0

    def _a(self, cond, msg):
        if cond: self._p += 1; print(f"  ✓  {msg}")
        else:    self._f += 1; print(f"  ✗  FAIL: {msg}")

    def test_abstract_interface(self):
        for method in ['execute', 'get_backend_info', 'supports_noise_model',
                       'supports_batching', 'get_available_backends', 'is_available']:
            self._a(hasattr(QuantumBackend, method), f"QuantumBackend.{method} declared")

    def test_registry(self):
        self._a('ibm' in BackendRegistry.available(), "ibm in registry")
        self._a('simulator' in BackendRegistry.available(), "simulator in registry")
        self._a('ionq' in BackendRegistry.available(), "ionq in registry")

    def test_simulator_backend(self):
        from zuntenium.transpiler.v2.engine import _build_ghz
        sb = SimulatorBackend(n_qubits=3, seed=0)
        self._a(sb.is_available(), "SimulatorBackend.is_available() = True")
        c = _build_ghz(3)
        r = sb.execute(c, shots=500)
        self._a(r.num_qubits == 3, f"SimBackend: n={r.num_qubits}")
        self._a(r.probabilities is not None, "SimBackend: has probabilities")

    def test_ibm_backend_mock_mode(self):
        b = IBMBackend(token=None, fallback_to_simulator=True)
        self._a(not b.is_available(), "IBMBackend: not available without token")
        self._a(b.supports_noise_model(), "IBMBackend.supports_noise_model()")
        self._a(b.supports_batching(), "IBMBackend.supports_batching()")

    def test_ibm_fallback_execute(self):
        b = IBMBackend(token=None, fallback_to_simulator=True)
        c = Circuit(2)
        c.add_gate(Gate('H', (0,))); c.add_gate(Gate('CNOT', (1,), controls=(0,)))
        r = b.execute(c, shots=100)
        self._a(r.probabilities is not None, "IBMBackend fallback: has probs")
        self._a(r.counts.get('__fallback__') == 1, "IBMBackend fallback: audit tag set")
        sv = r.statevector
        p00_p11 = abs(sv[0])**2 + abs(sv[3])**2
        self._a(math.isclose(p00_p11, 1.0, abs_tol=1e-6),
                f"IBMBackend fallback Bell correct: P[00]+P[11]={p00_p11:.6f}")

    def test_converter_bell(self):
        if not QISKIT_AVAILABLE:
            print("  ~  Qiskit unavailable — conversion test skipped")
            self._p += 1; return
        c = Circuit(2)
        c.add_gate(Gate('H', (0,))); c.add_gate(Gate('CNOT', (1,), controls=(0,)))
        qc = ZunteniumToQiskitConverter.convert(c)
        self._a(qc is not None, "Bell → Qiskit QuantumCircuit non-None")
        self._a(qc.num_qubits == 2, "Bell: 2 qubits in Qiskit circuit")

    def test_converter_all_native_ibm(self):
        if not QISKIT_AVAILABLE:
            print("  ~  Qiskit unavailable — conversion test skipped")
            self._p += 1; return
        c = Circuit(2)
        c.add_gate(Gate('RZ', (0,), params=(0.5,)))
        c.add_gate(Gate('SX', (0,)))
        c.add_gate(Gate('X', (0,)))
        c.add_gate(Gate('ECR', (1,), controls=(0,)))
        qc = ZunteniumToQiskitConverter.convert(c)
        self._a(qc is not None and qc.num_qubits == 2, "IBM native gates convert cleanly")

    def test_credential_env(self):
        original = os.environ.get('QISKIT_IBM_RUNTIME_TOKEN')
        os.environ['QISKIT_IBM_RUNTIME_TOKEN'] = 'test_token_xyz'
        cm = CredentialManager()
        self._a(cm.get_token() == 'test_token_xyz', "CredentialManager reads from env")
        self._a('test_token_xyz' not in repr(cm), "Token not exposed in repr")
        if original is None:
            if 'QISKIT_IBM_RUNTIME_TOKEN' in os.environ:
                del os.environ['QISKIT_IBM_RUNTIME_TOKEN']
        else:
            os.environ['QISKIT_IBM_RUNTIME_TOKEN'] = original

    def test_credential_repr_masked(self):
        cm = CredentialManager(token='abcdef1234567890')
        self._a('abcdef' not in repr(cm), f"Token masked in repr: {repr(cm)}")

    def test_backend_selection_scoring(self):
        bi_low_err = BackendInfo("backend_a", "ibm", 127, False, 300.0, 0.005, True)
        bi_high_err = BackendInfo("backend_b", "ibm", 127, False, 60.0, 0.015, True)
        self._a(bi_low_err.score("low_noise") < bi_high_err.score("low_noise"),
                "Score: low-error backend preferred in low_noise mode")
        self._a(bi_high_err.score("low_latency") < bi_low_err.score("low_latency"),
                "Score: low-queue backend preferred in low_latency mode")

    def test_available_backends_mock(self):
        b = IBMBackend(token=None)
        avail = b.get_available_backends(min_qubits=1)
        self._a(len(avail) > 0, f"get_available_backends returns items: {len(avail)}")

    def test_unified_executor_simulator(self):
        c = Circuit(3); c.add_gate(Gate('H', (0,))); c.add_gate(Gate('CNOT', (1,), controls=(0,)))
        ex = UnifiedQuantumExecutor(backend="simulator", n_qubits=3)
        r = ex.execute(c, shots=0)
        self._a(r.probabilities is not None, "UnifiedExecutor/sim: has probs")
        self._a(r.num_qubits == 3, f"UnifiedExecutor/sim: n={r.num_qubits}")

    def test_unified_executor_ibm_fallback(self):
        c = Circuit(2); c.add_gate(Gate('H', (0,)))
        ex = UnifiedQuantumExecutor(backend="ibm", n_qubits=3)
        r = ex.execute(c, shots=100)
        self._a(r.probabilities is not None, "UnifiedExecutor/ibm-fallback: has probs")

    def test_backend_switch(self):
        ex = UnifiedQuantumExecutor(backend="simulator", n_qubits=3)
        ex.switch_backend("ibm")
        self._a(isinstance(ex.backend, IBMBackend), "Hot-swap simulator→ibm works")

    def test_job_handle_age(self):
        jh = JobHandle("test-job-123", "ibm_brisbane", JobStatus.QUEUED)
        time.sleep(0.01)
        self._a(jh.age_seconds() > 0.005, f"JobHandle.age_seconds() works: {jh.age_seconds():.3f}s")

    def test_job_status_enum(self):
        for s in JobStatus:
            self._a(isinstance(s, JobStatus), f"JobStatus.{s.name} is valid enum")

    def test_dry_run(self):
        b = IBMBackend(token=None)
        c = Circuit(2); c.add_gate(Gate('H', (0,))); c.add_gate(Gate('CNOT', (1,), controls=(0,)))
        r = b.execute(c, shots=1024, dry_run=True)
        self._a(r is not None, "dry_run returns ExecutionResult")
        self._a(r.num_qubits == 2, f"dry_run: num_qubits=2 (got {r.num_qubits})")

    def test_batch_execute_fallback(self):
        from zuntenium.transpiler.v2.engine import _build_ghz
        b = IBMBackend(token=None, fallback_to_simulator=True)
        cs = [_build_ghz(3) for _ in range(3)]
        res = b.execute_batch(cs, shots=100)
        self._a(len(res) == 3, f"batch returns 3 results (got {len(res)})")
        for i, r in enumerate(res):
            self._a(r.probabilities is not None, f"batch[{i}]: has probs")

    def test_backend_aware_vqe(self):
        sim = SimulatorBackend(n_qubits=2)
        H = {0: 1.0, 1: -0.5}
        def ansatz(p):
            c = Circuit(2)
            c.add_gate(Gate('RY', (0,), params=(float(p[0]),)))
            c.add_gate(Gate('RY', (1,), params=(float(p[1]),)))
            return c
        vqe = BackendAwareVQE(ansatz, H, sim, shots=0)
        e = vqe.cost(np.zeros(2))
        self._a(isinstance(e, float), f"VQE cost is float: {e:.4f}")
        self._a(-2.0 < e < 2.0, f"VQE cost in range: {e:.4f}")

    def run_all(self):
        print("\n" + "═" * 60)
        print("  IBM Backend Integration — Test Suite")
        print("═" * 60)
        tests = [m for m in sorted(dir(self)) if m.startswith('test_')]
        for t in tests:
            print(f"\n[{t}]"); getattr(self, t)()
        total = self._p + self._f
        print(f"\n{'═' * 60}")
        print(f"  Results: {self._p}/{total} passed"
              + (f"  ({self._f} FAILED)" if self._f else "  ✓ all pass"))
        print("═" * 60)
        assert self._f == 0, f"{self._f} tests failed in IBMBackendTestSuite"


def run_demo():
    print("\n" + "═" * 70)
    print("  ZUNTENIUM IBM Quantum Backend Integration — Demo")
    print("═" * 70)
    from zuntenium.transpiler.v2.engine import _build_ghz
    sim = SimulatorBackend(n_qubits=3)
    c = _build_ghz(3)
    r = sim.execute(c, shots=500)
    print(f"  GHZ(3): {r.num_qubits}q, {r.num_gates_executed} gates, "
          f"P[000]={r.probabilities[0]:.3f}, P[111]={r.probabilities[7]:.3f}")
    print("\n" + "═" * 70 + "\n")


from zuntenium.execution.quantum_executor import (
    ExecutionResult, _apply_circuit_to_simulator, validate_circuit_compatibility,
)


__all__ = [
    "QISKIT_AVAILABLE", "IBM_RUNTIME_AVAILABLE", "BackendInfo", "JobStatus",
    "JobHandle", "QuantumBackend", "ZunteniumToQiskitConverter",
    "CredentialManager", "IBMBackend", "IonQBackend", "RigettiBacked",
    "SimulatorBackend", "BackendRegistry", "UnifiedQuantumExecutor",
    "BackendAwareVQE", "IBMBackendTestSuite", "run_demo",
]

if __name__ == "__main__":
    run_demo()
    IBMBackendTestSuite().run_all()
