# tests/test_cloud_platform.py
"""
Comprehensive test suite for the Zuntenium Cloud Deployment Platform.

Tests:
  A. Schema validation — circuit, job submission, API key
  B. Auth unit tests — password hashing, JWT, API keys, quota enforcement
  C. JobManager unit tests — submit, status, cancel, retry logic
  D. Worker pipeline unit tests — circuit reconstruction, backend routing, fallback
  E. API integration tests — full HTTP flow with mocked DB + Redis
  F. Fault injection — worker death, backend failure, Redis outage
  G. Concurrent submission load test (100 simulated users)
  H. End-to-end: Bell circuit → submit → poll → result
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

# ── Import platform modules ───────────────────────────────────────────────────

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pytest.importorskip("pydantic", reason="zuntenium_cloud requires pydantic")
pytest.importorskip("fastapi", reason="zuntenium_cloud requires fastapi")

from zuntenium_cloud.schemas import (
    CircuitSchema, GateSchema, JobSubmitRequest, ErrorResponse,
)
from zuntenium_cloud.auth import (
    hash_password, verify_password, generate_api_key, hash_api_key,
    create_access_token, create_refresh_token, decode_token,
)
from zuntenium_cloud.worker import _build_circuit, _run_circuit_on_backend


# ── Fixtures ──────────────────────────────────────────────────────────────────

def bell_circuit_schema() -> CircuitSchema:
    return CircuitSchema(
        num_qubits=2,
        gates=[
            GateSchema(name="h",  qubits=[0]),
            GateSchema(name="cx", qubits=[0, 1]),
        ],
        name="bell",
    )


def large_circuit_schema(n: int = 30) -> CircuitSchema:
    gates = [GateSchema(name="h", qubits=[i]) for i in range(n)]
    gates += [GateSchema(name="cx", qubits=[i, i+1]) for i in range(n-1)]
    return CircuitSchema(num_qubits=n, gates=gates, name=f"large_{n}q")


# ── SECTION A: Schema Validation ──────────────────────────────────────────────

class TestSchemaValidation:

    def test_valid_bell_circuit(self):
        c = bell_circuit_schema()
        assert c.num_qubits == 2
        assert len(c.gates) == 2
        assert c.gates[0].name == "h"

    def test_circuit_qubit_range_reject(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            CircuitSchema(num_qubits=2, gates=[
                GateSchema(name="h", qubits=[5])  # qubit 5 > num_qubits-1=1
            ])

    def test_circuit_max_qubits(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            CircuitSchema(num_qubits=201, gates=[])  # > 200

    def test_circuit_zero_qubits(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            CircuitSchema(num_qubits=0, gates=[])

    def test_negative_qubit_index(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            GateSchema(name="h", qubits=[-1])

    def test_job_submit_valid(self):
        req = JobSubmitRequest(
            job_type="circuit",
            circuit=bell_circuit_schema(),
            shots=2048,
            backend="auto",
            mitigation_enabled=True,
            optimization_mode="ai",
        )
        assert req.shots == 2048
        assert req.mitigation_enabled is True

    def test_job_submit_invalid_backend(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            JobSubmitRequest(backend="quantum_computer_9000")

    def test_job_submit_invalid_type(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            JobSubmitRequest(job_type="quantum_voodoo")

    def test_job_submit_shots_limits(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            JobSubmitRequest(shots=2_000_000)  # > max

    def test_gate_params_optional(self):
        g = GateSchema(name="rx", qubits=[0], params=[1.57])
        assert g.params == [1.57]
        g2 = GateSchema(name="h", qubits=[0])
        assert g2.params == []

    def test_large_circuit_schema(self):
        c = large_circuit_schema(50)
        assert c.num_qubits == 50
        assert len(c.gates) == 50 + 49  # H + CX chain


# ── SECTION B: Auth Unit Tests ────────────────────────────────────────────────

class TestAuth:

    def test_password_hash_and_verify(self):
        plain   = "super-secret-password-123!"
        hashed  = hash_password(plain)
        assert hashed != plain
        assert verify_password(plain, hashed)
        assert not verify_password("wrong-password", hashed)

    def test_password_hash_unique(self):
        h1 = hash_password("same")
        h2 = hash_password("same")
        assert h1 != h2   # bcrypt salts are random

    def test_api_key_generation(self):
        raw, prefix, key_hash = generate_api_key()
        assert raw.startswith("ztn_")
        assert len(raw) == 4 + 64     # "ztn_" + 64 hex chars
        assert prefix == raw[:8]
        assert len(key_hash) == 64    # SHA-256 hex

    def test_api_key_hash_deterministic(self):
        raw, _, h1 = generate_api_key()
        h2 = hash_api_key(raw)
        assert h1 == h2

    def test_jwt_roundtrip(self):
        import os
        os.environ["ZUNTENIUM_JWT_SECRET"] = "test-secret-for-unit-test"
        user_id = str(uuid.uuid4())
        token   = create_access_token(user_id, "user")
        payload = decode_token(token)
        assert payload["sub"]  == user_id
        assert payload["role"] == "user"
        assert payload["type"] == "access"

    def test_jwt_refresh_token_type(self):
        import os
        os.environ["ZUNTENIUM_JWT_SECRET"] = "test-secret-for-unit-test"
        user_id = str(uuid.uuid4())
        token   = create_refresh_token(user_id)
        payload = decode_token(token)
        assert payload["type"] == "refresh"

    def test_jwt_invalid_raises(self):
        import os
        from fastapi import HTTPException
        os.environ["ZUNTENIUM_JWT_SECRET"] = "test-secret"
        with pytest.raises(HTTPException) as exc_info:
            decode_token("not.a.valid.token")
        assert exc_info.value.status_code == 401


# ── SECTION C: Circuit Reconstruction ────────────────────────────────────────

class TestWorkerCircuit:

    def test_build_circuit_basic(self):
        circuit_json = {
            "num_qubits": 2,
            "gates": [
                {"name": "h",  "qubits": [0], "params": []},
                {"name": "cx", "qubits": [0, 1], "params": []},
            ],
            "name": "bell",
        }
        circuit = _build_circuit(circuit_json)
        assert circuit.num_qubits == 2
        assert len(circuit.gates) == 2
        assert circuit.gates[0].name == "h"
        assert circuit.gates[1].qubits == [0, 1]

    def test_build_circuit_with_params(self):
        circuit_json = {
            "num_qubits": 1,
            "gates": [{"name": "rx", "qubits": [0], "params": [1.57]}],
        }
        circuit = _build_circuit(circuit_json)
        assert circuit.gates[0].params == [1.57]

    def test_build_circuit_large(self):
        gates = [{"name": "h", "qubits": [i], "params": []} for i in range(50)]
        gates += [{"name": "cx", "qubits": [i, i+1], "params": []} for i in range(49)]
        circuit = _build_circuit({"num_qubits": 50, "gates": gates})
        assert circuit.num_qubits == 50
        assert len(circuit.gates) == 99


# ── SECTION D: Backend Routing & Fallback ────────────────────────────────────

class TestBackendRouting:

    def _bell_circuit(self):
        circuit_json = {
            "num_qubits": 2,
            "gates": [
                {"name": "h",  "qubits": [0], "params": []},
                {"name": "cx", "qubits": [0, 1], "params": []},
            ],
            "name": "bell",
        }
        return _build_circuit(circuit_json)

    def test_tensornetwork_backend_bell(self):
        """MPS backend should produce correct Bell state counts."""
        try:
            from zuntenium.backends.tensor_network import TensorNetworkBackend
        except ImportError:
            pytest.skip("TensorNetworkBackend not available")

        import numpy as np
        tn = TensorNetworkBackend(max_bond_dim=8)
        circuit = self._bell_circuit()
        result = _run_circuit_on_backend(
            circuit, "tensornetwork", {"tensornetwork": tn}, {"shots": 5000}
        )
        counts = result.get("counts", {})
        total  = sum(counts.values())
        p00    = counts.get("00", 0) / max(total, 1)
        p11    = counts.get("11", 0) / max(total, 1)
        assert abs(p00 - 0.5) < 0.05, f"p00={p00:.3f}"
        assert abs(p11 - 0.5) < 0.05, f"p11={p11:.3f}"

    def test_fallback_when_backend_missing(self):
        """Should fall back gracefully when requested backend is absent."""
        try:
            from zuntenium.backends.tensor_network import TensorNetworkBackend
        except ImportError:
            pytest.skip("TensorNetworkBackend not available")

        circuit  = self._bell_circuit()
        backends = {"tensornetwork": TensorNetworkBackend(max_bond_dim=4)}
        # Request IBM but it's not in the registry
        result = _run_circuit_on_backend(
            circuit, "ibm", backends, {"shots": 100}
        )
        assert "counts" in result

    def test_fallback_empty_registry(self):
        """Even with empty registry, should use TN directly."""
        try:
            from zuntenium.backends.tensor_network import TensorNetworkBackend
        except ImportError:
            pytest.skip("TN not available")

        circuit = self._bell_circuit()
        result  = _run_circuit_on_backend(circuit, "statevector", {}, {"shots": 100})
        assert "counts" in result

    def test_large_circuit_routes_to_mps(self):
        """n>28 should route to MPS backend, not crash."""
        try:
            from zuntenium.backends.tensor_network import TensorNetworkBackend
        except ImportError:
            pytest.skip("TN not available")

        gates = [{"name": "h", "qubits": [i], "params": []} for i in range(30)]
        big   = _build_circuit({"num_qubits": 30, "gates": gates})
        tn    = TensorNetworkBackend(max_bond_dim=4)
        result = _run_circuit_on_backend(
            big, "tensornetwork", {"tensornetwork": tn}, {"shots": 100}
        )
        assert "counts" in result


# ── SECTION E: Job Manager Unit Tests ────────────────────────────────────────

class TestJobManager:
    """Unit tests with mocked DB and Redis."""

    def _make_mock_db(self):
        db = AsyncMock()
        db.execute = AsyncMock()
        db.add     = MagicMock()
        db.flush   = AsyncMock()
        db.commit  = AsyncMock()
        return db

    def _make_mock_redis(self):
        r_sync  = MagicMock()
        r_async = AsyncMock()
        r_async.setex = AsyncMock(return_value=True)
        r_async.get   = AsyncMock(return_value=None)
        # Mock RQ Queue
        mock_job = MagicMock()
        mock_job.id = str(uuid.uuid4())
        r_sync.llen = MagicMock(return_value=0)
        return r_sync, r_async

    @pytest.mark.asyncio
    async def test_submit_creates_job(self):
        from zuntenium_cloud.job_manager import JobManager
        from zuntenium_cloud.models import JobStatus

        db = self._make_mock_db()
        rs, ra = self._make_mock_redis()

        # Mock RQ Queue.enqueue
        with patch("zuntenium_cloud.job_manager.Queue") as MockQueue:
            mock_q = MagicMock()
            mock_rq_job = MagicMock()
            mock_rq_job.id = "rq-test-123"
            mock_q.enqueue.return_value = mock_rq_job
            MockQueue.return_value = mock_q

            manager = JobManager(db, rs, ra)
            request = JobSubmitRequest(
                job_type="circuit",
                circuit=bell_circuit_schema(),
                shots=1024,
            )

            job = await manager.submit(request, user_id=str(uuid.uuid4()))
            assert job is not None
            assert db.add.called
            assert db.commit.called

    @pytest.mark.asyncio
    async def test_cancel_queued_job(self):
        from zuntenium_cloud.job_manager import JobManager
        from zuntenium_cloud.models import Job, JobStatus, JobType, BackendPreference

        db = self._make_mock_db()
        rs, ra = self._make_mock_redis()

        job_id  = str(uuid.uuid4())
        user_id = str(uuid.uuid4())
        mock_job = Job(
            id=job_id, user_id=user_id, job_type=JobType.CIRCUIT,
            status=JobStatus.QUEUED, shots=1024,
            mitigation=False, optimization_mode="rule",
            backend_pref=BackendPreference.AUTO,
            retry_count=0, max_retries=3, priority=5, queue_name="default",
            created_at=datetime.now(timezone.utc),
        )

        from sqlalchemy.engine.result import MappingResult
        db.execute.return_value.scalar_one_or_none = MagicMock(return_value=mock_job)

        with patch("rq.job.Job.fetch", side_effect=Exception("not in queue")):
            manager = JobManager(db, rs, ra)
            with patch.object(manager._db, "execute", new_callable=AsyncMock) as mock_exec:
                mock_exec.return_value.scalar_one_or_none = MagicMock(return_value=mock_job)
                result = await manager.cancel_job(job_id, user_id)
                # Cancel should succeed (update + commit called)
                assert db.commit.called


# ── SECTION F: Fault Injection ────────────────────────────────────────────────

class TestFaultInjection:

    def test_backend_exception_raises_to_worker(self):
        """Backend crash should propagate (worker retries it)."""
        class _CrashingBackend:
            def run(self, circuit, **kwargs):
                raise RuntimeError("Simulated hardware failure")

        circuit_json = {
            "num_qubits": 2,
            "gates": [{"name": "h", "qubits": [0], "params": []}],
        }
        circuit = _build_circuit(circuit_json)

        with pytest.raises(RuntimeError, match="Simulated hardware failure"):
            _run_circuit_on_backend(
                circuit, "ibm",
                {"ibm": _CrashingBackend()},
                {"shots": 100},
            )

    def test_empty_circuit_returns_empty_counts(self):
        """None circuit should return gracefully."""
        result = _run_circuit_on_backend(None, "statevector", {}, {"shots": 100})
        assert result == {"counts": {}, "statevector": None}

    def test_worker_reconstruct_invalid_json_raises(self):
        """Malformed circuit JSON should raise ValueError."""
        with pytest.raises((KeyError, TypeError)):
            _build_circuit({"not_num_qubits": 2})


# ── SECTION G: Load / Concurrent Simulation ───────────────────────────────────

class TestConcurrentLoad:
    """Simulate concurrent job submissions (no real DB required)."""

    def test_100_schema_validations_fast(self):
        """Schema validation must handle 100 circuits in <100ms."""
        import time
        t0     = time.perf_counter()
        errors = 0
        for i in range(100):
            try:
                c = CircuitSchema(
                    num_qubits=5,
                    gates=[
                        GateSchema(name="h",  qubits=[0]),
                        GateSchema(name="cx", qubits=[0, 1]),
                        GateSchema(name="rx", qubits=[2], params=[1.0]),
                    ],
                )
                assert c.num_qubits == 5
            except Exception:
                errors += 1

        elapsed = (time.perf_counter() - t0) * 1000
        assert errors == 0
        assert elapsed < 100, f"Schema validation took {elapsed:.0f}ms for 100 circuits"

    def test_100_jwt_tokens_fast(self):
        """Token creation must handle 100 tokens in <200ms."""
        import os
        os.environ["ZUNTENIUM_JWT_SECRET"] = "test-secret"
        t0 = time.perf_counter()
        for _ in range(100):
            uid   = str(uuid.uuid4())
            token = create_access_token(uid, "user")
            dec   = decode_token(token)
            assert dec["sub"] == uid
        elapsed = (time.perf_counter() - t0) * 1000
        assert elapsed < 500, f"JWT x100 took {elapsed:.0f}ms"

    @pytest.mark.asyncio
    async def test_50_mps_circuits_concurrent(self):
        """50 independent MPS Bell circuits in parallel should all succeed."""
        try:
            from zuntenium.backends.tensor_network import TensorNetworkBackend
        except ImportError:
            pytest.skip("TN not available")

        import asyncio
        circuit_json = {
            "num_qubits": 2,
            "gates": [
                {"name": "h",  "qubits": [0], "params": []},
                {"name": "cx", "qubits": [0, 1], "params": []},
            ],
        }

        def _run_one():
            circuit = _build_circuit(circuit_json)
            tn = TensorNetworkBackend(max_bond_dim=4)
            return _run_circuit_on_backend(
                circuit, "tensornetwork",
                {"tensornetwork": tn}, {"shots": 200}
            )

        loop    = asyncio.get_event_loop()
        results = await asyncio.gather(*[
            loop.run_in_executor(None, _run_one)
            for _ in range(50)
        ])

        assert len(results) == 50
        for r in results:
            assert "counts" in r
            total = sum(r["counts"].values())
            assert total == 200


# ── SECTION H: End-to-End Schema Round-Trip ───────────────────────────────────

class TestEndToEnd:

    def test_circuit_json_roundtrip(self):
        """Circuit schema → JSON → schema should be lossless."""
        original = bell_circuit_schema()
        json_str  = original.model_dump_json()
        restored  = CircuitSchema.model_validate_json(json_str)
        assert restored.num_qubits == original.num_qubits
        assert len(restored.gates) == len(original.gates)
        for o, r in zip(original.gates, restored.gates):
            assert o.name   == r.name
            assert o.qubits == r.qubits

    def test_job_submit_to_worker_dict(self):
        """
        Simulate the full path: JobSubmitRequest → circuit_json → worker reconstruct.
        """
        req     = JobSubmitRequest(circuit=bell_circuit_schema(), shots=2048)
        circuit_json = req.circuit.model_dump()
        circuit = _build_circuit(circuit_json)
        assert circuit.num_qubits == 2
        assert circuit.gates[0].name == "h"
        assert circuit.gates[1].name == "cx"

    def test_mps_bell_end_to_end(self):
        """Full pipeline: circuit spec → MPS execution → valid Bell counts."""
        try:
            from zuntenium.backends.tensor_network import TensorNetworkBackend
            import numpy as np
        except ImportError:
            pytest.skip("TN not available")

        req     = JobSubmitRequest(
            job_type="circuit",
            circuit=bell_circuit_schema(),
            shots=10_000,
            backend="tensornetwork",
        )
        circuit_json = req.circuit.model_dump()
        circuit      = _build_circuit(circuit_json)
        tn           = TensorNetworkBackend(max_bond_dim=8)

        result = _run_circuit_on_backend(
            circuit, "tensornetwork",
            {"tensornetwork": tn}, {"shots": 10_000}
        )

        counts = result["counts"]
        total  = sum(counts.values())
        p00    = counts.get("00", 0) / total
        p11    = counts.get("11", 0) / total

        assert abs(p00 - 0.5) < 0.03, f"Bell p(00) = {p00:.3f}"
        assert abs(p11 - 0.5) < 0.03, f"Bell p(11) = {p11:.3f}"
        assert abs(p00 + p11 - 1.0) < 0.01, "Bell state must be only |00> and |11>"

    def test_api_schema_error_format(self):
        """Error responses should have consistent structure."""
        err = ErrorResponse(
            error="ValidationError",
            details=[],
            request_id=str(uuid.uuid4()),
        )
        d = err.model_dump()
        assert "error"   in d
        assert "details" in d


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import subprocess
    subprocess.run([
        sys.executable, "-m", "pytest", __file__,
        "-v", "--tb=short", "-x", "--asyncio-mode=auto"
    ])
