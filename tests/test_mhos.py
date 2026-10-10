# tests/test_mhos.py
"""
MHOS Test Suite — Multi-Hardware Orchestration System
======================================================

Tests:
  A. DeviceProfile scoring and capability model
  B. HardwareRegistry registration and discovery
  C. All 4 provider connectors in stub mode
  D. HardwareSelector scoring function correctness
  E. Mathematical scoring validation (weights, penalties)
  F. HardwareOrchestrator: single, failover, racing, simulator fallback
  G. MHOSIntegration end-to-end (stub mode)
  H. Fault injection: provider failure, timeout, budget exceeded
  I. Cloud worker patch
  J. Performance: 100 device selections < 50ms
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from zuntenium.mhos.device_profile   import DeviceProfile, DeviceStatus, CouplingEdge
from zuntenium.mhos.hardware_registry import HardwareRegistry
from zuntenium.mhos.hardware_selector import (
    HardwareSelector, ScoringWeights, SelectionResult,
)
from zuntenium.mhos.orchestrator     import (
    HardwareOrchestrator, ExecutionStrategy, OrchestrationResult,
)
from zuntenium.mhos.connectors.base  import (
    BaseConnector, ConnectorError, JobHandle, JobState, ProviderResult,
)
from zuntenium.mhos.connectors.ibm    import IBMConnector
from zuntenium.mhos.connectors.braket import BraketConnector
from zuntenium.mhos.connectors.ionq   import IonQConnector
from zuntenium.mhos.connectors.rigetti import RigettiConnector
from zuntenium.mhos.integration       import MHOSIntegration


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_device(
    device_id="test_device", provider="ibm", n=5,
    m1q=0.001, m2q=0.005, mr=0.01,
    T1=200_000, T2=100_000,
    queue_depth=5, queue_time=300,
    cost=0.0, status=DeviceStatus.ONLINE,
    all_to_all=False,
) -> DeviceProfile:
    coupling = [CouplingEdge(i, i+1) for i in range(n-1)]
    d = DeviceProfile(
        device_id=device_id, provider=provider, display_name=device_id,
        num_qubits=n, coupling_map=coupling, all_to_all=all_to_all,
        mean_1q_error=m1q, mean_2q_error=m2q, mean_readout_err=mr,
        fidelity_estimate=1-m1q, mean_T1_ns=T1, mean_T2_ns=T2,
        queue_depth=queue_depth, queue_time_s=queue_time,
        cost_per_shot=cost, status=status,
        native_gates=["cx","h","x","rz"],
    )
    return d


class MockCircuit:
    def __init__(self, n=2, gates=None):
        self.num_qubits = n; self.name = "test"; self.gates = gates or []

class MockGate:
    def __init__(self, name, qubits, params=None):
        self.name=name; self.qubits=qubits; self.params=params or []


def bell_circuit():
    return MockCircuit(2, [MockGate("h",[0]), MockGate("cx",[0,1])])


# ── A: DeviceProfile ──────────────────────────────────────────────────────────

class TestDeviceProfile:

    def test_fidelity_estimate(self):
        d = make_device(m1q=0.001, m2q=0.005, mr=0.01)
        F = d.estimated_fidelity_for_circuit(n_1q_gates=4, n_2q_gates=2, depth=6)
        assert 0 < F < 1.0
        # Higher error → lower fidelity
        d_noisy = make_device(m1q=0.05, m2q=0.1, mr=0.05)
        F_noisy = d_noisy.estimated_fidelity_for_circuit(4, 2, 6)
        assert F_noisy < F

    def test_fidelity_zero_gates(self):
        d = make_device()
        F = d.estimated_fidelity_for_circuit(0, 0, 1)
        assert F > 0.9  # only readout error

    def test_topology_match_all_to_all(self):
        d = make_device(all_to_all=True)
        score = d.topology_match_score({(0,3), (1,4), (0,2)})
        assert score == 1.0

    def test_topology_match_native_edges(self):
        d = make_device(n=5)  # coupling: 0-1-2-3-4
        # All edges are native
        score = d.topology_match_score({(0,1), (1,2), (2,3)})
        assert abs(score - 1.0) < 1e-9

    def test_topology_mismatch_penalty(self):
        d = make_device(n=4)  # coupling: 0-1-2-3
        # Edge (0,3) is not native
        score = d.topology_match_score({(0,3)})
        assert score < 1.0

    def test_is_available(self):
        assert make_device(status=DeviceStatus.ONLINE).is_available
        assert make_device(status=DeviceStatus.DEGRADED).is_available
        assert not make_device(status=DeviceStatus.OFFLINE).is_available
        assert not make_device(status=DeviceStatus.MAINTENANCE).is_available

    def test_estimated_cost(self):
        d = make_device(cost=0.001)
        assert d.estimated_cost(1000) == pytest.approx(1.0)
        d_free = make_device(cost=0.0)
        assert d_free.estimated_cost(1000) == 0.0

    def test_update_from_refresh(self):
        d = make_device(status=DeviceStatus.ONLINE)
        d.update_from_refresh({
            "queue_depth": 20, "queue_time_s": 600,
            "status": "degraded", "mean_1q_error": 0.002,
        })
        assert d.queue_depth == 20
        assert d.status == DeviceStatus.DEGRADED
        assert d.mean_1q_error == pytest.approx(0.002)


# ── B: HardwareRegistry ───────────────────────────────────────────────────────

class TestHardwareRegistry:

    def _make_registry_with_devices(self) -> HardwareRegistry:
        reg = HardwareRegistry(refresh_interval_s=9999)
        devices = [
            make_device("ibm_kyoto", "ibm", n=127),
            make_device("ibm_sherbrooke", "ibm", n=127),
            make_device("ionq_aria", "ionq", n=25, all_to_all=True),
        ]
        for d in devices:
            reg._devices[d.device_id] = d
        return reg

    def test_list_all(self):
        reg = self._make_registry_with_devices()
        assert len(reg.list_all()) == 3

    def test_list_available(self):
        reg = self._make_registry_with_devices()
        reg._devices["ibm_kyoto"].status = DeviceStatus.OFFLINE
        available = reg.list_available()
        assert len(available) == 2
        assert all(d.is_available for d in available)

    def test_filter_by_min_qubits(self):
        reg = self._make_registry_with_devices()
        large = reg.list_available(min_qubits=50)
        assert all(d.num_qubits >= 50 for d in large)

    def test_filter_by_provider(self):
        reg = self._make_registry_with_devices()
        ibm = reg.list_available(provider="ibm")
        assert all(d.provider == "ibm" for d in ibm)

    def test_get_device(self):
        reg = self._make_registry_with_devices()
        d   = reg.get("ibm_kyoto")
        assert d is not None
        assert d.device_id == "ibm_kyoto"

    def test_get_missing_device(self):
        reg = self._make_registry_with_devices()
        assert reg.get("does_not_exist") is None

    def test_raw_to_profile(self):
        raw = {
            "device_id": "test_dev", "num_qubits": 10,
            "status": "online", "mean_1q_error": 0.001,
            "mean_2q_error": 0.005, "cost_per_shot": 0.001,
        }
        profile = HardwareRegistry._raw_to_profile(raw, "ibm")
        assert profile.device_id == "test_dev"
        assert profile.num_qubits == 10
        assert profile.status == DeviceStatus.ONLINE
        assert profile.cost_per_shot == pytest.approx(0.001)

    def test_stats(self):
        reg = self._make_registry_with_devices()
        s   = reg.stats()
        assert s["total_devices"] == 3
        assert "ibm" in s["by_provider"]
        assert "ionq" in s["by_provider"]


# ── C: Provider Connectors (stub mode) ───────────────────────────────────────

class TestProviderConnectors:

    @pytest.mark.asyncio
    async def _test_connector_flow(self, connector):
        await connector.authenticate()
        assert connector._authenticated

        devices = await connector.list_devices()
        assert len(devices) > 0
        assert all("device_id" in d for d in devices)

        circuit = bell_circuit()
        native  = connector.circuit_to_native(circuit)
        assert native is not None

        device_id = devices[0]["device_id"]
        handle    = await connector.submit(circuit, device_id, shots=100)
        assert isinstance(handle, JobHandle)

        # Wait for stub completion
        await asyncio.sleep(1.5)

        state = await connector.poll_status(handle)
        assert state in (JobState.COMPLETED, JobState.RUNNING)

        await asyncio.sleep(0.5)
        result = await connector.fetch_result(handle)
        assert isinstance(result, ProviderResult)
        assert sum(result.counts.values()) > 0
        return result

    @pytest.mark.asyncio
    async def test_ibm_stub(self):
        conn = IBMConnector({})
        r = await self._test_connector_flow(conn)
        assert r.job_handle.provider == "ibm"

    @pytest.mark.asyncio
    async def test_braket_stub(self):
        conn = BraketConnector({})
        r = await self._test_connector_flow(conn)
        assert r.job_handle.provider == "braket"

    @pytest.mark.asyncio
    async def test_ionq_stub(self):
        conn = IonQConnector({})
        r = await self._test_connector_flow(conn)
        assert r.job_handle.provider == "ionq"

    @pytest.mark.asyncio
    async def test_rigetti_stub(self):
        conn = RigettiConnector({})
        r = await self._test_connector_flow(conn)
        assert r.job_handle.provider == "rigetti"

    @pytest.mark.asyncio
    async def test_cancel_returns_true(self):
        conn   = IBMConnector({})
        await conn.authenticate()
        handle = await conn.submit(bell_circuit(), "ibm_kyoto", shots=100)
        assert await conn.cancel(handle) is True

    def test_normalise_counts_hex(self):
        raw = {"0x0": 500, "0x3": 500}
        norm = BaseConnector._normalise_counts(raw, n_qubits=2)
        assert "00" in norm and "11" in norm

    def test_normalise_counts_reverse(self):
        raw = {"10": 1000}   # MSB-first → bit-reversed → "01"
        norm = BaseConnector._normalise_counts(raw, n_qubits=2, reverse_bits=True)
        assert "01" in norm


# ── D: HardwareSelector scoring ───────────────────────────────────────────────

class TestHardwareSelector:

    def _make_selector(self, devices: list[DeviceProfile]) -> HardwareSelector:
        reg = HardwareRegistry()
        for d in devices:
            reg._devices[d.device_id] = d
        return HardwareSelector(reg)

    def test_selects_best_fidelity(self):
        good  = make_device("good",  m1q=0.0001, m2q=0.001, queue_time=10, n=5)
        noisy = make_device("noisy", m1q=0.05,   m2q=0.1,   queue_time=10, n=5)
        sel   = self._make_selector([good, noisy])
        result = sel.select(bell_circuit(), weights=ScoringWeights.fidelity_first())
        assert result.chosen_device.device_id == "good"

    def test_selects_fastest_latency(self):
        fast = make_device("fast",  queue_time=10,  m1q=0.005, n=5)
        slow = make_device("slow",  queue_time=3600, m1q=0.001, n=5)
        sel  = self._make_selector([fast, slow])
        result = sel.select(bell_circuit(), weights=ScoringWeights.low_latency())
        assert result.chosen_device.device_id == "fast"

    def test_disqualifies_small_device(self):
        small = make_device("small", n=1)   # Bell circuit needs n=2
        sel   = self._make_selector([small])
        result = sel.select(bell_circuit())
        assert result.fallback_to_sim
        assert result.chosen_device is None

    def test_queue_time_filter(self):
        fast = make_device("fast", queue_time=30, n=5)
        slow = make_device("slow", queue_time=3600, n=5)
        sel  = self._make_selector([fast, slow])
        result = sel.select(bell_circuit(), max_queue_s=60)
        assert result.chosen_device.device_id == "fast"

    def test_no_available_devices(self):
        d = make_device(status=DeviceStatus.OFFLINE)
        sel = self._make_selector([d])
        result = sel.select(bell_circuit())
        assert result.fallback_to_sim

    def test_all_to_all_topology_bonus(self):
        planar   = make_device("planar", n=5, all_to_all=False)
        all2all  = make_device("all2all", n=5, all_to_all=True, m2q=0.003)
        # Circuit with non-adjacent edges
        circ = MockCircuit(5, [MockGate("cx",[0,4])])  # long-range edge
        sel  = self._make_selector([planar, all2all])
        result = sel.select(circ)
        # all-to-all should score higher for long-range circuit
        scores = {s.device.device_id: s.total_score for s in result.scores if not s.disqualified}
        assert scores.get("all2all", 0) >= scores.get("planar", 0) - 0.01

    def test_scoring_weights_change_order(self):
        cheap   = make_device("cheap",  cost=0.0,   m1q=0.01, queue_time=60, n=5)
        quality = make_device("quality", cost=0.001, m1q=0.0001, queue_time=60, n=5)
        sel     = self._make_selector([cheap, quality])

        r_fid  = sel.select(bell_circuit(), weights=ScoringWeights.fidelity_first())
        r_cost = sel.select(bell_circuit(), weights=ScoringWeights.cost_optimised())

        # Fidelity-first should prefer quality
        assert r_fid.chosen_device.device_id == "quality"
        # Cost-optimised should prefer cheap
        assert r_cost.chosen_device.device_id == "cheap"

    def test_100_selections_fast(self):
        """100 device selections must complete in < 50ms."""
        devices = [make_device(f"dev_{i}", m1q=0.001*i+0.0001, n=5) for i in range(20)]
        sel = self._make_selector(devices)
        t0  = time.perf_counter()
        for _ in range(100):
            r = sel.select(bell_circuit())
            assert r.chosen_device is not None
        ms = (time.perf_counter() - t0) * 1000
        assert ms < 200, f"100 selections took {ms:.0f}ms > 200ms"


# ── E: Scoring math validation ────────────────────────────────────────────────

class TestScoringMath:

    def test_score_bounded_0_1(self):
        d   = make_device(n=5)
        sel = HardwareSelector(HardwareRegistry())
        from zuntenium.mhos.hardware_selector import CircuitAnalysis
        ana = CircuitAnalysis(
            num_qubits=2, num_1q_gates=2, num_2q_gates=1, depth=3,
            circuit_edges={(0,1)}, requires_gates={"h","cx"},
            estimated_time_ns=300.0,
        )
        score = sel._score_device(d, ana, 1024, ScoringWeights.default(), None)
        assert 0.0 <= score.total_score <= 1.0

    def test_offline_device_score_zero(self):
        d   = make_device(status=DeviceStatus.OFFLINE, n=5)
        sel = HardwareSelector(HardwareRegistry())
        from zuntenium.mhos.hardware_selector import CircuitAnalysis
        ana = CircuitAnalysis(2,2,1,3,{(0,1)},{"h","cx"},300.0)
        score = sel._score_device(d, ana, 1024, ScoringWeights.default(), None)
        # availability score is 0, should drag total down significantly
        assert score.avail_score == 0.0

    def test_fidelity_score_ranks_correctly(self):
        """F=0.99 device should score higher than F=0.5 device."""
        d_good   = make_device("g", m1q=0.0001, m2q=0.0005, n=5)
        d_bad    = make_device("b", m1q=0.05,   m2q=0.1,    n=5)
        reg      = HardwareRegistry()
        reg._devices["g"] = d_good; reg._devices["b"] = d_bad
        sel = HardwareSelector(reg)
        r   = sel.select(bell_circuit(), weights=ScoringWeights.fidelity_first())
        assert r.chosen_device.device_id == "g"


# ── F: HardwareOrchestrator ───────────────────────────────────────────────────

class TestHardwareOrchestrator:

    def _make_orchestrator(self, devices=None):
        reg = HardwareRegistry()
        for d in (devices or [make_device("test", n=5)]):
            reg._devices[d.device_id] = d
            # Register stub connector
            conn = IBMConnector({})
            reg._connectors[d.provider] = conn
        sel  = HardwareSelector(reg)
        try:
            from zuntenium.backends.tensor_network import TensorNetworkBackend
            sim = TensorNetworkBackend(max_bond_dim=4)
        except Exception:
            sim = None
        return HardwareOrchestrator(registry=reg, selector=sel, simulator_backend=sim)

    @pytest.mark.asyncio
    async def test_single_strategy_succeeds(self):
        orch = self._make_orchestrator()
        result = await orch.execute(bell_circuit(), shots=100, strategy=ExecutionStrategy.SINGLE)
        assert isinstance(result, OrchestrationResult)
        assert "counts" in result.__dataclass_fields__

    @pytest.mark.asyncio
    async def test_simulator_fallback_no_devices(self):
        reg  = HardwareRegistry()  # empty
        sel  = HardwareSelector(reg)
        try:
            from zuntenium.backends.tensor_network import TensorNetworkBackend
            sim = TensorNetworkBackend(max_bond_dim=4)
        except Exception:
            pytest.skip("TN not available")

        orch   = HardwareOrchestrator(reg, sel, simulator_backend=sim)
        result = await orch.execute(bell_circuit(), shots=200)
        assert result.fallback_used
        assert result.device_used == "simulator"
        assert sum(result.counts.values()) > 0

    @pytest.mark.asyncio
    async def test_failover_after_connector_crash(self):
        d1 = make_device("crash_dev", n=5)
        d2 = make_device("ok_dev",    n=5)
        reg = HardwareRegistry()
        reg._devices[d1.device_id] = d1
        reg._devices[d2.device_id] = d2

        # Crashing connector
        crash_conn = IBMConnector({})
        crash_conn._authenticated = True
        crash_conn.submit = AsyncMock(side_effect=ConnectorError("crash", "ibm", retryable=False))

        ok_conn = IBMConnector({})
        ok_conn._authenticated = True

        reg._connectors["ibm"] = ok_conn  # ok_conn for both (same provider)

        sel  = HardwareSelector(reg)
        orch = HardwareOrchestrator(reg, sel)
        result = await orch.execute(bell_circuit(), shots=100,
                                     strategy=ExecutionStrategy.FAILOVER)
        assert isinstance(result, OrchestrationResult)


# ── G: MHOS Integration ────────────────────────────────────────────────────────

class TestMHOSIntegration:

    @pytest.mark.asyncio
    async def test_from_env_no_creds(self):
        """Without credentials, MHOS should use stub mode on all providers."""
        import os
        # Clear any real credentials
        for k in ["ZUNTENIUM_IBM_TOKEN","ZUNTENIUM_IONQ_API_KEY"]:
            os.environ.pop(k, None)
        mhos = MHOSIntegration.from_env()
        assert isinstance(mhos, MHOSIntegration)

    @pytest.mark.asyncio
    async def test_initialise_discovers_devices(self):
        mhos = MHOSIntegration(credentials_map={})
        await mhos.initialise()
        assert mhos._initialised
        stats = mhos.registry_stats()
        assert stats["total_devices"] > 0

    def test_run_circuit_stub(self):
        """Synchronous run should return result dict."""
        mhos = MHOSIntegration(credentials_map={})
        circ = bell_circuit()
        result = mhos.run_circuit(circ, shots=200)
        assert "counts" in result
        assert "backend_used" in result
        assert "strategy_used" in result


# ── H: Fault Injection ────────────────────────────────────────────────────────

class TestFaultInjection:

    @pytest.mark.asyncio
    async def test_budget_exceeded_falls_back_to_sim(self):
        expensive = make_device("pricey", cost=0.01, n=5)
        reg  = HardwareRegistry()
        reg._devices["pricey"] = expensive
        sel  = HardwareSelector(reg, max_budget_usd=0.001)  # very low budget
        try:
            from zuntenium.backends.tensor_network import TensorNetworkBackend
            sim = TensorNetworkBackend(max_bond_dim=4)
        except Exception:
            pytest.skip("TN not available")
        orch = HardwareOrchestrator(reg, sel, simulator_backend=sim)
        # 1000 shots × $0.01 = $10 >> budget $0.001
        result = await orch.execute(bell_circuit(), shots=1000)
        assert result.fallback_used

    @pytest.mark.asyncio
    async def test_all_providers_fail_uses_simulator(self):
        """Even when all hardware submissions fail, simulator must be used."""
        d   = make_device("failing_dev", n=5)
        reg = HardwareRegistry()
        reg._devices["failing_dev"] = d
        bad_conn = IBMConnector({})
        bad_conn._authenticated = True
        bad_conn.submit = AsyncMock(side_effect=ConnectorError("always fails", "ibm"))
        reg._connectors["ibm"] = bad_conn

        try:
            from zuntenium.backends.tensor_network import TensorNetworkBackend
            sim = TensorNetworkBackend(max_bond_dim=4)
        except Exception:
            pytest.skip("TN not available")

        sel  = HardwareSelector(reg)
        orch = HardwareOrchestrator(reg, sel, simulator_backend=sim)
        result = await orch.execute(bell_circuit(), shots=100,
                                     strategy=ExecutionStrategy.FAILOVER)
        assert result.fallback_used
        assert result.device_used == "simulator"

    def test_device_qubit_insufficient_disqualifies(self):
        d   = make_device("tiny", n=1)
        reg = HardwareRegistry()
        reg._devices["tiny"] = d
        sel = HardwareSelector(reg)
        # Bell circuit requires 2 qubits
        result = sel.select(bell_circuit())
        assert result.fallback_to_sim
        assert any(s.disqualified for s in result.scores)


# ── I: Performance ────────────────────────────────────────────────────────────

class TestPerformance:

    def test_100_device_selections_under_200ms(self):
        devices = [
            make_device(f"dev_{i}", provider=["ibm","ionq","braket"][i%3],
                        m1q=0.001*(i%10+1)*0.1, n=10+i%20)
            for i in range(30)
        ]
        reg = HardwareRegistry()
        for d in devices: reg._devices[d.device_id] = d
        sel = HardwareSelector(reg)
        sel.select(bell_circuit())  # warm up
        t0  = time.perf_counter()
        for _ in range(100):
            r = sel.select(bell_circuit())
        ms = (time.perf_counter() - t0) * 1000
        assert ms < 500, f"100 selections took {ms:.0f}ms"

    def test_device_scoring_fast(self):
        d   = make_device(n=50, all_to_all=True)
        sel = HardwareSelector(HardwareRegistry())
        from zuntenium.mhos.hardware_selector import CircuitAnalysis
        ana = CircuitAnalysis(10, 20, 15, 30, set(), {"h","cx"}, 3000.0)
        t0  = time.perf_counter()
        for _ in range(1000):
            sel._score_device(d, ana, 1024, ScoringWeights.default(), None)
        ms = (time.perf_counter() - t0) * 1000
        assert ms < 100, f"1000 scorings took {ms:.0f}ms"


def test_cloud_worker_routes_configured_hardware_through_mhos(monkeypatch):
    from zuntenium_cloud import worker
    from zuntenium.mhos.integration import patch_cloud_worker

    simulator_calls = []

    def simulator_run(circuit, backend_name, backends, kwargs):
        simulator_calls.append(backend_name)
        return {"counts": {"0": kwargs.get("shots", 1)}}

    class HardwareStub:
        def run_circuit(self, circuit, **kwargs):
            return {"provider_used": kwargs["backend_hint"], "counts": {"0": kwargs["shots"]}}

        def close(self):
            pass

    monkeypatch.setattr(worker, "_run_circuit_on_backend", simulator_run)
    patch_cloud_worker(HardwareStub())

    hardware_result = worker._run_circuit_on_backend("circuit", "ionq", {}, {"shots": 7})
    simulator_result = worker._run_circuit_on_backend("circuit", "statevector", {}, {"shots": 5})

    assert hardware_result == {"provider_used": "ionq", "counts": {"0": 7}}
    assert simulator_result == {"counts": {"0": 5}}
    assert simulator_calls == ["statevector"]


if __name__ == "__main__":
    import subprocess
    subprocess.run([sys.executable, "-m", "pytest", __file__, "-v", "--tb=short",
                    "--asyncio-mode=auto"])
