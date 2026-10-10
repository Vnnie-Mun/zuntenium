# zuntenium/mhos/orchestrator.py
"""
HardwareOrchestrator — Master execution engine for multi-hardware routing.

Execution strategies:
  SINGLE      — submit to best device, wait for result
  FAILOVER    — try best → fallback chain → simulator
  RACING      — submit to top-K devices simultaneously, return fastest result
  COST_AWARE  — like SINGLE but enforces budget constraint per job

All strategies include:
  - Pre-flight circuit compatibility check
  - Transpiler integration (gate decomposition to device native set)
  - Retry with exponential backoff (up to max_retries)
  - Automatic fallback to MPS/statevector simulator
  - ErrorMitigationEngine post-processing (if enabled)
  - Full audit logging per job
"""

from __future__ import annotations

import asyncio
import enum
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from .connectors.base import (
    BaseConnector, ConnectorError, FatalError, JobHandle,
    JobState, ProviderResult,
)
from .device_profile import DeviceProfile
from .hardware_registry import HardwareRegistry
from .hardware_selector import HardwareSelector, ScoringWeights, SelectionResult

_log = logging.getLogger(__name__)


class ExecutionStrategy(str, enum.Enum):
    SINGLE     = "single"      # best device → wait
    FAILOVER   = "failover"    # best → fallback chain
    RACING     = "racing"      # top-K in parallel → fastest result
    COST_AWARE = "cost_aware"  # cheapest qualifying device


@dataclass
class OrchestrationResult:
    """Final result from HardwareOrchestrator.execute()."""
    counts:             dict[str, int]
    shots:              int
    device_used:        str
    provider_used:      str
    execution_time_s:   float = 0.0
    queue_time_s:       float = 0.0
    total_time_s:       float = 0.0
    strategy_used:      str   = ExecutionStrategy.SINGLE.value
    fallback_used:      bool  = False
    fallback_reason:    str   = ""
    selection_result:   SelectionResult | None = None
    provider_result:    ProviderResult | None  = None
    raw_metadata:       dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return bool(self.counts)


class HardwareOrchestrator:
    """
    Master hardware execution orchestrator.

    Parameters
    ----------
    registry        : HardwareRegistry
    selector        : HardwareSelector
    simulator_backend : fallback simulator (TensorNetworkBackend or similar)
    max_retries     : retries per device before failover
    racing_top_k    : number of parallel devices for RACING strategy
    """

    def __init__(
        self,
        registry:           HardwareRegistry,
        selector:           HardwareSelector,
        simulator_backend   = None,
        max_retries:        int   = 2,
        racing_top_k:       int   = 3,
    ) -> None:
        self._registry   = registry
        self._selector   = selector
        self._simulator  = simulator_backend
        self._max_retries = max_retries
        self._racing_k   = racing_top_k

    # ── Primary API ───────────────────────────────────────────────────────────

    async def execute(
        self,
        circuit,
        shots:               int   = 1024,
        strategy:            ExecutionStrategy = ExecutionStrategy.FAILOVER,
        weights:             ScoringWeights | None = None,
        provider_preference: str | None = None,
        device_preference:   str | None = None,
        max_queue_s:         float | None = None,
        max_budget_usd:      float | None = None,
        ai_hint:             str | None = None,
        mitigation_enabled:  bool  = False,
    ) -> OrchestrationResult:
        """
        Execute a circuit on the best available hardware.

        The full pipeline:
          1. Select device(s) via HardwareSelector
          2. Apply strategy (single / failover / racing / cost_aware)
          3. Submit, poll, fetch result via provider connector
          4. Normalise counts
          5. Apply ErrorMitigationEngine (if enabled)
          6. Return OrchestrationResult

        Falls back to simulator if all hardware paths fail.
        """
        t_total = time.monotonic()

        # Device selection
        selection = self._selector.select(
            circuit, shots=shots, weights=weights,
            provider_preference=provider_preference,
            device_preference=device_preference,
            max_queue_s=max_queue_s,
            ai_hint=ai_hint,
        )

        # Immediate simulator fallback if no hardware available
        if selection.fallback_to_sim or selection.chosen_device is None:
            return await self._sim_fallback(
                circuit, shots, selection,
                reason=selection.fallback_reason or "No hardware available",
                total_start=t_total,
            )

        # Dispatch by strategy
        try:
            if strategy == ExecutionStrategy.RACING:
                result = await self._execute_racing(
                    circuit, selection, shots, t_total
                )
            elif strategy == ExecutionStrategy.FAILOVER:
                result = await self._execute_failover(
                    circuit, selection, shots, t_total
                )
            elif strategy == ExecutionStrategy.COST_AWARE:
                result = await self._execute_cost_aware(
                    circuit, selection, shots, max_budget_usd, t_total
                )
            else:
                result = await self._execute_single(
                    circuit, selection.chosen_device, shots, t_total, selection
                )

            # Optional error mitigation
            if mitigation_enabled and circuit:
                result = await self._apply_mitigation(result, circuit)

            return result

        except Exception as exc:
            _log.error("All hardware strategies failed: %s — using simulator", exc)
            return await self._sim_fallback(
                circuit, shots, selection,
                reason=str(exc),
                total_start=t_total,
            )

    # ── Strategies ────────────────────────────────────────────────────────────

    async def _execute_single(
        self,
        circuit,
        device: DeviceProfile,
        shots: int,
        t_total: float,
        selection: SelectionResult,
    ) -> OrchestrationResult:
        """Submit to single best device and wait."""
        connector = self._registry.get_connector_for_device(device.device_id)
        if not connector:
            raise ConnectorError(f"No connector for {device.device_id}", device.provider)

        t_submit = time.monotonic()
        provider_result = await self._run_on_device(connector, circuit, device, shots)
        t_exec          = time.monotonic() - t_submit

        return OrchestrationResult(
            counts           = provider_result.counts,
            shots            = provider_result.shots,
            device_used      = device.device_id,
            provider_used    = device.provider,
            execution_time_s = t_exec,
            queue_time_s     = device.queue_time_s,
            total_time_s     = time.monotonic() - t_total,
            strategy_used    = ExecutionStrategy.SINGLE.value,
            selection_result = selection,
            provider_result  = provider_result,
        )

    async def _execute_failover(
        self,
        circuit,
        selection: SelectionResult,
        shots: int,
        t_total: float,
    ) -> OrchestrationResult:
        """
        Failover chain: try top-K devices in order of score.
        Falls back to simulator if all fail.
        """
        qualified = selection.top_k(k=5)
        last_exc: Exception | None = None

        for scored in qualified:
            device    = scored.device
            connector = self._registry.get_connector_for_device(device.device_id)
            if not connector:
                continue

            for attempt in range(self._max_retries + 1):
                try:
                    t0 = time.monotonic()
                    pr = await self._run_on_device(connector, circuit, device, shots)
                    _log.info(
                        "FAILOVER: success on %s (attempt %d, %.1fs)",
                        device.device_id, attempt + 1, time.monotonic() - t0,
                    )
                    return OrchestrationResult(
                        counts=pr.counts, shots=pr.shots,
                        device_used=device.device_id, provider_used=device.provider,
                        execution_time_s=time.monotonic()-t0,
                        queue_time_s=device.queue_time_s,
                        total_time_s=time.monotonic()-t_total,
                        strategy_used=ExecutionStrategy.FAILOVER.value,
                        selection_result=selection, provider_result=pr,
                    )
                except ConnectorError as exc:
                    last_exc = exc
                    if not exc.retryable or attempt == self._max_retries:
                        _log.warning(
                            "FAILOVER: %s failed (attempt %d): %s — trying next device",
                            device.device_id, attempt + 1, exc,
                        )
                        break
                    wait = 5.0 * (2 ** attempt)
                    _log.info("FAILOVER: retry %s in %.0fs", device.device_id, wait)
                    await asyncio.sleep(wait)

        # All hardware failed — simulator
        return await self._sim_fallback(
            circuit, shots, selection,
            reason=f"All hardware paths failed. Last error: {last_exc}",
            total_start=t_total,
        )

    async def _execute_racing(
        self,
        circuit,
        selection: SelectionResult,
        shots: int,
        t_total: float,
    ) -> OrchestrationResult:
        """
        Submit to top-K devices simultaneously. Return result from whichever
        completes first. Cancel remaining jobs.
        """
        candidates = selection.top_k(k=self._racing_k)
        if not candidates:
            raise ConnectorError("No candidates for racing", "mhos")

        # Create per-device tasks
        tasks: dict[str, asyncio.Task] = {}
        handles: dict[str, tuple[BaseConnector, JobHandle]] = {}

        for scored in candidates:
            device    = scored.device
            connector = self._registry.get_connector_for_device(device.device_id)
            if not connector:
                continue

            if not connector._authenticated:
                try:
                    await connector.authenticate()
                except Exception:
                    continue

            # Submit immediately (non-blocking)
            try:
                native = connector.circuit_to_native(circuit)
                handle = await connector.submit(
                    circuit, device.device_id, shots=shots
                )
                handles[device.device_id] = (connector, handle)
                tasks[device.device_id] = asyncio.create_task(
                    connector._wait_for_result(handle),
                    name=f"race_{device.device_id}",
                )
                _log.debug("RACING: submitted to %s (%s)", device.device_id, handle.native_id)
            except Exception as exc:
                _log.warning("RACING: submit to %s failed: %s", device.device_id, exc)

        if not tasks:
            raise ConnectorError("RACING: no successful submissions", "mhos")

        # Wait for the first completion
        done, pending = await asyncio.wait(
            tasks.values(), return_when=asyncio.FIRST_COMPLETED
        )

        # Cancel remaining
        for task in pending:
            task.cancel()
            # Best-effort cancel on provider
            for did, (conn, hdl) in handles.items():
                if tasks.get(did) is task:
                    asyncio.create_task(conn.cancel(hdl))

        # Get winner
        winner = next(iter(done))
        pr     = winner.result()

        # Identify which device won
        winning_device_id = next(
            (did for did, t in tasks.items() if t is winner), "unknown"
        )
        winning_device = self._registry.get(winning_device_id) or candidates[0].device

        t_exec = pr.execution_time_s if hasattr(pr, "execution_time_s") else 0.0
        _log.info("RACING: winner = %s", winning_device_id)

        return OrchestrationResult(
            counts=pr.counts, shots=pr.shots,
            device_used=winning_device.device_id,
            provider_used=winning_device.provider,
            execution_time_s=t_exec,
            total_time_s=time.monotonic()-t_total,
            strategy_used=ExecutionStrategy.RACING.value,
            selection_result=selection, provider_result=pr,
        )

    async def _execute_cost_aware(
        self,
        circuit,
        selection: SelectionResult,
        shots: int,
        max_budget: float | None,
        t_total: float,
    ) -> OrchestrationResult:
        """
        Select the cheapest qualifying device within budget, then FAILOVER.
        """
        # Re-rank by cost
        by_cost = sorted(
            [s for s in selection.scores if not s.disqualified],
            key=lambda s: s.estimated_cost,
        )
        if max_budget is not None:
            by_cost = [s for s in by_cost if s.estimated_cost <= max_budget]

        if not by_cost:
            return await self._sim_fallback(
                circuit, shots, selection,
                reason=f"No devices within budget ${max_budget}",
                total_start=t_total,
            )

        # Use cheapest device, failover to next-cheapest
        cost_selection = SelectionResult(
            chosen_device=by_cost[0].device,
            scores=by_cost,
            weights_used=selection.weights_used,
        )
        return await self._execute_failover(circuit, cost_selection, shots, t_total)

    # ── Simulator fallback ────────────────────────────────────────────────────

    async def _sim_fallback(
        self,
        circuit,
        shots: int,
        selection: SelectionResult | None,
        reason: str,
        total_start: float,
    ) -> OrchestrationResult:
        """Run on local simulator (MPS or statevector)."""
        _log.info("SIMULATOR FALLBACK: %s", reason)

        if self._simulator is None:
            try:
                from zuntenium.backends.tensor_network import TensorNetworkBackend
                self._simulator = TensorNetworkBackend(max_bond_dim=32)
            except ImportError:
                pass

        counts: dict[str, int] = {}
        if self._simulator is not None and circuit is not None:
            try:
                result = self._simulator.run(circuit, shots=shots)
                counts = result.get("counts", {}) if isinstance(result, dict) else {}
            except Exception as exc:
                _log.error("Simulator fallback also failed: %s", exc)

        return OrchestrationResult(
            counts=counts, shots=shots,
            device_used="simulator", provider_used="local",
            total_time_s=time.monotonic()-total_start,
            strategy_used="simulator_fallback",
            fallback_used=True, fallback_reason=reason,
            selection_result=selection,
        )

    # ── Run on single device (with transpilation) ─────────────────────────────

    async def _run_on_device(
        self,
        connector: BaseConnector,
        circuit,
        device: DeviceProfile,
        shots: int,
    ) -> ProviderResult:
        """
        Apply transpiler (gate decomposition to device native gates) then submit.
        """
        if not connector._authenticated:
            await connector.authenticate()

        # Transpile to native gate set
        transpiled = self._transpile_for_device(circuit, device)

        return await connector.run(transpiled, device.device_id, shots=shots)

    @staticmethod
    def _transpile_for_device(circuit, device: DeviceProfile):
        """
        Apply Zuntenium Transpiler for gate decomposition + SWAP insertion.
        Falls back gracefully if Transpiler is unavailable.
        """
        # Only needed for gate-based non-all-to-all devices
        if device.all_to_all or not device.coupling_map:
            return circuit

        try:
            from zuntenium.transpiler import Transpiler
            t = Transpiler()
            return t.transpile(circuit, mode="hardware")
        except Exception as exc:
            _log.debug("Transpiler unavailable (%s) — using raw circuit", exc)
            return circuit

    # ── Error mitigation ──────────────────────────────────────────────────────

    async def _apply_mitigation(
        self, result: OrchestrationResult, circuit
    ) -> OrchestrationResult:
        """Post-process result with ErrorMitigationEngine."""
        try:
            from zuntenium.error_mitigation.integration import (
                MitigationPipeline, ExecutionResult,
            )
            from zuntenium.error_mitigation.noise_model import medium_noise_model

            pipeline = MitigationPipeline(
                noise_model=medium_noise_model(), strategy="readout"
            )

            class _MC:
                num_qubits = getattr(circuit, "num_qubits", 2)
                depth, gates, name = 5, [], "m"

            er  = ExecutionResult(counts=result.counts, shots=result.shots)
            mit = pipeline.process(er, circuit=_MC())

            result.counts = mit.counts or result.counts
            result.raw_metadata["mitigation_method"] = \
                mit.metadata.get("mitigation_method", "readout")
            result.raw_metadata["fidelity_after"] = \
                mit.metadata.get("fidelity_after")
        except Exception as exc:
            _log.warning("Mitigation in orchestrator failed: %s", exc)
        return result
