# zuntenium/mhos/integration.py
"""
MHOS Integration — Glue code connecting MHOS to the Cloud Platform.

Patches the cloud worker's _run_circuit_on_backend() to route hardware
jobs through the full HardwareOrchestrator pipeline.

Usage (in worker.py or cloud platform setup):
    from zuntenium.mhos.integration import MHOSIntegration, patch_cloud_worker
    
    mhos = MHOSIntegration.from_env()
    patch_cloud_worker(mhos)

After patching, all "ibm"/"ionq"/"braket"/"rigetti" backend requests
automatically go through MHOS multi-hardware orchestration.
Local simulator jobs are unchanged.
"""

from __future__ import annotations

import asyncio
import atexit
import logging
import os
from typing import Any

_log = logging.getLogger(__name__)

# Hardware provider backend names that MHOS handles
_HARDWARE_PROVIDERS = frozenset({"ibm", "ionq", "braket", "rigetti"})


class MHOSIntegration:
    """
    Single integration point for MHOS within the Zuntenium Cloud Platform.

    Initialise once at worker startup; re-use across jobs.
    Thread-safe: uses asyncio internally.

    Parameters
    ----------
    credentials_map : {provider: {key: value}} — injected from K8s Secrets
    strategy        : ExecutionStrategy for all hardware jobs
    scoring_weights : ScoringWeights preset name or dict
    racing_k        : number of parallel devices for RACING strategy
    """

    def __init__(
        self,
        credentials_map: dict[str, dict[str, str]] | None = None,
        strategy:         str = "failover",
        scoring_preset:   str = "default",
        racing_k:         int = 3,
    ) -> None:
        self._credentials  = credentials_map or {}
        self._strategy_str = strategy
        self._scoring_preset = scoring_preset
        self._racing_k     = racing_k
        self._orchestrator = None
        self._loop         = None
        self._initialised  = False

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_env(cls) -> "MHOSIntegration":
        """
        Build integration from environment variables (Kubernetes Secrets).

        Environment variables:
          ZUNTENIUM_IBM_TOKEN           — IBM Quantum API token
          ZUNTENIUM_IBM_INSTANCE        — IBM Quantum instance (hub/group/project)
          ZUNTENIUM_AWS_ACCESS_KEY_ID   — AWS access key
          ZUNTENIUM_AWS_SECRET          — AWS secret
          ZUNTENIUM_AWS_S3_BUCKET       — S3 bucket for Braket results
          ZUNTENIUM_IONQ_API_KEY        — IonQ API key
          ZUNTENIUM_RIGETTI_USER_ID     — Rigetti QCS user ID
          ZUNTENIUM_MHOS_STRATEGY       — execution strategy (default: failover)
          ZUNTENIUM_MHOS_SCORING        — scoring preset (default: default)
        """
        creds: dict[str, dict[str, str]] = {}

        ibm_token = os.environ.get("ZUNTENIUM_IBM_TOKEN", "")
        if ibm_token:
            creds["ibm"] = {
                "token":    ibm_token,
                "instance": os.environ.get("ZUNTENIUM_IBM_INSTANCE", "ibm-q/open/main"),
            }

        aws_key = os.environ.get("ZUNTENIUM_AWS_ACCESS_KEY_ID", "")
        if aws_key:
            creds["braket"] = {
                "aws_access_key_id":     aws_key,
                "aws_secret_access_key": os.environ.get("ZUNTENIUM_AWS_SECRET", ""),
                "s3_bucket":             os.environ.get("ZUNTENIUM_AWS_S3_BUCKET", "zuntenium-results"),
            }

        ionq_key = os.environ.get("ZUNTENIUM_IONQ_API_KEY", "")
        if ionq_key:
            creds["ionq"] = {"api_key": ionq_key}

        rigetti_uid = os.environ.get("ZUNTENIUM_RIGETTI_USER_ID", "")
        if rigetti_uid:
            creds["rigetti"] = {"user_id": rigetti_uid}

        return cls(
            credentials_map=creds,
            strategy=os.environ.get("ZUNTENIUM_MHOS_STRATEGY", "failover"),
            scoring_preset=os.environ.get("ZUNTENIUM_MHOS_SCORING", "default"),
        )

    # ── Initialisation ────────────────────────────────────────────────────────

    async def initialise(self) -> None:
        """Build registry, connectors, and discover devices. Async."""
        if self._initialised:
            return

        from .connectors.ibm     import IBMConnector
        from .connectors.braket  import BraketConnector
        from .connectors.ionq    import IonQConnector
        from .connectors.rigetti import RigettiConnector
        from .hardware_registry  import HardwareRegistry
        from .hardware_selector  import HardwareSelector, ScoringWeights

        registry = HardwareRegistry(refresh_interval_s=300.0)

        # Register connectors for each provider with credentials
        connector_map = {
            "ibm":     IBMConnector,
            "braket":  BraketConnector,
            "ionq":    IonQConnector,
            "rigetti": RigettiConnector,
        }
        for provider, ConnClass in connector_map.items():
            creds = self._credentials.get(provider, {})
            conn  = ConnClass(credentials=creds)
            registry.register_connector(provider, conn)

        # Always register stub connectors for providers without creds (stub mode)
        for provider, ConnClass in connector_map.items():
            if provider not in self._credentials:
                conn = ConnClass(credentials={})
                registry.register_connector(provider, conn)

        # Discover all devices
        n = await registry.discover_devices()
        _log.info("MHOS: discovered %d devices", n)

        # Start background refresh
        await registry.start_refresh_loop()

        # Build selector
        scoring_fn = {
            "default":        ScoringWeights.default,
            "fidelity_first": ScoringWeights.fidelity_first,
            "cost_optimised": ScoringWeights.cost_optimised,
            "low_latency":    ScoringWeights.low_latency,
        }.get(self._scoring_preset, ScoringWeights.default)

        selector = HardwareSelector(registry)

        # Build orchestrator with MPS simulator fallback
        sim = None
        try:
            from zuntenium.backends.tensor_network import TensorNetworkBackend
            sim = TensorNetworkBackend(max_bond_dim=32)
        except ImportError:
            pass

        from .orchestrator import HardwareOrchestrator
        self._orchestrator = HardwareOrchestrator(
            registry=registry, selector=selector,
            simulator_backend=sim, racing_top_k=self._racing_k,
        )
        self._initialised = True
        _log.info("MHOS: fully initialised (strategy=%s)", self._strategy_str)

    # ── Job execution ─────────────────────────────────────────────────────────

    def run_circuit(
        self,
        circuit,
        shots: int = 1024,
        backend_hint: str = "auto",
        mitigation_enabled: bool = False,
        **kwargs,
    ) -> dict:
        """
        Synchronous wrapper for worker processes (RQ workers are sync).
        Runs the async orchestration in a new event loop.
        """
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
        return self._loop.run_until_complete(
            self._run_async(circuit, shots, backend_hint, mitigation_enabled, **kwargs)
        )

    def close(self) -> None:
        """Stop registry background work and release the worker event loop."""
        if self._loop is None or self._loop.is_closed():
            return
        if self._orchestrator is not None:
            registry = self._orchestrator._registry
            self._loop.run_until_complete(registry.stop_refresh_loop())
        self._loop.close()
        self._loop = None
        self._orchestrator = None
        self._initialised = False

    async def _run_async(
        self,
        circuit,
        shots: int,
        backend_hint: str,
        mitigation_enabled: bool,
        **kwargs,
    ) -> dict:
        await self.initialise()

        from .orchestrator import ExecutionStrategy

        strategy_map = {
            "single":     ExecutionStrategy.SINGLE,
            "failover":   ExecutionStrategy.FAILOVER,
            "racing":     ExecutionStrategy.RACING,
            "cost_aware": ExecutionStrategy.COST_AWARE,
        }
        strategy = strategy_map.get(self._strategy_str, ExecutionStrategy.FAILOVER)

        # Provider preference from backend_hint
        provider_pref = backend_hint if backend_hint in _HARDWARE_PROVIDERS else None

        result = await self._orchestrator.execute(
            circuit,
            shots=shots,
            strategy=strategy,
            provider_preference=provider_pref,
            ai_hint=backend_hint,
            mitigation_enabled=mitigation_enabled,
            **kwargs,
        )

        return {
            "counts":           result.counts,
            "shots":            result.shots,
            "backend_used":     result.device_used,
            "provider_used":    result.provider_used,
            "execution_time_s": result.execution_time_s,
            "queue_time_s":     result.queue_time_s,
            "total_time_s":     result.total_time_s,
            "fallback_used":    result.fallback_used,
            "fallback_reason":  result.fallback_reason,
            "strategy_used":    result.strategy_used,
            "metadata":         result.raw_metadata,
        }

    def registry_stats(self) -> dict:
        if not self._initialised or not self._orchestrator:
            return {"initialised": False}
        return {
            **self._orchestrator._registry.stats(),
            "strategy": self._strategy_str,
            "initialised": True,
        }


# ── Cloud worker patch ────────────────────────────────────────────────────────

_GLOBAL_MHOS: MHOSIntegration | None = None


def patch_cloud_worker(mhos: MHOSIntegration) -> None:
    """
    Monkey-patch the Cloud Platform worker to route hardware jobs
    through MHOS. Call once at worker startup.

    After patching:
      _run_circuit_on_backend(circuit, "ibm", ...) → MHOS.run_circuit()
      _run_circuit_on_backend(circuit, "statevector", ...) → unchanged
    """
    global _GLOBAL_MHOS
    _GLOBAL_MHOS = mhos
    atexit.register(mhos.close)

    import zuntenium_cloud.worker as worker_module

    _original_run = worker_module._run_circuit_on_backend

    def _patched_run(circuit, backend_name: str, backends: dict, kwargs: dict) -> dict:
        if backend_name in _HARDWARE_PROVIDERS and _GLOBAL_MHOS is not None:
            _log.info("MHOS routing: %s → hardware orchestrator", backend_name)
            return _GLOBAL_MHOS.run_circuit(
                circuit,
                shots=kwargs.get("shots", 1024),
                backend_hint=backend_name,
                mitigation_enabled=kwargs.get("mitigation_enabled", False),
            )
        # Simulator paths unchanged
        return _original_run(circuit, backend_name, backends, kwargs)

    worker_module._run_circuit_on_backend = _patched_run
    _log.info("Cloud worker patched with MHOS hardware routing")
