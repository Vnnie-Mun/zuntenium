# zuntenium/mhos/hardware_registry.py
"""
HardwareRegistry — Central store of all registered quantum devices.

Maintains DeviceProfile objects for every known device across all providers.
Background refresh task polls providers every N seconds for live queue depths,
calibration updates, and availability status.

Thread/async safe: all mutations go through asyncio.Lock().
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from .device_profile import DeviceProfile, DeviceStatus, CouplingEdge, QubitProperties
from .connectors.base import BaseConnector

_log = logging.getLogger(__name__)


class HardwareRegistry:
    """
    Central registry of all quantum hardware devices.

    Parameters
    ----------
    refresh_interval_s : how often to refresh device metrics (default 300s)
    """

    def __init__(self, refresh_interval_s: float = 300.0) -> None:
        self._devices:    dict[str, DeviceProfile] = {}
        self._connectors: dict[str, BaseConnector] = {}   # provider → connector
        self._refresh_interval = refresh_interval_s
        self._lock    = asyncio.Lock()
        self._refresh_task: asyncio.Task | None = None

    # ── Registration ──────────────────────────────────────────────────────────

    def register_connector(self, provider: str, connector: BaseConnector) -> None:
        """
        Register a provider connector. Multiple connectors per provider allowed
        (e.g. IBM open + IBM premium). Provider name must match connector.PROVIDER_NAME.
        """
        self._connectors[provider] = connector
        _log.info("Registered connector for provider '%s'", provider)

    async def discover_devices(self) -> int:
        """
        Query all registered connectors and populate the device registry.
        Returns number of devices discovered.
        """
        total = 0
        for provider, connector in self._connectors.items():
            try:
                if not connector._authenticated:
                    await connector.authenticate()
                devices = await connector.list_devices()
                for raw in devices:
                    profile = self._raw_to_profile(raw, provider)
                    async with self._lock:
                        self._devices[profile.device_id] = profile
                    total += 1
                _log.info("[Registry] %s: discovered %d devices", provider, len(devices))
            except Exception as exc:
                _log.warning("[Registry] Failed to discover %s devices: %s", provider, exc)
        return total

    # ── Query ─────────────────────────────────────────────────────────────────

    def get(self, device_id: str) -> DeviceProfile | None:
        return self._devices.get(device_id)

    def list_all(self) -> list[DeviceProfile]:
        return list(self._devices.values())

    def list_available(
        self,
        min_qubits: int = 0,
        provider: str | None = None,
    ) -> list[DeviceProfile]:
        """Return devices that are currently ONLINE or DEGRADED."""
        return [
            d for d in self._devices.values()
            if d.is_available
            and d.num_qubits >= min_qubits
            and (provider is None or d.provider == provider)
        ]

    def get_connector(self, provider: str) -> BaseConnector | None:
        return self._connectors.get(provider)

    def get_connector_for_device(self, device_id: str) -> BaseConnector | None:
        profile = self.get(device_id)
        if profile:
            return self._connectors.get(profile.provider)
        return None

    # ── Background refresh ────────────────────────────────────────────────────

    async def start_refresh_loop(self) -> None:
        """Start the background device-status refresh task."""
        if self._refresh_task and not self._refresh_task.done():
            return
        self._refresh_task = asyncio.create_task(self._refresh_loop())
        _log.info("[Registry] Background refresh started (interval=%ds)",
                  int(self._refresh_interval))

    async def stop_refresh_loop(self) -> None:
        if self._refresh_task:
            self._refresh_task.cancel()
            try:
                await self._refresh_task
            except asyncio.CancelledError:
                pass
        _log.info("[Registry] Background refresh stopped")

    async def _refresh_loop(self) -> None:
        while True:
            await asyncio.sleep(self._refresh_interval)
            await self._do_refresh()

    async def _do_refresh(self) -> None:
        """Refresh all devices that need updating."""
        stale = [
            d for d in self._devices.values()
            if d.needs_refresh
        ]
        if not stale:
            return

        _log.debug("[Registry] Refreshing %d devices", len(stale))
        tasks = [self._refresh_one(d) for d in stale]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for d, r in zip(stale, results):
            if isinstance(r, Exception):
                _log.warning("[Registry] Refresh failed for %s: %s", d.device_id, r)

    async def _refresh_one(self, device: DeviceProfile) -> None:
        connector = self._connectors.get(device.provider)
        if not connector:
            return
        try:
            data = await connector.get_device_status(device.device_id)
            async with self._lock:
                device.update_from_refresh(data)
        except Exception as exc:
            # Mark as UNKNOWN on refresh failure
            async with self._lock:
                device.status = DeviceStatus.UNKNOWN
            raise exc

    # ── Forced refresh ────────────────────────────────────────────────────────

    async def refresh_device(self, device_id: str) -> DeviceProfile | None:
        """Force-refresh a specific device and return updated profile."""
        device = self.get(device_id)
        if not device:
            return None
        await self._refresh_one(device)
        return self.get(device_id)

    # ── Stats ─────────────────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        all_d = self.list_all()
        by_provider: dict[str, int] = {}
        for d in all_d:
            by_provider[d.provider] = by_provider.get(d.provider, 0) + 1
        return {
            "total_devices":     len(all_d),
            "available_devices": len(self.list_available()),
            "by_provider":       by_provider,
            "providers":         list(self._connectors.keys()),
        }

    # ── Raw dict → DeviceProfile ──────────────────────────────────────────────

    @staticmethod
    def _raw_to_profile(raw: dict, provider: str) -> DeviceProfile:
        """Convert provider-specific device dict to a DeviceProfile."""
        coupling_map = []
        for edge in raw.get("coupling_map", []):
            if isinstance(edge, (list, tuple)) and len(edge) >= 2:
                coupling_map.append(CouplingEdge(
                    control=int(edge[0]), target=int(edge[1]),
                    cx_error_rate=float(edge[2]) if len(edge) > 2 else 0.0,
                ))

        m1q = float(raw.get("mean_1q_error", 0.001))
        m2q = float(raw.get("mean_2q_error", 0.01))

        profile = DeviceProfile(
            device_id          = str(raw.get("device_id", "")),
            provider           = provider,
            display_name       = str(raw.get("display_name", raw.get("device_id", ""))),
            num_qubits         = int(raw.get("num_qubits", 0)),
            coupling_map       = coupling_map,
            all_to_all         = bool(raw.get("all_to_all", False)),
            native_gates       = list(raw.get("native_gates", [])),
            mean_1q_error      = m1q,
            mean_2q_error      = m2q,
            mean_readout_err   = float(raw.get("mean_readout_err", 0.01)),
            fidelity_estimate  = max(0.0, 1.0 - m1q),
            mean_T1_ns         = float(raw.get("mean_T1_ns", 0.0)),
            mean_T2_ns         = float(raw.get("mean_T2_ns", 0.0)),
            cost_per_shot      = float(raw.get("cost_per_shot", 0.0)),
            queue_depth        = int(raw.get("queue_depth", 0)),
            queue_time_s       = float(raw.get("queue_time_s",
                                                raw.get("queue_depth", 0) * 60.0)),
            last_refreshed_epoch = time.time(),
            last_calibrated_epoch = time.time(),
        )

        # Determine status
        status_str = str(raw.get("status", "unknown")).lower()
        from .device_profile import DeviceStatus
        try:
            profile.status = DeviceStatus(status_str)
        except ValueError:
            profile.status = DeviceStatus.UNKNOWN if status_str == "unknown" else DeviceStatus.ONLINE

        return profile
