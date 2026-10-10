"""
BackendProvider — Registry and factory for all ZUNTENIUM backends.

Backends register themselves via BackendProvider.register().
Users discover backends via BackendProvider.available_backends()
and instantiate them via BackendProvider.get_backend(name).
"""
from __future__ import annotations

from typing import Any

from zuntenium.backend.base_backend import AbstractBackend
from zuntenium.quantum_core.exceptions import BackendNotFoundError


class BackendProvider:
    """
    Registry and factory for all ZUNTENIUM backends.

    Usage::

        backend = BackendProvider.get_backend("statevector_simulator")
        result = backend.run(qc, shots=1024)
    """

    _registry: dict[str, type[AbstractBackend]] = {}

    @classmethod
    def register(cls, backend_cls: type[AbstractBackend]) -> None:
        """Register a backend class under its .name attribute."""
        cls._registry[backend_cls.name] = backend_cls

    @classmethod
    def get_backend(cls, name: str, **kwargs: Any) -> AbstractBackend:
        """
        Instantiate a backend by name.

        Args:
            name: Backend name string (e.g. "statevector_simulator").
            **kwargs: Passed to the backend constructor.

        Returns:
            An instantiated AbstractBackend.

        Raises:
            BackendNotFoundError: If the name is not in the registry.
        """
        # Resolve common aliases
        aliases = {
            "statevector_simulator": "zuntenium_statevector_simulator",
            "statevector": "zuntenium_statevector_simulator",
            "unitary": "zuntenium_unitary_simulator",
            "snapshot": "zuntenium_snapshot_simulator",
            "mock": "mock_hardware",
        }
        resolved = aliases.get(name, name)

        if resolved not in cls._registry:
            raise BackendNotFoundError(name)
        return cls._registry[resolved](**kwargs)

    @classmethod
    def available_backends(cls) -> list[str]:
        """Return list of all registered backend names."""
        return sorted(cls._registry.keys())

    @classmethod
    def _auto_register_defaults(cls) -> None:
        """Register built-in backends. Called on first get_backend()."""
        from zuntenium.backend.hardware.mock.mock_hardware import MockHardwareBackend
        from zuntenium.backend.simulator.snapshot_backend import SnapshotBackend
        from zuntenium.backend.simulator.statevector_backend import StatevectorBackend
        from zuntenium.backend.simulator.unitary_backend import UnitaryBackend

        backends: list[type[AbstractBackend]] = [
            StatevectorBackend, UnitaryBackend, SnapshotBackend, MockHardwareBackend,
        ]
        for cls_ in backends:
            if cls_.name not in cls._registry:
                cls.register(cls_)


# Auto-register on module load
BackendProvider._auto_register_defaults()
