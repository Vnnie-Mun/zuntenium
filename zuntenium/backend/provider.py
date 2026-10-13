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
            "densitymatrix": "zuntenium_density_matrix_simulator",
            "density_matrix": "zuntenium_density_matrix_simulator",
            "density_matrix_simulator": "zuntenium_density_matrix_simulator",
            "stabilizer": "zuntenium_stabilizer_simulator",
            "stabilizer_simulator": "zuntenium_stabilizer_simulator",
            "clifford": "zuntenium_stabilizer_simulator",
            "tensornetwork": "zuntenium_tensor_network_simulator",
            "tensor_network": "zuntenium_tensor_network_simulator",
            "tensor_network_backend": "zuntenium_tensor_network_simulator",
            "mps": "zuntenium_tensor_network_simulator",
            "gpu": "zuntenium_gpu_statevector_simulator",
            "gpu_statevector": "zuntenium_gpu_statevector_simulator",
            "gpu_backend": "zuntenium_gpu_statevector_simulator",
        }
        resolved = aliases.get(name.lower().strip(), name)

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
        from zuntenium.backend.simulator.extended_backends import (
            DensityMatrixBackend,
            GPUStatevectorBackend,
            StabilizerBackend,
            TensorNetworkBackendAdapter,
        )
        from zuntenium.backend.simulator.snapshot_backend import SnapshotBackend
        from zuntenium.backend.simulator.statevector_backend import StatevectorBackend
        from zuntenium.backend.simulator.unitary_backend import UnitaryBackend

        backends: list[type[AbstractBackend]] = [
            StatevectorBackend,
            UnitaryBackend,
            SnapshotBackend,
            MockHardwareBackend,
            DensityMatrixBackend,
            StabilizerBackend,
            TensorNetworkBackendAdapter,
            GPUStatevectorBackend,
        ]
        for cls_ in backends:
            if cls_.name not in cls._registry:
                cls.register(cls_)


# Auto-register on module load
BackendProvider._auto_register_defaults()
