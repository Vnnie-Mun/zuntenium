"""
quantum_api — Top-level user-facing API functions.

This is what users import from the top-level ``zuntenium`` package.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from zuntenium.backend.base_backend import AbstractBackend
    from zuntenium.circuit.circuit import QuantumCircuit
    from zuntenium.circuit.ir.quantum_ir import QuantumIR
    from zuntenium.quantum_core.executor.result import ExecutionResult


def get_backend(name: str = "statevector_simulator", **kwargs: Any) -> AbstractBackend:
    """
    Get a backend instance by name.

    Args:
        name: Backend name string. Common values:
              "statevector_simulator" (default)
              "unitary"
              "snapshot"
              "mock"
        **kwargs: Passed to the backend constructor.

    Returns:
        An instantiated AbstractBackend.

    Example::

        backend = get_backend("statevector_simulator", seed=42)
        result = execute(qc, backend=backend, shots=1024)
    """
    from zuntenium.backend.provider import BackendProvider
    return BackendProvider.get_backend(name, **kwargs)


def execute(
    circuit: QuantumCircuit | QuantumIR,
    backend: AbstractBackend | None = None,
    shots: int = 1024,
    **options: Any,
) -> ExecutionResult:
    """
    Execute a quantum circuit and return results.

    The primary execution entry point for ZUNTENIUM users.

    Args:
        circuit: The QuantumCircuit or QuantumIR to execute.
        backend: Backend to use. Default: statevector_simulator.
        shots: Number of measurement shots.
        **options: Backend-specific options (optimization_level, seed, etc.).

    Returns:
        ExecutionResult with counts, statevector, and metadata.

    Example::

        from zuntenium import QuantumCircuit, execute, get_backend

        qc = QuantumCircuit(2)
        qc.h(0).cx(0, 1).measure_all()

        result = execute(qc, shots=1024)
        print(result.counts)          # {'00': ~512, '11': ~512}
        result.plot_histogram()
    """
    if backend is None:
        backend = get_backend("statevector_simulator")

    return backend.run(circuit, shots=shots, **options)


def available_backends() -> list[str]:
    """Return a list of all registered backend names."""
    from zuntenium.backend.provider import BackendProvider
    return BackendProvider.available_backends()
