"""
zuntenium.simulators — Quantum execution simulators.

Provides high-level simulator wrappers for the quantum_api backend system.

.. note::
    The canonical ``StatevectorSimulator`` is at
    ``zuntenium.quantum_core.statevector.StatevectorSimulator``.
    This module provides ``StatevectorBackendRunner``, a lightweight
    wrapper around ``quantum_api.get_backend("statevector")`` for
    ergonomic circuit execution with shots.
"""
from __future__ import annotations

from typing import Any
from zuntenium.api.quantum_api import get_backend
from zuntenium.circuit.circuit import QuantumCircuit


class StatevectorBackendRunner:
    """High-level statevector simulator runner (via quantum_api backend)."""

    def __init__(self, **kwargs: Any) -> None:
        self.backend = get_backend("statevector")
        self.config = kwargs

    def run(self, circuit: QuantumCircuit, shots: int = 1024, **kwargs: Any) -> Any:
        opts = {**self.config, **kwargs}
        return self.backend.run(circuit, shots=shots, **opts)


# Backward-compatibility alias — deprecation planned for v0.2
StatevectorSimulator = StatevectorBackendRunner

from zuntenium.api.visualization_api import plot_histogram

__all__ = ["StatevectorBackendRunner", "StatevectorSimulator", "plot_histogram"]
