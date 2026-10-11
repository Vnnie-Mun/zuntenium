"""
zuntenium.simulators — Quantum execution simulators.

Provides StatevectorSimulator, MPS Simulator, and stabilizer execution wrappers.
"""
from __future__ import annotations

from typing import Any
from zuntenium.api.quantum_api import get_backend
from zuntenium.circuit.circuit import QuantumCircuit


class StatevectorSimulator:
    """Statevector quantum simulator backend."""

    def __init__(self, **kwargs: Any) -> None:
        self.backend = get_backend("statevector")
        self.config = kwargs

    def run(self, circuit: QuantumCircuit, shots: int = 1024, **kwargs: Any) -> Any:
        opts = {**self.config, **kwargs}
        return self.backend.run(circuit, shots=shots, **opts)

from zuntenium.api.visualization_api import plot_histogram

__all__ = ["StatevectorSimulator", "plot_histogram"]
