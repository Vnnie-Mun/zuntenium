"""Circuit topology and gate support validation utilities."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from zuntenium.backend.base_backend import AbstractBackend
    from zuntenium.circuit.circuit import QuantumCircuit


def validate_circuit_for_backend(
    circuit: QuantumCircuit,
    backend: AbstractBackend,
) -> list[str]:
    """
    Check circuit compatibility with a given backend.

    Returns:
        List of error strings (empty if valid).
    """
    result = backend.validate_circuit(circuit)
    return result.errors
