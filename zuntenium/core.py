"""
zuntenium.core — Core quantum circuit and state abstractions.

Provides top-level aliases for Circuit, QuantumCircuit, and core quantum primitives.
"""
from __future__ import annotations

from zuntenium.circuit.circuit import QuantumCircuit

# Standard alias: Circuit -> QuantumCircuit
Circuit = QuantumCircuit

__all__ = ["Circuit", "QuantumCircuit"]
