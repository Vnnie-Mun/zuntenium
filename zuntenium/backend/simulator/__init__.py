"""
backend/simulator sub-package.

Exports all built-in simulator backends for easy access.

Usage::

    from zuntenium.backend.simulator import StatevectorBackend, UnitaryBackend
    backend = StatevectorBackend(seed=42)
    result = backend.run(qc, shots=1024)
"""
from __future__ import annotations

from zuntenium.backend.simulator.snapshot_backend import SnapshotBackend
from zuntenium.backend.simulator.statevector_backend import StatevectorBackend
from zuntenium.backend.simulator.unitary_backend import UnitaryBackend

__all__ = [
    "StatevectorBackend",
    "UnitaryBackend",
    "SnapshotBackend",
]
