"""
ResultCache — LRU cache for repeated circuit execution.

Caches ExecutionResult objects keyed by (circuit_fingerprint, shots).
Avoids re-executing identical circuits — especially useful in VQE where
the same ansatz structure is evaluated with thousands of parameter sets.
"""
from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from typing import Any

from zuntenium.quantum_core.executor.result import ExecutionResult


def _circuit_fingerprint(circuit: Any, shots: int) -> str:
    """Compute a stable hash key for a circuit + shots combination."""
    try:
        from zuntenium.circuit.circuit import QuantumCircuit
        from zuntenium.circuit.ir.quantum_ir import QuantumIR
        from zuntenium.circuit.ir.serialization.json_serializer import ir_to_dict

        if isinstance(circuit, QuantumCircuit):
            ir = circuit.to_ir()
        elif isinstance(circuit, QuantumIR):
            ir = circuit
        else:
            return ""

        d = ir_to_dict(ir)
        d["shots"] = shots
        canonical = json.dumps(d, sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]
    except Exception:
        return ""


class ResultCache:
    """
    Thread-safe LRU cache for ExecutionResult objects.

    Args:
        max_size: Maximum number of results to cache (default 256).
    """

    def __init__(self, max_size: int = 256) -> None:
        self._max_size = max_size
        self._cache: OrderedDict[str, ExecutionResult] = OrderedDict()

    def get(self, circuit: Any, shots: int) -> ExecutionResult | None:
        key = _circuit_fingerprint(circuit, shots)
        if not key or key not in self._cache:
            return None
        # Move to end (most recently used)
        self._cache.move_to_end(key)
        return self._cache[key]

    def put(self, circuit: Any, shots: int, result: ExecutionResult) -> None:
        key = _circuit_fingerprint(circuit, shots)
        if not key:
            return
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = result
        if len(self._cache) > self._max_size:
            self._cache.popitem(last=False)  # evict LRU

    def clear(self) -> None:
        self._cache.clear()

    @property
    def size(self) -> int:
        return len(self._cache)

    @property
    def hit_rate(self) -> float:
        return 0.0   # v0.1: simple cache, no hit tracking yet
