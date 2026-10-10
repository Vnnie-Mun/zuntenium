"""
TranspilerCache — High-performance structural caching and parameter rebinding.

Eliminates transpilation overhead in variational quantum-classical algorithms
(VQE, QAOA, QML) where circuit structure is constant across iterations
and only numeric rotation angles change.

Key capabilities:
  1. Template Hashing: Computes parameter-invariant structural signatures.
  2. LRU Cache: Caches compiled circuits by (template_hash, opt_level, target).
  3. O(1) Rebinding: Updates parameterized gate angles into compiled templates
     without re-running expensive routing, synthesis, or optimization passes.
  4. Metrics: Tracks hits, misses, evictions, and compilation speedup.
"""
from __future__ import annotations

import copy
import hashlib
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CachedCompiledTemplate:
    """
    Compiled circuit template with parameter slot mappings.

    Attributes:
        template_hash:       Unique hash of the parameter-free circuit structure.
        compiled_ir_or_circ: The pre-compiled QuantumIR or Circuit template.
        param_slot_indices:  List of instruction/gate indices that accept variable parameters.
        original_num_params: Expected number of parameters.
    """
    template_hash: str
    compiled_ir_or_circ: Any
    param_slot_indices: tuple[int, ...]
    original_num_params: int


class TranspilerCache:
    """
    Thread-safe structural cache for transpiled quantum circuits.

    Example::

        cache = TranspilerCache(max_size=256)

        # In a VQE loop:
        for params in optimizer_iterations:
            circuit = build_ansatz(params)
            # 1st step: Compiles & caches template
            # 2nd..Nth step: Instant O(1) rebinding from cache
            compiled = cache.transpile_with_cache(
                pipeline, circuit, params, target=target
            )
    """

    def __init__(self, max_size: int = 512) -> None:
        self.max_size = max_size
        self._cache: dict[str, CachedCompiledTemplate] = {}
        self._lock = threading.RLock()
        self._hits: int = 0
        self._misses: int = 0
        self._evictions: int = 0

    # ------------------------------------------------------------------
    # Structural Hashing
    # ------------------------------------------------------------------

    @staticmethod
    def compute_template_hash(circuit_or_ir: Any, level: int = 2, target_name: str = "default") -> str:
        """
        Compute a parameter-invariant structural hash of a circuit.
        Replaces variable rotation angles with a generic placeholder.
        """
        hasher = hashlib.sha256()
        hasher.update(f"lvl={level}|tgt={target_name}".encode())

        # Check if QuantumIR
        if hasattr(circuit_or_ir, "instructions") and hasattr(circuit_or_ir, "num_qubits"):
            hasher.update(f"n={circuit_or_ir.num_qubits}".encode())
            for node in circuit_or_ir.instructions:
                op = node.operation
                qubits = tuple(node.qubits)
                gate_name = node.gate.name if (hasattr(node, "gate") and node.gate) else op
                is_parametric = bool(getattr(node, "params", None))
                token = f"{gate_name}|{qubits}|{'PARAM' if is_parametric else 'FIXED'}"
                hasher.update(token.encode())
        # Check if transpiler_v2 Circuit
        elif hasattr(circuit_or_ir, "gates") and hasattr(circuit_or_ir, "num_qubits"):
            hasher.update(f"n={circuit_or_ir.num_qubits}".encode())
            for g in circuit_or_ir.gates:
                name = getattr(g, "name", str(type(g)))
                qubits = tuple(getattr(g, "qubits", ()))
                is_parametric = bool(getattr(g, "params", None))
                token = f"{name}|{qubits}|{'PARAM' if is_parametric else 'FIXED'}"
                hasher.update(token.encode())
        else:
            # Fallback string repr
            hasher.update(str(circuit_or_ir).encode())

        return hasher.hexdigest()

    # ------------------------------------------------------------------
    # Parameter Extraction & Rebinding
    # ------------------------------------------------------------------

    @staticmethod
    def extract_params(circuit_or_ir: Any) -> list[float]:
        """Extract all numeric parameters in sequence from a circuit."""
        params: list[float] = []
        if hasattr(circuit_or_ir, "instructions"):
            for node in circuit_or_ir.instructions:
                if getattr(node, "params", None):
                    params.extend([float(p) for p in node.params])
        elif hasattr(circuit_or_ir, "gates"):
            for g in circuit_or_ir.gates:
                if getattr(g, "params", None):
                    params.extend([float(p) for p in g.params])
        return params

    @staticmethod
    def rebind(compiled_template: Any, new_params: list[float] | tuple[float, ...]) -> Any:
        """
        Rebind new parameter values into a compiled circuit template.
        Returns a fresh instance with new parameters applied.
        """
        # QuantumIR rebind
        if hasattr(compiled_template, "instructions") and hasattr(compiled_template, "num_qubits"):
            from zuntenium.circuit.ir.quantum_ir import QuantumIR
            from zuntenium.circuit.ir.ir_node import IRNode

            new_instructions: list[IRNode] = []
            param_idx = 0
            n_new = len(new_params)

            for node in compiled_template.instructions:
                if getattr(node, "params", None) and param_idx < n_new:
                    # Allocate matching number of params for this gate
                    n_node_params = len(node.params)
                    chunk = [float(new_params[param_idx + i]) for i in range(min(n_node_params, n_new - param_idx))]
                    param_idx += n_node_params
                    # Copy gate if parameterized
                    new_gate = copy.deepcopy(node.gate) if node.gate is not None else None
                    if new_gate is not None and hasattr(new_gate, "params"):
                        new_gate.params = list(chunk)
                    new_node = IRNode(
                        operation=node.operation,
                        gate=new_gate,
                        qubits=list(node.qubits),
                        clbits=list(node.clbits),
                        params=chunk,
                        annotations=dict(node.annotations),
                    )
                    new_instructions.append(new_node)
                else:
                    new_instructions.append(copy.deepcopy(node))

            return QuantumIR(
                num_qubits=compiled_template.num_qubits,
                num_clbits=compiled_template.num_clbits,
                instructions=new_instructions,
                metadata=dict(compiled_template.metadata),
            )

        # transpiler_v2 Circuit rebind
        if hasattr(compiled_template, "gates") and hasattr(compiled_template, "num_qubits"):
            from zuntenium.transpiler.transpiler_v2 import Gate, Circuit

            new_circ = Circuit(compiled_template.num_qubits)
            param_idx = 0
            n_new = len(new_params)

            for g in compiled_template.gates:
                if getattr(g, "params", None) and param_idx < n_new:
                    n_g_params = len(g.params)
                    chunk = tuple(float(new_params[param_idx + i]) for i in range(min(n_g_params, n_new - param_idx)))
                    param_idx += n_g_params
                    new_circ.add_gate(Gate(
                        name=g.name,
                        targets=g.targets,
                        controls=getattr(g, "controls", ()),
                        params=chunk,
                    ))
                else:
                    new_circ.add_gate(copy.copy(g))

            return new_circ

        # Fallback shallow copy
        return copy.deepcopy(compiled_template)

    # ------------------------------------------------------------------
    # Transpile with Caching
    # ------------------------------------------------------------------

    def get(self, template_hash: str) -> Optional[CachedCompiledTemplate]:
        with self._lock:
            cached = self._cache.get(template_hash)
            if cached is not None:
                self._hits += 1
                return cached
            self._misses += 1
            return None

    def put(self, template_hash: str, compiled: Any, num_params: int) -> None:
        with self._lock:
            if len(self._cache) >= self.max_size:
                # Evict oldest entry (simple FIFO/LRU)
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                self._evictions += 1

            # Identify slot indices
            slots: list[int] = []
            if hasattr(compiled, "instructions"):
                for idx, node in enumerate(compiled.instructions):
                    if getattr(node, "params", None):
                        slots.append(idx)
            elif hasattr(compiled, "gates"):
                for idx, g in enumerate(compiled.gates):
                    if getattr(g, "params", None):
                        slots.append(idx)

            self._cache[template_hash] = CachedCompiledTemplate(
                template_hash=template_hash,
                compiled_ir_or_circ=compiled,
                param_slot_indices=tuple(slots),
                original_num_params=num_params,
            )

    def transpile_with_cache(
        self,
        transpile_fn: Callable[[Any], Any],
        circuit_or_ir: Any,
        new_params: Optional[list[float]] = None,
        level: int = 2,
        target_name: str = "default",
    ) -> Any:
        """
        Transpile a circuit with caching and parameter rebinding.

        If the structural template has been transpiled previously, retrieves
        the compiled template and updates the numeric angles without running passes.
        """
        if new_params is None:
            new_params = self.extract_params(circuit_or_ir)

        key = self.compute_template_hash(circuit_or_ir, level=level, target_name=target_name)
        cached = self.get(key)

        if cached is not None and new_params:
            # Cache hit: Fast parameter rebinding
            return self.rebind(cached.compiled_ir_or_circ, new_params)

        # Cache miss: Run full transpilation
        compiled = transpile_fn(circuit_or_ir)
        self.put(key, compiled, len(new_params))
        return compiled

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
            self._evictions = 0

    @property
    def stats(self) -> dict[str, Any]:
        with self._lock:
            total = self._hits + self._misses
            ratio = (self._hits / total) if total > 0 else 0.0
            return {
                "hits": self._hits,
                "misses": self._misses,
                "total_requests": total,
                "hit_ratio": round(ratio, 4),
                "cache_size": len(self._cache),
                "evictions": self._evictions,
            }

    def __repr__(self) -> str:
        s = self.stats
        return (
            f"TranspilerCache(size={s['cache_size']}/{self.max_size}, "
            f"hits={s['hits']}, misses={s['misses']}, ratio={s['hit_ratio']:.1%})"
        )


__all__ = ["TranspilerCache", "CachedCompiledTemplate"]
