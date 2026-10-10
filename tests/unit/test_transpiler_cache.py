"""
Unit tests for TranspilerCache.

Verifies:
  - Structural template hashing (parameter-invariant)
  - Parameter extraction and O(1) rebinding
  - Cache hits vs misses and speedup
  - Capacity eviction and clear()
  - Thread-safety
"""

import threading
import pytest
from zuntenium.transpiler import TranspilerCache, CachedCompiledTemplate
from zuntenium.transpiler.transpiler_v2 import Circuit, Gate
from zuntenium.circuit.ir.quantum_ir import QuantumIR
from zuntenium.circuit.ir.ir_node import IRNode


def _make_circuit(num_qubits: int, gates: list[Gate]) -> Circuit:
    c = Circuit(num_qubits)
    for g in gates:
        c.add_gate(g)
    return c


class TestTranspilerCache:

    def test_cache_initialization(self):
        cache = TranspilerCache(max_size=10)
        assert cache.max_size == 10
        assert cache.stats["hits"] == 0
        assert cache.stats["misses"] == 0
        assert cache.stats["cache_size"] == 0
        assert "TranspilerCache" in repr(cache)

    def test_structural_hashing_parameter_invariance(self):
        # Two circuits with identical structure but different rotation angles
        c1 = _make_circuit(2, [
            Gate("rx", (0,), params=(0.25,)),
            Gate("cx", (0, 1)),
            Gate("rz", (1,), params=(1.57,)),
        ])
        c2 = _make_circuit(2, [
            Gate("rx", (0,), params=(3.14,)),
            Gate("cx", (0, 1)),
            Gate("rz", (1,), params=(-0.78,)),
        ])

        h1 = TranspilerCache.compute_template_hash(c1, level=2, target_name="h100")
        h2 = TranspilerCache.compute_template_hash(c2, level=2, target_name="h100")

        assert h1 == h2, "Structural hashes must be identical when only continuous parameters differ"

    def test_structural_hashing_sensitivity(self):
        # Circuits with different topology or gate types must produce different hashes
        c1 = _make_circuit(2, [
            Gate("rx", (0,), params=(0.25,)),
            Gate("cx", (0, 1)),
        ])
        c2 = _make_circuit(2, [
            Gate("ry", (0,), params=(0.25,)),
            Gate("cx", (0, 1)),
        ])
        c3 = _make_circuit(2, [
            Gate("rx", (0,), params=(0.25,)),
            Gate("cx", (1, 0)),  # Reversed direction
        ])

        h1 = TranspilerCache.compute_template_hash(c1)
        h2 = TranspilerCache.compute_template_hash(c2)
        h3 = TranspilerCache.compute_template_hash(c3)

        assert h1 != h2
        assert h1 != h3
        assert h2 != h3

    def test_extract_params(self):
        c = _make_circuit(3, [
            Gate("rx", (0,), params=(0.1,)),
            Gate("h", (1,)),
            Gate("rz", (2,), params=(0.5, 0.9)),
        ])
        params = TranspilerCache.extract_params(c)
        assert params == [0.1, 0.5, 0.9]

    def test_rebind_transpiler_v2_circuit(self):
        c = _make_circuit(2, [
            Gate("rx", (0,), params=(0.0,)),
            Gate("cx", (0, 1)),
            Gate("rz", (1,), params=(0.0,)),
        ])
        rebound = TranspilerCache.rebind(c, [1.23, 4.56])
        assert rebound.num_qubits == 2
        assert len(rebound.gates) == 3
        assert rebound.gates[0].name.upper() == "RX"
        assert rebound.gates[0].params == (1.23,)
        assert rebound.gates[1].name.upper() == "CX"
        assert rebound.gates[1].params == ()
        assert rebound.gates[2].name.upper() == "RZ"
        assert rebound.gates[2].params == (4.56,)

    def test_rebind_quantum_ir(self):
        node_rx = IRNode(operation="rx", qubits=[0], params=[0.0])
        node_cx = IRNode(operation="cx", qubits=[0, 1], params=[])
        node_rz = IRNode(operation="rz", qubits=[1], params=[0.0])
        ir = QuantumIR(num_qubits=2, num_clbits=0, instructions=[node_rx, node_cx, node_rz])

        rebound = TranspilerCache.rebind(ir, [0.77, 2.14])
        assert rebound.num_qubits == 2
        assert len(rebound.instructions) == 3
        assert rebound.instructions[0].params == [0.77]
        assert rebound.instructions[1].params == []
        assert rebound.instructions[2].params == [2.14]

    def test_transpile_with_cache_workflow(self):
        cache = TranspilerCache(max_size=16)
        call_count = 0

        def dummy_transpile(circ):
            nonlocal call_count
            call_count += 1
            # Return a compiled representation
            return copy_circ(circ)

        def copy_circ(circ):
            c = Circuit(circ.num_qubits)
            for g in circ.gates:
                c.add_gate(g)
            return c

        # Iteration 1: initial ansatz (cache miss)
        c1 = _make_circuit(2, [
            Gate("rx", (0,), params=(0.1,)),
            Gate("cx", (0, 1)),
        ])
        out1 = cache.transpile_with_cache(dummy_transpile, c1, [0.1])
        assert call_count == 1
        assert cache.stats["misses"] == 1
        assert cache.stats["hits"] == 0
        assert out1.gates[0].params == (0.1,)

        # Iteration 2: same structure, updated angles (cache hit!)
        c2 = _make_circuit(2, [
            Gate("rx", (0,), params=(0.95,)),
            Gate("cx", (0, 1)),
        ])
        out2 = cache.transpile_with_cache(dummy_transpile, c2, [0.95])
        assert call_count == 1, "Transpile function should NOT be re-executed on cache hit"
        assert cache.stats["misses"] == 1
        assert cache.stats["hits"] == 1
        assert out2.gates[0].params == (0.95,)

        # Iteration 3: different circuit structure (cache miss)
        c3 = _make_circuit(2, [
            Gate("ry", (0,), params=(0.5,)),
            Gate("cx", (0, 1)),
        ])
        out3 = cache.transpile_with_cache(dummy_transpile, c3, [0.5])
        assert call_count == 2
        assert cache.stats["misses"] == 2
        assert cache.stats["hits"] == 1
        assert out3.gates[0].params == (0.5,)

    def test_eviction(self):
        cache = TranspilerCache(max_size=2)
        c1 = _make_circuit(1, [Gate("x", (0,))])
        c2 = _make_circuit(1, [Gate("y", (0,))])
        c3 = _make_circuit(1, [Gate("z", (0,))])

        cache.put("h1", c1, 0)
        cache.put("h2", c2, 0)
        assert cache.stats["cache_size"] == 2
        assert cache.stats["evictions"] == 0

        # Adding 3rd entry triggers eviction of oldest
        cache.put("h3", c3, 0)
        assert cache.stats["cache_size"] == 2
        assert cache.stats["evictions"] == 1
        assert cache.get("h1") is None
        assert cache.get("h2") is not None
        assert cache.get("h3") is not None

    def test_clear(self):
        cache = TranspilerCache()
        cache.put("test", "circ", 0)
        cache.get("test")
        assert cache.stats["hits"] == 1
        assert cache.stats["cache_size"] == 1

        cache.clear()
        assert cache.stats["hits"] == 0
        assert cache.stats["cache_size"] == 0
        assert cache.get("test") is None

    def test_thread_safety(self):
        cache = TranspilerCache(max_size=100)
        errors = []

        def worker(thread_id: int):
            try:
                for i in range(50):
                    c = _make_circuit(2, [
                        Gate("rx", (0,), params=(float(i),)),
                        Gate("cx", (0, 1)),
                    ])
                    cache.transpile_with_cache(lambda x: x, c, [float(i)])
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Concurrent cache operations failed: {errors}"
        assert cache.stats["total_requests"] == 200
