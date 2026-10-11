"""
Verification test for ZQL-15 Top-Tier Benchmark Audit & 8-Layer Accuracy Stack.
Tests Layer 1 norm drift, Layer 2 endianness conventions, and Layer 7 seeded reproducibility.
"""
import math
import numpy as np
from zuntenium.circuit import QuantumCircuit
from zuntenium.api.quantum_api import execute


def test_docs_zql15_layer1_norm_drift():
    # Layer 1: Check norm drift after multi-gate sequence
    qc = QuantumCircuit(3)
    qc.h(0).h(1).h(2)
    qc.rx(0.45, 0).ry(0.82, 1).rz(1.23, 2)
    qc.cx(0, 1).cx(1, 2).swap(0, 2)

    result = execute(qc, shots=0)
    sv = np.array(result.statevector)
    norm = np.linalg.norm(sv)
    assert math.isclose(norm, 1.0, abs_tol=1e-12), f"Norm drift exceeded limit: {abs(norm - 1.0)}"


def test_docs_zql15_layer2_endianness_convention():
    # Layer 2: Verify little-endian convention (q0 is rightmost bit)
    qc = QuantumCircuit(2)
    qc.x(0)  # q0 = 1, q1 = 0 -> |q1 q0> = |01>
    qc.measure_all()

    result = execute(qc, shots=512)
    assert result.most_probable == "01", f"Expected '01' (q0 rightmost), got {result.most_probable}"


def test_docs_zql15_layer7_seeded_reproducibility():
    # Layer 7: Seeded runs must produce reproducible results
    from zuntenium.api.quantum_api import get_backend
    qc = QuantumCircuit(2)
    qc.h(0).cx(0, 1).measure_all()

    b1 = get_backend("statevector_simulator", seed=42)
    res1 = execute(qc, backend=b1, shots=1024)

    b2 = get_backend("statevector_simulator", seed=42)
    res2 = execute(qc, backend=b2, shots=1024)

    assert res1.counts == res2.counts, f"Seeded executions diverged: {res1.counts} != {res2.counts}"
