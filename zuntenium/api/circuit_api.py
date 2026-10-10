"""Circuit construction convenience API."""
from __future__ import annotations

from zuntenium.circuit.circuit import QuantumCircuit


def bell_state(qubit0: int = 0, qubit1: int = 1) -> QuantumCircuit:
    """Create a 2-qubit Bell state circuit."""
    qc = QuantumCircuit(2, name="bell")
    qc.h(qubit0).cx(qubit0, qubit1)
    return qc


def ghz_state(n: int) -> QuantumCircuit:
    """Create an n-qubit GHZ state circuit."""
    qc = QuantumCircuit(n, name=f"ghz_{n}")
    qc.h(0)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    return qc


def quantum_fourier_transform(n: int) -> QuantumCircuit:
    """Create an n-qubit Quantum Fourier Transform circuit."""
    import math
    qc = QuantumCircuit(n, name=f"qft_{n}")
    for j in range(n):
        qc.h(j)
        for k in range(j + 1, n):
            angle = math.pi / (2 ** (k - j))
            qc.crz(angle, k, j)   # controlled-Rz
    # Bit-reversal swaps
    for i in range(n // 2):
        qc.swap(i, n - 1 - i)
    return qc
