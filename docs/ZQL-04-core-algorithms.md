# ZQL-04 | Core Algorithms

## Learning Objectives
- Understand the mechanics behind fundamental quantum algorithms.
- Build a Quantum Fourier Transform (QFT) circuit manually using Zuntenium.

---

## 1. The Quantum Fourier Transform (QFT)

The QFT is the quantum analogue of the Discrete Fourier Transform. It is the core engine inside Phase Estimation (QPE) and Shor's Algorithm.

Instead of computing the transform on a classical array of numbers, the QFT computes it on the amplitudes of a quantum state.

### The Math
For a state $|x\rangle$:
$$ QFT|x\rangle = \frac{1}{\sqrt{N}} \sum_{y=0}^{N-1} e^{2\pi i x y / N} |y\rangle $$

## 2. Building a 3-Qubit QFT in Zuntenium

*Note: While Zuntenium plans to release a native `QFT` algorithm class in the future, constructing it manually is the best way to learn.*

The QFT uses Hadamard gates ($H$) and Controlled-Phase rotations ($CP$ or $CRz$ up to global phase).

```python
from zuntenium.circuit import QuantumCircuit
from zuntenium.api.quantum_api import execute
import math

# We will build a 3-qubit QFT
qc = QuantumCircuit(3, name="QFT3")

# We apply the QFT sequence qubit by qubit, starting from the Most Significant Qubit.

# Qubit 2
qc.h(2)
qc.crz(math.pi / 2, 1, 2)
qc.crz(math.pi / 4, 0, 2)

# Qubit 1
qc.h(1)
qc.crz(math.pi / 2, 0, 1)

# Qubit 0
qc.h(0)

# Finally, we must SWAP the qubits to match the standard endianness
qc.swap(0, 2)

# Execute the circuit and retrieve the statevector
result = execute(qc, shots=0)
print("QFT Statevector (Rounded):")
for i, amp in enumerate(result.statevector):
    print(f"|{i:03b}> : {amp:.3f}")
```

## 3. Grover's Algorithm (Conceptual Overview)

Grover's Algorithm provides a quadratic speedup for unstructured search. It consists of two components:
1. **The Oracle**: Flips the phase of the target state ($|x\rangle \rightarrow -|x\rangle$).
2. **The Diffuser**: Inverts all amplitudes about their mean, boosting the target state and suppressing the rest.

*Grover's algorithm components are actively being developed for the `zuntenium.algorithms` module.*

## Next Steps
Now that you have seen manual algorithmic construction, you are ready to learn about hybrid algorithms which combine classical optimizers with parameterized quantum circuits in [ZQL-05 | Variational and Hybrid](ZQL-05-variational-hybrid.md).
