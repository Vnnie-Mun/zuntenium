# ZQL-01 | Quantum Foundations

## Learning Objectives
- Understand Qubits, Superposition, and Entanglement.
- Learn the Matrix representations of quantum gates.
- Visualize quantum states using Zuntenium's Statevector.

---

## 1. The Qubit

A classical bit is either `0` or `1`. A quantum bit (qubit) can exist in a superposition of both states, represented as a column vector in a complex vector space $\mathbb{C}^2$:

$$ |\psi\rangle = \alpha |0\rangle + \beta |1\rangle = \begin{pmatrix} \alpha \\ \beta \end{pmatrix} $$

Where $|\alpha|^2 + |\beta|^2 = 1$. The probabilities of measuring 0 or 1 are $|\alpha|^2$ and $|\beta|^2$, respectively.

## 2. Quantum Gates

Quantum gates are unitary matrices $U$ ($U^\dagger U = I$) that rotate the qubit state. 

### Pauli-X (NOT)
Flips the computational basis states.
$$ X = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix} $$

### Hadamard (H)
Creates an equal superposition.
$$ H = \frac{1}{\sqrt{2}} \begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix} $$

### CNOT (Controlled-NOT)
A 2-qubit gate. If the control qubit is $|1\rangle$, it applies an $X$ gate to the target.
$$ CNOT = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & 1 & 0 \end{pmatrix} $$

## 3. Simulating Foundations in Zuntenium

Let's use Zuntenium's exact Statevector simulator to inspect these matrices in action.

```python
from zuntenium.circuit import QuantumCircuit
from zuntenium.api.quantum_api import execute
import numpy as np

# 1. Create a 1-qubit circuit
qc = QuantumCircuit(1)

# 2. Apply Hadamard to create superposition
qc.h(0)

# 3. Use the statevector simulator to get the exact amplitudes
result = execute(qc, shots=0)  # shots=0 returns the exact statevector
state = result.statevector

print("Statevector:", state)
# Output should be roughly [0.707+0.j, 0.707+0.j]
```

## Next Steps
Now that you understand the mathematical foundations of qubits and gates, move on to combining them for classical arithmetic operations in [ZQL-03 | Quantum Arithmetic](ZQL-03-quantum-arithmetic.md).
