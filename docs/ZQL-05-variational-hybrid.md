# ZQL-05 | Variational and Hybrid

## Learning Objectives
- Define quantum observables using `Hamiltonian` and `PauliString`.
- Execute the Variational Quantum Eigensolver (VQE).
- Execute the Quantum Approximate Optimization Algorithm (QAOA).

---

## 1. Defining Observables

In hybrid algorithms, we measure the expectation value of a target Hamiltonian (the energy landscape). Zuntenium provides native objects to define these using Pauli strings.

```python
from zuntenium.hybrid.observable import Hamiltonian, PauliString

# H = 0.5 * ZZ - 0.3 * XX
H = Hamiltonian([
    (0.5, PauliString("ZZ")),
    (-0.3, PauliString("XX"))
])
```

## 2. Variational Quantum Eigensolver (VQE)

VQE aims to find the ground state energy (the lowest eigenvalue) of a Hamiltonian using a parameterized quantum circuit (the Ansatz) and a classical optimizer (like COBYLA).

### Running VQE natively

```python
from zuntenium.circuit import QuantumCircuit
from zuntenium.api.hybrid_api import run_vqe
from zuntenium.hybrid.observable import Hamiltonian, PauliString
import numpy as np

# 1. Define Hamiltonian
H = Hamiltonian([(0.5, PauliString("ZZ")), (-0.3, PauliString("XX"))])

# 2. Define Parameterized Ansatz
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
qc.rx(0.0, 0) # Parameterized angle (init at 0.0)
qc.ry(0.0, 1) # Parameterized angle (init at 0.0)

# 3. Initial Parameters
initial_params = np.array([0.1, 0.2])

# 4. Execute VQE
result = run_vqe(
    ansatz=qc, 
    hamiltonian=H, 
    initial_params=initial_params, 
    max_iter=100, 
    shots=1024
)

print(f"Optimal Energy: {result.energy:.4f}")
print(f"Optimal Parameters: {result.optimal_params}")
```

## 3. QAOA (Optimization)

QAOA is specialized for combinatorial optimization tasks (like MaxCut). It alternates between applying the problem Hamiltonian and a mixing Hamiltonian across $p$ layers.

Zuntenium provides a specialized wrapper for QAOA that automatically constructs the ansatz:

```python
from zuntenium.api.hybrid_api import run_qaoa
from zuntenium.hybrid.observable import Hamiltonian, PauliString

# Define the Cost Hamiltonian for a graph (e.g., MaxCut)
H_cost = Hamiltonian([
    (1.0, PauliString("ZZ")), # Edge between qubit 0 and 1
])

# Run QAOA with p=1 layers
result = run_qaoa(
    hamiltonian=H_cost,
    p=1,
    shots=1024,
    max_iter=50
)

print(f"Optimal Cost: {result.energy:.4f}")
```

## Next Steps
Now that you can run variational loops, learn how these concepts extend to Neural Networks in [ZQL-06 | Quantum AI and Machine Learning](ZQL-06-quantum-ai-ml.md).
