# ZQL-08 | Optimization, Finance, and Logistics

## Learning Objectives
- Formulate the MaxCut problem as a Hamiltonian.
- Solve combinatorial optimization using Zuntenium's QAOA.

---

## 1. Combinatorial Optimization

Many problems in finance (portfolio optimization) and logistics (vehicle routing) can be mapped to a Quadratic Unconstrained Binary Optimization (QUBO) problem. QUBOs map directly to Ising Hamiltonians, which can be solved on a quantum computer using QAOA.

## 2. The MaxCut Problem

The Maximum Cut (MaxCut) problem asks: Given a graph, how can we partition the vertices into two sets to maximize the number of edges between the sets?

For an edge between vertex $i$ and $j$, the cost Hamiltonian term is $Z_i Z_j$. If the spins (qubits) are aligned (same partition), $Z_i Z_j = 1$. If they are anti-aligned (different partitions, cut edge), $Z_i Z_j = -1$. 
To maximize the cut, we minimize the energy of the Hamiltonian:
$$ H_C = \sum_{(i,j) \in E} Z_i Z_j $$

## 3. Solving MaxCut with Zuntenium

Let's solve a simple triangle graph (3 nodes, 3 edges).

```python
from zuntenium.api.hybrid_api import run_qaoa
from zuntenium.hybrid.observable import Hamiltonian, PauliString

# 1. Define the graph edges
# Edges: (0,1), (1,2), (0,2)

# 2. Construct the Cost Hamiltonian
# Each edge contributes a ZZ term to the cost.
# In PauliStrings, "ZZ" acts on qubits 0 and 1, etc.
H_cost = Hamiltonian([
    (1.0, PauliString("ZZI")), # Edge (0, 1) -> Z on 0 and 1, I on 2
    (1.0, PauliString("IZZ")), # Edge (1, 2) -> I on 0, Z on 1 and 2
    (1.0, PauliString("ZIZ"))  # Edge (0, 2) -> Z on 0, I on 1, Z on 2
])

# 3. Run QAOA (Depth p=2 for better accuracy)
result = run_qaoa(
    hamiltonian=H_cost,
    p=2,
    shots=0, # Exact statevector simulation
    max_iter=100
)

print(f"Optimal Energy (MaxCut Cost): {result.energy:.4f}")
print("Optimal Parameters (Gamma, Beta):", result.optimal_params)
```

## Next Steps
Optimization algorithms like QAOA are resilient to near-term noise. In the next section, we explore long-term strategies in cryptography and secure communications in [ZQL-09 | Cryptography and Security](ZQL-09-crypto-security.md).
