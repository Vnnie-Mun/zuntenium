# ZUNTENIUM User Guides

## Quick Start

```python
from zuntenium import QuantumCircuit, execute, get_backend

# Create a Bell pair
qc = QuantumCircuit(2)
qc.h(0).cx(0, 1).measure_all()

# Execute
result = execute(qc, shots=1024)
print(result.counts)      # {'00': ~512, '11': ~512}
print(result.most_probable)  # '00' or '11'
```

## Guides

| Guide | Description |
|---|---|
| [Getting Started](getting_started.md) | Installation, first circuit, first run |
| [Circuit Building](circuit_building.md) | Fluent API, custom gates, composition |
| [VQE Tutorial](vqe_tutorial.md) | Variational Quantum Eigensolver walkthrough |
| [Transpiler](transpiler_guide.md) | Optimization levels, custom passes |
| [C++ Backend](cpp_backend.md) | Building and using the C++ extension |
