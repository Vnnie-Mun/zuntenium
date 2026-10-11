# ZQL-00 | Start Here

Welcome to **Zuntenium Quantum Learning (ZQL)**, the official curriculum and documentation for the Zuntenium Hybrid Quantum SDK and Workspace.

## Learning Objectives
- Install the Zuntenium SDK.
- Set up your development environment.
- Construct and execute your first quantum circuit in under 5 minutes.
- Choose your learning path.

## Prerequisites
- Python 3.10+
- Basic understanding of Python programming

## 1. Installation

Install the core SDK directly via `pip`:
```bash
pip install zuntenium
```

*(Optional) Install Zuntenium Cloud microservices and Workspace:*
```bash
pip install "zuntenium[cloud]"
```

## 2. Your First Circuit (Hello Quantum World)

Let's build a circuit that creates a Bell state (a maximally entangled state of two qubits) and measures it.

### Step-by-Step Code

```python
from zuntenium.circuit import QuantumCircuit
from zuntenium.api.quantum_api import execute

# 1. Initialize a 2-qubit circuit with a fluent builder
circuit = QuantumCircuit(2, name="BellState")

# 2. Add a Hadamard to qubit 0, a CNOT from 0->1, and measure all
circuit.h(0).cx(0, 1).measure_all()

# 3. Execute the circuit using the default Statevector simulator
# We'll take 1024 measurement "shots"
result = execute(circuit, shots=1024)

# 4. Print the results
print("Most probable state:", result.most_probable)
print("Measurement counts:", result.counts)
```

### Expected Output
Because this creates the state $\frac{|00\rangle + |11\rangle}{\sqrt{2}}$, you should see roughly equal measurements of `00` and `11`:
```text
Most probable state: 00
Measurement counts: {'00': 503, '11': 521}
```

## 3. The Zuntenium CLI

Zuntenium provides a powerful Command-Line Interface (CLI). You can run QASM 2.0 circuits directly from your terminal:

```bash
# Check your installation and system specs
zuntenium info

# List available backend simulators and hardware connectors
zuntenium backends

# Run a QASM file
zuntenium run my_circuit.qasm --backend statevector_simulator --shots 2048
```

## 4. Choose Your Learning Path

*   **Developer:** Proceed to [ZQL-02 | SDK Core Reference](ZQL-02-sdk-core-reference.md) to deeply understand the architecture, transpiler, and APIs.
*   **Researcher:** Proceed to [ZQL-05 | Variational and Hybrid](ZQL-05-variational-hybrid.md) to explore VQE, QAOA, and error mitigation (ZNE/PEC).
*   **Workspace User:** Jump to [ZQL-14 | Workspace Guide](ZQL-14-workspace-guide.md) to learn how to use the interactive QCanvas UI and Next.js portal.
