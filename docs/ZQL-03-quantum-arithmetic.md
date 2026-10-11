# ZQL-03 | Quantum Arithmetic

## Learning Objectives
- Understand how classical arithmetic translates to quantum circuits.
- Build a Quantum Half-Adder manually using Zuntenium's native gates.

---

## 1. Reversible Computation

Classical computers use irreversible logic gates (e.g., AND, OR) which destroy information, shedding heat (Landauer's Principle). Quantum gates are unitary and therefore perfectly reversible. 

To perform classical addition on a quantum computer, we must construct a reversible **Half-Adder**.

### The Classical Half-Adder
- **Inputs**: A, B
- **Sum**: A XOR B
- **Carry**: A AND B

### The Quantum Half-Adder
We map these logic gates to reversible quantum gates:
- **XOR** is equivalent to the **CNOT** gate.
- **AND** is equivalent to the **Toffoli (CCX)** gate.

Because quantum gates are reversible, we need a dedicated "Carry" qubit and a "Sum" qubit.

## 2. Building a Half-Adder in Zuntenium

*Note: While Zuntenium plans to offer high-level `Adder` objects in future releases, you can construct them manually using the `QuantumCircuit` API today.*

```python
from zuntenium.circuit import QuantumCircuit
from zuntenium.api.quantum_api import execute

# We need 3 qubits:
# q0: Input A
# q1: Input B (will hold the Sum)
# q2: Carry out
qc = QuantumCircuit(3, name="HalfAdder")

# -------------------------------
# Setup Inputs (Let's add 1 + 1)
# -------------------------------
qc.x(0)  # A = 1
qc.x(1)  # B = 1

qc.barrier() # Optional: visual separation

# -------------------------------
# Compute Carry = A AND B
# -------------------------------
# The Toffoli (CCX) gate targets q2, controlled by q0 and q1.
qc.ccx(0, 1, 2)

# -------------------------------
# Compute Sum = A XOR B
# -------------------------------
# The CNOT gate targets q1, controlled by q0.
qc.cx(0, 1)

# Measure all qubits
qc.measure_all()

# Execute
result = execute(qc, shots=1024)
print("Arithmetic Result:", result.most_probable)
```

### Expected Output
If you add $1 + 1$ in binary, the result is `10` (Sum = 0, Carry = 1).
Zuntenium returns bitstrings in $q_2 q_1 q_0$ order.
- $q_2$ (Carry) = `1`
- $q_1$ (Sum) = `0`
- $q_0$ (Input A) = `1`

Thus, `result.most_probable` will return `'101'`.

## Next Steps
With arithmetic out of the way, you are ready to explore quantum algorithms that leverage superposition and entanglement in [ZQL-04 | Core Algorithms](ZQL-04-core-algorithms.md).
