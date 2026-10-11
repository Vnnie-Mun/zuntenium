# ZUNTENIUM API Reference

> Auto-generated API documentation placeholder.
> Run `sphinx-apidoc -o docs/api zuntenium/` to generate full API reference.

## Core Modules

### `zuntenium.quantum_core`

| Class | Description |
|---|---|
| `StateVector` | Primary quantum state abstraction (NumPy + C++ dispatch) |
| `DensityMatrix` | Mixed state density matrix with partial trace |
| `CircuitExecutor` | Core execution engine |
| `MeasurementSampler` | Sampling logic from statevector amplitudes |
| `NoiseModel` | Noise channels: depolarizing, amplitude/phase damping |

### `zuntenium.quantum_core.gates`

| Class | Qubits | Description |
|---|---|---|
| `HGate` | 1 | Hadamard |
| `XGate`, `YGate`, `ZGate` | 1 | Pauli gates |
| `SGate`, `TGate` | 1 | Phase gates |
| `RxGate`, `RyGate`, `RzGate` | 1 | Rotation gates |
| `U1Gate`, `U2Gate`, `U3Gate` | 1 | IBM universal gates |
| `CXGate`, `CZGate`, `CYGate` | 2 | Controlled Pauli |
| `SWAPGate`, `ISWAPGate` | 2 | Swap gates |
| `ToffoliGate`, `FredkinGate` | 3 | Multi-controlled gates |

### `zuntenium.circuit`

| Class | Description |
|---|---|
| `QuantumCircuit` | Fluent builder for quantum circuits |
| `QuantumRegister` | Qubit register |
| `ClassicalRegister` | Classical bit register |
| `QuantumIR` | Canonical intermediate representation |
| `TextDrawer` | ASCII circuit diagrams |
| `LatexDrawer` | LaTeX quantikz renderer |

### `zuntenium.hybrid`

| Class | Description |
|---|---|
| `VQE` | Variational Quantum Eigensolver |
| `QAOA` | Quantum Approximate Optimization Algorithm |
| `Hamiltonian` | Linear combination of Pauli operators |
| `PauliString` | Tensor product of Pauli matrices |
| `ClassicalOptimizer` | scipy.optimize wrapper |
| `GradientEstimator` | Parameter-shift rule and finite differences |

### `zuntenium.transpiler`

| Class | Description |
|---|---|
| `TranspilerPipeline` | Ordered pass execution (levels 0–3) |
| `GateCancellationPass` | Remove H·H, X·X identity pairs |
| `RotationMergingPass` | Merge adjacent Rz rotations |
| `SingleQubitUnifyPass` | Compile 1Q sequences → U3 |
| `CouplingMap` | Hardware connectivity graph |
| `Target` | Device capabilities and constraints |

### `zuntenium.backend`

| Class | Description |
|---|---|
| `AbstractBackend` | Universal backend interface |
| `StatevectorBackend` | Primary simulator (v0.1) |
| `UnitaryBackend` | Full unitary matrix simulation |
| `SnapshotBackend` | Statevector snapshots |
| `MockHardwareBackend` | Deterministic test backend |
| `BackendProvider` | Registry and factory |

## Public Functions

```python
from zuntenium import QuantumCircuit, execute, get_backend

qc = QuantumCircuit(2)
qc.h(0).cx(0, 1).measure_all()
result = execute(qc, shots=1024)
print(result.counts)
```
