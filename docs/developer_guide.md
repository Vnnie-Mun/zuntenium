# ZUNTENIUM — Developer Guide

## Quick Start

```bash
# Clone & install in editable mode
git clone https://github.com/zuntenium/zuntenium.git
cd zuntenium
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# (Optional) Build C++ bindings for extra performance
make build-bindings
```

## Running Your First Circuit

```python
from zuntenium import QuantumCircuit, execute

# 2-qubit Bell state
qc = QuantumCircuit(2)
qc.h(0).cx(0, 1).measure_all()

result = execute(qc, shots=1024)
print(result.counts)          # {'00': ~512, '11': ~512}
result.plot_histogram()
```

## Running Tests

```bash
make test               # full suite
make test-unit          # unit tests only
make test-integration   # integration tests only
pytest tests/ -v        # verbose
```

## Project Layout

```
zuntenium/         Python package — core library
bindings/          pybind11 C++ extension build files
bindings_core/     Standalone C++/Rust simulation core
docs/              Architecture docs and ADRs
tests/             Unit, integration, and benchmark tests
scripts/           Developer utility scripts
docker/            Container build files
```

## Key Concepts

### Circuit → IR → Backend

1. User builds `QuantumCircuit` with fluent gate methods
2. `circuit.to_ir()` compiles to `QuantumIR` (canonical IR)
3. `TranspilerPipeline` optimizes the IR
4. `Backend.run(ir, shots=N)` executes and returns `ExecutionResult`

### Optimization Levels

| Level | Passes Applied |
|-------|---------------|
| 0 | None (debug) |
| 1 | GateCancellation |
| 2 | GateCancellation + RotationMerging + SingleQubitUnify |
| 3 | Level 2 + ConstantFolding |

### VQE Example

```python
import numpy as np
from zuntenium import QuantumCircuit, execute
from zuntenium.hybrid import VQE, Hamiltonian, PauliString

H = Hamiltonian([(0.5, PauliString("ZZ")), (-0.3, PauliString("XX"))])

ansatz = QuantumCircuit(2)
ansatz.ry(0.0, 0).ry(0.0, 1).cx(0, 1)

vqe = VQE(ansatz=ansatz, hamiltonian=H)
result = vqe.run(initial_params=np.array([0.1, 0.2]))
print(f"Ground state energy: {result.energy:.6f}")
```
