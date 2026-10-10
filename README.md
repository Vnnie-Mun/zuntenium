# ZUNTENIUM

**AI-Native Hybrid Quantum Simulation Framework**

[![CI](https://github.com/zuntenium/zuntenium/actions/workflows/ci.yml/badge.svg)](https://github.com/zuntenium/zuntenium/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/zuntenium/zuntenium/branch/main/graph/badge.svg)](https://codecov.io/gh/zuntenium/zuntenium)
[![PyPI version](https://badge.fury.io/py/zuntenium.svg)](https://badge.fury.io/py/zuntenium)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

---

## Overview

ZUNTENIUM is a production-grade, AI-native hybrid quantum simulation framework designed to scale from laptop simulations to real quantum hardware execution. Built with a clean layered architecture, ZUNTENIUM provides:

- **High-performance statevector simulation** — NumPy for ≤20 qubits, C++ (pybind11) above that
- **Fluent circuit builder API** — Chainable, expressive circuit construction
- **Canonical IR + Transpiler** — Rule-based optimization pipeline with ML-ready interface
- **Hardware abstraction** — Swap simulator/hardware with zero code changes
- **Hybrid algorithms** — VQE, QAOA with parameter-shift gradients
- **AI-ready architecture** — Training data collection from v0.1 for future ML transpiler

---

## Quick Start

```bash
pip install zuntenium
```

```python
from zuntenium import QuantumCircuit, execute, get_backend

# Build a Bell state circuit
qc = QuantumCircuit(2)
qc.h(0).cx(0, 1).measure_all()

# Run on the statevector simulator
backend = get_backend("statevector_simulator")
result = execute(qc, backend=backend, shots=1024)

print(result.counts)        # {'00': 512, '11': 512}
result.plot_histogram()
```

### VQE Example

```python
from zuntenium.hybrid import VQE
from zuntenium.hybrid.observable import Hamiltonian, PauliString

H = Hamiltonian([
    (0.5,  PauliString("ZZ")),
    (-0.3, PauliString("XX")),
])

vqe = VQE(ansatz=qc, hamiltonian=H, backend=backend)
result = vqe.run(initial_params=[0.1, 0.2])
print(f"Ground state energy: {result.energy:.6f}")
```

---

## Installation

### From PyPI (stable)
```bash
pip install zuntenium
```

### From source (with C++ bindings)
```bash
git clone https://github.com/zuntenium/zuntenium.git
cd zuntenium
pip install -e ".[dev]"
bash scripts/build_bindings.sh
```

### Docker (recommended for development)
```bash
docker-compose -f docker/docker-compose.yml up dev
```

---

## Architecture

```
User Code → Public API → Circuit Builder → Transpiler → Backend → Result
                                                ↓
                                        C++/Rust Bindings (high qubit count)
```

See [`docs/architecture/overview.md`](docs/architecture/overview.md) for the full design.

---

## Roadmap

| Version | Focus |
|---------|-------|
| **v0.1** | NumPy statevector, rule-based transpiler, VQE/QAOA |
| v0.2 | C++ statevector, noise model, IBM/IonQ hardware |
| v0.3 | MPS simulator, GPU acceleration, Rust bindings |
| v0.4 | AI transpiler (RL-based), circuit fingerprinting |
| v0.5 | ZUNTENIUM Cloud, multi-tenant job queue, Plugin SDK |

---

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md). All PRs require passing CI and ≥90% unit test coverage.

---

## License

Apache 2.0 — See [`LICENSE`](LICENSE).

*ZUNTENIUM v0.1 — Designed for scale from day one.*
# zuntenium-Qutritflow-sdk
