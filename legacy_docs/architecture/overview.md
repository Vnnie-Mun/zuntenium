# ZUNTENIUM Architecture Overview

## Design Philosophy

ZUNTENIUM is built around three core principles:

1. **Layered separation** — each layer has one job and talks only to the layer below it
2. **Backend transparency** — user code is identical whether targeting a simulator or real hardware
3. **AI-native from day one** — training data collection and ML hooks are wired in at v0.1, not bolted on later

---

## Layer Diagram

```
┌─────────────────────────────────────────────────────────┐
│  USER CODE  (Python)                                    │
│  from zuntenium import QuantumCircuit, execute          │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│  PUBLIC API  (zuntenium/api/)                           │
│  execute() · get_backend() · VQE() · QAOA()             │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│  CIRCUIT BUILDER  (zuntenium/circuit/)                  │
│  QuantumCircuit builder → Instruction list → QuantumIR  │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│  TRANSPILER  (zuntenium/transpiler/)                    │
│  QuantumIR → optimization passes → optimized QuantumIR  │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│  BACKEND  (zuntenium/backend/)                          │
│  AbstractBackend → simulator or hardware adapter        │
└───────────┬──────────────────────────────┬──────────────┘
            │                              │
┌───────────▼──────────┐    ┌─────────────▼──────────────┐
│  STATEVECTOR SIM     │    │  HARDWARE ADAPTER           │
│  quantum_core/       │    │  IBM / IonQ / mock          │
│  NumPy ≤20q          │    │  Translates IR → native fmt │
│  C++ >20q            │    └────────────────────────────┘
└───────────┬──────────┘
            │
┌───────────▼──────────────────────────────────────────────┐
│  NATIVE BINDINGS  (bindings_core/)                       │
│  C++ pybind11 statevector · SIMD · OpenMP · CUDA (v0.4) │
└───────────────────────────────────────────────────────────┘
```

---

## Module Responsibilities

| Module | Responsibility |
|---|---|
| `api/` | User-facing public interface — orchestrates all layers |
| `circuit/` | Circuit construction, registers, IR generation |
| `circuit/ir/` | Canonical intermediate representation + serialization |
| `transpiler/` | IR optimization, mapping, device adaptation |
| `quantum_core/` | Statevector math, gates, measurement, execution |
| `backend/` | Backend abstraction, job management, result caching |
| `hybrid/` | VQE/QAOA classical-quantum optimization loop |
| `utils/` | Linear algebra, validation, logging, profiling |
| `bindings/` | Python ↔ C++/Rust binding loader + fallback |
| `bindings_core/` | Native C++ / Rust performance implementations |

---

## Key Design Decisions

### Why a Custom IR?

Using a custom `QuantumIR` dataclass rather than adopting Qiskit's DAGCircuit:
- **Zero external dependency** on Qiskit (no version lock)
- **Full control** over shape, serialization, and ML training data embedding
- **Clean interface** to the transpiler pass system

### Backend Strategy Pattern

All backends (`StatevectorBackend`, `IBMBackend`, `IonQBackend`) implement `AbstractBackend`. When `execute()` is called:
1. The circuit is compiled to `QuantumIR`
2. The transpiler optimizes the IR for the target backend
3. The backend executes the IR and returns an `ExecutionResult`

Switching backends is one line of code.

### AI-Ready Architecture

- The transpiler's `ai_pass/` module accepts a drop-in ML model
- Every pass logs IR in/out pairs for training data collection
- Circuit fingerprinting (v0.4) will enable compilation caching with learned policies

---

## Versioning & Stability

- **v0.1** — All public APIs in `api/` are stable and SemVer-guaranteed
- Internal modules (`transpiler/passes/`, `quantum_core/executor/`) are implementation detail and may change
- C extension ABI is **not** stable until v0.3
