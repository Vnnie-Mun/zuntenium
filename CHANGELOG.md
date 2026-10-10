# CHANGELOG

All notable changes to ZUNTENIUM are documented here.
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- Full architecture scaffolding for v0.2 hardware backends (IBM, IonQ)
- SABRE routing pass stub for v0.2

---

## [0.1.0] — 2026-04-15

### Added

**Core Simulation Engine**
- `StateVector` with NumPy backend (auto-dispatches to C++ when built)
- Full gate library: H, X, Y, Z, S, T, Rx, Ry, Rz, U1, U2, U3, CX, CZ, CY, SWAP, iSWAP, CRx, CRy, CRz, CCX (Toffoli), CSWAP (Fredkin)
- `CircuitExecutor` with post-hoc measurement sampling

**Circuit Layer**
- `QuantumCircuit` fluent builder with full gate set
- `QuantumRegister` and `ClassicalRegister`
- Circuit depth computation, compose, bind_parameters
- ASCII circuit visualization (`draw()`)
- `QuantumIR` canonical intermediate representation
- `IRNode` typed instruction nodes

**Transpiler**
- `TranspilerPipeline` with optimization levels 0–3
- `GateCancellationPass`: removes H·H, X·X, CX·CX pairs
- `RotationMergingPass`: merges Rz(a)·Rz(b) → Rz(a+b)
- `SingleQubitUnifyPass`: ZYZ Euler decomposition
- `ConstantFoldingPass`: replaces known-angle rotations with fixed gates
- `DepthAnalysisPass`, `GateCountPass`, `QubitConnectivityPass`
- `LayoutPass` (identity, v0.1), `RoutingPass` (greedy SWAP insertion)
- `AIOptimizerPass` data collection stub for v0.4

**Backends**
- `StatevectorBackend`: primary NumPy/C++ simulator (≤30 qubits)
- `UnitaryBackend`: full unitary matrix simulator (≤14 qubits)
- `SnapshotBackend`: per-layer statevector snapshots
- `MockHardwareBackend`: test fixture for hardware code paths
- `BackendProvider` registry with alias resolution

**Hybrid Algorithms**
- `VQE` with parameter-shift gradient and scipy optimizer
- `QAOA` with p-layer ansatz construction
- `Hamiltonian` (linear combination of `PauliString`s)
- `ClassicalOptimizer` (scipy wrapper with gradient descent fallback)
- `GradientEstimator` (parameter-shift and finite difference)
- `ConvergenceTracker` with patience-based early stopping
- `HybridResult` with convergence history

**Serialization & I/O**
- OpenQASM 2.0 and 3.0 serializer
- OpenQASM 2.0 parser (standard gate set)
- JSON serializer/deserializer for QuantumIR
- `load_qasm()`, `save_qasm()`, `load_circuit()`, `save_circuit()` API

**C++ Extension**
- pybind11 binding skeleton with `statevector_core.hpp`
- Graceful NumPy fallback when C++ extension is not built
- `loader.py` with `is_cpp_available()` and `warn_if_numpy_fallback()`

**Utilities**
- Linear algebra: `tensor_product`, `partial_trace`, `commutator`, `is_unitary`, `is_hermitian`
- Distance measures: `state_fidelity`, `trace_distance`, `bures_distance`
- `matrix_exponential`, ZYZ Euler decomposition
- Bloch sphere: `bloch_vector`, `spherical_to_cartesian`
- Bit utilities: `int_to_bitstring`, `hamming_distance`, `parity`, `flip_bit`
- `timed` / `memory_tracked` decorators, `benchmark()` runner
- Structured logging via `get_logger()`, `configure_logging()`

**Tests**
- 60+ unit tests: gates (unitarity, dimensions, per-gate properties)
- Circuit tests: construction, depth, compose, IR, parameter binding
- StateVector tests: init, gate application, normalization
- Transpiler tests: all passes, pipeline levels, add/remove pass
- Integration tests: Bell/GHZ execution, rotation gates, API
- QASM/JSON round-trip integration tests
- Performance benchmarks for multi-qubit circuits

**CI/CD**
- GitHub Actions: CI (multi-Python, multi-OS), Release, Benchmarks
- Docker: dev, prod, GPU (CUDA) Dockerfiles + docker-compose
- Makefile with install, test, lint, format, typecheck, benchmark targets

### Known Limitations (v0.1)

- Hardware backends (IBM, IonQ) are stubs — activate in v0.2
- Noise models are stubs — activate in v0.2
- MPS/sparse simulation not yet implemented (v0.3)
- SABRE routing is a stub — activate in v0.2
- GPU acceleration is a stub — activate in v0.4
- AI transpiler pass is a stub — activate in v0.4
