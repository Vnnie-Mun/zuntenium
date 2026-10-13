# QuFlow Quantum SDK & Workspace: System Architecture

## 1. System Topology Overview

```
                               ┌───────────────────────────────────────────────────────────┐
                               │                    Developer Surfaces                     │
                               │                                                           │
                               │   QuFlow Web IDE      VS Code Extension      Python /    │
                               │   (Next.js 15)         (VSIX 0.1.0)           CLI         │
                               └─────────────┬──────────────────┬───────────────┬──────────┘
                                             │                  │               │
                                             ▼                  ▼               ▼
┌───────────────────────────┐   WebSocket   ┌──────────────────────────────────────────────┐
│ Real-Time Collab Server   │◄─────────────►│             FastAPI Cloud Gateway            │
│ (ws, CRDT State Sync)     │   (Port 3001) │         (AsyncPG, Redis, Prometheus)         │
└───────────────────────────┘               └──────────────────────┬───────────────────────┘
                                                                   │
                                                                   ▼
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                                   QuFlow Python SDK Core                              │
│                                                                                          │
│  ┌────────────────────────┐  ┌────────────────────────┐  ┌─────────────────────────────┐ │
│  │   Circuit IR & Types   │  │  Hardware Transpiler   │  │   Adaptive Engine (MHOS)    │ │
│  │  - Circuit / Gate (v2) │─►│  - Cancellation / Merge│─►│  - Rule & Scoring Selector  │ │
│  │  - OpenQASM 2.0 / 3.0  │  │  - SingleQubitFusion   │  │  - Feature Extraction       │ │
│  │  - DAG / Fingerprint   │  │  - SABRE Layout/Routing│  │  - Pre-flight Memory Est.   │ │
│  └────────────────────────┘  └────────────────────────┘  └──────────────┬──────────────┘ │
│                                                                         │                │
│  ┌──────────────────────────────────────────────────────────────────────┴─────────────┐  │
│  │                                 Simulation & Execution Backends                    │  │
│  │                                                                                    │  │
│  │  ┌───────────────────────┐ ┌───────────────────────┐ ┌──────────────────────────┐ │  │
│  │  │ StatevectorSimulator  │ │  StabilizerSimulator  │ │  TensorNetworkBackend    │ │  │
│  │  │ - C++ Native Core     │ │ - Aaronson-Gottesman  │ │ - Matrix Product State   │ │  │
│  │  │ - einsum contraction  │ │ - O(N²) Clifford Tab. │ │ - Low-entanglement MPS   │ │  │
│  │  └───────────────────────┘ └───────────────────────┘ └──────────────────────────┘ │  │
│  │  ┌───────────────────────┐ ┌───────────────────────┐ ┌──────────────────────────┐ │  │
│  │  │ GPUStatevector (CuPy) │ │   Hardware Adapters   │ │ Error Mitigation Stack   │ │  │
│  │  │ - CUDA kernel dispatch│ │ - IBM Quantum Runtime │ │ - ZNE (Gate Folding)     │ │  │
│  │  │ - CPU NumPy fallback  │ │ - IonQ / Rigetti Mocks│ │ - PEC & Readout Matrix   │ │  │
│  │  └───────────────────────┘ └───────────────────────┘ └──────────────────────────┘ │  │
│  └────────────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                          │
│  ┌────────────────────────────────────────────────────────────────────────────────────┐  │
│  │                             Hybrid Quantum-Classical Layer                         │  │
│  │  - VQE Solver (`zuntenium.hybrid.solver.vqe`)                                      │  │
│  │  - QAOA Solver (`zuntenium.hybrid.solver.qaoa`)                                    │  │
│  │  - Parameter-Shift Gradient Estimator (`zuntenium.hybrid.optimizer`)               │  │
│  │  - PyTorch Tensor Integration (`zuntenium.pytorch`)                                │  │
│  └────────────────────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Core Subsystems & File Layout

### 2.1 Quantum IR & Data Structures
- **v2 Circuit IR**: [`zuntenium/transpiler/v2/core.py`](file:///home/vincent/Downloads/zuntenium/zuntenium/transpiler/v2/core.py)  
  High-performance typed data models for quantum gates, operations, parameter maps, and DAG dependencies.
- **Circuit Representation & Registries**: [`zuntenium/circuit/circuit.py`](file:///home/vincent/Downloads/zuntenium/zuntenium/circuit/circuit.py)  
  Quantum and classical register tracking, circuit serialization, bitwise fingerprinting, and validation.
- **OpenQASM Parser & Serializer**: [`zuntenium/circuit/ir/serialization/`](file:///home/vincent/Downloads/zuntenium/zuntenium/circuit/ir/serialization/)  
  Bidirectional transpilation between OpenQASM 2.0 / 3.0 and internal DAG structures.

### 2.2 Compilation, Routing & Transpilation Pipeline
- **Pass Engine**: [`zuntenium/transpiler/v2/engine.py`](file:///home/vincent/Downloads/zuntenium/zuntenium/transpiler/v2/engine.py)  
  Pass manager supporting optimization levels `0`, `1`, `2`, and `3`.
- **Optimization Passes**:
  - `RedundantGateCancellationPass`: Cancels inverse gate pairs ($H \cdot H = I$, $X \cdot X = I$).
  - `RotationMergingPass`: Fuses adjacent rotations along identical axes ($R_z(\alpha) \cdot R_z(\beta) = R_z(\alpha+\beta)$).
  - `SingleQubitFusionPass`: Collapses sequences of single-qubit unitaries into a single $ZYZ$ decomposition.
- **Hardware-Aware SABRE Routing**: [`zuntenium/transpiler/passes/mapping/sabre_router.py`](file:///home/vincent/Downloads/zuntenium/zuntenium/transpiler/passes/mapping/sabre_router.py)  
  Forward and backward lookahead heuristic router to satisfy coupling map topology constraints with minimum SWAP insertions.

### 2.3 Simulation Cores & Backends
- **C++ Native Extension**: [`bindings_core/cpp/`](file:///home/vincent/Downloads/zuntenium/bindings_core/cpp/)  
  Precompiled native C++ statevector core (`_zuntenium_core.so`) applying gates in-place.
- **Statevector Simulator**: [`zuntenium/quantum_core/statevector/`](file:///home/vincent/Downloads/zuntenium/zuntenium/quantum_core/statevector/)  
  Pure Python/NumPy statevector engine with `np.einsum` contraction and periodic norm drift monitoring.  
  Decomposed (O6) into focused modules: `simulator_backend.py` (ABC), `statevector_state.py` (array mgmt), `gate_library.py` (21+ gates), `gate_applicator.py` (tensor contraction), `sim_circuit.py` (fluent builder), `measurement_engine.py` (Born-rule sampling), and `statevector_simulator.py` (public facade).
- **Density Matrix Simulator**: [`zuntenium/backends/density_matrix_simulator.py`](file:///home/vincent/Downloads/zuntenium/zuntenium/backends/density_matrix_simulator.py)  
  Mixed-state simulation via ρ → UρU† evolution with Kraus channel support for noisy circuits (up to 15 qubits).
- **Stabilizer Simulator**: [`zuntenium/backends/stabilizer.py`](file:///home/vincent/Downloads/zuntenium/zuntenium/backends/stabilizer.py)  
  Aaronson-Gottesman binary tableau simulator ($O(N^2)$ gate scaling) for Clifford circuits up to 10,000 qubits.
- **Matrix Product State (MPS) Backend**: [`zuntenium/backends/tensor_network_backend.py`](file:///home/vincent/Downloads/zuntenium/zuntenium/backends/tensor_network_backend.py)  
  Tensor network simulation with configurable bond dimension $\chi$ and singular value truncation statistics.
- **GPU Accelerated Simulator**: [`zuntenium/backends/gpu_backend.py`](file:///home/vincent/Downloads/zuntenium/zuntenium/backends/gpu_backend.py)  
  CuPy / CUDA 12.x backend providing accelerated state sweeps with CPU NumPy fallback.

### 2.4 Multi-Hardware Orchestration System (MHOS) & Adaptive Engine
- **Circuit Feature Analyzer**: [`zuntenium/execution/adaptive_engine.py`](file:///home/vincent/Downloads/zuntenium/zuntenium/execution/adaptive_engine.py)  
  Extracts qubit count, depth, gate counts, Clifford-purity, entanglement metric, and estimated memory footprint.
- **Rule & Scoring Engine**: Dynamically routes workloads to Stabilizer, Tensor Network, Statevector, or Physical Hardware depending on circuit structure and user optimization preferences (`speed`, `accuracy`, `cost`).

### 2.5 Error Mitigation Stack
- **Zero Noise Extrapolation (ZNE)**: [`zuntenium/error_mitigation/mitigation.py`](file:///home/vincent/Downloads/zuntenium/zuntenium/error_mitigation/mitigation.py)  
  Unitary gate-folding ($G \to G (G^\dagger G)^k$) with polynomial and exponential extrapolation models.
- **Probabilistic Error Cancellation (PEC)**: Quasi-probability decomposition with explicit sampling overhead tracking ($\gamma^2$).
- **Readout Error Mitigation**: Inversion of calibration confusion matrices with Tikhonov regularization and simplex projection.

### 2.6 Hybrid Quantum-Classical & ML
- **VQE & QAOA**: [`zuntenium/hybrid/solver/`](file:///home/vincent/Downloads/zuntenium/zuntenium/hybrid/solver/)  
  Variational algorithms using classical optimizers (COBYLA, L-BFGS-B, SLSQP, Adam).
- **Gradient Estimators**: [`zuntenium/hybrid/optimizer/gradient_estimator.py`](file:///home/vincent/Downloads/zuntenium/zuntenium/hybrid/optimizer/gradient_estimator.py)  
  Parameter-shift analytic gradient evaluation ($2M$ circuit executions) and finite differences.
- **PyTorch Integration**: [`zuntenium/pytorch/zuntenium_pytorch.py`](file:///home/vincent/Downloads/zuntenium/zuntenium/pytorch/zuntenium_pytorch.py)  
  `torch.autograd.Function` bridge allowing quantum circuits inside differentiable neural networks.

### 2.7 Workspace & Cloud Infrastructure
- **FastAPI Cloud Gateway**: [`zuntenium_cloud/`](file:///home/vincent/Downloads/zuntenium/zuntenium_cloud/)  
  REST and OpenAPI documentation endpoints, JWT authentication, backend health checks, and Redis job queues.
- **QuFlow Workspace (Web UI)**: [`zuntenium-ui/`](file:///home/vincent/Downloads/zuntenium/zuntenium-ui/)  
  Next.js 15 visual circuit composer, gate palette, and interactive statevector inspector.
- **CRDT Real-Time Collaboration**: [`zuntenium-collab-server/`](file:///home/vincent/Downloads/zuntenium/zuntenium-collab-server/)  
  Conflict-free replicated data type WebSocket server for simultaneous multi-user circuit editing.
- **VS Code Extension**: [`zuntenium-vscode/`](file:///home/vincent/Downloads/zuntenium/zuntenium-vscode/)  
  Language server, syntax highlighting, circuit runner, and diagram preview packaged into `zuntenium-vscode-0.1.0.vsix`.
