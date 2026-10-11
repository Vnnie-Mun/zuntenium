# ZUNTENIUM Data Flow

## End-to-End Circuit Execution

```
User Code
  │
  │  qc = QuantumCircuit(2)
  │  qc.h(0).cx(0, 1).measure_all()
  ▼
QuantumCircuit.compile()
  │  Validates qubit bounds, gate support
  │  Builds list[Instruction] → QuantumIR
  ▼
QuantumIR
  │  num_qubits=2, instructions=[H(0), CX(0,1), Measure(0,1)]
  ▼
TranspilerPipeline.run(ir, target=backend.target)
  │  Pass 1: GateCancellation    → removes H·H pairs
  │  Pass 2: RotationMerging     → merges adjacent Rz
  │  Pass 3: SingleQubitUnify    → 1-qubit sequence → U3
  │  Pass 4: AI stub (no-op v0.1)
  ▼
optimized QuantumIR
  ▼
AbstractBackend.run(optimized_ir, shots=1024)
  │
  ├─(simulator path)──────────────────────────────────────────┐
  │                                                           │
  │  CircuitExecutor.execute(ir, shots)                       │
  │    StateVector(2)  → |00⟩                                │
  │    apply H(0)      → (|00⟩ + |10⟩)/√2                   │
  │    apply CX(0,1)   → (|00⟩ + |11⟩)/√2  (Bell state)    │
  │    measure(shots)  → sample amplitudes²                  │
  │                                                           │
  │    <20q → NumPy kernel                                    │
  │    >20q → C++ StateVectorCPP via pybind11                │
  │                                                           │
  └─(hardware path)────────────────────────────────────────┐ │
                                                           │ │
     HardwareAdapter.translate_ir()                        │ │
       IR → native circuit format (QASM / JSON)            │ │
     QuantumJob.submit() → poll status → fetch result      │ │
                                                           │ │
  ◄──────────────────────────────────────────────────────── ┘ ┘
  │
ExecutionResult
  │  .counts     = {'00': 512, '11': 512}
  │  .statevector = numpy complex128 array (sim only)
  │  .metadata   = {shots, backend_name, exec_time_ms, ...}
  ▼
User Code
  result.counts
  result.plot_histogram()
```

---

## Hybrid VQE Data Flow

```
VQE.run(initial_params)
  │
  ╔══════════════════════════════════════════════════════╗
  ║  Classical Optimization Loop  (scipy.optimize)       ║
  ║                                                      ║
  ║  params ──► ansatz.bind_parameters(params)           ║
  ║               │                                      ║
  ║               ▼                                      ║
  ║         QuantumCircuit (bound)                       ║
  ║               │                                      ║
  ║               ▼                                      ║
  ║         Backend.run(circuit, shots)                  ║
  ║               │                                      ║
  ║               ▼                                      ║
  ║         ExecutionResult                              ║
  ║               │                                      ║
  ║               ▼                                      ║
  ║  Hamiltonian.expectation_value(result)               ║
  ║    ⟨ψ|H|ψ⟩ = Σ coeff_i · ⟨PauliString_i⟩          ║
  ║               │                                      ║
  ║               ▼                                      ║
  ║         energy (float) ──► optimizer.step()          ║
  ║                                                      ║
  ║         Gradient via parameter-shift rule:           ║
  ║         ∂E/∂θ = [E(θ+π/2) - E(θ-π/2)] / 2          ║
  ║         → 2M circuit evaluations per gradient step  ║
  ╚══════════════════════════════════════════════════════╝
  │
HybridResult
  │  .energy          = -1.137 (converged)
  │  .optimal_params  = [0.42, 1.17, ...]
  │  .convergence_history = [list of energies per iter]
  │  .num_iterations  = 87
  ▼
User Code
  print(result.energy)
  result.plot_convergence()
```

---

## IR Serialization Flow

```
QuantumIR
  │
  ├── to_qasm2()  →  OpenQASM 2.0 string  →  .qasm file
  ├── to_qasm3()  →  OpenQASM 3.0 string  →  .qasm file
  ├── to_json()   →  JSON dict  →  .json file  →  REST API transport
  └── to_dag()    →  DAGCircuit  →  depth/critical path analysis
```
