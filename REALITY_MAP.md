# REALITY MAP: Zuntenium Quantum SDK vs. Computational & Physical Realities

This document maps the Zuntenium Quantum SDK and Workspace against the 12 governing computational and physical facts (R1–R12). Every claim is backed by repository evidence, code locations, and empirical verification.

---

## Reality Matrix Summary

| ID | Governing Reality | Current SDK Status | Evidence Citation | Gap Rating |
|:---|:---|:---|:---|:---:|
| **R1** | Memory is exponential ($2^n$, $4^n$) | Pre-flight memory estimator exists; `MAX_QUBITS = 30` (configurable via `max_qubits` kwarg); no `complex64` mode | [`statevector_simulator.py:208,276`](file:///home/vincent/Downloads/zuntenium/zuntenium/quantum_core/statevector/statevector_simulator.py#L208) | **moderate** |
| **R2** | Dense simulation is memory-bandwidth-bound | Transpiler single-qubit fusion exists; runtime gate fusion absent; C++ kernel loops branch on every index | [`statevector.cpp:45`](file:///home/vincent/Downloads/zuntenium/bindings_core/cpp/src/statevector.cpp#L45), [`gate_ops.cpp:45`](file:///home/vincent/Downloads/zuntenium/bindings_core/cpp/src/gate_ops.cpp#L45) | **weak** |
| **R3** | Structure beats brute force | Stabilizer (Aaronson-Gottesman) & MPS backends implemented; adaptive engine auto-selects | [`adaptive_engine.py:137-156`](file:///home/vincent/Downloads/zuntenium/zuntenium/execution/adaptive_engine.py#L137), [`stabilizer.py:24`](file:///home/vincent/Downloads/zuntenium/zuntenium/backends/stabilizer.py#L24) | **good** |
| **R4** | Distribution has communication cost | Single-node CPU/GPU (CuPy) only; zero multi-node/MPI/NCCL distributed execution | [`gpu_backend.py:30-50`](file:///home/vincent/Downloads/zuntenium/zuntenium/backends/gpu_backend.py#L30) | **weak** |
| **R5** | Interpreter overhead is real | C++ core exists (`_zuntenium_core.so`); cold-start import is 221 ms; AST traversal on execution | Benchmark: 221.01 ms cold import; [`statevector.cpp`](file:///home/vincent/Downloads/zuntenium/bindings_core/cpp/src/statevector.cpp) | **weak** |
| **R6** | Compilation problems are NP-hard | SABRE heuristic router implemented; rotation & cancellation passes; rebind requires AST sweep | [`sabre_router.py:50`](file:///home/vincent/Downloads/zuntenium/zuntenium/transpiler/passes/mapping/sabre_router.py#L50), [`cache.py`](file:///home/vincent/Downloads/zuntenium/zuntenium/transpiler/cache.py) | **good** |
| **R7** | Compiled output must be equivalent | Unitary distance metrics exist; endianness documented in `ExecutionResult`; automated CI equivalence missing | [`distance.py:9-39`](file:///home/vincent/Downloads/zuntenium/zuntenium/utils/linalg/distance.py#L9), [`result.py:98-101`](file:///home/vincent/Downloads/zuntenium/zuntenium/quantum_core/executor/result.py#L98) | **weak** |
| **R8** | Floating point has limits | Drift monitor runs every 100 gates (`norm_sq - 1.0 > 1e-12`); defensive re-normalization active | [`statevector_simulator.py:663-668`](file:///home/vincent/Downloads/zuntenium/zuntenium/quantum_core/statevector/statevector_simulator.py#L663) | **good** |
| **R9** | Sampling noise is statistical | Standard errors on `ExecutionResult` counts; expectation values return scalar without error bars | [`result.py:103-118`](file:///home/vincent/Downloads/zuntenium/zuntenium/quantum_core/executor/result.py#L103), [`expectation.py:14-30`](file:///home/vincent/Downloads/zuntenium/zuntenium/hybrid/observable/expectation.py#L14) | **weak** |
| **R10**| Hardware error compounds multiplicatively | Coupling maps & link error scoring in SABRE; IBM runtime connected; no dynamical decoupling | [`hardware_transpiler.py:19-60`](file:///home/vincent/Downloads/zuntenium/zuntenium/transpiler/hardware_transpiler.py#L19) | **weak** |
| **R11**| Variational repetition dominates | Parameter-shift gradients cost $2M$ runs; no adjoint differentiation on simulator | [`gradient_estimator.py:47-60`](file:///home/vincent/Downloads/zuntenium/zuntenium/hybrid/optimizer/gradient_estimator.py#L47) | **weak** |
| **R12**| Mitigation trades bias for variance | ZNE, PEC, and Readout mitigation implemented; sampling overhead $\gamma^2$ explicitly reported | [`mitigation.py:15-33,75`](file:///home/vincent/Downloads/zuntenium/zuntenium/error_mitigation/mitigation.py#L15) | **good** |

---

## Detailed Audit by Reality ID

### R1: Memory is Exponential ($2^n$ Amplitudes, $4^n$ Density Matrices)
- **Physical Law**: A statevector of $n$ qubits requires $2^n \times 16$ bytes in `complex128` (or 8 bytes in `complex64`). 30 qubits requires 16 GiB; 34 qubits requires 256 GiB. Density matrices require $4^n$ entries (15 qubits = 16 GiB in `complex128`).
- **Current Behavior**:
  - `CircuitFeatures` in [`adaptive_engine.py:92`](file:///home/vincent/Downloads/zuntenium/zuntenium/execution/adaptive_engine.py#L92) estimates memory as `16.0 * (2 ** circuit.num_qubits) / (1024.0 ** 2)` MB.
  - `StatevectorState` in [`statevector_simulator.py:208`](file:///home/vincent/Downloads/zuntenium/zuntenium/quantum_core/statevector/statevector_simulator.py#L208) has a configurable guard: `MAX_QUBITS: int = 30` (overridable via `max_qubits=` kwarg). Psutil-based pre-flight memory check is the primary constraint.
  - Precision mode is fixed to `complex128`. No `complex64` mode is exposed to halve memory usage.
- **Gap Rating**: **weak**
- **Actionable Levers**:
  1. Add configurable precision modes (`precision="single"` for `complex64`, `precision="double"` for `complex128`).
  2. ~~Replace hardcoded `MAX_QUBITS = 20`~~ **DONE** — `MAX_QUBITS` raised to 30 and made configurable via `StatevectorState(n, max_qubits=25)`. Psutil pre-flight check is the primary memory guard.

---

### R2: Dense Simulation is Memory-Bandwidth-Bound
- **Physical Law**: Arithmetic intensity of quantum gate application is tiny ($\approx 1$ complex multiply-add per 16 bytes moved from RAM). Performance is bounded by memory bus throughput (GB/s), not CPU peak TFLOPs.
- **Current Behavior**:
  - In pure Python, [`statevector_simulator.py:740`](file:///home/vincent/Downloads/zuntenium/zuntenium/quantum_core/statevector/statevector_simulator.py#L740) calls `np.einsum(...)` for each 2-qubit gate, generating intermediate arrays and performing full memory sweeps per gate.
  - In C++, [`gate_ops.cpp:45-60`](file:///home/vincent/Downloads/zuntenium/bindings_core/cpp/src/gate_ops.cpp#L45) implements `apply_two_qubit_op` with `for (size_t i = 0; i < n; ++i)` testing `if ((i & bit0) == 0 && (i & bit1) == 0)`—branching on every single index of $2^n$ rather than stepping only across valid index strides.
  - Gate fusion occurs only in the transpiler via `SingleQubitFusionPass`. Dynamic runtime fusion of 2-qubit gate blocks before state sweeping is not implemented.
- **Gap Rating**: **weak**
- **Actionable Levers**:
  1. Rewrite C++ 2-qubit gate loops to iterate directly over $2^{n-2}$ indices with bit-manipulation strides, eliminating branch mispredictions.
  2. Implement runtime gate batching/fusion (combining consecutive commuting operations on disjoint/overlapping qubits into fused matrix operators).
  3. Add AVX2/AVX-512 vectorization and OpenMP multithreading to C++ kernels.

---

### R3: Structure Beats Brute Force
- **Physical Law**: Stabilizer (Clifford-only) circuits simulate in polynomial time $O(n^2)$ via Gottesman-Knill theorem. Low-entanglement states simulate in polynomial time via Matrix Product States (MPS) where cost scales with bond dimension $\chi$, not $2^n$.
- **Current Behavior**:
  - `StabilizerSimulator` in [`stabilizer.py:24`](file:///home/vincent/Downloads/zuntenium/zuntenium/backends/stabilizer.py#L24) implements the complete Aaronson-Gottesman binary tableau simulator ($2n \times (2n+1)$).
  - `TensorNetworkBackend` in [`tensor_network_backend.py:1`](file:///home/vincent/Downloads/zuntenium/zuntenium/backends/tensor_network_backend.py#L1) simulates MPS with singular-value decomposition (SVD) and truncation tracking.
  - `RuleEngine` in [`adaptive_engine.py:143-155`](file:///home/vincent/Downloads/zuntenium/zuntenium/execution/adaptive_engine.py#L143) inspects circuit features and automatically routes Clifford circuits to `stabilizer`, low-entanglement circuits to `tensornetwork`, and noisy circuits to `densitymatrix`.
  - `DensityMatrixSimulator` in [`density_matrix_simulator.py`](file:///home/vincent/Downloads/zuntenium/zuntenium/backends/density_matrix_simulator.py) provides mixed-state ρ → UρU† simulation with Kraus noise channels (up to 15 qubits).
- **Gap Rating**: **good**
- **Actionable Levers**:
  1. Expand Clifford angle detection to include decomposed Phase, T-gate Clifford synthesis where applicable.

---

### R4: Distribution Has a Communication Cost
- **Physical Law**: Distributing statevectors across cluster nodes or multi-GPU environments requires cross-node interconnect transfers whenever gates act on "global" qubits. Scaling is bottlenecked by network/PCIe bandwidth.
- **Current Behavior**:
  - `GPUStatevectorSimulator` in [`gpu_backend.py`](file:///home/vincent/Downloads/zuntenium/zuntenium/backends/gpu_backend.py) supports single-GPU via CuPy with fallback to NumPy CPU.
  - No multi-GPU or multi-node MPI/NCCL distributed execution exists in the SDK.
- **Gap Rating**: **weak**
- **Actionable Levers**:
  1. Profile single-node baseline before investing in distributed execution.
  2. When implementing distributed execution, schedule gate passes to reorder qubits so gates act locally on device chunks before exchanging global halves.

---

### R5: Interpreter Overhead is Real
- **Physical Law**: Python per-gate function dispatch, object allocation, and dictionary lookups introduce microsecond latencies that swamp simulation time on shallow circuits and variational iterations.
- **Current Behavior**:
  - Cold import time measured at **221.01 ms** (`import zuntenium`).
  - Precompiled C++ extension `_zuntenium_core.so` is present, but Python API wrappers construct intermediate tuples and dicts during gate submission.
- **Gap Rating**: **weak**
- **Actionable Levers**:
  1. Reduce cold-start import time by lazy-loading heavy scientific libraries (`qiskit`, `networkx`, `torch`).
  2. Add batched gate evaluation APIs in C++ to accept a flat circuit instruction buffer, bypassing Python dispatch per gate.

---

### R6: Compilation Problems are NP-Hard
- **Physical Law**: Optimal qubit mapping onto constrained hardware coupling maps is NP-hard. Exact compilation requires exponential search; scalable transpilers require heuristics with predictable trade-offs between compile time and circuit quality.
- **Current Behavior**:
  - SABRE heuristic router in [`sabre_router.py`](file:///home/vincent/Downloads/zuntenium/zuntenium/transpiler/passes/mapping/sabre_router.py) provides forward/backward lookahead heuristic scoring.
  - `Transpiler` supports levels `0`, `1`, `2`, `3` with gate cancellation, rotation merging, and single-qubit fusion.
  - Parameter rebinding currently re-traverses circuit instructions.
- **Gap Rating**: **good**
- **Actionable Levers**:
  1. Benchmark SABRE SWAP overhead against Qiskit SABRE and tket on identical coupling graphs.
  2. Implement zero-recompile parameter binding tables for fast VQE/QAOA parameter updates.

---

### R7: Compiled Output Must Be Equivalent
- **Physical Law**: Transpiler passes must preserve state fidelity ($F = |\langle\psi|\phi\rangle|^2 = 1.0$) up to global phase. Common bugs arise from qubit endianness inversions and rotation sign conventions.
- **Current Behavior**:
  - `distance.py` provides `state_fidelity`, `trace_distance`, and `bures_distance`.
  - `ExecutionResult` explicitly declares `endianness: "little-endian (q0 is rightmost bit)"` in [`result.py:98-101`](file:///home/vincent/Downloads/zuntenium/zuntenium/quantum_core/executor/result.py#L98).
  - Automated unitary distance assertions are present in transpiler tests, but not yet integrated as a CI gate verifying random circuits across all passes.
- **Gap Rating**: **weak**
- **Actionable Levers**:
  1. Build a strict property-based equivalence CI gate enforcing unitary distance $< 10^{-10}$ on random and structured circuits.
  2. Add explicit convention tests validating little-endian basis states against Qiskit and Cirq.

---

### R8: Floating Point Has Limits
- **Physical Law**: Standard IEEE 754 float64 machine epsilon is $\approx 2.22 \times 10^{-16}$. Numerical errors accumulate across deep circuits, leading to norm drift and loss of unitarity.
- **Current Behavior**:
  - `StatevectorSimulator` in [`statevector_simulator.py:663-668`](file:///home/vincent/Downloads/zuntenium/zuntenium/quantum_core/statevector/statevector_simulator.py#L663) monitors norm drift every 100 gates:
    ```python
    norm_sq = float(np.real(np.dot(state.conj(), state)))
    if abs(norm_sq - 1.0) > 1e-12:
        state /= math.sqrt(norm_sq)
    ```
  - `ExecutionResult.norm_drift` exposes $|\Vert\psi\Vert - 1|$ directly to the user.
- **Gap Rating**: **good**
- **Actionable Levers**:
  1. Extend drift monitoring to GPU and MPS simulation backends.
  2. Document accumulation bounds for deep circuit depths.

---

### R9: Sampling Noise is Statistical and Unavoidable
- **Physical Law**: Expectation values computed from $N$ measurement shots have standard error $\sigma \le 1/\sqrt{N}$. Error $0.01$ requires $\approx 10^4$ shots; error $0.001$ requires $\approx 10^6$ shots. Reporting point estimates without error bars is unscientific.
- **Current Behavior**:
  - `ExecutionResult` in [`result.py:103-118`](file:///home/vincent/Downloads/zuntenium/zuntenium/quantum_core/executor/result.py#L103) provides `standard_errors` and `error_bars` based on $\sigma_k = \sqrt{p_k(1-p_k)/N}$.
  - `compute_expectation()` in [`expectation.py:14-30`](file:///home/vincent/Downloads/zuntenium/zuntenium/hybrid/observable/expectation.py#L14) only calculates the exact statevector inner product $\langle\psi|O|\psi\rangle$ and lacks shot-based variance and confidence intervals.
- **Gap Rating**: **weak**
- **Actionable Levers**:
  1. Update all expectation value APIs to return tuple `(value, stderr)` with sample variance $\sigma/\sqrt{N}$ when shots $> 0$.
  2. Implement Pauli grouping (commuting Pauli cliques) to measure multiple Hamiltonian terms simultaneously from the same shots.

---

### R10: Hardware Error Compounds Multiplicatively
- **Physical Law**: Physical circuit fidelity degrades as $F \approx \prod (1 - \epsilon_g) \times \prod (1 - \epsilon_{readout})$. Reducing two-qubit gates directly boosts hardware output fidelity.
- **Current Behavior**:
  - `HardwareConfig` and `HardwareAwareTranspiler` incorporate coupling maps and link error rates into SABRE path weights in [`hardware_transpiler.py:19-60`](file:///home/vincent/Downloads/zuntenium/zuntenium/transpiler/hardware_transpiler.py#L19).
  - Dynamic polling of live calibration data is static/mock-based.
  - Dynamical decoupling (DD) for idle qubits is not implemented.
- **Gap Rating**: **weak**
- **Actionable Levers**:
  1. Implement dynamical decoupling (XY4 / CPMG pulse insertion on idle qubit wires).
  2. Add pre-flight estimated circuit fidelity calculation ($F_{est} = \prod (1 - \epsilon_i)$) warning users when expected fidelity falls below 1%.

---

### R11: Variational Workloads are Dominated by Repetition
- **Physical Law**: Parameter-shift rule requires $2$ circuit evaluations per parameter per variational step. For a 50-parameter ansatz, each gradient evaluation requires 100 circuit executions. Queue time and classical overhead dominate total runtime.
- **Current Behavior**:
  - `GradientEstimator` in [`gradient_estimator.py:51-60`](file:///home/vincent/Downloads/zuntenium/zuntenium/hybrid/optimizer/gradient_estimator.py#L51) implements standard parameter-shift rule requiring $2M$ circuit runs.
  - Adjoint differentiation is absent on simulators.
- **Gap Rating**: **weak**
- **Actionable Levers**:
  1. Implement simulator adjoint differentiation (evaluates all $\nabla_\theta \langle H \rangle$ in a single backward pass through statevector ops, yielding $10\times\text{--}50\times$ speedup on VQE/QAOA).
  2. Implement circuit batching in `execute_batch()` to submit all shifted parameter points in a single vectorized call.

---

### R12: Mitigation Trades Bias for Variance
- **Physical Law**: Quantum error mitigation (ZNE, PEC) suppresses systematic hardware bias at the expense of inflating variance. PEC shot overhead scales exponentially as $\gamma^2$ with error rate and depth.
- **Current Behavior**:
  - `MitigatedResult` in [`mitigation.py:75`](file:///home/vincent/Downloads/zuntenium/zuntenium/error_mitigation/mitigation.py#L75) explicitly tracks `overhead_factor` ($\gamma^2$ for PEC, number of scales for ZNE).
  - Both raw probabilities, corrected probabilities, and fidelity before/after are preserved and logged.
- **Gap Rating**: **good**
- **Actionable Levers**:
  1. Add pre-flight variance overhead check warning users if $\gamma^2 > 1000$.
