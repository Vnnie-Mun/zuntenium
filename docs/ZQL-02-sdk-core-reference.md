# ZQL-02 | SDK Core Reference

## Learning Objectives
- Understand Zuntenium's Core Architecture.
- Master the `Circuit` and `Gate` builders.
- Use the Transpiler v2 (SABRE, KAK, Solovay-Kitaev).
- Execute jobs across Multi-Hardware Orchestration System (MHOS).

---

## 1. Core Architecture

Zuntenium relies on a decoupled architecture:
1. **Frontend (IR):** The `Circuit` class builds an intermediate representation (IR).
2. **Optimization (Transpiler):** Engine v2 rewrites the IR for specific hardware constraints (Coupling Maps, Native Gates).
3. **Execution (Backends/MHOS):** Simulators (Statevector, Tensor Network) or Real QPUs (IBM, IonQ, Rigetti) execute the optimized IR.

---

## 2. Circuit Building and IR

Zuntenium's `Circuit` object is the primary frontend structure.

### Supported Gates
Import gates from `zuntenium.quantum_core.gates`:

- **Single Qubit:** `H`, `X`, `Y`, `Z`, `S`, `T`, `RX`, `RY`, `RZ`
- **Two Qubit:** `CNOT`, `CZ`, `SWAP`, `RXX`, `RYY`, `RZZ`
- **Parameterized:** Use `RX(params=(theta,))` for variational algorithms.

### Serialization
Zuntenium seamlessly parses and exports QASM 2.0:
```python
from zuntenium.circuit.ir.serialization.qasm_parser import parse_qasm2
# Load from string
circuit = parse_qasm2("OPENQASM 2.0; qreg q[2]; H q[0]; CX q[0], q[1];")
```

---

## 3. Transpiler Engine v2

Zuntenium features a highly optimized transpiler suite.

### SABRE Routing
If you run on hardware with limited connectivity (e.g., IBM heavy-hex), you must route your qubits. Zuntenium uses a SABRE (Swap-based BidiRectional heuristic) algorithm.

```python
from zuntenium.transpiler.passes.mapping.sabre_router import SABRERouter
from zuntenium.transpiler.coupling_map import CouplingMap

# Define a 3-qubit linear chain: 0-1-2
cmap = CouplingMap([(0,1), (1,0), (1,2), (2,1)])
router = SABRERouter(coupling_map=cmap)

routed_circuit = router.run(original_circuit)
```

### Transpiler Cache
In VQE and QAOA, the circuit structure is constant while parameters change. `TranspilerCache` avoids O(N) re-transpilation by performing O(1) parameter rebinding.

---

## 4. Multi-Hardware Orchestration System (MHOS)

MHOS manages unified execution across vendors.

### Supported Providers
- `IBM`: IBM Quantum Runtime (requires `IBM_QUANTUM_TOKEN`)
- `IonQ`: IonQ Cloud (requires `IONQ_API_KEY`)
- `Rigetti`: Rigetti QCS (requires `RIGETTI_API_KEY`)
- `Braket`: AWS Braket (requires AWS credentials)

### Example: Running on IBM
```python
from zuntenium.mhos.orchestrator import MHOSOrchestrator

orchestrator = MHOSOrchestrator()
# Assuming credentials are in env vars
job = orchestrator.submit(circuit, provider="ibm", device="ibm_kyoto", shots=2000)
result = orchestrator.wait_for_job(job.job_id)
print("Hardware counts:", result.counts)
```

---

## 5. Simulators

If running locally, you can choose specialized engines:

- `statevector_simulator`: Dense matrix multiplication (fastest for ≤30 qubits; limit is configurable via `max_qubits`).
- `gpu_backend` (CuPy): GPU accelerated statevector.
- `tensor_network_backend`: Matrix Product State (MPS) ideal for low-entanglement QAOA.
- `stabilizer`: Clifford-only simulator capable of $10^4$ qubits.

```python
from zuntenium.api.quantum_api import execute, get_backend
# Use MPS for large shallow circuits
execute(circuit, backend=get_backend("tensor_network_backend"))
```

## Next Steps
- Learn how to use Zuntenium's built-in Error Mitigation (ZNE/PEC) in [ZQL-11 | Error Mitigation](ZQL-11-error-mitigation.md).
- Dive into hybrid optimization with [ZQL-05 | Variational and Hybrid](ZQL-05-variational-hybrid.md).
