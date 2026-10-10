# ZUNTENIUM API Reference

## Top-Level API

### `execute(circuit, backend=None, shots=1024, **options)`

Execute a quantum circuit and return results.

**Args:**
- `circuit`: `QuantumCircuit` or `QuantumIR`
- `backend`: `AbstractBackend` instance. Default: `StatevectorBackend()`
- `shots`: Number of measurement samples
- `**options`: Backend-specific kwargs (`optimization_level`, `seed`, ...)

**Returns:** `ExecutionResult`

### `get_backend(name, **kwargs)`

| Name | Backend |
|------|---------|
| `"statevector_simulator"` | `StatevectorBackend` |
| `"unitary"` | `UnitaryBackend` |
| `"snapshot"` | `SnapshotBackend` |
| `"mock"` | `MockHardwareBackend` |

---

## QuantumCircuit

```python
qc = QuantumCircuit(num_qubits, num_clbits=0, name="")
```

**Single-qubit gates:** `h`, `x`, `y`, `z`, `s`, `t`, `rx`, `ry`, `rz`, `u`, `u1`, `u2`

**Two-qubit gates:** `cx`, `cz`, `cy`, `swap`, `iswap`, `crx`, `cry`, `crz`

**Three-qubit gates:** `ccx`, `cswap`

**Operations:** `measure`, `measure_all`, `barrier`, `reset`, `compose`, `append`

**Properties:** `num_qubits`, `num_clbits`, `num_gates`, `depth`, `name`

---

## ExecutionResult

```python
result.counts          # dict[str, int] — bitstring → count
result.shots           # int
result.statevector     # np.ndarray | None
result.probabilities   # dict[str, float]
result.most_probable   # str — most frequent outcome
result.metadata        # dict — backend, exec_time_ms, …
result.expectation_value(observable)
result.plot_histogram()
```

---

## TranspilerPipeline

```python
pipeline = TranspilerPipeline(level=2)   # 0–3
ir_opt = pipeline.run(ir, target=backend.target)
pipeline.add_pass(MyCustomPass())
pipeline.remove_pass("GateCancellation")
```

---

## VQE / QAOA

```python
from zuntenium.hybrid import VQE, QAOA, Hamiltonian, PauliString

H = Hamiltonian([(0.5, PauliString("ZZ")), (-1.0, PauliString("XI"))])

result = VQE(ansatz, H).run(initial_params)
result.energy           # float
result.optimal_params   # np.ndarray
result.convergence_history  # list[float]
result.plot_convergence()
```

---

## Exceptions

| Exception | When raised |
|-----------|------------|
| `QubitIndexError` | Qubit index out of range |
| `CircuitValidationError` | Circuit fails pre-compile checks |
| `CircuitCompilationError` | IR compilation failure |
| `BackendNotFoundError` | Unknown backend name |
| `QASMParseError` | Invalid QASM input |
| `GateNotSupportedError` | Gate not in backend's native set |
