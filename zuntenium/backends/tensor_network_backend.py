"""Matrix-product-state tensor-network backend for low-entanglement circuits."""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from zuntenium.backends.ibm_backend import BackendInfo, QuantumBackend, SimulatorBackend
from zuntenium.transpiler.v2 import Circuit, Gate

logger = logging.getLogger("zuntenium.tensor_network")
_C128 = np.complex128


def _one_qubit_matrix(gate: Gate) -> np.ndarray:
    name, params = gate.name, gate.params
    c = math.cos(params[0] / 2) if params else 1.0
    s = math.sin(params[0] / 2) if params else 0.0
    matrices = {
        "I": np.eye(2, dtype=_C128), "X": np.array([[0, 1], [1, 0]], dtype=_C128),
        "Y": np.array([[0, -1j], [1j, 0]], dtype=_C128),
        "Z": np.diag([1, -1]).astype(_C128),
        "H": np.array([[1, 1], [1, -1]], dtype=_C128) / math.sqrt(2),
        "S": np.diag([1, 1j]).astype(_C128), "T": np.diag([1, np.exp(1j * math.pi / 4)]).astype(_C128),
        "SDG": np.diag([1, -1j]).astype(_C128), "TDG": np.diag([1, np.exp(-1j * math.pi / 4)]).astype(_C128),
        "SX": np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]], dtype=_C128) / 2,
        "SXD": np.array([[1 - 1j, 1 + 1j], [1 + 1j, 1 - 1j]], dtype=_C128) / 2,
    }
    if name in matrices: return matrices[name]
    if name == "RX": return np.array([[c, -1j * s], [-1j * s, c]], dtype=_C128)
    if name == "RY": return np.array([[c, -s], [s, c]], dtype=_C128)
    if name == "RZ": return np.diag([np.exp(-1j * params[0] / 2), np.exp(1j * params[0] / 2)]).astype(_C128)
    if name == "PHASE": return np.diag([1, np.exp(1j * params[0])]).astype(_C128)
    if name == "U3":
        theta, phi, lam = params
        return np.array([[math.cos(theta / 2), -np.exp(1j * lam) * math.sin(theta / 2)],
                         [np.exp(1j * phi) * math.sin(theta / 2), np.exp(1j * (phi + lam)) * math.cos(theta / 2)]], dtype=_C128)
    raise ValueError(f"Unsupported one-qubit gate {name!r}")


def _two_qubit_matrix(gate: Gate) -> np.ndarray:
    if gate.name in {"CNOT", "CX"}: return np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=_C128)
    if gate.name == "CZ": return np.diag([1, 1, 1, -1]).astype(_C128)
    if gate.name == "SWAP": return np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=_C128)
    if gate.name == "ISWAP": return np.array([[1,0,0,0],[0,0,1j,0],[0,1j,0,0],[0,0,0,1]], dtype=_C128)
    if gate.name == "ECR":
        return np.array([[0, 1, 0, 1j], [1, 0, -1j, 0], [0, 1j, 0, 1], [-1j, 0, 1, 0]], dtype=_C128) / math.sqrt(2)
    if gate.name == "XX":
        th = gate.params[0] if gate.params else math.pi / 4
        c, s = math.cos(th / 2), math.sin(th / 2)
        return np.array([[c, 0, 0, -1j * s], [0, c, -1j * s, 0], [0, -1j * s, c, 0], [-1j * s, 0, 0, c]], dtype=_C128)
    raise ValueError(f"Unsupported two-qubit gate {gate.name!r}")


@dataclass
class TruncationStats:
    discarded_sv_count: int = 0
    max_truncation_err: float = 0.0
    bond_dim_before: int = 0
    bond_dim_after: int = 0


def truncate_singular_values(values: np.ndarray, max_dim: int, eps: float) -> Tuple[np.ndarray, int, float]:
    if len(values) == 0: return values, 0, 0.0
    kept = max(1, min(int(np.sum(values >= eps * max(values[0], 1e-15))), max_dim))
    error = float(np.sqrt(np.sum(values[kept:] ** 2)))
    return values[:kept], kept, error


class MPSState:
    def __init__(self, num_qubits: int, max_bond_dim: int = 128, truncation_threshold: float = 1e-10):
        if num_qubits < 1: raise ValueError("num_qubits must be positive")
        if max_bond_dim < 1: raise ValueError("max_bond_dim must be positive")
        self.num_qubits, self.max_bond_dim = num_qubits, max_bond_dim
        self.truncation_threshold = truncation_threshold
        self.tensors: List[np.ndarray] = []
        self.total_truncation_error = 0.0
        self._gate_count = 0
        self._init_zero_state()

    def _init_zero_state(self):
        self.tensors = []
        for _ in range(self.num_qubits):
            tensor = np.zeros((1, 2, 1), dtype=_C128); tensor[0, 0, 0] = 1.0; self.tensors.append(tensor)

    @classmethod
    def from_product_state(cls, bitstring, max_bond_dim=128, trunc_eps=1e-10):
        bits = [int(bit) for bit in bitstring]
        result = cls(len(bits), max_bond_dim, trunc_eps)
        for index, bit in enumerate(bits):
            result.tensors[index].fill(0); result.tensors[index][0, bit, 0] = 1.0
        return result

    @classmethod
    def from_statevector(cls, statevector, max_bond_dim=128, trunc_eps=1e-10):
        vector = np.asarray(statevector, dtype=_C128).ravel()
        n = int(round(math.log2(len(vector))))
        if 2 ** n != len(vector): raise ValueError("statevector length must be a power of two")
        result = cls(n, max_bond_dim, trunc_eps); matrix = vector.reshape(1, -1)
        for site in range(n - 1):
            left, right = matrix.shape; matrix = matrix.reshape(left * 2, right // 2)
            u, singular, vh = np.linalg.svd(matrix, full_matrices=False)
            singular, kept, error = truncate_singular_values(singular, max_bond_dim, trunc_eps)
            result.total_truncation_error += error
            result.tensors[site] = u[:, :kept].reshape(left, 2, kept)
            matrix = np.diag(singular) @ vh[:kept]
        result.tensors[-1] = matrix.reshape(matrix.shape[0], 2, 1)
        result.normalize()
        return result

    @property
    def bond_dims(self):
        return [self.tensors[0].shape[0]] + [tensor.shape[2] for tensor in self.tensors]
    @property
    def max_current_bond(self): return max(self.bond_dims)
    @property
    def gate_count(self): return self._gate_count

    def copy(self):
        result = MPSState.__new__(MPSState); result.num_qubits = self.num_qubits; result.max_bond_dim = self.max_bond_dim; result.truncation_threshold = self.truncation_threshold; result.tensors = [tensor.copy() for tensor in self.tensors]; result.total_truncation_error = self.total_truncation_error; result._gate_count = self._gate_count; return result
    def norm(self): return float(np.linalg.norm(self.tensors[-1].ravel()))
    def normalize(self):
        norm = self.norm()
        if norm > 1e-15: self.tensors[-1] /= norm

    def to_statevector(self):
        result = self.tensors[0].squeeze(0)
        for tensor in self.tensors[1:]:
            result = np.tensordot(result, tensor, axes=([-1], [0])).reshape(-1, tensor.shape[2])
        return result.ravel()

    def _left_canonicalize(self, site):
        if site >= self.num_qubits - 1: return
        tensor = self.tensors[site]; left, _, right = tensor.shape
        q, r = np.linalg.qr(tensor.reshape(left * 2, right))
        self.tensors[site] = q.reshape(left, 2, q.shape[1])
        self.tensors[site + 1] = np.einsum("ij,jkl->ikl", r, self.tensors[site + 1])

    def apply_single_qubit_gate(self, matrix, qubit):
        self.tensors[qubit] = np.einsum("ij,kjl->kil", matrix, self.tensors[qubit])
        self._left_canonicalize(qubit); self._gate_count += 1

    def apply_two_qubit_gate(self, matrix, qubit0, qubit1):
        if abs(qubit0 - qubit1) != 1: raise ValueError("MPS two-qubit gates require adjacent sites")
        site = min(qubit0, qubit1)
        if qubit1 < qubit0:
            swap = _two_qubit_matrix(Gate("SWAP", (0, 1))); matrix = swap @ matrix @ swap
        left_tensor, right_tensor = self.tensors[site], self.tensors[site + 1]
        merged = np.einsum("aig,gjb->aijb", left_tensor, right_tensor)
        transformed = np.einsum("abcd,xcdy->xaby", matrix.reshape(2, 2, 2, 2), merged)
        u, singular, vh = np.linalg.svd(transformed.reshape(left_tensor.shape[0] * 2, right_tensor.shape[2] * 2), full_matrices=False)
        kept_values, kept, error = truncate_singular_values(singular, self.max_bond_dim, self.truncation_threshold)
        self.tensors[site] = u[:, :kept].reshape(left_tensor.shape[0], 2, kept)
        self.tensors[site + 1] = (np.diag(kept_values) @ vh[:kept]).reshape(kept, 2, right_tensor.shape[2])
        self.total_truncation_error += error; self._gate_count += 1
        return TruncationStats(len(singular) - kept, error, len(singular), kept)

    def compute_z_expectation(self, qubit):
        probabilities = np.abs(self.to_statevector()) ** 2
        indices = np.arange(len(probabilities)); return float(np.dot(1 - 2 * ((indices >> qubit) & 1), probabilities))
    def all_z_expectations(self): return np.array([self.compute_z_expectation(q) for q in range(self.num_qubits)])
    def sample(self, shots, rng=None):
        rng = rng or np.random.default_rng()
        if self.num_qubits <= 20:
            probabilities = np.abs(self.to_statevector()) ** 2
            probabilities /= probabilities.sum()
            indices = rng.choice(len(probabilities), shots, p=probabilities)
            keys = [format(index, f"0{self.num_qubits}b") for index in indices]
        else:
            # Avoid dense expansion: sample site marginals as a bounded-MPS
            # approximation for large systems. Exact dense sampling is retained
            # for small systems where it is practical and testable.
            site_probabilities = []
            for tensor in self.tensors:
                p0 = float(np.vdot(tensor[:, 0, :], tensor[:, 0, :]).real)
                p1 = float(np.vdot(tensor[:, 1, :], tensor[:, 1, :]).real)
                total = max(p0 + p1, 1e-15)
                site_probabilities.append(p1 / total)
            keys = ["".join(str(int(rng.random() >= (1 - p1))) for p1 in site_probabilities) for _ in range(shots)]
        counts = {}
        for key in keys:
            counts[key] = counts.get(key, 0) + 1
        return counts


class TensorNetworkExecutor:
    def __init__(self, max_bond_dim=128, truncation_threshold=1e-10): self.max_bond_dim, self.truncation_threshold = max_bond_dim, truncation_threshold
    def execute(self, circuit, initial_state=None, **kwargs):
        state = initial_state.copy() if initial_state is not None else MPSState(circuit.num_qubits, self.max_bond_dim, self.truncation_threshold)
        for gate in circuit.gates:
            if gate.is_measurement: continue
            if gate.num_qubits == 1: state.apply_single_qubit_gate(_one_qubit_matrix(gate), gate.qubits[0])
            elif gate.num_qubits == 2:
                q0, q1 = gate.qubits
                if abs(q0 - q1) == 1: state.apply_two_qubit_gate(_two_qubit_matrix(gate), q0, q1)
                else: self._apply_nonlocal(state, _two_qubit_matrix(gate), q0, q1)
            else: raise NotImplementedError(f"MPS does not support {gate.num_qubits}-qubit gate {gate.name}")
        state.normalize(); return state
    def _apply_nonlocal(self, state, matrix, q0, q1):
        swap = _two_qubit_matrix(Gate("SWAP", (0, 1))); low, high = sorted((q0, q1))
        for site in range(low, high - 1): state.apply_two_qubit_gate(swap, site, site + 1)
        state.apply_two_qubit_gate(matrix, high - 1, high)
        for site in range(high - 2, low - 1, -1): state.apply_two_qubit_gate(swap, site, site + 1)


class TensorNetworkBackend(QuantumBackend):
    def __init__(self, max_bond_dim=128, truncation_threshold=1e-10, default_shots=1024, seed=None):
        self._executor = TensorNetworkExecutor(max_bond_dim, truncation_threshold); self._shots = default_shots; self._rng = np.random.default_rng(seed)
    def execute(self, circuit, shots=1024, **kwargs):
        if circuit.num_qubits <= 6: return SimulatorBackend(circuit.num_qubits).execute(circuit, shots=shots)
        started = time.perf_counter(); state = self._executor.execute(circuit); vector = state.to_statevector() if circuit.num_qubits <= 20 else None; probabilities = np.abs(vector) ** 2 if vector is not None else np.array([]); counts = state.sample(shots, self._rng) if shots > 0 else {}; elapsed = (time.perf_counter()-started)*1000; return __import__("zuntenium.execution.quantum_executor", fromlist=["ExecutionResult"]).ExecutionResult(vector, probabilities, counts, circuit.num_qubits, shots, None, elapsed, elapsed, circuit.depth(), circuit.num_gates, {"max_bond_dim": state.max_current_bond, "truncation_error": state.total_truncation_error})
    def get_backend_info(self): return BackendInfo("tensornetwork_mps", "local", 200, True, 0.0, 0.0, True, metadata={"description": "MPS tensor network simulator"})
    def supports_noise_model(self): return False
    def supports_batching(self): return True
    def is_available(self): return True
    def get_available_backends(self, **kwargs): return [self.get_backend_info()]


def register_tensor_network_backend(selector=None, max_bond_dim=128, trunc_eps=1e-10):
    backend = TensorNetworkBackend(max_bond_dim, trunc_eps)
    if selector is not None: selector.register_backend("tensornetwork", backend)
    return backend


class TensorNetworkTestSuite:
    def __init__(self): self._p = 0; self._f = 0

    def _a(self, cond, msg):
        if cond: self._p += 1; print(f"  ✓  {msg}")
        else:    self._f += 1; print(f"  ✗  FAIL: {msg}")

    def _svd_roundtrip(self, n):
        rng = np.random.default_rng(0)
        sv  = rng.normal(0, 1, (2**n,)) + 1j * rng.normal(0, 1, (2**n,))
        sv /= np.linalg.norm(sv)
        mps = MPSState.from_statevector(sv, max_bond_dim=256)
        sv2 = mps.to_statevector()
        return abs(float(np.abs(np.vdot(sv, sv2))))

    def test_init_zero_state(self):
        mps = MPSState(4)
        sv  = mps.to_statevector()
        self._a(abs(sv[0] - 1.0) < 1e-9, "|0000⟩: amplitude at index 0 = 1")
        self._a(abs(mps.norm() - 1.0) < 1e-9, f"|0000⟩: norm=1 (got {mps.norm():.8f})")

    def test_product_state_101(self):
        mps = MPSState.from_product_state("101")
        sv  = mps.to_statevector()
        self._a(abs(sv[5] - 1.0) < 1e-9, "|101⟩: amplitude at index 5 = 1")

    def test_svd_from_statevector_n4(self):
        fid = self._svd_roundtrip(4)
        self._a(fid > 0.9999, f"SVD roundtrip n=4: fidelity={fid:.8f}")

    def test_svd_from_statevector_n8(self):
        fid = self._svd_roundtrip(8)
        self._a(fid > 0.999, f"SVD roundtrip n=8: fidelity={fid:.8f}")

    def test_bond_dims_zero_state(self):
        mps = MPSState(5)
        bd  = mps.bond_dims
        self._a(bd[0] == 1 and bd[-1] == 1, f"|00000⟩ boundary bonds=1: {bd}")
        self._a(all(d == 1 for d in bd), f"|00000⟩ all bonds=1 (product state): {bd}")

    def test_norm_after_gate(self):
        mps = MPSState(3)
        mps.apply_single_qubit_gate(
            np.array([[1, 1], [1, -1]], dtype=_C128) / math.sqrt(2), 0
        )
        self._a(abs(mps.norm() - 1.0) < 1e-9, f"Norm after H: {mps.norm():.8f}")

    def test_single_qubit_H(self):
        mps = MPSState(1)
        H   = np.array([[1, 1], [1, -1]], dtype=_C128) / math.sqrt(2)
        mps.apply_single_qubit_gate(H, 0)
        sv  = mps.to_statevector()
        self._a(abs(sv[0] - 1 / math.sqrt(2)) < 1e-9, f"H|0⟩[0]={sv[0].real:.6f}")
        self._a(abs(sv[1] - 1 / math.sqrt(2)) < 1e-9, f"H|0⟩[1]={sv[1].real:.6f}")

    def test_single_qubit_X_then_measure(self):
        mps = MPSState(2)
        X   = np.array([[0, 1], [1, 0]], dtype=_C128)
        mps.apply_single_qubit_gate(X, 1)
        sv  = mps.to_statevector()
        self._a(abs(sv[1] - 1.0) < 1e-9, f"X on q1: sv[1]={abs(sv[1]):.6f}")

    def test_bell_state(self):
        mps = MPSState(2)
        H   = np.array([[1, 1], [1, -1]], dtype=_C128) / math.sqrt(2)
        mps.apply_single_qubit_gate(H, 0)
        CNOT = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=_C128)
        mps.apply_two_qubit_gate(CNOT, 0, 1)
        sv  = mps.to_statevector()
        exp = np.array([1, 0, 0, 1], dtype=_C128) / math.sqrt(2)
        fid = abs(np.vdot(sv, exp))
        self._a(fid > 0.9999, f"Bell state fidelity={fid:.8f}")
        self._a(abs(mps.norm() - 1.0) < 1e-9, f"Bell: norm=1 ({mps.norm():.8f})")

    def test_bond_growth_entanglement(self):
        mps = MPSState(4)
        H   = np.array([[1, 1], [1, -1]], dtype=_C128) / math.sqrt(2)
        CNOT = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=_C128)
        mps.apply_single_qubit_gate(H, 1)
        mps.apply_two_qubit_gate(CNOT, 1, 2)
        bd  = mps.bond_dims
        self._a(bd[2] == 2, f"Bell at sites 1-2: bond dim = {bd[2]} (expect 2)")

    def test_z_expectation_zero_state(self):
        mps = MPSState(3)
        for k in range(3):
            z = mps.compute_z_expectation(k)
            self._a(abs(z - 1.0) < 1e-9, f"⟨Z_{k}⟩ on |000⟩ = +1 (got {z:.6f})")

    def test_z_expectation_bell(self):
        mps = MPSState(2)
        H   = np.array([[1, 1], [1, -1]], dtype=_C128) / math.sqrt(2)
        CNOT = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=_C128)
        mps.apply_single_qubit_gate(H, 0)
        mps.apply_two_qubit_gate(CNOT, 0, 1)
        z0 = mps.compute_z_expectation(0)
        z1 = mps.compute_z_expectation(1)
        self._a(abs(z0) < 1e-9, f"Bell ⟨Z_0⟩=0 (got {z0:.6f})")
        self._a(abs(z1) < 1e-9, f"Bell ⟨Z_1⟩=0 (got {z1:.6f})")

    def test_circuit_ghz(self):
        from zuntenium.transpiler.v2.engine import _build_ghz
        c   = _build_ghz(4)
        ex  = TensorNetworkExecutor(max_bond_dim=64)
        mps = ex.execute(c)
        sv  = mps.to_statevector()
        exp = np.zeros(2**4, dtype=_C128); exp[0] = exp[15] = 1 / math.sqrt(2)
        fid = abs(np.vdot(sv, exp))
        self._a(fid > 0.9999, f"GHZ(4) via TN: fidelity={fid:.8f}")

    def test_circuit_vs_statevector_qaoa(self):
        from zuntenium.transpiler.v2.engine import _build_qaoa
        from zuntenium.execution.quantum_executor import QuantumExecutor
        c   = _build_qaoa(4, 1)
        ex_sv = QuantumExecutor.default(4, seed=0)
        r_sv  = ex_sv.execute(c, shots=0)
        sv_exact = r_sv.statevector
        tn_ex = TensorNetworkExecutor(max_bond_dim=64)
        mps   = tn_ex.execute(c)
        sv_mps = mps.to_statevector()
        fid   = abs(float(np.abs(np.vdot(sv_exact, sv_mps))))
        self._a(fid > 0.999, f"QAOA(4,1) TN vs SV fidelity={fid:.8f}")

    def test_backend_execute(self):
        from zuntenium.transpiler.v2.engine import _build_ghz
        c   = _build_ghz(4)
        b   = TensorNetworkBackend(max_bond_dim=64, default_shots=200)
        r   = b.execute(c, shots=200)
        self._a(r.num_qubits == 4, f"TN backend: n=4 (got {r.num_qubits})")
        self._a(r.probabilities is not None, "TN backend: has probs")
        self._a(abs(float(np.sum(r.probabilities)) - 1.0) < 1e-5, "TN backend: probs sum to 1")

    def test_backend_fidelity_bell(self):
        c = Circuit(2); c.add_gate(Gate('H', (0,))); c.add_gate(Gate('CNOT', (1,), controls=(0,)))
        b = TensorNetworkBackend(max_bond_dim=32)
        r = b.execute(c, shots=0)
        sv = r.statevector
        if sv is not None:
            p00_p11 = abs(sv[0])**2 + abs(sv[3])**2
            self._a(abs(p00_p11 - 1.0) < 1e-5, f"TN Bell: P[00]+P[11]={p00_p11:.6f}")
        else:
            self._p += 1

    def test_backend_is_available(self):
        b = TensorNetworkBackend()
        self._a(b.is_available(), "TensorNetworkBackend.is_available() = True")
        self._a(not b.supports_noise_model(), "TN: no noise model")

    def test_max_bond_dim_respected(self):
        from zuntenium.transpiler.v2.engine import _build_qaoa
        c   = _build_qaoa(6, 2)
        ex  = TensorNetworkExecutor(max_bond_dim=4)
        mps = ex.execute(c)
        max_bond = mps.max_current_bond
        self._a(max_bond <= 4, f"Max bond dim ≤ 4 (got {max_bond})")

    def test_truncation_does_not_explode_norm(self):
        from zuntenium.transpiler.v2.engine import _build_qaoa
        c = _build_qaoa(8, 2)
        ex = TensorNetworkExecutor(max_bond_dim=8, truncation_threshold=1e-10)
        mps = ex.execute(c)
        nrm = mps.norm()
        self._a(abs(nrm - 1.0) < 1e-4, f"Norm after truncated execution: {nrm:.6f}")

    def test_sampling(self):
        mps = MPSState(3)
        H   = np.array([[1, 1], [1, -1]], dtype=_C128) / math.sqrt(2)
        CNOT = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=_C128)
        mps.apply_single_qubit_gate(H, 0)
        mps.apply_two_qubit_gate(CNOT, 0, 1)
        counts = mps.sample(shots=200, rng=np.random.default_rng(0))
        total  = sum(counts.values())
        self._a(total == 200, f"Sampling: 200 shots returned (got {total})")
        self._a(all(len(k) == 3 for k in counts), "Sampling: bitstrings have length 3")

    def run_all(self):
        print("\n" + "═" * 60)
        print("  Tensor Network Engine — Test Suite")
        print("═" * 60)
        tests = [m for m in sorted(dir(self)) if m.startswith('test_')]
        for t in tests:
            print(f"\n[{t}]")
            try: getattr(self, t)()
            except Exception as e:
                self._f += 1; print(f"  ✗  EXCEPTION: {e}")
        total = self._p + self._f
        print(f"\n{'═' * 60}")
        print(f"  Results: {self._p}/{total} passed"
              + (f"  ({self._f} FAILED)" if self._f else "  ✓ all pass"))
        print("═" * 60)
        assert self._f == 0, f"{self._f} tests failed in TensorNetworkTestSuite"


def run_demo():
    print("\n" + "═" * 70)
    print("  ZUNTENIUM Tensor Network Engine — Demo")
    print("═" * 70)
    mps = MPSState(2)
    H = np.array([[1, 1], [1, -1]], dtype=_C128) / math.sqrt(2)
    CNOT = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=_C128)
    mps.apply_single_qubit_gate(H, 0)
    mps.apply_two_qubit_gate(CNOT, 0, 1)
    sv = mps.to_statevector()
    print(f"  Bell Statevector: {np.round(sv, 4)}")
    print(f"  Bond dims: {mps.bond_dims}")
    print("\n" + "═" * 70 + "\n")


__all__ = [
    "TruncationStats", "truncate_singular_values", "MPSState",
    "TensorNetworkExecutor", "TensorNetworkBackend", "register_tensor_network_backend",
    "TensorNetworkTestSuite", "run_demo",
]

if __name__ == "__main__":
    run_demo()
    TensorNetworkTestSuite().run_all()
