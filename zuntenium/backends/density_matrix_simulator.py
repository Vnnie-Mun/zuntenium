"""
zuntenium/backends/density_matrix_simulator.py
==============================================
Density matrix simulator for noisy quantum circuit simulation.

Implements ρ → U ρ U† evolution for unitary gates, with support for
Kraus channels (noise injection) and computational-basis measurement
via Tr(|x⟩⟨x| ρ) = ρ_{x,x}.

Key advantages over statevector simulation:
- Natively handles mixed states (noise channels, partial measurements)
- Supports decoherence modeling (amplitude damping, phase damping)
- Density matrix stays physical (positive semidefinite, unit trace)

Key disadvantage:
- Memory: O(4^n) vs O(2^n) — limits to ~15 qubits on standard hardware
  (4^15 × 16 bytes ≈ 16 GiB)

Author:  ZUNTENIUM Core Team (B4 — Option B)
"""
from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from zuntenium.quantum_core.statevector.gate_library import GateLibrary
from zuntenium.quantum_core.statevector.density_matrix import DensityMatrix


# Maximum qubits for density matrix simulation (memory guard)
# 4^n × 16 bytes:  n=12 → 256 MiB, n=15 → 16 GiB
MAX_DM_QUBITS = 15


class DensityMatrixSimulator:
    """
    Full density matrix quantum simulator.

    Evolves ρ = |ψ⟩⟨ψ| (or mixed state) through unitary gates and
    optionally through noise channels (Kraus operators).

    Mathematical operations
    -----------------------
    - Unitary gate U on qubits: ρ → (I⊗...⊗U⊗...⊗I) ρ (I⊗...⊗U†⊗...⊗I)
      Implemented via tensor contraction on reshape(ρ, (2,)*2n).
    - Measurement: P(x) = ρ_{x,x} (diagonal elements)
    - Noise channel: ρ → Σ_k K_k ρ K_k†
    """

    def __init__(self, num_qubits: int, seed: Optional[int] = None) -> None:
        if num_qubits > MAX_DM_QUBITS:
            raise ValueError(
                f"DensityMatrixSimulator supports at most {MAX_DM_QUBITS} qubits "
                f"(requested {num_qubits}). Memory cost: 4^{num_qubits} × 16 bytes = "
                f"{(4**num_qubits * 16) / (1024**3):.1f} GiB."
            )
        self._dm = DensityMatrix(num_qubits)
        self._num_qubits = num_qubits
        self._dim = 2 ** num_qubits
        self._rng = np.random.default_rng(seed)

    @property
    def num_qubits(self) -> int:
        return self._num_qubits

    @property
    def density_matrix(self) -> DensityMatrix:
        return self._dm

    def initialize(self, num_qubits: int) -> None:
        """Reset to |0...0⟩⟨0...0|."""
        if num_qubits != self._num_qubits:
            self._dm = DensityMatrix(num_qubits)
            self._num_qubits = num_qubits
            self._dim = 2 ** num_qubits
        else:
            self._dm = DensityMatrix(num_qubits)

    def apply_gate(self, matrix: np.ndarray, qubits: Sequence[int]) -> None:
        """
        Apply unitary gate U to specified qubits: ρ → U_full ρ U_full†.

        For a k-qubit gate U on qubits q₀,...,q_{k-1}:
            U_full = I ⊗ ... ⊗ U ⊗ ... ⊗ I
        computed via tensor contraction (same idea as GateApplicator but on ρ).
        """
        n = self._num_qubits
        k = len(qubits)
        dim_k = 1 << k

        if matrix.shape != (dim_k, dim_k):
            raise ValueError(
                f"Gate matrix shape {matrix.shape} doesn't match "
                f"{k} qubits (expected ({dim_k},{dim_k}))."
            )

        # Build full unitary via tensor product
        U_full = self._build_full_unitary(matrix, qubits, n)

        # ρ → U ρ U†
        rho = self._dm.matrix  # copy
        rho_new = U_full @ rho @ U_full.conj().T
        self._dm = DensityMatrix(n, data=rho_new)

    def _build_full_unitary(
        self, U: np.ndarray, qubits: Sequence[int], n: int
    ) -> np.ndarray:
        """
        Build the full 2^n × 2^n unitary from a k-qubit gate matrix U
        acting on the specified qubits.

        Uses tensor product assembly:
          1. Reshape U to (2,)*2k tensor
          2. Contract into an identity tensor of shape (2,)*2n
        """
        k = len(qubits)
        dim = 1 << n

        if k == 1:
            # Single-qubit: fast path
            q = qubits[0]
            # Build via Kronecker product
            result = np.eye(1, dtype=np.complex128)
            for i in range(n):
                result = np.kron(result, U if i == q else np.eye(2, dtype=np.complex128))
            return result
        elif k == 2:
            # Two-qubit gate: use the sorted qubit ordering
            # Need to handle non-adjacent and non-ordered qubits
            return self._build_two_qubit_full(U, qubits[0], qubits[1], n)
        else:
            # General k-qubit: use einsum-based tensor contraction
            return self._build_general_full(U, qubits, n)

    def _build_two_qubit_full(
        self, U: np.ndarray, q0: int, q1: int, n: int
    ) -> np.ndarray:
        """Build full unitary for a 2-qubit gate."""
        dim = 1 << n
        U_full = np.eye(dim, dtype=np.complex128)

        # For each pair of basis states, apply U
        for i in range(dim):
            for j in range(dim):
                # Extract bits at q0, q1
                b0_i = (i >> q0) & 1
                b1_i = (i >> q1) & 1
                b0_j = (j >> q0) & 1
                b1_j = (j >> q1) & 1

                # Check if all OTHER bits match between i and j
                mask = ~((1 << q0) | (1 << q1)) & ((1 << n) - 1)
                if (i & mask) != (j & mask):
                    continue

                # Index into 4x4 U matrix
                row = (b0_i << 0) | (b1_i << 1)
                col = (b0_j << 0) | (b1_j << 1)
                U_full[i, j] = U[row, col]

        return U_full

    def _build_general_full(
        self, U: np.ndarray, qubits: Sequence[int], n: int
    ) -> np.ndarray:
        """Build full unitary for a general k-qubit gate via basis enumeration."""
        k = len(qubits)
        dim = 1 << n
        dim_k = 1 << k
        U_full = np.eye(dim, dtype=np.complex128)

        for i in range(dim):
            for j in range(dim):
                # Check non-target bits match
                mask = 0
                for q in qubits:
                    mask |= (1 << q)
                complement_mask = (~mask) & ((1 << n) - 1)
                if (i & complement_mask) != (j & complement_mask):
                    continue

                # Extract target-qubit bits
                row = 0
                col = 0
                for bit_pos, q in enumerate(qubits):
                    row |= (((i >> q) & 1) << bit_pos)
                    col |= (((j >> q) & 1) << bit_pos)

                U_full[i, j] = U[row, col]

        return U_full

    def apply_noise_channel(
        self,
        kraus_ops: List[np.ndarray],
        qubit: Optional[int] = None,
    ) -> None:
        """
        Apply a noise channel: ρ → Σ_k K_k ρ K_k†.

        Parameters
        ----------
        kraus_ops : list of Kraus operator matrices
                    If qubit is specified, these should be 2×2 (single-qubit).
                    If qubit is None, these must match the full system dimension.
        qubit     : target qubit for single-qubit channels (optional)
        """
        if qubit is not None and kraus_ops[0].shape == (2, 2):
            # Expand single-qubit Kraus ops to full system
            n = self._num_qubits
            full_ops = []
            for K in kraus_ops:
                K_full = np.eye(1, dtype=np.complex128)
                for i in range(n):
                    K_full = np.kron(K_full, K if i == qubit else np.eye(2, dtype=np.complex128))
                full_ops.append(K_full)
            self._dm = self._dm.apply_kraus(full_ops)
        else:
            self._dm = self._dm.apply_kraus(kraus_ops)

    def get_probabilities(self) -> np.ndarray:
        """Return P(x) = ρ_{x,x} for all basis states."""
        return np.real(np.diag(self._dm.matrix)).copy()

    def sample(
        self, shots: int, qubits: Optional[Sequence[int]] = None
    ) -> Dict[str, int]:
        """
        Sample measurement outcomes from P(x) = ρ_{x,x}.

        Parameters
        ----------
        shots  : number of samples
        qubits : subset of qubits to measure (None = all)

        Returns
        -------
        Dict[bitstring → count]
        """
        probs = self.get_probabilities()
        probs = probs / probs.sum()  # defensive renormalization

        n = self._num_qubits
        indices = self._rng.choice(len(probs), size=shots, p=probs)

        if qubits is None:
            qubits_to_use = list(range(n))
        else:
            qubits_to_use = list(qubits)

        if qubits is None:
            fmt = f'{{:0{n}b}}'
            result: Dict[str, int] = {}
            for idx in indices:
                key = fmt.format(idx)
                result[key] = result.get(key, 0) + 1
        else:
            result = {}
            for idx in indices:
                key = ''.join(str((idx >> q) & 1) for q in qubits_to_use)
                result[key] = result.get(key, 0) + 1

        return result

    def purity(self) -> float:
        """Return Tr(ρ²) — 1.0 for pure states, < 1 for mixed."""
        return self._dm.purity()

    def von_neumann_entropy(self) -> float:
        """Return S(ρ) = -Tr(ρ log ρ) in nats."""
        return self._dm.von_neumann_entropy()

    def expectation_value(self, observable: np.ndarray) -> float:
        """Compute ⟨O⟩ = Tr(ρ O)."""
        return self._dm.expectation_value(observable)

    def execute(self, circuit, shots: int = 1024, **kwargs) -> Any:
        """
        Execute a transpiler v2 Circuit on this density matrix simulator.

        Compatible with the QuantumBackend interface used by the adaptive engine.
        """
        from zuntenium.execution.quantum_executor import (
            TranspiledExecutionResult as ExecutionResult,
            _apply_circuit_to_simulator,
        )

        started = time.perf_counter()
        self.initialize(circuit.num_qubits)

        # Apply each gate as a unitary
        gate_lib = GateLibrary
        for gate in circuit.gates:
            if gate.is_measurement:
                continue
            name = gate.name.upper()
            qubits = list(gate.qubits)
            try:
                if gate.params:
                    if name == 'RX':
                        mat = gate_lib.rx(gate.params[0])
                    elif name == 'RY':
                        mat = gate_lib.ry(gate.params[0])
                    elif name == 'RZ':
                        mat = gate_lib.rz(gate.params[0])
                    elif name == 'PHASE' or name == 'P':
                        mat = gate_lib.phase(gate.params[0])
                    elif name == 'U3' or name == 'U':
                        mat = gate_lib.u3(*gate.params[:3])
                    else:
                        mat = gate_lib.get(name)
                else:
                    mat = gate_lib.get(name)
            except ValueError:
                # Unknown gate — skip (measurement, barrier, etc.)
                continue
            self.apply_gate(mat, qubits)

        # Extract results
        probs = self.get_probabilities()
        counts = self.sample(shots) if shots > 0 else {}
        elapsed_ms = (time.perf_counter() - started) * 1000

        return ExecutionResult(
            statevector=None,  # density matrix sim doesn't produce a pure statevector
            probabilities=probs,
            counts=counts,
            num_qubits=circuit.num_qubits,
            shots=shots,
            opt_report=None,
            execution_time_ms=elapsed_ms,
            total_time_ms=elapsed_ms,
            circuit_depth=circuit.depth(),
            num_gates_executed=circuit.num_gates,
            metadata={
                "backend": "density_matrix",
                "purity": self.purity(),
                "von_neumann_entropy": self.von_neumann_entropy(),
            },
        )

    def __repr__(self) -> str:
        return (
            f"DensityMatrixSimulator(n={self._num_qubits}, "
            f"purity={self.purity():.6f}, "
            f"S(ρ)={self.von_neumann_entropy():.4f})"
        )
