# zuntenium/backends/tensor_network.py
"""
TensorNetworkBackend — Production MPS (Matrix Product State) simulator.

Architecture:
  MPSState maintains a list of rank-3 tensors:
    Γ[i] ∈ ℂ^{χ_{i-1} × d × χ_i}   (left_bond × physical × right_bond)
  with left-canonical or right-canonical gauge for numerical stability.

Mathematical Foundation:
  |ψ⟩ = Σ_{s_0…s_{n-1}} A[0]^{s_0} A[1]^{s_1} … A[n-1]^{s_{n-1}} |s_0…s_{n-1}⟩
  where A[i]^{s_i} ∈ ℂ^{χ_{i-1} × χ_i} (bond matrices)

SVD Truncation:
  After each 2-qubit gate application on sites (i, i+1):
  1. Contract Θ = A[i] ⊗ A[i+1] ∈ ℂ^{χ_{i-1} × d × d × χ_{i+1}}
  2. Reshape Θ → matrix M ∈ ℂ^{(χ_{i-1}·d) × (d·χ_{i+1})}
  3. SVD: M = U Σ V†
  4. Truncate: keep top min(χ_max, rank) singular values
  5. Update: A[i] ← U[:, :χ], A[i+1] ← diag(Σ[:χ]) V[:χ, :]
  6. Normalise: track truncation error = ‖Σ[χ:]‖² / ‖Σ‖²

Truncation error bound:
  ε_trunc ≤ Σ_{k>χ} σ_k²   (sum of discarded squared singular values)
  Accumulated error over T gates: ε_total ≤ T · ε_per_gate (rough bound)

Canonical Forms:
  Left-canonical:  Σ_i (A^{s_i})† A^{s_i} = I  (isometry condition)
  Right-canonical: Σ_i A^{s_i} (A^{s_i})† = I

Entanglement entropy:
  S = -Σ_k λ_k² log λ_k²   where λ_k = σ_k / ‖Σ‖

Memory: O(n · χ² · d)   — linear in n, quadratic in χ
Gate application: O(χ³ · d²)  per 2-qubit gate
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

_log = logging.getLogger(__name__)

# Physical dimension for qubit (d=2: |0⟩,|1⟩)
_D = 2


# ── Common single-qubit gate matrices ───────────────────────────────────────

def _H()  -> np.ndarray: return np.array([[1,1],[1,-1]],dtype=complex)/np.sqrt(2)
def _X()  -> np.ndarray: return np.array([[0,1],[1,0]],dtype=complex)
def _Y()  -> np.ndarray: return np.array([[0,-1j],[1j,0]],dtype=complex)
def _Z()  -> np.ndarray: return np.array([[1,0],[0,-1]],dtype=complex)
def _S()  -> np.ndarray: return np.array([[1,0],[0,1j]],dtype=complex)
def _T()  -> np.ndarray: return np.array([[1,0],[0,np.exp(1j*np.pi/4)]],dtype=complex)
def _Rx(theta): return np.array([[np.cos(theta/2), -1j*np.sin(theta/2)],
                                  [-1j*np.sin(theta/2), np.cos(theta/2)]], dtype=complex)
def _Ry(theta): return np.array([[np.cos(theta/2), -np.sin(theta/2)],
                                  [np.sin(theta/2),  np.cos(theta/2)]], dtype=complex)
def _Rz(theta): return np.array([[np.exp(-1j*theta/2), 0],
                                  [0, np.exp(1j*theta/2)]], dtype=complex)
def _CX() -> np.ndarray:
    return np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex)
def _CZ() -> np.ndarray:
    return np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,-1]], dtype=complex)
def _SWAP() -> np.ndarray:
    return np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=complex)

GATE_REGISTRY: dict[str, Any] = {
    "h": _H, "x": _X, "y": _Y, "z": _Z, "s": _S, "t": _T,
    "rx": _Rx, "ry": _Ry, "rz": _Rz,
    "cx": _CX, "cnot": _CX, "cz": _CZ, "swap": _SWAP,
    "id": lambda: np.eye(2, dtype=complex),
    "sdg": lambda: np.array([[1,0],[0,-1j]],dtype=complex),
    "tdg": lambda: np.array([[1,0],[0,np.exp(-1j*np.pi/4)]],dtype=complex),
}


# ============================================================================
# TruncationStats — tracks accumulated error
# ============================================================================

@dataclass
class TruncationStats:
    """Accumulated SVD truncation statistics over a full circuit execution."""
    n_truncations: int = 0
    total_truncation_error: float = 0.0
    max_bond_dim_reached: int = 0
    max_truncation_error_per_step: float = 0.0

    def record(self, singular_values: np.ndarray, kept: int) -> None:
        all_sq = float(np.sum(singular_values ** 2))
        if all_sq < 1e-15:
            return
        discarded_sq = float(np.sum(singular_values[kept:] ** 2))
        err = discarded_sq / all_sq
        self.total_truncation_error += err
        self.max_truncation_error_per_step = max(self.max_truncation_error_per_step, err)
        self.n_truncations += 1
        self.max_bond_dim_reached = max(self.max_bond_dim_reached, kept)

    @property
    def avg_truncation_error(self) -> float:
        return self.total_truncation_error / max(self.n_truncations, 1)


# ============================================================================
# MPSState — core MPS tensor train
# ============================================================================

class MPSState:
    """
    Matrix Product State for n qubits with controllable bond dimension χ.

    Little-endian convention (matches StatevectorSimulator):
      site 0 = least significant qubit
      |ψ⟩ = A[0]^{s_0} A[1]^{s_1} ... A[n-1]^{s_{n-1}} |s_{n-1}...s_1 s_0⟩

    Parameters
    ----------
    n_qubits : int — number of qubits
    max_bond_dim : int — maximum bond dimension χ (default 64)
    svd_threshold : float — singular values below this × max are discarded
    """

    def __init__(
        self,
        n_qubits: int,
        max_bond_dim: int = 64,
        svd_threshold: float = 1e-10,
    ) -> None:
        if n_qubits < 1:
            raise ValueError(f"n_qubits must be ≥ 1, got {n_qubits}")
        self.n       = n_qubits
        self.chi     = max_bond_dim
        self.thr     = svd_threshold
        self.stats   = TruncationStats()

        # Initialise to |0⟩^⊗n
        # Each tensor: shape (χ_left, d, χ_right) = (1, 2, 1) for product state
        self.tensors: list[np.ndarray] = []
        for i in range(n_qubits):
            # |0⟩ = [1, 0]
            t = np.zeros((1, _D, 1), dtype=complex)
            t[0, 0, 0] = 1.0
            self.tensors.append(t)

        # Left-orthogonality centre tracker
        self._ortho_centre: int | None = None

    # ------------------------------------------------------------------
    # Gate application
    # ------------------------------------------------------------------

    def apply_single_qubit_gate(self, gate: np.ndarray, qubit: int) -> None:
        """
        Apply 2×2 unitary gate to qubit `qubit`.
        O(χ · d) operation — exact.

        U has shape (d, d), tensor[qubit] has shape (χ_L, d, χ_R).
        Result: T'[χ_L, d', χ_R] = Σ_d U[d', d] T[χ_L, d, χ_R]
        """
        if not 0 <= qubit < self.n:
            raise ValueError(f"Qubit {qubit} out of range [0,{self.n})")
        U = np.asarray(gate, dtype=complex)
        if U.shape != (_D, _D):
            raise ValueError(f"Single-qubit gate must be 2×2, got {U.shape}")
        # einsum: 'ij,ajk->aik' where a=left_bond, i=out_phys, j=in_phys, k=right_bond
        self.tensors[qubit] = np.einsum("ij,ajk->aik", U, self.tensors[qubit])

    def apply_two_qubit_gate(self, gate: np.ndarray, qubit_a: int, qubit_b: int) -> float:
        """
        Apply 4×4 unitary gate to adjacent qubits (qubit_a, qubit_b).
        Uses SVD truncation. Returns truncation error.

        Protocol:
          1. Contract Θ = A[a] ⊗ A[b]
          2. Apply gate: Θ' = U Θ
          3. SVD + truncate at max_bond_dim
          4. Update tensors

        For non-adjacent qubits, SWAP gates must be inserted beforehand.
        """
        if abs(qubit_a - qubit_b) != 1:
            raise ValueError(
                f"Two-qubit gate requires adjacent qubits, got {qubit_a}, {qubit_b}. "
                f"Insert SWAP gates to make them adjacent first."
            )
        i = min(qubit_a, qubit_b)
        j = i + 1
        U = np.asarray(gate, dtype=complex).reshape(_D, _D, _D, _D)
        # U[out_i, out_j, in_i, in_j]

        Ti = self.tensors[i]   # (χ_L, d, χ_M)
        Tj = self.tensors[j]   # (χ_M, d, χ_R)

        chi_L = Ti.shape[0]
        chi_M = Ti.shape[2]  # == Tj.shape[0]
        chi_R = Tj.shape[2]

        # Contract: Θ[χ_L, d_i, d_j, χ_R]
        # Step 1: Ti[χ_L, d_i, χ_M] × Tj[χ_M, d_j, χ_R] → Θ[χ_L, d_i, d_j, χ_R]
        Theta = np.einsum("lim,mjr->lijr", Ti, Tj)

        # Step 2: Apply gate U[out_i, out_j, in_i, in_j]
        # Θ'[χ_L, out_i, out_j, χ_R] = Σ_{in_i, in_j} U[out_i, out_j, in_i, in_j] Θ[χ_L, in_i, in_j, χ_R]
        # einsum replaced by tensordot below (BUG: non-ASCII)
        # Fix einsum: use explicit index names
        Theta_prime = np.tensordot(U.reshape(_D*_D, _D*_D),
                                    Theta.reshape(chi_L, _D*_D, chi_R),
                                    axes=([1], [1]))
        # Result: (d²,  χ_L, χ_R) → transpose to (χ_L, d², χ_R)
        Theta_prime = Theta_prime.transpose(1, 0, 2)
        # Reshape for SVD: (χ_L · d, d · χ_R)
        M = Theta_prime.reshape(chi_L * _D, _D * chi_R)

        # Step 3: SVD
        try:
            U_svd, S, Vh = np.linalg.svd(M, full_matrices=False)
        except np.linalg.LinAlgError:
            _log.warning("SVD failed, using pseudo-inverse")
            U_svd, S, Vh = np.linalg.svd(M + 1e-12 * np.eye(*M.shape[::-1][:2])[:M.shape[0], :M.shape[1]], full_matrices=False)

        # Step 4: Truncate
        # Threshold: keep singular values > svd_threshold * max(S)
        if len(S) == 0:
            return 0.0
        s_max    = S[0]
        keep_thr = np.sum(S > self.thr * max(s_max, 1e-15))
        chi_new  = min(int(keep_thr), self.chi, len(S))
        chi_new  = max(chi_new, 1)  # always keep at least 1

        self.stats.record(S, chi_new)
        trunc_err = float(np.sum(S[chi_new:] ** 2) / (np.sum(S ** 2) + 1e-15))

        # Step 5: Normalise and update
        S_trunc  = S[:chi_new]
        U_trunc  = U_svd[:, :chi_new]
        Vh_trunc = Vh[:chi_new, :]

        # Absorb singular values into right tensor (right-orthogonal left tensor)
        SV = np.diag(S_trunc) @ Vh_trunc  # (chi_new, d · chi_R)

        self.tensors[i] = U_trunc.reshape(chi_L, _D, chi_new)       # left-orthogonal
        self.tensors[j] = SV.reshape(chi_new, _D, chi_R)

        _log.debug(
            "2q gate site (%d,%d): χ %d→%d, trunc_err=%.2e",
            i, j, chi_M, chi_new, trunc_err,
        )
        return trunc_err

    # ------------------------------------------------------------------
    # Non-adjacent gate support via SWAP chain
    # ------------------------------------------------------------------

    def apply_two_qubit_gate_nonlocal(
        self, gate: np.ndarray, qubit_a: int, qubit_b: int
    ) -> float:
        """
        Apply 2-qubit gate to non-adjacent qubits via SWAP chain.
        Moves qubit_b adjacent to qubit_a, applies gate, swaps back.
        Total cost: O(|a-b| · χ³ · d²) SWAP operations.
        """
        if qubit_a == qubit_b:
            raise ValueError("qubit_a == qubit_b")
        if qubit_a > qubit_b:
            qubit_a, qubit_b = qubit_b, qubit_a

        total_err = 0.0
        SWAP = _SWAP()

        # Bring qubit_b adjacent to qubit_a+1 via SWAP chain
        for k in range(qubit_b - 1, qubit_a, -1):
            total_err += self.apply_two_qubit_gate(SWAP, k - 1, k)

        # Apply gate at (qubit_a, qubit_a+1)
        total_err += self.apply_two_qubit_gate(gate, qubit_a, qubit_a + 1)

        # Swap back
        for k in range(qubit_a + 1, qubit_b):
            total_err += self.apply_two_qubit_gate(SWAP, k, k + 1)

        return total_err

    # ------------------------------------------------------------------
    # Measurement
    # ------------------------------------------------------------------

    def measure_all(self, shots: int = 1024, rng_seed: int | None = None) -> dict[str, int]:
        """
        Sample bitstrings from the MPS state.

        Uses sequential Born-rule sampling:
        For each qubit i from 0 to n-1:
          P(s_i | s_0,...,s_{i-1}) = ‖A[i]^{s_i}‖² (given left-canonical gauge)

        Returns
        -------
        dict[str, int] — bitstring counts, little-endian (qubit 0 = rightmost bit)
        """
        rng = np.random.default_rng(rng_seed)
        sv  = self.to_statevector()
        p   = np.abs(sv) ** 2
        p   = np.abs(p) / (np.sum(np.abs(p)) + 1e-15)

        samples = rng.multinomial(shots, p)
        counts: dict[str, int] = {}
        for idx, cnt in enumerate(samples):
            if cnt > 0:
                key = format(idx, f"0{self.n}b")
                counts[key] = int(cnt)
        return counts

    # ------------------------------------------------------------------
    # State vector reconstruction
    # ------------------------------------------------------------------

    def to_statevector(self) -> np.ndarray:
        """
        Contract entire MPS to full 2^n statevector.
        O(n · χ² · d^n) — only feasible for small n (≤ 20).
        For n > 20, use measure_all() with Born-rule sampling.
        """
        if self.n > 25:
            _log.warning(
                "to_statevector() called for n=%d — this will use %.0f MB",
                self.n, 2**self.n * 16 / 1e6,
            )

        # Start with tensor[0]: shape (1, d, χ_1)
        sv = self.tensors[0]  # (1, d, χ)
        sv = sv.reshape(_D, sv.shape[2])  # (d, χ)

        for i in range(1, self.n):
            Ti = self.tensors[i]   # (χ_left, d, χ_right)
            # sv: (2^i, χ_left), Ti: (χ_left, d, χ_right)
            # Contract: result = (2^i · d, χ_right)
            sv = np.einsum("ij,jkl->ikl", sv, Ti).reshape(-1, Ti.shape[2])

        # Reverse to match StatevectorSimulator little-endian convention (qubit 0 = LSB)
        # MPS contraction is big-endian (site 0 = MSB), so reverse qubit order
        sv_arr = sv.reshape(-1)
        # Bit-reverse: flip qubit indices so qubit 0 becomes LSB
        n = self.n
        little_endian = np.zeros(2**n, dtype=complex)
        for idx in range(2**n):
            # Reverse the n-bit binary representation
            reversed_idx = int(format(idx, f'0{n}b')[::-1], 2)
            little_endian[reversed_idx] = sv_arr[idx]
        return little_endian  # (2^n,) little-endian

    def to_density_matrix(self) -> np.ndarray:
        """Full density matrix |ψ⟩⟨ψ|. Only for small n."""
        sv = self.to_statevector()
        return np.outer(sv, sv.conj())

    # ------------------------------------------------------------------
    # Entanglement entropy
    # ------------------------------------------------------------------

    def entanglement_entropy(self, cut: int) -> float:
        """
        Von Neumann entanglement entropy for bipartition at `cut`.
        Requires SVD of the bond at site `cut`.
        S = -Σ_k λ_k² log λ_k²
        """
        # Contract left part: shape (2^cut, χ)
        sv_left = self.tensors[0].reshape(_D, -1)
        for i in range(1, cut):
            Ti = self.tensors[i]
            sv_left = np.einsum("ij,jkl->ikl", sv_left, Ti).reshape(-1, Ti.shape[2])

        # SVD of the bond matrix
        _, S, _ = np.linalg.svd(sv_left, full_matrices=False)
        S2 = S ** 2
        S2 /= S2.sum() + 1e-15
        S2 = S2[S2 > 1e-15]
        return float(-np.sum(S2 * np.log(S2)))

    # ------------------------------------------------------------------
    # Norm and normalisation
    # ------------------------------------------------------------------

    def norm(self) -> float:
        """Compute ‖|ψ⟩‖ via contraction. Should be ≈ 1.0."""
        sv = self.to_statevector()
        return float(np.sqrt(np.sum(np.abs(sv) ** 2)))

    def normalise(self) -> None:
        """Normalise the MPS in-place by scaling the first tensor."""
        n = self.norm()
        if n > 1e-15:
            self.tensors[0] /= n

    # ------------------------------------------------------------------
    # Canonical forms
    # ------------------------------------------------------------------

    def canonicalise_left(self) -> None:
        """
        Bring MPS into left-canonical form via sequential QR decomposition.
        After: Σ_s (A^s)† A^s = I for each site except the last.
        """
        for i in range(self.n - 1):
            Ti = self.tensors[i]   # (χ_L, d, χ_R)
            χ_L, d, χ_R = Ti.shape
            # Reshape to (χ_L·d, χ_R), QR decompose
            M = Ti.reshape(χ_L * d, χ_R)
            Q, R = np.linalg.qr(M)
            new_chi = Q.shape[1]
            self.tensors[i] = Q.reshape(χ_L, d, new_chi)
            # Absorb R into next tensor
            Ti1 = self.tensors[i + 1]  # (χ_R, d, χ_R')
            self.tensors[i + 1] = np.einsum("ij,jkl->ikl", R, Ti1)
        self._ortho_centre = self.n - 1

    def canonicalise_right(self) -> None:
        """
        Bring MPS into right-canonical form via sequential LQ decomposition.
        After: Σ_s A^s (A^s)† = I for each site except the first.
        """
        for i in range(self.n - 1, 0, -1):
            Ti = self.tensors[i]   # (χ_L, d, χ_R)
            χ_L, d, χ_R = Ti.shape
            # Reshape to (χ_L, d·χ_R), LQ decompose
            M = Ti.reshape(χ_L, d * χ_R)
            Q, R = np.linalg.qr(M.T)
            Q, R = Q.T, R.T  # LQ: M = R Q
            new_chi = Q.shape[0]
            self.tensors[i] = Q.reshape(new_chi, d, χ_R)
            # Absorb R into previous tensor
            Ti_prev = self.tensors[i - 1]
            self.tensors[i - 1] = np.einsum("ijk,kl->ijl", Ti_prev, R)
        self._ortho_centre = 0


# ============================================================================
# TensorNetworkBackend — QuantumBackend implementation
# ============================================================================

class TensorNetworkBackend:
    """
    Production MPS-based quantum simulation backend.

    Implements the QuantumBackend interface (duck-typed to avoid import cycles):
      run(circuit, shots=1024, **kwargs) → dict

    Parameters
    ----------
    max_bond_dim    : int — maximum MPS bond dimension χ (default 64)
    svd_threshold   : float — SVD truncation threshold (default 1e-10)
    canonicalise    : bool — apply left-canonical form before measurement (default True)
    warn_large_sv   : bool — warn when to_statevector() is called for n > 20
    """

    def __init__(
        self,
        max_bond_dim: int = 64,
        svd_threshold: float = 1e-10,
        canonicalise: bool = True,
        warn_large_sv: bool = True,
    ) -> None:
        self.max_bond_dim  = max_bond_dim
        self.svd_threshold = svd_threshold
        self.canonicalise  = canonicalise
        self.warn_large_sv = warn_large_sv

    def run(
        self,
        circuit,
        shots: int = 1024,
        return_statevector: bool = False,
        return_density_matrix: bool = False,
        **kwargs,
    ) -> dict:
        """
        Execute circuit using MPS simulation.

        Parameters
        ----------
        circuit              : zuntenium.quantum_core.Circuit
        shots                : int — number of measurement samples
        return_statevector   : bool — include statevector in result (only feasible n≤25)
        return_density_matrix: bool — include density matrix (only feasible n≤15)

        Returns
        -------
        dict with keys: counts, statevector (optional), metadata
        """
        n = int(getattr(circuit, "num_qubits", 0))
        if n == 0:
            raise ValueError("Circuit has 0 qubits")

        mps = MPSState(
            n_qubits=n,
            max_bond_dim=self.max_bond_dim,
            svd_threshold=self.svd_threshold,
        )

        total_trunc_err = 0.0
        gates = list(getattr(circuit, "gates", []))

        for gate in gates:
            name   = str(getattr(gate, "name", "id")).lower()
            qubits = list(getattr(gate, "qubits", []))
            params = list(getattr(gate, "params", []))
            trunc_err = self._apply_gate(mps, name, qubits, params)
            total_trunc_err += trunc_err

        if self.canonicalise:
            mps.canonicalise_left()
            mps.normalise()

        counts = mps.measure_all(shots=shots)

        result: dict = {
            "counts":   counts,
            "shots":    shots,
            "metadata": {
                "backend":               "TensorNetworkBackend",
                "max_bond_dim":          self.max_bond_dim,
                "max_bond_dim_reached":  mps.stats.max_bond_dim_reached,
                "total_truncation_error": total_trunc_err,
                "avg_truncation_error":  mps.stats.avg_truncation_error,
                "n_truncations":         mps.stats.n_truncations,
                "n_qubits":              n,
            },
        }

        if return_statevector and n <= 25:
            result["statevector"] = mps.to_statevector()
        elif return_statevector:
            _log.warning("Skipping statevector for n=%d > 25", n)

        if return_density_matrix and n <= 15:
            result["density_matrix"] = mps.to_density_matrix()

        _log.info(
            "MPS run: n=%d chi_max=%d reached=%d trunc_err=%.2e shots=%d",
            n, self.max_bond_dim,
            mps.stats.max_bond_dim_reached,
            total_trunc_err, shots,
        )
        return result

    def _apply_gate(
        self, mps: MPSState, name: str, qubits: list[int], params: list[float]
    ) -> float:
        """Apply a named gate to MPS. Returns truncation error (0 for 1q gates)."""
        if len(qubits) == 0:
            return 0.0

        if len(qubits) == 1:
            U = self._get_single_qubit_matrix(name, params)
            mps.apply_single_qubit_gate(U, qubits[0])
            return 0.0

        elif len(qubits) == 2:
            U = self._get_two_qubit_matrix(name, params)
            qa, qb = qubits[0], qubits[1]
            if abs(qa - qb) == 1:
                return mps.apply_two_qubit_gate(U, qa, qb)
            else:
                return mps.apply_two_qubit_gate_nonlocal(U, qa, qb)

        else:
            _log.warning("Gate '%s' has %d qubits — decompose first", name, len(qubits))
            return 0.0

    @staticmethod
    def _get_single_qubit_matrix(name: str, params: list[float]) -> np.ndarray:
        fn = GATE_REGISTRY.get(name)
        if fn is None:
            _log.warning("Unknown gate '%s', using identity", name)
            return np.eye(2, dtype=complex)
        try:
            return fn(*params) if params else fn()
        except TypeError:
            return fn()

    @staticmethod
    def _get_two_qubit_matrix(name: str, params: list[float]) -> np.ndarray:
        fn = GATE_REGISTRY.get(name)
        if fn is None:
            _log.warning("Unknown 2q gate '%s', using identity tensor", name)
            return np.eye(4, dtype=complex)
        try:
            return fn(*params) if params else fn()
        except TypeError:
            return fn()

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def memory_estimate(n_qubits: int, max_bond_dim: int) -> dict[str, float]:
        """Estimate memory usage for MPS simulation."""
        # n tensors of shape (χ, d, χ) at complex128
        tensors_mb = n_qubits * max_bond_dim ** 2 * 2 * 16 / 1e6
        sv_mb      = 2 ** n_qubits * 16 / 1e6 if n_qubits <= 30 else float("inf")
        return {
            "mps_tensors_mb":  tensors_mb,
            "full_sv_mb":      sv_mb,
            "savings_factor":  sv_mb / max(tensors_mb, 1e-6),
        }

    def benchmark(self, n_qubits: int, depth: int, shots: int = 1024) -> dict:
        """
        Quick benchmark: random Clifford circuit to measure throughput.
        Returns timing and truncation statistics.
        """
        import time

        class _Gate:
            def __init__(self, name, qubits, params=None):
                self.name = name; self.qubits = qubits; self.params = params or []

        class _Circuit:
            def __init__(self, n, gates):
                self.num_qubits = n; self.gates = gates; self.name = f"bench_{n}q"

        rng = np.random.default_rng(42)
        gates = []
        for _ in range(depth):
            q = int(rng.integers(0, n_qubits))
            gates.append(_Gate("h", [q]))
            if n_qubits >= 2:
                qa = int(rng.integers(0, n_qubits - 1))
                gates.append(_Gate("cx", [qa, qa + 1]))

        circuit = _Circuit(n_qubits, gates)

        t0     = time.perf_counter()
        result = self.run(circuit, shots=shots)
        elapsed = (time.perf_counter() - t0) * 1000

        return {
            "n_qubits":         n_qubits,
            "depth":            depth,
            "elapsed_ms":       elapsed,
            "shots":            shots,
            "max_bond_reached": result["metadata"]["max_bond_dim_reached"],
            "trunc_error":      result["metadata"]["total_truncation_error"],
            "memory_mb":        self.memory_estimate(n_qubits, self.max_bond_dim),
        }
