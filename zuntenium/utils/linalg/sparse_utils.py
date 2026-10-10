"""
sparse_utils — Sparse state handling and MPS tensor utilities.

v0.1: Sparse dict conversion utilities.
v0.3: Full MPS tensor operations — SVD truncation, bond dimension
      control, single/two-qubit gate application to MPS.

The MPS (Matrix Product State) representation stores a quantum state
as a chain of tensors:
    |ψ⟩ = Σ A[0]_{i₀} · A[1]_{i₁} · ... · A[n-1]_{iₙ₋₁} |i₀i₁...iₙ₋₁⟩

where each A[k] is a tensor of shape (χ_left, 2, χ_right) and χ is
the bond dimension controlling the amount of entanglement captured.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


# ═══════════════════════════════════════════════════════════════════════
#  v0.1 — Sparse dict utilities (unchanged)
# ═══════════════════════════════════════════════════════════════════════

def to_sparse_dict(amplitudes: np.ndarray, tol: float = 1e-12) -> dict[int, complex]:
    """
    Convert a dense statevector to a sparse dict of {index: amplitude}.

    Only includes basis states with |amplitude| > tol.
    """
    return {
        i: complex(amp)
        for i, amp in enumerate(amplitudes)
        if abs(amp) > tol
    }


def from_sparse_dict(sparse: dict[int, complex], dim: int) -> np.ndarray:
    """Convert a sparse dict back to a dense statevector."""
    amplitudes = np.zeros(dim, dtype=np.complex128)
    for idx, amp in sparse.items():
        amplitudes[idx] = amp
    return amplitudes


def sparsity(amplitudes: np.ndarray, tol: float = 1e-12) -> float:
    """Return the fraction of zero amplitudes (higher = sparser)."""
    non_zero = np.sum(np.abs(amplitudes) > tol)
    return 1.0 - float(non_zero) / len(amplitudes)


# ═══════════════════════════════════════════════════════════════════════
#  v0.3 — MPS Tensor Utilities
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class MPSTensor:
    """
    A single tensor in an MPS chain.

    Shape convention: (chi_left, physical_dim, chi_right)
    For the leftmost tensor:  (1, physical_dim, chi_right)
    For the rightmost tensor: (chi_left, physical_dim, 1)

    Attributes:
        data: The tensor data as a 3D numpy array.
        site: The qubit index this tensor corresponds to.
    """
    data: np.ndarray
    site: int = 0

    @property
    def chi_left(self) -> int:
        return self.data.shape[0]

    @property
    def physical_dim(self) -> int:
        return self.data.shape[1]

    @property
    def chi_right(self) -> int:
        return self.data.shape[2]

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    def __repr__(self) -> str:
        return (
            f"MPSTensor(site={self.site}, shape={self.shape}, "
            f"χ_L={self.chi_left}, χ_R={self.chi_right})"
        )


@dataclass
class MPSState:
    """
    Matrix Product State representation of an n-qubit quantum state.

    Attributes:
        tensors: List of MPSTensor objects, one per qubit.
        num_qubits: Number of qubits.
        max_bond_dim: Maximum allowed bond dimension (chi).
        truncation_error: Accumulated truncation error from SVD truncations.
    """
    tensors: list[MPSTensor] = field(default_factory=list)
    num_qubits: int = 0
    max_bond_dim: int = 64
    truncation_error: float = 0.0

    @classmethod
    def zero_state(cls, n: int, max_bond_dim: int = 64) -> MPSState:
        """
        Create the MPS for |00...0⟩.

        Each tensor is shape (1, 2, 1) with the |0⟩ amplitude set to 1.
        """
        tensors = []
        for i in range(n):
            data = np.zeros((1, 2, 1), dtype=np.complex128)
            data[0, 0, 0] = 1.0  # |0⟩ amplitude
            tensors.append(MPSTensor(data=data, site=i))

        return cls(
            tensors=tensors,
            num_qubits=n,
            max_bond_dim=max_bond_dim,
        )

    @classmethod
    def from_statevector(
        cls, sv: np.ndarray, max_bond_dim: int = 64
    ) -> MPSState:
        """
        Convert a dense statevector to MPS form via successive SVDs.

        This is the canonical way to initialize an MPS from a known state.
        The bond dimensions are determined by the entanglement structure
        of the state, truncated to max_bond_dim.
        """
        n = int(np.log2(len(sv)))
        assert 2 ** n == len(sv), "Statevector length must be a power of 2"

        tensors: list[MPSTensor] = []
        residual = sv.reshape(1, -1)  # shape: (1, 2^n)
        total_trunc = 0.0

        for i in range(n - 1):
            # Reshape: (chi_left * 2, 2^(n-i-1))
            chi_left = residual.shape[0]
            residual = residual.reshape(chi_left * 2, -1)

            # SVD
            U, S, Vh = np.linalg.svd(residual, full_matrices=False)

            # Truncate to max_bond_dim
            chi = min(len(S), max_bond_dim)
            if chi < len(S):
                total_trunc += float(np.sum(S[chi:] ** 2))
            U = U[:, :chi]
            S = S[:chi]
            Vh = Vh[:chi, :]

            # Tensor for site i: reshape U to (chi_left, 2, chi)
            tensor_data = U.reshape(chi_left, 2, chi)
            tensors.append(MPSTensor(data=tensor_data, site=i))

            # Residual for next site: S @ Vh
            residual = np.diag(S) @ Vh

        # Last tensor: (chi_left, 2, 1)
        chi_left = residual.shape[0]
        tensor_data = residual.reshape(chi_left, 2, 1)
        tensors.append(MPSTensor(data=tensor_data, site=n - 1))

        return cls(
            tensors=tensors,
            num_qubits=n,
            max_bond_dim=max_bond_dim,
            truncation_error=total_trunc,
        )

    def to_statevector(self) -> np.ndarray:
        """
        Contract the MPS to a dense statevector.

        WARNING: Exponential in n. Only use for small qubit counts.
        """
        if not self.tensors:
            return np.array([1.0], dtype=np.complex128)

        result = self.tensors[0].data  # (chi_L, 2, chi_R)
        for i in range(1, len(self.tensors)):
            # Contract: result_{a,i₀...iₖ,b} · tensor_{b,iₖ₊₁,c}
            # = result_{a,i₀...iₖ₊₁,c}
            tensor = self.tensors[i].data
            result = np.einsum("...b,bpc->...pc", result, tensor)

        # Flatten to 1D: squeeze out boundary dims (should be 1)
        return result.reshape(-1)

    @property
    def bond_dimensions(self) -> list[int]:
        """Return the bond dimension at each bond."""
        if len(self.tensors) < 2:
            return []
        return [t.chi_right for t in self.tensors[:-1]]

    @property
    def max_current_bond(self) -> int:
        """Current maximum bond dimension."""
        dims = self.bond_dimensions
        return max(dims) if dims else 1


# ═══════════════════════════════════════════════════════════════════════
#  SVD Truncation
# ═══════════════════════════════════════════════════════════════════════

def svd_truncate(
    matrix: np.ndarray,
    max_chi: int,
    cutoff: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Perform SVD with truncation to max_chi bond dimension.

    Args:
        matrix: 2D matrix to decompose.
        max_chi: Maximum number of singular values to keep.
        cutoff: Discard singular values smaller than this.

    Returns:
        (U, S, Vh, truncation_error) where truncation_error is
        the sum of squared discarded singular values.
    """
    U, S, Vh = np.linalg.svd(matrix, full_matrices=False)

    # Apply cutoff
    mask = S > cutoff
    if not np.any(mask):
        mask[0] = True  # Keep at least one

    U, S, Vh = U[:, mask], S[mask], Vh[mask, :]

    # Truncate to max_chi
    chi = min(len(S), max_chi)
    trunc_error = float(np.sum(S[chi:] ** 2)) if chi < len(S) else 0.0

    return U[:, :chi], S[:chi], Vh[:chi, :], trunc_error


# ═══════════════════════════════════════════════════════════════════════
#  Gate Application to MPS
# ═══════════════════════════════════════════════════════════════════════

def apply_single_qubit_gate(
    mps: MPSState,
    gate: np.ndarray,
    qubit: int,
) -> None:
    """
    Apply a single-qubit gate to the MPS in-place.

    This is cheap: O(χ²·d) where d=2 is the physical dimension.
    Simply contracts the gate matrix with the physical index.

    Args:
        mps: The MPS state (modified in-place).
        gate: 2×2 unitary matrix.
        qubit: Target qubit index.
    """
    tensor = mps.tensors[qubit]
    # tensor.data shape: (chi_L, 2, chi_R)
    # Contract: new_data[a, j, b] = Σ_i gate[j, i] · data[a, i, b]
    new_data = np.einsum("ji,aib->ajb", gate, tensor.data)
    mps.tensors[qubit] = MPSTensor(data=new_data, site=qubit)


def apply_two_qubit_gate(
    mps: MPSState,
    gate: np.ndarray,
    qubit0: int,
    qubit1: int,
) -> None:
    """
    Apply a two-qubit gate to adjacent MPS sites.

    For non-adjacent qubits, SWAP gates should be inserted first.
    This operation contracts the two site tensors, applies the gate,
    and re-decomposes via SVD with truncation.

    Cost: O(χ³·d²) where d=2, χ = bond dimension.

    Args:
        mps: The MPS state (modified in-place).
        gate: 4×4 unitary matrix.
        qubit0: First qubit (must be adjacent to qubit1).
        qubit1: Second qubit (= qubit0 + 1 for adjacent).
    """
    # Ensure qubit0 < qubit1
    if qubit0 > qubit1:
        qubit0, qubit1 = qubit1, qubit0
        # Swap the gate accordingly
        swap_perm = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ], dtype=np.complex128)
        gate = swap_perm @ gate @ swap_perm

    # Handle non-adjacent qubits with SWAPs
    if qubit1 != qubit0 + 1:
        _swap_to_adjacent(mps, qubit0, qubit1)
        # After swapping, apply gate to (qubit1-1, qubit1)
        # Then swap back
        apply_two_qubit_gate(mps, gate, qubit1 - 1, qubit1)
        _swap_to_adjacent(mps, qubit0, qubit1)
        return

    t0 = mps.tensors[qubit0]  # (chi_L, 2, chi_mid)
    t1 = mps.tensors[qubit1]  # (chi_mid, 2, chi_R)

    chi_L = t0.chi_left
    chi_R = t1.chi_right

    # Contract the two tensors into a combined tensor
    # theta[a, i, j, b] = Σ_m t0[a, i, m] · t1[m, j, b]
    theta = np.einsum("aim,mjb->aijb", t0.data, t1.data)

    # Apply the gate
    # gate is 4×4, acting on (i, j) → (i', j')
    gate_4d = gate.reshape(2, 2, 2, 2)
    # theta_new[a, i', j', b] = Σ_{i,j} gate[i', j', i, j] · theta[a, i, j, b]
    theta_new = np.einsum("pqij,aijb->apqb", gate_4d, theta)

    # Reshape for SVD: (chi_L * 2, 2 * chi_R)
    theta_mat = theta_new.reshape(chi_L * 2, 2 * chi_R)

    # SVD with truncation
    U, S, Vh, trunc_err = svd_truncate(theta_mat, mps.max_bond_dim)
    mps.truncation_error += trunc_err

    chi_new = len(S)

    # Absorb singular values into U (left-canonical form)
    US = U @ np.diag(S)

    # New tensors
    new_t0 = US.reshape(chi_L, 2, chi_new)
    new_t1 = Vh.reshape(chi_new, 2, chi_R)

    mps.tensors[qubit0] = MPSTensor(data=new_t0, site=qubit0)
    mps.tensors[qubit1] = MPSTensor(data=new_t1, site=qubit1)


def _swap_to_adjacent(mps: MPSState, q0: int, q1: int) -> None:
    """Insert SWAP gates to bring q0 adjacent to q1."""
    SWAP = np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
    ], dtype=np.complex128)

    if q0 < q1:
        for i in range(q0, q1 - 1):
            apply_two_qubit_gate(mps, SWAP, i, i + 1)
    else:
        for i in range(q0, q1 + 1, -1):
            apply_two_qubit_gate(mps, SWAP, i - 1, i)


# ═══════════════════════════════════════════════════════════════════════
#  MPS Analysis Utilities
# ═══════════════════════════════════════════════════════════════════════

def mps_fidelity(mps1: MPSState, mps2: MPSState) -> float:
    """
    Compute the fidelity |⟨ψ₁|ψ₂⟩|² between two MPS states.

    For small qubit counts, converts to statevector.
    For larger states, uses transfer matrix contraction.
    """
    if mps1.num_qubits != mps2.num_qubits:
        raise ValueError("MPS states must have the same number of qubits")

    n = mps1.num_qubits

    # For small n, use statevector
    if n <= 20:
        sv1 = mps1.to_statevector()
        sv2 = mps2.to_statevector()
        overlap = abs(np.dot(sv1.conj(), sv2)) ** 2
        return float(overlap)

    # Transfer matrix contraction for large n
    # T[a,b] = Σ_i conj(A1[a,i,a']) · A2[b,i,b']
    T = np.ones((1, 1), dtype=np.complex128)
    for i in range(n):
        t1 = mps1.tensors[i].data  # (χ1_L, 2, χ1_R)
        t2 = mps2.tensors[i].data  # (χ2_L, 2, χ2_R)
        # Contract: T_new[a', b'] = Σ_{a,b,i} T[a,b] · conj(t1[a,i,a']) · t2[b,i,b']
        T = np.einsum("ab,aip,biq->pq", T, t1.conj(), t2)

    overlap = abs(T[0, 0]) ** 2
    return float(overlap)


def mps_entanglement_entropy(mps: MPSState, bond: int) -> float:
    """
    Compute the von Neumann entanglement entropy at a given bond.

    S = -Σ λᵢ² log(λᵢ²)

    where λᵢ are the singular values at the bond between sites
    `bond` and `bond+1`.
    """
    if bond < 0 or bond >= mps.num_qubits - 1:
        raise ValueError(f"Bond index {bond} out of range [0, {mps.num_qubits - 2}]")

    # Contract tensors up to the bond to get the bipartition
    left = mps.tensors[0].data
    for i in range(1, bond + 1):
        left = np.einsum("...b,bpc->...pc", left, mps.tensors[i].data)

    # Reshape for SVD
    left_flat = left.reshape(-1, left.shape[-1])
    _, S, _ = np.linalg.svd(left_flat, full_matrices=False)

    # Compute entropy
    probs = S ** 2
    probs = probs[probs > 1e-30]  # avoid log(0)
    probs = probs / np.sum(probs)  # normalize

    entropy = -float(np.sum(probs * np.log2(probs)))
    return entropy


def mps_norm(mps: MPSState) -> float:
    """Compute the norm ⟨ψ|ψ⟩ of the MPS state."""
    T = np.ones((1, 1), dtype=np.complex128)
    for tensor in mps.tensors:
        t = tensor.data
        T = np.einsum("ab,aip,bip->pq", T, t.conj(), t,
                       optimize=True).reshape(t.shape[2], t.shape[2])
    return float(np.sqrt(abs(T[0, 0])))


def normalize_mps(mps: MPSState) -> None:
    """Normalize the MPS state in-place."""
    norm = mps_norm(mps)
    if norm > 1e-15:
        # Scale the first tensor
        mps.tensors[0] = MPSTensor(
            data=mps.tensors[0].data / norm,
            site=0,
        )


__all__ = [
    # v0.1
    "to_sparse_dict",
    "from_sparse_dict",
    "sparsity",
    # v0.3
    "MPSTensor",
    "MPSState",
    "svd_truncate",
    "apply_single_qubit_gate",
    "apply_two_qubit_gate",
    "mps_fidelity",
    "mps_entanglement_entropy",
    "mps_norm",
    "normalize_mps",
]
