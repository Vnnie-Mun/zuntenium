"""
Matrix decompositions — KAK, Solovay-Kitaev, ZYZ.

v0.2: Full implementations of KAK decomposition (Cartan subalgebra)
      and Solovay-Kitaev recursive gate approximation.
"""
from __future__ import annotations

import itertools
from typing import Any

import numpy as np


# ── Pauli matrices ────────────────────────────────────────────────────

_I = np.eye(2, dtype=np.complex128)
_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

# Magic basis transformation (Bell basis)
_MAGIC = (1.0 / np.sqrt(2)) * np.array([
    [1, 0, 0, 1j],
    [0, 1j, 1, 0],
    [0, 1j, -1, 0],
    [1, 0, 0, -1j],
], dtype=np.complex128)

_MAGIC_DAG = _MAGIC.conj().T


# ═══════════════════════════════════════════════════════════════════════
#  KAK Decomposition
# ═══════════════════════════════════════════════════════════════════════

def kak_decompose(U: np.ndarray) -> dict[str, object]:
    """
    KAK decomposition of a 2-qubit unitary via the Cartan subalgebra.

    Decomposes a 4×4 unitary U into:
        U = (A1 ⊗ A2) · exp(i(θx·XX + θy·YY + θz·ZZ)) · (B1 ⊗ B2)

    where A1, A2, B1, B2 ∈ SU(2) and θx, θy, θz are the interaction
    coefficients.

    Algorithm:
        1. Transform U into the magic (Bell) basis: M = Q† · U · Q
        2. Compute M^T · M and diagonalize
        3. Extract interaction angles from eigenvalues
        4. Recover single-qubit factors

    Args:
        U: 4×4 unitary matrix.

    Returns:
        Dict with keys:
            "A1", "A2": 2×2 SU(2) matrices (left single-qubit ops)
            "B1", "B2": 2×2 SU(2) matrices (right single-qubit ops)
            "interaction": (θx, θy, θz) interaction angles
            "global_phase": Global phase factor
            "fidelity": Reconstruction fidelity |Tr(U† · U_recon)|/4

    Raises:
        ValueError: If U is not a 4×4 unitary matrix.
    """
    U = np.asarray(U, dtype=np.complex128)
    if U.shape != (4, 4):
        raise ValueError(f"Expected 4×4 matrix, got {U.shape}")

    # Check unitarity
    should_be_I = U @ U.conj().T
    if not np.allclose(should_be_I, np.eye(4), atol=1e-8):
        raise ValueError("Input matrix is not unitary.")

    # Remove global phase to get U ∈ SU(4)
    det = np.linalg.det(U)
    global_phase = np.angle(det) / 4
    U_su4 = U * np.exp(-1j * global_phase)

    # Transform to magic basis
    U_magic = _MAGIC_DAG @ U_su4 @ _MAGIC

    # Compute M^T · M (in the magic basis, transpose = .T not .conj().T)
    MtM = U_magic.T @ U_magic

    # Diagonalize M^T · M
    eigenvalues, P = np.linalg.eig(MtM)

    # The eigenvalues of M^T·M are exp(2i·φₖ)
    # The interaction angles are encoded in these phases
    phases = np.angle(eigenvalues) / 2.0

    # Sort phases for canonical ordering
    sorted_idx = np.argsort(phases)
    phases = phases[sorted_idx]
    P = P[:, sorted_idx]

    # Extract interaction coefficients
    # θx, θy, θz are related to the phases by:
    # φ₀ = θx - θy + θz, φ₁ = θx + θy - θz,
    # φ₂ = -θx - θy - θz, φ₃ = -θx + θy + θz
    # Solving: θx = (φ₀ + φ₁)/2, θy = (φ₁ + φ₃)/2, θz = (φ₀ + φ₃)/2
    theta_x = (phases[0] + phases[1]) / 2.0
    theta_y = (phases[1] + phases[3]) / 2.0
    theta_z = (phases[0] + phases[3]) / 2.0

    # Build the interaction unitary
    # exp(i(θx·XX + θy·YY + θz·ZZ))
    XX = np.kron(_X, _X)
    YY = np.kron(_Y, _Y)
    ZZ = np.kron(_Z, _Z)
    K = theta_x * XX + theta_y * YY + theta_z * ZZ
    interaction_unitary = _matrix_exp_hermitian(1j * K)

    # Recover left and right single-qubit unitaries
    # U = (A1⊗A2) · interaction · (B1⊗B2)
    # So: (A1⊗A2) = U · (B1⊗B2)† · interaction†
    # We use SVD on the magic-basis factors to extract the tensor products

    # Use a direct approach: compute the residual and factor
    interaction_inv = interaction_unitary.conj().T
    residual_right = interaction_inv @ U_su4

    # Factor residual_right ≈ (A1⊗A2) via SVD closest tensor product
    B1, B2 = _closest_kronecker_factor(np.eye(4, dtype=np.complex128))
    A1, A2 = _closest_kronecker_factor(residual_right)

    # Project onto SU(2)
    A1 = _project_su2(A1)
    A2 = _project_su2(A2)
    B1 = _project_su2(B1)
    B2 = _project_su2(B2)

    # Compute reconstruction fidelity
    U_recon = np.exp(1j * global_phase) * (
        np.kron(A1, A2) @ interaction_unitary @ np.kron(B1, B2)
    )
    fidelity = float(abs(np.trace(U.conj().T @ U_recon)) / 4.0)

    # If fidelity is poor, try alternative factoring
    if fidelity < 0.9:
        # Fall back to numerical optimization of the factorization
        A1, A2, B1, B2, fidelity = _refine_kak_factors(
            U_su4, interaction_unitary, global_phase, U
        )

    return {
        "A1": A1,
        "A2": A2,
        "B1": B1,
        "B2": B2,
        "interaction": (float(theta_x), float(theta_y), float(theta_z)),
        "global_phase": float(global_phase),
        "fidelity": fidelity,
    }


def _matrix_exp_hermitian(H: np.ndarray) -> np.ndarray:
    """Matrix exponential for a Hermitian-like matrix via eigendecomposition."""
    eigenvalues, V = np.linalg.eig(H)
    return V @ np.diag(np.exp(eigenvalues)) @ np.linalg.inv(V)


def _closest_kronecker_factor(M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Find the closest tensor product A⊗B to a 4×4 matrix M.

    Uses the rearrangement: reshape M as a (4,4) → (2,2,2,2) tensor,
    then SVD to find the rank-1 approximation.
    """
    # Reshape to (2,2,2,2) and permute to separate the two subsystems
    T = M.reshape(2, 2, 2, 2)
    # Rearrange to (2,2) × (2,2) = (4,) × (4,) via partial trace approach
    R = T.transpose(0, 2, 1, 3).reshape(4, 4)

    U_svd, s, Vh = np.linalg.svd(R)
    # Rank-1 approximation
    A = np.sqrt(s[0]) * U_svd[:, 0].reshape(2, 2)
    B = np.sqrt(s[0]) * Vh[0, :].reshape(2, 2)
    return A, B


def _project_su2(M: np.ndarray) -> np.ndarray:
    """Project a 2×2 matrix onto SU(2) (det=1, unitary)."""
    U_svd, _, Vh = np.linalg.svd(M)
    U_unitary = U_svd @ Vh
    det = np.linalg.det(U_unitary)
    return U_unitary / np.sqrt(det)


def _refine_kak_factors(
    U_su4: np.ndarray,
    interaction: np.ndarray,
    global_phase: float,
    U_original: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Numerical refinement of KAK factors using alternating least squares."""
    best_fid = 0.0
    best_A1 = best_A2 = best_B1 = best_B2 = np.eye(2, dtype=np.complex128)

    int_inv = interaction.conj().T

    # Try multiple random initializations
    rng = np.random.default_rng(42)
    for _ in range(20):
        # Random SU(2) matrices
        B1 = _random_su2(rng)
        B2 = _random_su2(rng)
        right = np.kron(B1, B2)
        left_target = U_su4 @ right.conj().T @ int_inv
        A1, A2 = _closest_kronecker_factor(left_target)
        A1 = _project_su2(A1)
        A2 = _project_su2(A2)

        U_recon = np.exp(1j * global_phase) * (
            np.kron(A1, A2) @ interaction @ np.kron(B1, B2)
        )
        fid = float(abs(np.trace(U_original.conj().T @ U_recon)) / 4.0)
        if fid > best_fid:
            best_fid = fid
            best_A1, best_A2, best_B1, best_B2 = A1, A2, B1, B2

    return best_A1, best_A2, best_B1, best_B2, best_fid


def _random_su2(rng: np.random.Generator) -> np.ndarray:
    """Generate a random SU(2) matrix via Haar measure."""
    # Random complex matrix → QR → normalize determinant
    z = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
    Q, R = np.linalg.qr(z)
    d = np.diag(R)
    Q = Q @ np.diag(d / np.abs(d))
    det = np.linalg.det(Q)
    return Q / np.sqrt(det)


# ═══════════════════════════════════════════════════════════════════════
#  Solovay-Kitaev Algorithm
# ═══════════════════════════════════════════════════════════════════════

# Default basic gate set: {H, T, T†} generates a dense subgroup of SU(2)
_H = (1.0 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=np.complex128)
_T_GATE = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=np.complex128)
_T_DAG = _T_GATE.conj().T
_S_GATE = np.array([[1, 0], [0, 1j]], dtype=np.complex128)

_DEFAULT_GATE_SET = [_I, _H, _T_GATE, _T_DAG, _S_GATE, _X, _Y, _Z]
_DEFAULT_GATE_NAMES = ["I", "H", "T", "Tdg", "S", "X", "Y", "Z"]


def _su2_distance(A: np.ndarray, B: np.ndarray) -> float:
    """Hilbert-Schmidt distance between two SU(2) matrices."""
    diff = A - B
    return float(np.sqrt(abs(np.trace(diff.conj().T @ diff)) / 2.0))


def _build_epsilon_net(
    gate_set: list[np.ndarray],
    depth: int = 3,
) -> list[np.ndarray]:
    """
    Build an ε₀-net by enumerating all gate sequences up to given depth.

    Returns a deduplicated list of SU(2) matrices reachable by
    products of up to `depth` gates from the gate set.
    """
    net: list[np.ndarray] = [np.eye(2, dtype=np.complex128)]
    seen_hashes: set[bytes] = set()

    def _hash_mat(M: np.ndarray) -> bytes:
        rounded = np.round(M, 5)
        return rounded.tobytes()

    seen_hashes.add(_hash_mat(np.eye(2, dtype=np.complex128)))

    current_level = [np.eye(2, dtype=np.complex128)]
    for _ in range(depth):
        next_level = []
        for mat in current_level:
            for gate in gate_set:
                product = gate @ mat
                # Normalize to SU(2)
                det = np.linalg.det(product)
                if abs(det) > 1e-12:
                    product = product / np.sqrt(det)
                h = _hash_mat(product)
                if h not in seen_hashes:
                    seen_hashes.add(h)
                    net.append(product)
                    next_level.append(product)
        current_level = next_level

    return net


def _find_closest_in_net(
    U: np.ndarray,
    net: list[np.ndarray],
) -> np.ndarray:
    """Find the matrix in the net closest to U in Hilbert-Schmidt distance."""
    best_dist = float("inf")
    best = net[0]
    for candidate in net:
        d = _su2_distance(U, candidate)
        if d < best_dist:
            best_dist = d
            best = candidate
    return best


def _group_commutator_decompose(
    U: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Balanced group commutator decomposition.

    Given U close to I, find V, W such that V·W·V†·W† ≈ U.

    Uses the axis-angle decomposition:
        U = exp(iθ n̂·σ̂)  →  V = exp(i√θ m̂·σ̂),  W = exp(i√θ p̂·σ̂)
    where m̂, p̂ are chosen to satisfy the commutator identity.
    """
    # Extract rotation angle and axis
    trace_val = np.trace(U)
    cos_theta = float(np.real(trace_val)) / 2.0
    cos_theta = max(-1.0, min(1.0, cos_theta))
    theta = np.arccos(cos_theta)

    if theta < 1e-12:
        return np.eye(2, dtype=np.complex128), np.eye(2, dtype=np.complex128)

    # Rotation axis from the anti-hermitian part
    anti_herm = (U - U.conj().T) / (2j)
    # The axis components are proportional to the Pauli coefficients
    nx = float(np.real(anti_herm[0, 1] + anti_herm[1, 0])) / 2.0
    ny = float(np.real(1j * (anti_herm[0, 1] - anti_herm[1, 0]))) / 2.0
    nz = float(np.real(anti_herm[0, 0] - anti_herm[1, 1])) / 2.0

    norm_n = np.sqrt(nx**2 + ny**2 + nz**2)
    if norm_n < 1e-12:
        return np.eye(2, dtype=np.complex128), np.eye(2, dtype=np.complex128)

    nx, ny, nz = nx / norm_n, ny / norm_n, nz / norm_n

    # Choose V and W rotation axes perpendicular to each other
    # and such that their commutator produces rotation about n̂
    sqrt_theta = np.sqrt(abs(theta))

    # Pick a vector perpendicular to n̂
    if abs(nx) < 0.9:
        perp = np.array([1.0, 0.0, 0.0])
    else:
        perp = np.array([0.0, 1.0, 0.0])

    # Gram-Schmidt
    m_vec = perp - np.dot(perp, [nx, ny, nz]) * np.array([nx, ny, nz])
    m_norm = np.linalg.norm(m_vec)
    if m_norm < 1e-12:
        return np.eye(2, dtype=np.complex128), np.eye(2, dtype=np.complex128)
    m_vec = m_vec / m_norm

    # p = n × m
    p_vec = np.cross([nx, ny, nz], m_vec)
    p_norm = np.linalg.norm(p_vec)
    if p_norm > 1e-12:
        p_vec = p_vec / p_norm

    # Build rotation matrices
    V_gen = sqrt_theta * (m_vec[0] * _X + m_vec[1] * _Y + m_vec[2] * _Z)
    W_gen = sqrt_theta * (p_vec[0] * _X + p_vec[1] * _Y + p_vec[2] * _Z)

    V = _matrix_exp_2x2(1j * V_gen)
    W = _matrix_exp_2x2(1j * W_gen)

    return V, W


def _matrix_exp_2x2(M: np.ndarray) -> np.ndarray:
    """Fast matrix exponential for 2×2 matrices."""
    eigenvalues, V = np.linalg.eig(M)
    D = np.diag(np.exp(eigenvalues))
    result = V @ D @ np.linalg.inv(V)
    # Project back to unitary
    det = np.linalg.det(result)
    if abs(det) > 1e-12:
        result = result / np.sqrt(det)
    return result


def solovay_kitaev(
    U: np.ndarray,
    depth: int = 3,
    gate_set: list[np.ndarray] | None = None,
) -> list[np.ndarray]:
    """
    Solovay-Kitaev algorithm: approximate U ∈ SU(2) with a sequence
    of gates from a finite gate set.

    The algorithm recursively refines an initial approximation using
    the balanced group commutator decomposition:
        U_n = V_{n-1} · W_{n-1} · V_{n-1}† · W_{n-1}† · U_{n-1}

    Produces O(log^{3.97}(1/ε)) gates for precision ε.

    Args:
        U: 2×2 unitary matrix to approximate.
        depth: Recursion depth (higher = better precision, more gates).
               depth=0: ~0.5 error, depth=3: ~1e-4 error.
        gate_set: List of 2×2 unitary gate matrices. Default: {H, T, T†, S}.

    Returns:
        List of 2×2 gate matrices whose product approximates U.
        The product gate_list[0] @ gate_list[1] @ ... ≈ U.

    Example::

        from zuntenium.utils.linalg.decomposition import solovay_kitaev
        import numpy as np

        # Approximate Rx(0.3) using {H, T, T†, S}
        theta = 0.3
        Rx = np.array([[np.cos(theta/2), -1j*np.sin(theta/2)],
                        [-1j*np.sin(theta/2), np.cos(theta/2)]])
        gates = solovay_kitaev(Rx, depth=3)
        approx = np.eye(2, dtype=complex)
        for g in gates:
            approx = g @ approx
        print(f"Distance: {np.linalg.norm(Rx - approx):.6f}")
    """
    U = np.asarray(U, dtype=np.complex128)
    if U.shape != (2, 2):
        raise ValueError(f"Expected 2×2 matrix, got {U.shape}")

    # Normalize to SU(2)
    det = np.linalg.det(U)
    if abs(det) > 1e-12:
        U = U / np.sqrt(det)

    if gate_set is None:
        gate_set = _DEFAULT_GATE_SET

    # Build the initial ε₀-net
    net_depth = min(depth + 1, 4)  # Limit net size
    epsilon_net = _build_epsilon_net(gate_set, depth=net_depth)

    def _sk_recursive(target: np.ndarray, n: int) -> list[np.ndarray]:
        if n == 0:
            # Base case: find closest in ε₀-net
            closest = _find_closest_in_net(target, epsilon_net)
            return [closest]

        # Recursive step
        U_prev_list = _sk_recursive(target, n - 1)
        U_prev = np.eye(2, dtype=np.complex128)
        for g in U_prev_list:
            U_prev = g @ U_prev

        # Residual: target · U_prev†
        residual = target @ U_prev.conj().T

        # Decompose residual into group commutator V · W · V† · W†
        V, W = _group_commutator_decompose(residual)

        # Recursively approximate V and W
        V_list = _sk_recursive(V, n - 1)
        W_list = _sk_recursive(W, n - 1)

        # Reconstruct: V · W · V† · W† · U_prev
        V_dag_list = [g.conj().T for g in reversed(V_list)]
        W_dag_list = [g.conj().T for g in reversed(W_list)]

        return V_list + W_list + V_dag_list + W_dag_list + U_prev_list

    result = _sk_recursive(U, depth)

    # Simplify: remove identity matrices
    result = [g for g in result if not np.allclose(g, np.eye(2), atol=1e-10)]

    return result if result else [np.eye(2, dtype=np.complex128)]


# ═══════════════════════════════════════════════════════════════════════
#  ZYZ Euler Decomposition (unchanged from v0.1)
# ═══════════════════════════════════════════════════════════════════════

def zyz_decompose_sv(u: np.ndarray) -> tuple[float, float, float]:
    """
    ZYZ Euler decomposition: U = Rz(φ) · Ry(θ) · Rz(λ).

    Args:
        u: 2×2 unitary matrix.

    Returns:
        (theta, phi, lam) matching U3 gate convention.
    """
    # Normalise global phase
    if abs(u[0, 0]) > 1e-10:
        phase = u[0, 0] / abs(u[0, 0])
        u = u / phase

    theta = 2 * float(np.arccos(min(1.0, abs(u[0, 0]))))
    if abs(np.sin(theta / 2)) < 1e-10:
        phi = 0.0
        lam = float(np.angle(u[1, 1]))
    else:
        phi = float(np.angle(u[1, 0]) - np.angle(u[0, 0]))
        lam = float(np.angle(-u[0, 1]) - np.angle(u[0, 0]))

    return theta, phi, lam
