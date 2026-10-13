"""
zuntenium/quantum_core/statevector/gate_applicator.py
=====================================================
Core gate application logic via tensor contraction (reshape + einsum).

Extracted from statevector_simulator.py as part of O6 God-class decomposition.
"""

from __future__ import annotations

import math
from typing import Dict, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
StateArray = np.ndarray   # shape (2**n,), dtype complex128
Matrix2x2  = np.ndarray   # shape (2, 2),   dtype complex128
MatrixNxN  = np.ndarray   # shape (N, N),   dtype complex128


class GateApplicator:
    """
    Core gate application logic.

    Algorithm: tensor contraction via reshape + matmul
    --------------------------------------------------
    Let the statevector have shape S = (2^n,).

    Step 1 — reshape S → tensor T with shape (2, 2, ..., 2)  [n axes].
              Each axis k corresponds to qubit k.
              Axis 0 = qubit 0 (LSB), axis n−1 = qubit n−1 (MSB).

    Step 2 — for a k-qubit gate with matrix U (shape 2^k × 2^k):
              reshape U → U_tensor (shape (2,)*2k).
              Contract the k target axes of T with the k "input" axes of U_tensor.

    Step 3 — reshape result back to (2^n,).

    For a single-qubit gate U on qubit q, this simplifies to:

        T = state.reshape((2,)*n)        # no data copy — view
        T' = np.tensordot(U, T, axes=([1], [q]))   # 2×2 × T contraction
        T' = np.moveaxis(T', 0, q)      # restore axis order

    Time complexity:  O(2 × 2^n)  per single-qubit gate.
    Space complexity: O(2^n)      — reshape is a zero-copy view.

    Cache behavior
    --------------
    The reshape trick creates a view of the same contiguous C-order array.
    np.tensordot contracts along the chosen axis, traversing memory with stride
    2^q × 16 bytes.  For q < n/2 this is mostly sequential; for large q some
    stride irregularity occurs but remains within L3 cache for n ≤ 20.

    GPU migration path
    ------------------
    Replace np.tensordot with cupy.tensordot (or torch.tensordot).
    The statevector array should be transferred to device once and kept there.
    Only measurement results need host transfer.
    """

    # Periodic normalization drift monitoring (Critical Fix #1)
    _DRIFT_CHECK_INTERVAL: int = 100
    _apply_count: int = 0

    # Einsum string cache for repeated gate patterns (High Priority Fix #5)
    _einsum_cache_2q: Dict[Tuple[int, int, int], str] = {}
    _einsum_cache_kq: Dict[Tuple[int, ...], str] = {}

    @staticmethod
    def apply(state: StateArray, matrix: MatrixNxN, qubits: Sequence[int]) -> None:
        """
        Apply a unitary gate to `state` IN-PLACE.

        Includes periodic normalization drift monitoring every 100 gates.
        If drift exceeds 1e-12, the statevector is defensively renormalized.

        Parameters
        ----------
        state   : statevector, shape (2^n,), dtype complex128
        matrix  : gate unitary, shape (2^k, 2^k) where k = len(qubits)
        qubits  : target qubit indices (0-based, little-endian)

        Raises
        ------
        ValueError  : if matrix dimensions don't match qubit count
        IndexError  : if any qubit index is out of range
        """
        n_qubits_total = int(math.log2(state.shape[0]))
        k = len(qubits)
        dim_k = 1 << k

        if matrix.shape != (dim_k, dim_k):
            raise ValueError(
                f"Gate matrix shape {matrix.shape} does not match "
                f"{k} qubits (expected ({dim_k},{dim_k}))."
            )
        for q in qubits:
            if not (0 <= q < n_qubits_total):
                raise IndexError(
                    f"Qubit index {q} out of range for {n_qubits_total}-qubit system."
                )
        if len(set(qubits)) != k:
            raise ValueError(f"Duplicate qubit indices: {qubits}")

        # ── Dispatch by gate size ───────────────────────────────────────
        if k == 1:
            GateApplicator._apply_single(state, matrix, qubits[0], n_qubits_total)
        elif k == 2:
            GateApplicator._apply_two(state, matrix, qubits, n_qubits_total)
        else:
            GateApplicator._apply_general(state, matrix, qubits, n_qubits_total)

        # ── Periodic normalization drift check ──────────────────────────
        GateApplicator._apply_count += 1
        if GateApplicator._apply_count % GateApplicator._DRIFT_CHECK_INTERVAL == 0:
            norm_sq = float(np.real(np.dot(state.conj(), state)))
            if abs(norm_sq - 1.0) > 1e-12:
                state /= math.sqrt(norm_sq)

    @staticmethod
    def _apply_single(
        state: StateArray,
        U: Matrix2x2,
        qubit: int,
        n: int,
    ) -> None:
        """
        Apply a 2×2 unitary U to qubit `qubit` of an n-qubit statevector.

        Implementation detail
        ---------------------
        Reshape (2^n,) → (2,)*n, then contract axis `qubit` with U.

        tensordot(U, T, axes=([1],[qubit])) produces T' with shape
            (2, 2, ..., [qubit axis removed], ..., 2)
        where the new axis-0 is the "output" of U.
        moveaxis(T', 0, qubit) restores the qubit ordering.

        All reshapes are zero-copy views (no data movement).
        The only real work is the matmul inside tensordot: 2 × 2^n complex fmas.
        """
        tensor = state.reshape((2,) * n)         # view — O(1)
        tensor_out = np.tensordot(U, tensor, axes=([1], [qubit]))
        # tensordot puts the contracted "output" index first (axis 0)
        tensor_out = np.moveaxis(tensor_out, 0, qubit)
        np.copyto(state, tensor_out.reshape(state.shape))

    @staticmethod
    def _apply_two(
        state: StateArray,
        U: MatrixNxN,
        qubits: Sequence[int],
        n: int,
    ) -> None:
        """
        Apply a 4×4 unitary U to two qubits.

        Strategy
        --------
        Reshape state → (2,)*n tensor.
        Reshape U → (2,2,2,2) tensor U_t with axes [out0, out1, in0, in1].
        Contract axes (in0=qubits[0], in1=qubits[1]) of state tensor with
        axes (2,3) of U_t via einsum.

        The einsum string contracts the two target axes simultaneously,
        which is more efficient than two sequential single-qubit contractions
        for entangling gates (avoids creating intermediate tensors).
        """
        q0, q1 = qubits[0], qubits[1]
        tensor = state.reshape((2,) * n)
        U_t = U.reshape(2, 2, 2, 2)

        # Cached einsum string — avoids per-call string construction
        cache_key = (n, q0, q1)
        ein_str = GateApplicator._einsum_cache_2q.get(cache_key)
        if ein_str is None:
            letters = list('abcdefghijklmnopqrstuvwxyz')
            if n > len(letters):
                raise RuntimeError("n > 26 not supported with letter-based einsum.")
            idx = letters[:n]
            i0, i1 = idx[q0], idx[q1]
            out_idx = idx.copy()
            out_idx[q0] = 'P'
            out_idx[q1] = 'Q'
            tensor_str = ''.join(idx)
            out_str = ''.join(out_idx)
            ein_str = f'PQ{i0}{i1},{tensor_str}->{out_str}'
            GateApplicator._einsum_cache_2q[cache_key] = ein_str

        result = np.einsum(ein_str, U_t, tensor)
        np.copyto(state, result.reshape(state.shape))

    @staticmethod
    def _apply_general(
        state: StateArray,
        U: MatrixNxN,
        qubits: Sequence[int],
        n: int,
    ) -> None:
        """
        Apply a 2^k × 2^k unitary to k qubits using generalized einsum.

        This handles Toffoli (k=3) and arbitrary multi-qubit gates.

        Same approach as _apply_two, extended to k axes.
        Time complexity: O(2^k × 2^n) — grows with gate size but still
        linear in state dimension for fixed k.
        """
        k = len(qubits)
        letters = list('abcdefghijklmnopqrstuvwxyz')
        if n + k > len(letters):
            raise RuntimeError(
                f"n={n} + k={k} > 26: insufficient einsum letters."
            )

        state_idx = letters[:n]
        out_letters = letters[n:n+k]
        U_in  = [state_idx[q] for q in qubits]
        U_out = out_letters
        U_t = U.reshape((2,) * (2 * k))

        U_str = ''.join(U_out) + ''.join(U_in)
        tensor_str = ''.join(state_idx)
        out_idx = state_idx.copy()
        for i, q in enumerate(qubits):
            out_idx[q] = out_letters[i]
        out_str = ''.join(out_idx)

        ein_str = f'{U_str},{tensor_str}->{out_str}'
        result = np.einsum(ein_str, U_t, state.reshape((2,)*n))
        np.copyto(state, result.reshape(state.shape))
