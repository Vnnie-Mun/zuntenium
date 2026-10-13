"""
zuntenium/quantum_core/statevector/statevector_state.py
=======================================================
Manages the raw statevector array and initialization logic.

Extracted from statevector_simulator.py as part of O6 God-class decomposition.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
StateArray = np.ndarray   # shape (2**n,), dtype complex128


class StatevectorState:
    """
    Manages the raw statevector array and initialization logic.

    Mathematical detail
    -------------------
    The n-qubit zero state is:

        |0...0⟩ = e_0 ∈ C^{2^n}

    i.e., amplitude 1 at index 0, all others 0.  We store it as a 1-D
    np.complex128 array of length 2^n.

    Memory cost: 2^n × 16 bytes (complex128 = two float64).
        n=10 →  16 KiB
        n=20 →  16 MiB
        n=30 → ~16 GiB  (not supported in v0.1)

    Normalization invariant
    -----------------------
    After every gate application: ‖ψ‖₂ = 1 (maintained to float64 machine
    precision ~1e-15).  We check this in debug mode and re-normalize
    defensively after measurements.
    """

    MAX_QUBITS: int = 30  # Secondary guard; psutil pre-flight is the primary
    MIN_QUBITS: int = 1

    def __init__(self, num_qubits: int, *, max_qubits: int | None = None) -> None:
        if max_qubits is not None:
            self.MAX_QUBITS = max_qubits
        self._validate_num_qubits(num_qubits)
        self._num_qubits: int = num_qubits
        self._dim: int = 1 << num_qubits          # 2^n, fast bit-shift
        # Allocate once; reuse across resets via in-place fill
        self._data: StateArray = np.zeros(self._dim, dtype=np.complex128)
        self._data[0] = 1.0 + 0.0j                # |0...0⟩

    @property
    def num_qubits(self) -> int:
        return self._num_qubits

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def data(self) -> StateArray:
        """Direct view — callers must not resize."""
        return self._data

    def __array__(self, dtype=None) -> np.ndarray:
        """NumPy array protocol integration."""
        return self._data if dtype is None else self._data.astype(dtype)

    def reset(self) -> None:
        """
        Reset to |0...0⟩ in-place without reallocation.

        Performance: O(2^n) due to np.ndarray.fill — highly vectorized.
        """
        self._data.fill(0.0 + 0.0j)
        self._data[0] = 1.0 + 0.0j

    def set_statevector(self, vector: StateArray) -> None:
        """
        Load an external statevector (e.g., for state preparation benchmarks).

        Validates shape, dtype, and normalization before accepting.
        """
        if vector.shape != (self._dim,):
            raise ValueError(
                f"Shape mismatch: expected ({self._dim},), got {vector.shape}"
            )
        norm = np.linalg.norm(vector)
        if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(
                f"Statevector must be normalized; got ‖ψ‖ = {norm:.6e}"
            )
        np.copyto(self._data, vector.astype(np.complex128, copy=False))

    def norm(self) -> float:
        """Return ‖ψ‖₂ — should be 1.0 after every gate."""
        return float(np.sqrt(np.real(np.dot(self._data.conj(), self._data))))

    def renormalize(self) -> None:
        """Defensively renormalize after floating-point drift."""
        n = self.norm()
        if n < 1e-12:
            raise RuntimeError("Statevector collapsed to near-zero norm.")
        self._data /= n

    @staticmethod
    def _validate_num_qubits(n: int) -> None:
        if not isinstance(n, int) or n < StatevectorState.MIN_QUBITS:
            raise ValueError(
                f"num_qubits must be a positive integer ≥ {StatevectorState.MIN_QUBITS}; "
                f"got {n!r}"
            )
        # Pre-flight memory validation (F1, F9) — primary guard
        bytes_needed = (1 << n) * 16  # complex128
        try:
            import psutil
            available_bytes = psutil.virtual_memory().available
            if bytes_needed > available_bytes:
                raise MemoryError(
                    f"Statevector allocation requires {bytes_needed / (1024**3):.2f} GiB ({n} qubits), "
                    f"which exceeds available system memory of {available_bytes / (1024**3):.2f} GiB. "
                    f"Pre-flight guard prevented an out-of-memory crash. "
                    f"Consider using TensorNetworkEngine or StabilizerSimulator."
                )
        except ImportError:
            pass

        # Secondary fallback guard (configurable via max_qubits kwarg)
        if n > StatevectorState.MAX_QUBITS:
            raise ValueError(
                f"num_qubits={n} exceeds statevector limit of {StatevectorState.MAX_QUBITS}. "
                "Use TensorNetworkEngine for larger systems, or pass max_qubits= to override."
            )
