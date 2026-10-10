"""
StateVector — Primary quantum state abstraction.

Dispatches to NumPy or C++ backend based on qubit count and availability.
The C++ backend is preferred for num_qubits > CPP_THRESHOLD for performance.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from zuntenium.quantum_core.exceptions import QubitIndexError

if TYPE_CHECKING:
    from zuntenium.quantum_core.executor.result import MeasurementResult
    from zuntenium.quantum_core.gates.base_gate import BaseGate

# Qubit threshold above which C++ backend is preferred
CPP_THRESHOLD = 20


class StateVector:
    """
    Primary quantum state abstraction.

    Dispatches to NumPy or C++ backend based on qubit count and availability.
    The |0...0⟩ state is the initial state.

    Args:
        num_qubits: Number of qubits in the system.
        backend: One of "auto", "numpy", "cpp". "auto" selects C++ if available
                 and num_qubits > CPP_THRESHOLD, else NumPy.
    """

    def __init__(self, num_qubits: int, backend: str = "auto") -> None:
        if num_qubits < 1:
            raise ValueError(f"num_qubits must be >= 1, got {num_qubits}")
        self._num_qubits = num_qubits
        self._dim = 2 ** num_qubits
        self._backend = self._resolve_backend(backend)
        self._amplitudes: np.ndarray = self._initialize_zero_state()
        # Internal C++ handle (set when cpp backend is active)
        self._cpp_handle: Any = None
        if self._backend == "cpp":
            self._init_cpp()

    # ── Initialization ─────────────────────────────────────────────────────

    def _resolve_backend(self, requested: str) -> str:
        if requested == "numpy":
            return "numpy"
        if requested == "cpp":
            if self._cpp_available():
                return "cpp"
            return "numpy"  # graceful fallback
        # auto
        if self._num_qubits > CPP_THRESHOLD and self._cpp_available():
            return "cpp"
        return "numpy"

    @staticmethod
    def _cpp_available() -> bool:
        try:
            from zuntenium.bindings.binding_loader import is_cpp_available
            return is_cpp_available()
        except ImportError:
            return False

    def _initialize_zero_state(self) -> np.ndarray:
        """Return |0...0⟩ as a complex128 array."""
        amplitudes = np.zeros(self._dim, dtype=np.complex128)
        amplitudes[0] = 1.0
        return amplitudes

    def _init_cpp(self) -> None:
        """Initialize the C++ StateVector handle."""
        try:
            from zuntenium.bindings.binding_loader import load_cpp_module
            cpp = load_cpp_module()
            if cpp is None:
                raise ImportError("C++ module not loaded")
            # The pybind11-exposed class is 'StateVector' (not 'StateVectorCPP')
            self._cpp_handle = cpp.StateVector(self._num_qubits)
        except Exception:
            self._backend = "numpy"
            self._cpp_handle = None

    # ── Gate application ───────────────────────────────────────────────────

    def apply_gate(self, gate: BaseGate, qubits: list[int]) -> None:
        """
        Apply a gate to the specified qubits in-place.

        Args:
            gate: Gate object with a .matrix() method.
            qubits: List of qubit indices the gate acts on.

        Raises:
            QubitIndexError: If any qubit index is out of range.
        """
        self._validate_qubits(qubits)
        if self._backend == "cpp" and self._cpp_handle is not None:
            self._apply_gate_cpp(gate, qubits)
        else:
            self._apply_gate_numpy(gate, qubits)

    def _apply_gate_numpy(self, gate: BaseGate, qubits: list[int]) -> None:
        from zuntenium.quantum_core.statevector.statevector_numpy import (
            apply_single_qubit_gate,
            apply_two_qubit_gate,
        )
        matrix = gate.matrix()
        if len(qubits) == 1:
            self._amplitudes = apply_single_qubit_gate(
                self._amplitudes, matrix, qubits[0], self._num_qubits
            )
        elif len(qubits) == 2:
            self._amplitudes = apply_two_qubit_gate(
                self._amplitudes, matrix, qubits[0], qubits[1], self._num_qubits
            )
        else:
            # General multi-qubit gate via full tensor contraction
            self._amplitudes = self._apply_general_gate(matrix, qubits)

    def _apply_gate_cpp(self, gate: BaseGate, qubits: list[int]) -> None:
        matrix = gate.matrix().flatten().tolist()
        if len(qubits) == 1:
            self._cpp_handle.apply_single_qubit_gate(matrix, qubits[0])
        elif len(qubits) == 2:
            self._cpp_handle.apply_two_qubit_gate(matrix, qubits[0], qubits[1])

    def _apply_general_gate(self, matrix: np.ndarray, qubits: list[int]) -> np.ndarray:
        """Apply an N-qubit gate via full state-tensor contraction."""
        n = self._num_qubits
        state = self._amplitudes.reshape([2] * n)
        gate_tensor = matrix.reshape([2] * (2 * len(qubits)))
        # Contract over the target qubit axes
        axes_in = list(range(len(qubits), 2 * len(qubits)))
        state = np.tensordot(gate_tensor, state, axes=(axes_in, qubits))
        # Move result axes back to correct positions
        source = list(range(len(qubits)))
        dest = list(qubits)
        state = np.moveaxis(state, source, dest)
        return state.reshape(self._dim)

    # ── Measurement ────────────────────────────────────────────────────────

    def measure(self, qubits: list[int], shots: int) -> MeasurementResult:
        """
        Sample measurement outcomes over the given qubits.

        Args:
            qubits: Qubit indices to measure.
            shots: Number of measurement samples.

        Returns:
            MeasurementResult with counts dict and probability distribution.
        """
        from zuntenium.quantum_core.executor.measurement import sample_statevector
        self._validate_qubits(qubits)
        return sample_statevector(self._amplitudes, qubits, self._num_qubits, shots)

    # ── Queries ────────────────────────────────────────────────────────────

    def get_probabilities(self) -> np.ndarray:
        """Return the probability of each basis state: |amplitude|²."""
        if self._backend == "cpp" and self._cpp_handle is not None:
            probs = self._cpp_handle.get_probabilities()
            return np.array(probs)
        return np.abs(self._amplitudes) ** 2

    def get_amplitudes(self) -> np.ndarray:
        """Return the raw complex amplitude vector."""
        if self._backend == "cpp" and self._cpp_handle is not None:
            raw = self._cpp_handle.get_amplitudes()
            return np.array(raw, dtype=np.complex128)
        return self._amplitudes.copy()

    def inner_product(self, other: StateVector) -> complex:
        """Compute ⟨self|other⟩."""
        return complex(np.dot(self._amplitudes.conj(), other._amplitudes))

    def tensor_product(self, other: StateVector) -> StateVector:
        """Return |self⟩ ⊗ |other⟩."""
        combined_n = self._num_qubits + other._num_qubits
        result = StateVector(combined_n, backend="numpy")
        result._amplitudes = np.kron(self._amplitudes, other._amplitudes)
        return result

    def to_density_matrix(self) -> np.ndarray:
        """Convert pure state to density matrix ρ = |ψ⟩⟨ψ|."""
        psi = self._amplitudes.reshape(-1, 1)
        return psi @ psi.conj().T

    def snapshot(self) -> StateVector:
        """Return an immutable copy of the current state (for branching)."""
        copy = StateVector(self._num_qubits, backend="numpy")
        copy._amplitudes = self._amplitudes.copy()
        return copy

    # ── Validation ─────────────────────────────────────────────────────────

    def _validate_qubits(self, qubits: list[int]) -> None:
        for q in qubits:
            if not (0 <= q < self._num_qubits):
                raise QubitIndexError(q, self._num_qubits)

    # ── Properties ─────────────────────────────────────────────────────────

    @property
    def num_qubits(self) -> int:
        return self._num_qubits

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def active_backend(self) -> str:
        return self._backend

    def __repr__(self) -> str:
        return (
            f"StateVector(num_qubits={self._num_qubits}, "
            f"backend='{self._backend}', dim={self._dim})"
        )
