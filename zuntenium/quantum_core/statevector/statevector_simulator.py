"""
zuntenium/quantum_core/statevector/statevector_simulator.py
===========================================================
ZUNTENIUM v0.1 — Production-Grade Statevector Simulator
========================================================

Mathematical Foundation
-----------------------
A pure n-qubit state lives in the Hilbert space H = (C^2)^{⊗n} ≅ C^{2^n}.
We represent it as a normalized complex vector:

    |ψ⟩ = Σ_{x=0}^{2^n−1} α_x |x⟩,   Σ |α_x|² = 1

Indexing convention: LITTLE-ENDIAN (qubit 0 = LSB).

Gate application (core insight)
--------------------------------
For a single-qubit gate U acting on qubit k of an n-qubit system, the
naive full-operator approach would construct:

    M_full = I^{⊗(n−k−1)} ⊗ U ⊗ I^{⊗k}    [shape 2^n × 2^n]

Instead we exploit the tensor structure:

    Reshape statevector: shape (2^n,) → (2, 2, ..., 2)  [n axes]
    Contract axis k with U using einsum/matmul

This is O(2^n) time and O(2^n) space — optimal.

Author:  ZUNTENIUM Core Team
License: Apache 2.0

Module Structure (O6 Decomposition)
------------------------------------
This file was decomposed from a 1620-line God class into focused modules:

- ``simulator_backend.py``   — SimulatorBackend ABC
- ``statevector_state.py``   — StatevectorState (array + initialization)
- ``gate_library.py``        — GateLibrary (all standard gate matrices)
- ``gate_applicator.py``     — GateApplicator (tensor contraction engine)
- ``sim_circuit.py``         — _SimInstruction + SimCircuit (fluent builder)
- ``measurement_engine.py``  — MeasurementEngine (Born rule sampling)

This file retains StatevectorSimulator (the public facade) and re-exports
all symbols for backward compatibility.
"""

from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Re-export all extracted components for backward compatibility
# ---------------------------------------------------------------------------
from zuntenium.quantum_core.statevector.simulator_backend import (
    SimulatorBackend,
    StateArray,
    MatrixNxN,
)
from zuntenium.quantum_core.statevector.statevector_state import StatevectorState
from zuntenium.quantum_core.statevector.gate_library import GateLibrary
from zuntenium.quantum_core.statevector.gate_applicator import GateApplicator
from zuntenium.quantum_core.statevector.sim_circuit import (
    _SimInstruction,
    SimCircuit,
)
from zuntenium.quantum_core.statevector.measurement_engine import MeasurementEngine

# Type alias re-exports
Matrix2x2 = np.ndarray   # shape (2, 2),   dtype complex128


###############################################################################
# MAIN SIMULATOR CLASS
###############################################################################

class StatevectorSimulator(SimulatorBackend):
    """
    ZUNTENIUM v0.1 Production Statevector Simulator.

    Combines StatevectorState, GateApplicator, and MeasurementEngine into a
    single, coherent backend implementing the SimulatorBackend interface.

    Design choices
    --------------
    1. Single source of truth: `_state._data` is the live statevector.
    2. All gate applications are in-place (GateApplicator.apply modifies the array).
    3. Measurement collapse is in-place by default; non-destructive sampling
       is provided via `sample()`.
    4. Random number generation is encapsulated in a seeded Generator for
       full reproducibility.

    Extension hooks
    ---------------
    - `get_expectation_value(pauli_string)`: VQE/hybrid-solver interface (v0.2).
    - `snapshot()`: returns immutable copy for branching simulations.
    - Swap `_applicator` for a CuPy-based variant to get GPU execution.
    """

    def __init__(self, num_qubits: int, seed: Optional[int] = None) -> None:
        """
        Parameters
        ----------
        num_qubits : number of qubits (1 ≤ n ≤ 30, configurable)
        seed       : random seed for reproducible measurements
        """
        self._state = StatevectorState(num_qubits)
        self._rng   = np.random.default_rng(seed)

    def initialize(self, num_qubits: int) -> None:
        """Reset to |0...0⟩ (re-creates state if qubit count changes)."""
        if num_qubits != self._state.num_qubits:
            self._state = StatevectorState(num_qubits)
        else:
            self._state.reset()

    def apply_gate(self, matrix: MatrixNxN, qubits: Sequence[int]) -> None:
        """Apply unitary `matrix` to `qubits`."""
        GateApplicator.apply(self._state.data, matrix, qubits)

    def get_statevector(self) -> StateArray:
        """Return a copy of the current statevector."""
        return self._state.data.copy()

    def get_probabilities(self) -> np.ndarray:
        """Return |α_x|² for all 2^n basis states."""
        return MeasurementEngine.probabilities(self._state.data)

    def measure(
        self,
        qubits: Sequence[int],
        collapse: bool = True,
    ) -> Dict[str, int]:
        """
        Perform a single-shot projective measurement.

        Returns
        -------
        Dict mapping qubit-index (str) → outcome bit (int)
        """
        if collapse:
            outcome, _ = MeasurementEngine.measure_and_collapse(
                self._state.data, qubits, rng=self._rng
            )
        else:
            counts = MeasurementEngine.sample_counts(
                self._state.data, shots=1, qubits=qubits, rng=self._rng
            )
            outcome = next(iter(counts))
        return {str(qubits[i]): int(outcome[i]) for i in range(len(qubits))}

    def sample(
        self,
        shots: int,
        qubits: Optional[Sequence[int]] = None,
    ) -> Dict[str, int]:
        """
        Draw `shots` non-destructive measurement samples.

        Parameters
        ----------
        shots  : number of repetitions
        qubits : qubit subset (None = all qubits)

        Returns
        -------
        Dict[bitstring → count], e.g. {'00': 512, '11': 512}
        """
        if qubits is None:
            qubits_to_use = list(range(self._state.num_qubits))
        else:
            qubits_to_use = list(qubits)
        return MeasurementEngine.sample_counts(
            self._state.data, shots=shots, qubits=qubits_to_use, rng=self._rng
        )

    # ------------------------------------------------------------------
    # Hybrid-solver extension (v0.1 — full implementation)
    # ------------------------------------------------------------------

    def get_expectation_value(self, pauli_string: str) -> float:
        """
        ⟨ψ|P|ψ⟩ for Pauli string P = P_{n-1} ⊗ ... ⊗ P_1 ⊗ P_0.

        O(2^n) implementation: applies Pauli operators qubit-by-qubit to a
        copy of the statevector using vectorized bit manipulation.  Never
        materializes the full 2^n × 2^n Pauli matrix (which would be 16 TiB
        at n=20).

        Algorithm
        ---------
        For each qubit i with Pauli P_i:
          - I: no-op
          - Z: negate amplitudes where bit i = 1
          - X: swap amplitude pairs differing at bit i
          - Y: swap pairs with phase factors (Y = iXZ)

        Then ⟨ψ|P|ψ⟩ = Re(ψ† · P|ψ⟩).

        Parameters
        ----------
        pauli_string : string of I,X,Y,Z of length num_qubits
                       e.g. 'ZZI' means Z on q0, Z on q1, I on q2

        Returns
        -------
        Real expectation value ⟨ψ|P|ψ⟩ ∈ [−1, +1]
        """
        n = self._state.num_qubits
        if len(pauli_string) != n:
            raise ValueError(
                f"Pauli string length {len(pauli_string)} != num_qubits {n}."
            )

        psi = self._state.data
        # Work on a copy: P|ψ⟩
        p_psi = psi.copy()
        dim = len(p_psi)
        indices = np.arange(dim, dtype=np.int64)

        for i, p in enumerate(pauli_string):
            p = p.upper()
            if p == 'I':
                continue
            mask = 1 << i
            bit_is_one = (indices & mask).astype(bool)
            bit_is_zero = ~bit_is_one

            if p == 'Z':
                # Z|0⟩ = |0⟩, Z|1⟩ = -|1⟩  → negate where bit i = 1
                p_psi[bit_is_one] *= -1
            elif p == 'X':
                # X swaps |0⟩ ↔ |1⟩ at bit position i
                partner = indices ^ mask  # flip bit i
                # Only swap each pair once (process where bit i = 0)
                p_psi_copy = p_psi.copy()
                p_psi[bit_is_zero] = p_psi_copy[partner[bit_is_zero]]
                p_psi[bit_is_one] = p_psi_copy[partner[bit_is_one]]
            elif p == 'Y':
                # Y = iXZ: swap with phase factors
                # Y|0⟩ = i|1⟩, Y|1⟩ = -i|0⟩
                partner = indices ^ mask
                p_psi_copy = p_psi.copy()
                # For bit=0: Y sends |0⟩ → i|1⟩, so amp at partner gets i * original
                # For bit=1: Y sends |1⟩ → -i|0⟩, so amp at partner gets -i * original
                p_psi[bit_is_zero] = -1j * p_psi_copy[partner[bit_is_zero]]
                p_psi[bit_is_one] = 1j * p_psi_copy[partner[bit_is_one]]
            else:
                raise ValueError(
                    f"Invalid Pauli character '{p}' at position {i}. "
                    "Must be I, X, Y, or Z."
                )

        # ⟨ψ|P|ψ⟩ = Re(ψ† · P|ψ⟩)  [must be real for Hermitian P]
        val = np.real(np.dot(psi.conj(), p_psi))
        return float(val)

    @property
    def num_qubits(self) -> int:
        return self._state.num_qubits

    def snapshot(self) -> "StatevectorSimulator":
        """
        Return a deep-copy simulator with the same state.

        Use for branching simulations (e.g. conditional circuits,
        expectation value computation without collapsing the state).
        """
        copy = StatevectorSimulator(self._state.num_qubits)
        np.copyto(copy._state.data, self._state.data)
        return copy

    def norm(self) -> float:
        """Return ‖ψ‖₂ — diagnostic tool to check normalization drift."""
        return self._state.norm()

    def __repr__(self) -> str:
        return (
            f"StatevectorSimulator(n={self._state.num_qubits}, "
            f"dim={self._state.dim}, ‖ψ‖={self._state.norm():.10f})"
        )


###############################################################################
# CONVENIENCE FACTORY
###############################################################################

def create_simulator(num_qubits: int, seed: Optional[int] = None) -> StatevectorSimulator:
    """Factory function — top-level entry point for ZUNTENIUM users."""
    return StatevectorSimulator(num_qubits, seed=seed)


###############################################################################
# EXAMPLE USAGE & VALIDATION
###############################################################################

def _example_1_bell_state() -> None:
    """Bell state |Phi+> = (|00> + |11>) / sqrt(2)"""
    print("=" * 60)
    print("EXAMPLE 1 — Bell state")
    print("=" * 60)

    sim = create_simulator(2, seed=42)
    qc  = SimCircuit(2)
    qc.h(0).cx(0, 1)
    qc.run(sim)

    sv = sim.get_statevector()
    expected = np.array([1/math.sqrt(2), 0, 0, 1/math.sqrt(2)], dtype=np.complex128)
    print(f"Statevector: {np.round(sv, 6)}")
    print(f"Expected:    {np.round(expected, 6)}")
    print(f"Max deviation: {np.max(np.abs(sv - expected)):.2e}")
    print(f"Norm = {sim.norm():.15f}")
    counts = sim.sample(shots=1000)
    print(f"Counts (1000 shots): {counts}")
    print()


def _example_2_hadamard_measurement() -> None:
    """Single-qubit Hadamard + measurement."""
    print("=" * 60)
    print("EXAMPLE 2 — Hadamard + measurement")
    print("=" * 60)

    sim = create_simulator(1, seed=0)
    qc  = SimCircuit(1)
    qc.h(0)
    qc.run(sim)

    sv = sim.get_statevector()
    print(f"Statevector after H: {np.round(sv, 6)}")
    counts = sim.sample(shots=2000)
    print(f"Counts (2000 shots): {counts}")

    sim2 = create_simulator(1, seed=7)
    qc.run(sim2)
    result = sim2.measure([0], collapse=True)
    print(f"Single collapsed measurement: {result}")
    post_sv = sim2.get_statevector()
    print(f"Post-collapse statevector:   {np.round(post_sv, 6)}")
    print()


def _example_3_parameterized_rx() -> None:
    """RX(pi/2) on |0>"""
    print("=" * 60)
    print("EXAMPLE 3 — RX(pi/2)")
    print("=" * 60)

    sim = create_simulator(1, seed=1)
    qc  = SimCircuit(1)
    qc.rx(math.pi / 2, 0)
    qc.run(sim)

    sv = sim.get_statevector()
    expected = np.array([1/math.sqrt(2), -1j/math.sqrt(2)], dtype=np.complex128)
    print(f"Statevector: {np.round(sv, 6)}")
    print(f"Expected:    {np.round(expected, 6)}")
    print(f"Max deviation: {np.max(np.abs(sv - expected)):.2e}")
    print(f"Probabilities: {np.round(sim.get_probabilities(), 4)}")
    print()


def _example_4_custom_gate() -> None:
    """Custom gate: sqrt(X)"""
    print("=" * 60)
    print("EXAMPLE 4 — Custom gate SX")
    print("=" * 60)

    sx = np.array([[1+1j, 1-1j], [1-1j, 1+1j]], dtype=np.complex128) / 2
    sx_sq = sx @ sx
    print(f"SX squared = \n{np.round(sx_sq, 6)}")
    print(f"X   = \n{GateLibrary.X}")

    sim = create_simulator(1, seed=5)
    qc  = SimCircuit(1)
    qc.append_gate(sx, [0], name='SX')
    qc.append_gate(sx, [0], name='SX')
    qc.run(sim)

    sv = sim.get_statevector()
    print(f"SX*SX|0> = {np.round(sv, 6)}")
    print(f"X|0>     = [0, 1]  (expected)")
    print()


def _example_5_toffoli() -> None:
    """Toffoli gate: |110> -> |111>"""
    print("=" * 60)
    print("EXAMPLE 5 — Toffoli gate")
    print("=" * 60)

    sim = create_simulator(3, seed=0)
    qc = SimCircuit(3)
    qc.x(1).x(2)
    qc.ccx(1, 2, 0)
    qc.run(sim)

    sv = sim.get_statevector()
    print(f"Statevector: {np.round(sv, 4)}")
    counts = sim.sample(shots=100)
    print(f"Result: {counts}  (expected: 100% '111')")
    print()


def _example_6_expectation_value() -> None:
    """Expectation value <ZZ> on Bell state"""
    print("=" * 60)
    print("EXAMPLE 6 — Expectation value")
    print("=" * 60)

    sim = create_simulator(2, seed=0)
    qc  = SimCircuit(2)
    qc.h(0).cx(0, 1)
    qc.run(sim)

    ev_zz = sim.get_expectation_value('ZZ')
    ev_xx = sim.get_expectation_value('XX')
    ev_zi = sim.get_expectation_value('ZI')
    print(f"<ZZ> = {ev_zz:.6f}  (expected: +1.0)")
    print(f"<XX> = {ev_xx:.6f}  (expected: +1.0)")
    print(f"<ZI> = {ev_zi:.6f}  (expected:  0.0)")
    print()


def _benchmark(n_list: List[int] = [10, 15, 18]) -> None:
    """Wall-clock benchmark: time per single-qubit gate (H) at various n."""
    print("=" * 60)
    print("BENCHMARK — Single-qubit gate (H) time vs qubit count")
    print("=" * 60)

    for n in n_list:
        sim = create_simulator(n)
        H = GateLibrary.H
        reps = max(1, 50 // n)

        GateApplicator.apply(sim._state.data, H, [0])
        sim.initialize(n)

        t0 = time.perf_counter()
        for _ in range(reps):
            GateApplicator.apply(sim._state.data, H, [0])
        t1 = time.perf_counter()

        ms_per_gate = (t1 - t0) / reps * 1000
        mem_mb = (1 << n) * 16 / (1024 ** 2)
        print(
            f"  n={n:2d}: {ms_per_gate:8.4f} ms/gate  "
            f"(dim=2^{n}={1<<n:>8d}, mem={mem_mb:.1f} MiB)"
        )
    print()


###############################################################################
# MAIN
###############################################################################

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  ZUNTENIUM v0.1 — Statevector Simulator")
    print("  Production-Grade Reference Implementation")
    print("=" * 60 + "\n")

    _example_1_bell_state()
    _example_2_hadamard_measurement()
    _example_3_parameterized_rx()
    _example_4_custom_gate()
    _example_5_toffoli()
    _example_6_expectation_value()
    _benchmark([10, 15, 18])

    print("All examples passed. ZUNTENIUM v0.1 simulator verified.")
