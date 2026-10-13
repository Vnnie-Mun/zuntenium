"""
zuntenium/quantum_core/statevector/measurement_engine.py
========================================================
Computational-basis measurement for statevectors.

Extracted from statevector_simulator.py as part of O6 God-class decomposition.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
StateArray = np.ndarray   # shape (2**n,), dtype complex128


class MeasurementEngine:
    """
    Implements computational-basis measurement for statevectors.

    Mathematical background
    -----------------------
    For state |ψ⟩ = Σ α_x |x⟩, the probability of measuring qubit k = 0 is:

        P(qubit k = 0) = Σ_{x: bit_k(x)=0} |α_x|²

    Using NumPy advanced indexing, this is computed in O(2^n) without a loop.

    Post-measurement collapse (Born rule projection)
    ------------------------------------------------
    If outcome m ∈ {0,1} is observed on qubit k, the post-measurement state is:

        |ψ'⟩ = Π_k(m) |ψ⟩ / ‖Π_k(m) |ψ⟩‖

    where Π_k(m) = projector onto {x : bit_k(x) = m}.

    Implementation: zero out all amplitudes where bit k ≠ m, then renormalize.
    This is O(2^n) and uses in-place masking (no extra allocation for large n).

    Sampling (without collapse)
    ---------------------------
    np.random.choice(2^n, size=shots, p=probs) draws shots outcomes
    from the probability distribution in one vectorized call.
    """

    @staticmethod
    def probabilities(state: StateArray) -> np.ndarray:
        """
        Return the probability vector P = |α_x|² for all 2^n outcomes.

        Time:  O(2^n)
        Space: O(2^n) — new array
        """
        return np.abs(state) ** 2

    @staticmethod
    def marginal_probabilities(
        state: StateArray,
        qubits: Sequence[int],
    ) -> Dict[str, float]:
        """
        Compute marginal probabilities for a subset of qubits.

        Sums over all other (traced-out) qubits.
        Vectorized implementation using NumPy advanced indexing.

        Parameters
        ----------
        state  : full statevector (2^n,)
        qubits : qubit indices to keep

        Returns
        -------
        Dict mapping bitstring (length = len(qubits)) → probability
        """
        probs = MeasurementEngine.probabilities(state)
        dim = len(probs)
        indices = np.arange(dim, dtype=np.int64)
        n_meas = len(qubits)
        n_outcomes = 1 << n_meas

        marginals: Dict[str, float] = {}
        for outcome_idx in range(n_outcomes):
            # Build boolean mask for all basis states matching this outcome
            mask = np.ones(dim, dtype=bool)
            bits = []
            for bit_pos, q in enumerate(qubits):
                target_bit = (outcome_idx >> bit_pos) & 1
                mask &= ((indices >> q) & 1) == target_bit
                bits.append(str(target_bit))
            key = ''.join(bits)
            marginals[key] = float(np.sum(probs[mask]))

        return marginals

    @staticmethod
    def measure_and_collapse(
        state: StateArray,
        qubits: Sequence[int],
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[str, StateArray]:
        """
        Perform a projective measurement on `qubits`, collapsing the state.

        Parameters
        ----------
        state  : statevector (modified in-place)
        qubits : qubits to measure
        rng    : numpy random Generator (for reproducibility)

        Returns
        -------
        (outcome_bitstring, state)  — state is collapsed in-place
        """
        if rng is None:
            rng = np.random.default_rng()

        marginals = MeasurementEngine.marginal_probabilities(state, qubits)
        outcomes = list(marginals.keys())
        probs = np.array([marginals[o] for o in outcomes], dtype=np.float64)
        probs /= probs.sum()

        idx = rng.choice(len(outcomes), p=probs)
        outcome = outcomes[idx]

        # Vectorized collapse: build boolean mask of consistent amplitudes
        dim = len(state)
        indices = np.arange(dim, dtype=np.int64)
        mask = np.ones(dim, dtype=bool)
        for i, q in enumerate(qubits):
            target_bit = int(outcome[i])
            mask &= ((indices >> q) & 1) == target_bit
        # Zero all inconsistent amplitudes in-place
        state[~mask] = 0.0 + 0.0j

        norm_sq = float(np.real(np.dot(state.conj(), state)))
        if norm_sq < 1e-12:
            raise RuntimeError(
                f"Post-measurement norm² = {norm_sq:.2e}: state collapsed to zero."
            )
        state /= math.sqrt(norm_sq)

        return outcome, state

    @staticmethod
    def sample_counts(
        state: StateArray,
        shots: int,
        qubits: Optional[Sequence[int]] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> Dict[str, int]:
        """
        Draw `shots` measurement samples from the statevector distribution
        WITHOUT collapsing the state.

        Parameters
        ----------
        state  : statevector (not modified)
        shots  : number of samples
        qubits : if given, return marginal bitstrings; else full bitstrings
        rng    : numpy random Generator

        Returns
        -------
        Dict[bitstring → count]

        Performance
        -----------
        np.random.Generator.choice is O(2^n + shots) — the dominant cost is
        building the probability vector, not the sampling itself.
        At n=20, shots=10_000 takes ~5 ms on a modern CPU.
        """
        if shots <= 0:
            raise ValueError(f"shots must be positive; got {shots}")
        if rng is None:
            rng = np.random.default_rng()

        n = int(math.log2(len(state)))
        probs = MeasurementEngine.probabilities(state)
        probs = probs / probs.sum()

        indices = rng.choice(len(probs), size=shots, p=probs)

        if qubits is None:
            fmt = f'{{:0{n}b}}'
            result: Dict[str, int] = {}
            for idx in indices:
                key = fmt.format(idx)
                result[key] = result.get(key, 0) + 1
        else:
            result = {}
            for idx in indices:
                key = ''.join(str((idx >> q) & 1) for q in qubits)
                result[key] = result.get(key, 0) + 1

        return result
