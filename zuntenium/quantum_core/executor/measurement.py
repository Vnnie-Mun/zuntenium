"""
Measurement and sampling logic.

All measurement in ZUNTENIUM v0.1 is post-hoc: we run the circuit once to
obtain the final statevector, then sample from its probability distribution.
This gives statistically equivalent results to physical measurement for
ideal (noiseless) circuits.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from zuntenium.quantum_core.executor.result import MeasurementResult


def sample_statevector(
    amplitudes: np.ndarray,
    qubits: list[int],
    num_qubits: int,
    shots: int,
    seed: int | None = None,
) -> MeasurementResult:
    """
    Sample measurement outcomes from a statevector amplitude distribution.

    Args:
        amplitudes: Complex128 array of shape (2^N,).
        qubits: Qubit indices to measure (ordered LSB to MSB in output string).
        num_qubits: Total number of qubits.
        shots: Number of samples to draw.
        seed: Optional RNG seed for reproducibility.

    Returns:
        MeasurementResult with counts keyed by bitstring.

    Notes:
        Bitstring ordering: qubit 0 is the *rightmost* character.
        E.g., qubit 0 = |1⟩, qubit 1 = |0⟩ → "01".
    """
    rng = np.random.default_rng(seed)
    probs = np.abs(amplitudes) ** 2
    # Normalize (guard against numerical drift)
    probs /= probs.sum()

    # Sample full-state indices
    dim = len(amplitudes)
    indices = rng.choice(dim, size=shots, p=probs)

    # Marginalise to measured qubits
    qubit_set = sorted(qubits)
    bitstrings = []
    for idx in indices:
        bits = _extract_bits(idx, qubit_set, num_qubits)
        bitstrings.append(bits)

    counts = dict(Counter(bitstrings))

    # Build marginal probability array for the measured qubits
    marg_dim = 2 ** len(qubit_set)
    marg_probs = np.zeros(marg_dim)
    for full_idx, p in enumerate(probs):
        marg_idx = _marginal_index(full_idx, qubit_set, num_qubits)
        marg_probs[marg_idx] += p

    return MeasurementResult(
        counts=counts,
        qubits=qubits,
        shots=shots,
        probabilities=marg_probs,
    )


def _extract_bits(state_index: int, qubits: list[int], num_qubits: int) -> str:
    """
    Extract bits at specified qubit positions from a state index.

    Returns a bitstring where position 0 (rightmost) corresponds to qubits[0].
    """
    chars = []
    for q in reversed(qubits):
        bit = (state_index >> (num_qubits - 1 - q)) & 1
        chars.append(str(bit))
    return "".join(chars)


def _marginal_index(state_index: int, qubits: list[int], num_qubits: int) -> int:
    """Convert a full-state index to a marginal index over the given qubits."""
    result = 0
    for k, q in enumerate(reversed(qubits)):
        bit = (state_index >> (num_qubits - 1 - q)) & 1
        result |= bit << k
    return result


def measure_all(
    amplitudes: np.ndarray,
    num_qubits: int,
    shots: int,
    seed: int | None = None,
) -> MeasurementResult:
    """Convenience: measure all qubits."""
    return sample_statevector(
        amplitudes, list(range(num_qubits)), num_qubits, shots, seed
    )
