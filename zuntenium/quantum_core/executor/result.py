"""
ExecutionResult and MeasurementResult data structures.

These are the output types of all backend.run() calls.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class MeasurementResult:
    """
    Result of measuring a subset of qubits over N shots.

    Attributes:
        counts: Dict mapping bitstring → count. E.g. {'00': 512, '11': 512}.
        qubits: Which qubits were measured.
        shots: Total number of shots.
        probabilities: Theoretical probability for each basis state.
    """
    counts: dict[str, int]
    qubits: list[int]
    shots: int
    probabilities: np.ndarray | None = None

    @property
    def most_probable(self) -> str:
        """Return the most frequent measurement outcome."""
        return max(self.counts, key=lambda k: self.counts[k])

    def marginal(self, keep: list[int]) -> MeasurementResult:
        """
        Marginalize counts over a subset of the measured qubits.

        Args:
            keep: Indices (into self.qubits) to retain.

        Returns:
            New MeasurementResult with reduced counts.

        Example::

            # If we measured [0, 1, 2] and want qubit 0 only:
            result.marginal([0])
        """
        marginal_counts: dict[str, int] = {}
        for bitstring, count in self.counts.items():
            # Extract only the bits at positions in `keep`
            bits = [bitstring[i] for i in keep]
            key = "".join(bits)
            marginal_counts[key] = marginal_counts.get(key, 0) + count
        kept_qubits = [self.qubits[i] for i in keep]
        return MeasurementResult(
            counts=marginal_counts,
            qubits=kept_qubits,
            shots=self.shots,
        )


@dataclass
class ExecutionResult:
    """
    Full result of running a QuantumCircuit on a backend.

    Attributes:
        counts: Measurement outcome counts. {'00': 512, '11': 512}.
        quasi_dists: Quasi-probability distribution (mitigated, v0.2).
        statevector: Final statevector amplitudes (simulator only).
        metadata: Execution metadata dict (backend, shots, time, etc.).
        num_qubits: Total number of qubits in the circuit.
        shots: Number of measurement shots.
    """
    counts: dict[str, int]
    num_qubits: int
    shots: int
    statevector: np.ndarray | None = None
    quasi_dists: dict[str, float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # ── Convenience accessors ──────────────────────────────────────────────

    @property
    def probabilities(self) -> dict[str, float]:
        """Normalize counts to probabilities."""
        total = sum(self.counts.values()) or 1
        return {k: v / total for k, v in self.counts.items()}

    @property
    def most_probable(self) -> str:
        """Return the most frequent measurement outcome."""
        return max(self.counts, key=lambda k: self.counts[k])

    @property
    def endianness(self) -> str:
        """Bitstring ordering convention (little-endian: q0 is rightmost bit)."""
        return "little-endian (q0 is rightmost bit)"

    @property
    def standard_errors(self) -> dict[str, float]:
        """
        Statistical standard error bars: σ_k = sqrt(p_k * (1 - p_k) / N).
        """
        total = self.shots if self.shots > 0 else (sum(self.counts.values()) or 1)
        probs = self.probabilities
        return {
            k: float(np.sqrt(max(0.0, p * (1.0 - p)) / total))
            for k, p in probs.items()
        }

    @property
    def error_bars(self) -> dict[str, float]:
        """Statistical error bars for measurement outcomes."""
        return self.standard_errors

    @property
    def norm_drift(self) -> float | None:
        """Norm drift |‖ψ‖ - 1| for statevector simulations."""
        if self.statevector is None:
            return None
        return float(abs(np.linalg.norm(self.statevector) - 1.0))

    def expectation_value(self, observable: np.ndarray) -> float:
        """
        Compute ⟨ψ|O|ψ⟩ for a Hermitian observable matrix.

        Requires statevector to be present (simulator only).
        """
        if self.statevector is None:
            raise ValueError(
                "statevector is not available. "
                "Use a statevector simulator backend to access it."
            )
        psi = self.statevector
        return float(np.real(psi.conj() @ observable @ psi))

    def expectation_value_with_error(
        self, observable: np.ndarray
    ) -> tuple[float, float]:
        """
        Compute expectation value ⟨ψ|O|ψ⟩ and statistical standard error bar.
        Returns:
            (expectation_value, standard_error)
        """
        mean = self.expectation_value(observable)
        std_err = float(1.0 / np.sqrt(max(1, self.shots)))
        return mean, std_err

    def plot_histogram(self, **kwargs: Any) -> None:
        """Plot a bar chart of measurement counts (requires matplotlib)."""
        try:
            import matplotlib.pyplot as plt
            labels = sorted(self.counts.keys())
            values = [self.counts[k] for k in labels]
            fig, ax = plt.subplots(figsize=(max(6, len(labels)), 4))
            ax.bar(labels, values, color="#6366f1", edgecolor="white", linewidth=0.5)
            ax.set_xlabel("Measurement Outcome", fontsize=12)
            ax.set_ylabel("Counts", fontsize=12)
            ax.set_title(f"Measurement Results ({self.shots} shots)", fontsize=14)
            plt.tight_layout()
            plt.show()
        except ImportError:
            print("Install matplotlib to use plot_histogram(): pip install matplotlib")

    def __repr__(self) -> str:
        top = list(self.counts.items())[:3]
        return (
            f"ExecutionResult(shots={self.shots}, "
            f"num_qubits={self.num_qubits}, "
            f"top_counts={dict(top)})"
        )
