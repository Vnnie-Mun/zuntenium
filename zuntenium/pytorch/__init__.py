"""
Zuntenium PyTorch Integration — Hybrid Quantum-Classical Neural Network Layer.

Provides:
    QuantumLayer          — nn.Module with parameter-shift autograd
    AngleEncoder          — angle-based data encoding (O(n) gates)
    AmplitudeEncoder      — amplitude-based data encoding (O(2^n) expressivity)
    CircuitBuilder        — variational ansatz construction
    StatevectorSimulatorBackend — quantum execution backend
    ParameterShiftEngine  — exact analytic gradients
    BatchedQuantumExecutor— batched circuit execution
"""
from __future__ import annotations

from zuntenium.pytorch.zuntenium_pytorch import (
    # Encoding
    AngleEncoder,
    AmplitudeEncoder,
    Encoder,
    # Circuit
    CircuitBuilder,
    # Execution
    QuantumExecutor,
    StatevectorSimulatorBackend,
    BatchedQuantumExecutor,
    # Gradient
    ParameterShiftEngine,
    # Layer
    QuantumLayer,
    # Utilities
    normalize_to_range,
    normalize_l2,
    pauli_z_expectation,
    all_pauli_z_expectations,
    # Data
    generate_synthetic_data,
    # Constants
    TORCH_AVAILABLE,
)

__all__ = [
    "AngleEncoder",
    "AmplitudeEncoder",
    "Encoder",
    "CircuitBuilder",
    "QuantumExecutor",
    "StatevectorSimulatorBackend",
    "BatchedQuantumExecutor",
    "ParameterShiftEngine",
    "QuantumLayer",
    "normalize_to_range",
    "normalize_l2",
    "pauli_z_expectation",
    "all_pauli_z_expectations",
    "generate_synthetic_data",
    "TORCH_AVAILABLE",
]
