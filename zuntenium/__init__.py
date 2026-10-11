"""
ZUNTENIUM — AI-Native Hybrid Quantum Simulation Framework
v0.1

Public API surface. All symbols exported here are stable under SemVer.
"""
from __future__ import annotations

from zuntenium.api.quantum_api import execute, get_backend
from zuntenium.backend.provider import BackendProvider
from zuntenium.circuit.circuit import QuantumCircuit
from zuntenium.simulators import StatevectorSimulator, plot_histogram
from zuntenium.version import __version__

# Canonical alias: Circuit -> QuantumCircuit
Circuit = QuantumCircuit

from zuntenium.execution import (
    AdaptiveQuantumExecutor,
    BackendSelector,
    HybridExecutor,
    Preferences,
    QuantumExecutor,
)
from zuntenium.transpiler import HardwareAwareTranspiler
from zuntenium.backends import IBMBackend, TensorNetworkBackend
from zuntenium.ai_engine import AIOptimizer, AIDecision

__all__ = [
    "__version__",
    "QuantumCircuit",
    "Circuit",
    "execute",
    "get_backend",
    "BackendProvider",
    "StatevectorSimulator",
    "plot_histogram",
    "AdaptiveQuantumExecutor",
    "BackendSelector",
    "HybridExecutor",
    "Preferences",
    "QuantumExecutor",
    "HardwareAwareTranspiler",
    "IBMBackend",
    "TensorNetworkBackend",
    "AIOptimizer",
    "AIDecision",
]
