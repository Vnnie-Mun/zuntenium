"""
ZUNTENIUM — AI-Native Hybrid Quantum Simulation Framework
v0.1

Public API surface. All symbols exported here are stable under SemVer.
"""
from __future__ import annotations

from typing import Any

from zuntenium.api.quantum_api import execute, get_backend
from zuntenium.backend.provider import BackendProvider
from zuntenium.circuit.circuit import QuantumCircuit
from zuntenium.simulators import StatevectorSimulator, plot_histogram
from zuntenium.version import __version__

# Canonical alias: Circuit -> QuantumCircuit
Circuit = QuantumCircuit

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "HardwareAwareTranspiler": ("zuntenium.transpiler", "HardwareAwareTranspiler"),
    "IBMBackend": ("zuntenium.backends", "IBMBackend"),
    "TensorNetworkBackend": ("zuntenium.backends", "TensorNetworkBackend"),
    "AIOptimizer": ("zuntenium.ai_engine", "AIOptimizer"),
    "AIDecision": ("zuntenium.ai_engine", "AIDecision"),
    "AdaptiveQuantumExecutor": ("zuntenium.execution", "AdaptiveQuantumExecutor"),
    "BackendSelector": ("zuntenium.execution", "BackendSelector"),
    "HybridExecutor": ("zuntenium.execution", "HybridExecutor"),
    "Preferences": ("zuntenium.execution", "Preferences"),
    "QuantumExecutor": ("zuntenium.execution", "QuantumExecutor"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_EXPORTS:
        module_name, attr_name = _LAZY_EXPORTS[name]
        import importlib
        mod = importlib.import_module(module_name)
        val = getattr(mod, attr_name)
        globals()[name] = val
        return val
    raise AttributeError(f"module 'zuntenium' has no attribute '{name}'")


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
