"""
ZUNTENIUM — AI-Native Hybrid Quantum Simulation Framework
v0.1

Public API surface. All symbols exported here are stable under SemVer.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from zuntenium.ai_engine import AIDecision, AIOptimizer
    from zuntenium.backends import (
        DensityMatrixSimulator,
        GPUStatevectorSimulator,
        IBMBackend,
        StabilizerSimulator,
        TensorNetworkBackend,
    )
    from zuntenium.circuit.instruction import Instruction
    from zuntenium.execution import (
        AdaptiveQuantumExecutor,
        BackendSelector,
        HybridExecutor,
        Preferences,
        QuantumExecutor,
    )
    from zuntenium.quantum_core.executor.result import ExecutionResult, MeasurementResult
    from zuntenium.quantum_core.gates.base_gate import BaseGate
    from zuntenium.transpiler import HardwareAwareTranspiler
    from zuntenium.transpiler.coupling_map import CouplingMap
    from zuntenium.transpiler.v2.core import Gate

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
    "DensityMatrixSimulator": ("zuntenium.backends", "DensityMatrixSimulator"),
    "StabilizerSimulator": ("zuntenium.backends", "StabilizerSimulator"),
    "GPUStatevectorSimulator": ("zuntenium.backends", "GPUStatevectorSimulator"),
    "AIOptimizer": ("zuntenium.ai_engine", "AIOptimizer"),
    "AIDecision": ("zuntenium.ai_engine", "AIDecision"),
    "AdaptiveQuantumExecutor": ("zuntenium.execution", "AdaptiveQuantumExecutor"),
    "BackendSelector": ("zuntenium.execution", "BackendSelector"),
    "HybridExecutor": ("zuntenium.execution", "HybridExecutor"),
    "Preferences": ("zuntenium.execution", "Preferences"),
    "QuantumExecutor": ("zuntenium.execution", "QuantumExecutor"),
    "Gate": ("zuntenium.transpiler.v2.core", "Gate"),
    "Instruction": ("zuntenium.circuit.instruction", "Instruction"),
    "BaseGate": ("zuntenium.quantum_core.gates.base_gate", "BaseGate"),
    "CouplingMap": ("zuntenium.transpiler.coupling_map", "CouplingMap"),
    "ExecutionResult": ("zuntenium.quantum_core.executor.result", "ExecutionResult"),
    "MeasurementResult": ("zuntenium.quantum_core.executor.result", "MeasurementResult"),
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
    "DensityMatrixSimulator",
    "StabilizerSimulator",
    "GPUStatevectorSimulator",
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
    "Gate",
    "Instruction",
    "BaseGate",
    "CouplingMap",
    "ExecutionResult",
    "MeasurementResult",
]
