"""Quantum execution integration layer."""

from zuntenium.execution.quantum_executor import (
    CompatibilityError,
    ExecutionResult,
    IntegrationTestSuite,
    QuantumExecutor,
    TranspilerAwareQuantumLayer,
    TranspilerAwareVQE,
    validate_circuit_compatibility,
)
from zuntenium.execution.adaptive_engine import (
    AdaptiveEngineTestSuite,
    AdaptiveQuantumExecutor,
    BackendCapability,
    BackendSelector,
    CircuitAnalyzer,
    CircuitFeatures,
    RuleEngine,
    ScoringEngine,
    Preferences,
    SelectionDecision,
)
from zuntenium.execution.hybrid_executor import (
    HybridExecutor, HybridExecutionBackend, IntegratedQuantumLayer, HybridModel,
)

__all__ = [
    "CompatibilityError",
    "ExecutionResult",
    "IntegrationTestSuite",
    "QuantumExecutor",
    "TranspilerAwareQuantumLayer",
    "TranspilerAwareVQE",
    "validate_circuit_compatibility",
    "AdaptiveEngineTestSuite",
    "AdaptiveQuantumExecutor",
    "BackendSelector",
    "BackendCapability",
    "CircuitAnalyzer",
    "CircuitFeatures",
    "Preferences",
    "RuleEngine",
    "ScoringEngine",
    "SelectionDecision",
    "HybridExecutor",
    "HybridExecutionBackend",
    "IntegratedQuantumLayer",
    "HybridModel",
]
