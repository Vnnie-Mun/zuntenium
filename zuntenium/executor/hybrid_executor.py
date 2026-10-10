"""Compatibility module re-exporting zuntenium.execution.hybrid_executor."""
from __future__ import annotations

from zuntenium.execution.hybrid_executor import *
from zuntenium.execution.hybrid_executor import (
    convert_irgates_to_circuit,
    convert_circuit_to_irgates,
    enrich_result,
    HybridExecutor,
    HybridExecutionBackend,
    IntegratedQuantumLayer,
    HybridModel,
    validate_gradient_correctness,
    run_full_pipeline_check,
    Phase2IntegrationTestSuite,
    run_integration_demo,
)
