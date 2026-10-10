# zuntenium/error_mitigation/__init__.py
"""
Zuntenium Error Mitigation Engine (EME)
========================================
Production-grade quantum noise modeling, error detection,
and result mitigation for the Zuntenium Quantum OS.

Architecture position:
  Backend.execute() → [EME] → corrected result → HybridSolver / QuantumLayer

Activation:
  HybridExecutor(mitigation_enabled=True)

Zero-overhead when disabled (default False).
"""

from .noise_model  import (
    NoiseModel, DepolarizingNoise, AmplitudeDampingNoise,
    PhaseDampingNoise, ThermalRelaxationNoise, ReadoutNoise,
    GateNoise, NoiseModelBuilder,
)
from .noise_injector   import NoiseInjector
from .error_detector   import ErrorDetector, DetectionReport
from .mitigation       import (
    ErrorMitigationEngine,
    ZNEMitigator, PECMitigator, ReadoutMitigator,
    MitigatedResult,
)
from .fidelity         import FidelityEstimator, FidelityReport
from .integration      import MitigationPipeline

__all__ = [
    # Noise modeling
    "NoiseModel", "DepolarizingNoise", "AmplitudeDampingNoise",
    "PhaseDampingNoise", "ThermalRelaxationNoise", "ReadoutNoise",
    "GateNoise", "NoiseModelBuilder",
    # Injection
    "NoiseInjector",
    # Detection
    "ErrorDetector", "DetectionReport",
    # Mitigation
    "ErrorMitigationEngine", "ZNEMitigator", "PECMitigator",
    "ReadoutMitigator", "MitigatedResult",
    # Fidelity
    "FidelityEstimator", "FidelityReport",
    # Pipeline
    "MitigationPipeline",
]
