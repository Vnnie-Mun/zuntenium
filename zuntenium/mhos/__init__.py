# zuntenium/mhos/__init__.py
"""
Zuntenium Multi-Hardware Orchestration System (MHOS) — Phase 5
==============================================================

Universal quantum hardware orchestration layer. Vendor-agnostic routing,
intelligent device selection, failover, parallel execution, and cost control
across IBM Quantum, AWS Braket, IonQ, Rigetti, and future providers.

Canonical execution flow:
  Cloud API
    → JobManager
    → AIOptimizer.select()          [existing — Phase 3]
    → HardwareSelector.score()      [MHOS — ranks all available devices]
    → HardwareOrchestrator.run()    [MHOS — submits, polls, retries]
       ├─ IBMConnector              [MHOS — IBM Quantum Runtime V2]
       ├─ BraketConnector           [MHOS — AWS Braket]
       ├─ IonQConnector             [MHOS — IonQ Cloud]
       ├─ RigettiConnector          [MHOS — Rigetti QCS]
       └─ SimulatorFallback         [existing MPS/SV backends]
    → ErrorMitigationEngine         [existing — Phase 3]
    → Result Storage                [existing — Phase 4]

Design principles:
  - New provider = one new Connector class (no other changes)
  - All provider logic isolated in its connector
  - Scoring is pure function: configurable weights, no hidden state
  - Full async: all I/O non-blocking
  - Zero vendor lock-in: swappable at any layer
"""

from .device_profile    import DeviceProfile, DeviceStatus, GateErrorMap
from .hardware_registry import HardwareRegistry
from .hardware_selector import HardwareSelector, SelectionResult, ScoringWeights
from .orchestrator      import HardwareOrchestrator, ExecutionStrategy
from .connectors.base   import BaseConnector, ConnectorError
from .connectors.ibm    import IBMConnector
from .connectors.braket import BraketConnector
from .connectors.ionq   import IonQConnector
from .connectors.rigetti import RigettiConnector
from .integration       import MHOSIntegration, patch_cloud_worker

__all__ = [
    "DeviceProfile", "DeviceStatus", "GateErrorMap",
    "HardwareRegistry",
    "HardwareSelector", "SelectionResult", "ScoringWeights",
    "HardwareOrchestrator", "ExecutionStrategy",
    "BaseConnector", "ConnectorError",
    "IBMConnector", "BraketConnector", "IonQConnector", "RigettiConnector",
    "MHOSIntegration", "patch_cloud_worker",
]
