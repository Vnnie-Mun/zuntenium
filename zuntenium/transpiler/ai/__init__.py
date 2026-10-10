"""
Zuntenium AI Transpiler — Phase 1 (Rule-based) + Phase 2 (RL/PPO).

Sub-package providing intelligent quantum circuit optimization.
"""
from __future__ import annotations

from zuntenium.transpiler.ai.ir import (
    CircuitDAG,
    Gate,
    HardwareConfig,
    TranspilerResult,
    circuit_to_dag,
    circuit_to_zuntenium_format,
    dag_to_circuit,
)
from zuntenium.transpiler.ai.passes import (
    CommutativeReorderingPass,
    PauliCancellationPass,
    RotationFusionPass,
    RuleEngine,
    SingleQubitUnifyPass,
    SWAPRoutingPass,
)
from zuntenium.transpiler.ai.rl import (
    CircuitOptimizationEnv,
    CircuitStateEncoder,
    ReplayBuffer,
    RewardFunction,
    RLAgent,
    TransformAction,
)
from zuntenium.transpiler.ai.transpiler import (
    AITranspiler,
    DatasetGenerator,
    TranspilerAwareHybridSolver,
)

__all__ = [
    "Gate", "HardwareConfig", "CircuitDAG", "TranspilerResult",
    "circuit_to_dag", "dag_to_circuit", "circuit_to_zuntenium_format",
    "PauliCancellationPass", "RotationFusionPass", "SingleQubitUnifyPass",
    "CommutativeReorderingPass", "SWAPRoutingPass", "RuleEngine",
    "CircuitStateEncoder", "RewardFunction", "TransformAction",
    "CircuitOptimizationEnv", "ReplayBuffer", "RLAgent",
    "AITranspiler", "DatasetGenerator", "TranspilerAwareHybridSolver",
]
