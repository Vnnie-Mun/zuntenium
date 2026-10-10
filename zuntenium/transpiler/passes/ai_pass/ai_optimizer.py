"""
AIOptimizerPass — ML model plug-in for the transpiler pipeline.

v0.1: No-op data collection mode (logs IR for training).
v0.4: When a model checkpoint is available, invokes the RL agent
      to select optimization actions. Falls back to rule-based
      passes if confidence is below threshold.

This pass bridges between the QuantumIR transpiler pipeline and the
AI transpiler engine (zuntenium.transpiler.ai).
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from zuntenium.transpiler.passes.ai_pass.training_data import TrainingDataCollector
from zuntenium.transpiler.passes.base_pass import TranspilerPass

if TYPE_CHECKING:
    from zuntenium.circuit.ir.quantum_ir import QuantumIR

logger = logging.getLogger(__name__)


class AIOptimizerPass(TranspilerPass):
    """
    AI-based transpiler optimization pass.

    v0.1 mode (model_path=None):
        Logs IR before/after optimization for training data collection.
        Acts as a no-op pass (returns IR unchanged).

    v0.4 mode (model_path provided):
        Loads a trained RL agent checkpoint and uses it to select
        optimization actions (gate cancellation, commutation, fusion).
        Falls back to pass-through if confidence is below threshold.

    Args:
        collect_data: If True (default), log training data to the collector.
        model_path: Path to a trained model checkpoint directory.
        confidence_threshold: Minimum RL agent confidence to apply
                              optimizations (default 0.6).
        max_optimization_rounds: Maximum RL optimization steps per
                                 invocation (default 50).
    """

    name = "AIOptimizer"
    description = "AI-based transpiler pass. Data collection + RL optimization."

    def __init__(
        self,
        collect_data: bool = True,
        model_path: str | None = None,
        confidence_threshold: float = 0.6,
        max_optimization_rounds: int = 50,
    ) -> None:
        self._collect_data = collect_data
        self._model_path = model_path
        self._confidence_threshold = confidence_threshold
        self._max_rounds = max_optimization_rounds
        self._collector = TrainingDataCollector() if collect_data else None
        self._transpiler: Any = None  # Lazy-loaded AI transpiler

    def _ensure_transpiler(self) -> None:
        """Lazy-load the AI transpiler when a model path is provided."""
        if self._transpiler is not None:
            return
        if self._model_path is None:
            return

        try:
            from zuntenium.transpiler.ai.transpiler import AITranspiler
            self._transpiler = AITranspiler(use_ai=True)
            logger.info(
                "AI transpiler loaded from '%s'.", self._model_path
            )
        except ImportError:
            logger.warning(
                "AI transpiler module not available. "
                "Falling back to data-collection-only mode."
            )
            self._transpiler = None
        except Exception as exc:
            logger.warning(
                "Failed to load AI transpiler: %s. "
                "Falling back to data-collection-only mode.",
                exc,
            )
            self._transpiler = None

    def _ir_to_gate_list(self, ir: QuantumIR) -> list[Any]:
        """Convert QuantumIR to AI transpiler gate list format."""
        try:
            from zuntenium.transpiler.ai.ir import Gate
        except ImportError:
            return []

        gates = []
        for node in ir.instructions:
            if not node.is_gate() or node.gate is None:
                continue
            gates.append(Gate(
                name=node.gate.name,
                qubits=tuple(node.qubits),
                params=tuple(float(p) for p in node.params),
            ))
        return gates

    def _gate_list_to_ir(
        self,
        gates: list[Any],
        num_qubits: int,
        num_clbits: int,
        metadata: dict[str, Any] | None = None,
    ) -> QuantumIR:
        """Convert AI transpiler gate list back to QuantumIR."""
        from zuntenium.circuit.ir.ir_node import IRNode
        from zuntenium.circuit.ir.quantum_ir import QuantumIR as QIR
        from zuntenium.quantum_core.gates import (
            CXGate, CZGate, HGate, RxGate, RyGate, RzGate,
            SGate, SWAPGate, TGate, U3Gate,
            XGate, YGate, ZGate,
        )

        _GATE_MAP: dict[str, type] = {
            "H": HGate, "X": XGate, "Y": YGate, "Z": ZGate,
            "S": SGate, "T": TGate,
            "CX": CXGate, "CNOT": CXGate, "CZ": CZGate, "SWAP": SWAPGate,
        }
        _PARAM_MAP: dict[str, type] = {
            "RX": RxGate, "RY": RyGate, "RZ": RzGate, "U3": U3Gate,
        }

        nodes: list[IRNode] = []
        for g in gates:
            nm = g.name.upper()
            gate_cls = _GATE_MAP.get(nm)
            if gate_cls is not None:
                gate_obj = gate_cls()
            elif nm in _PARAM_MAP:
                gate_obj = _PARAM_MAP[nm](*g.params)
            else:
                gate_obj = None

            if gate_obj is not None:
                nodes.append(IRNode(
                    operation="gate",
                    gate=gate_obj,
                    qubits=list(g.qubits),
                    params=list(g.params),
                ))

        return QIR(
            num_qubits=num_qubits,
            num_clbits=num_clbits,
            instructions=nodes,
            metadata=dict(metadata or {}),
        )

    def run(self, ir: QuantumIR, **kwargs: object) -> QuantumIR:
        """
        Apply AI optimization to the IR.

        In data-collection mode: records IR snapshot and passes through.
        In model mode: runs the RL agent and applies optimizations
        if confidence exceeds the threshold.
        """
        # Record training data (before optimization)
        if self._collect_data and self._collector is not None:
            self._collector.record(ir)

        # If no model path, pass through (data collection only)
        if self._model_path is None:
            return ir

        # Attempt AI optimization
        self._ensure_transpiler()
        if self._transpiler is None:
            return ir

        try:
            gates = self._ir_to_gate_list(ir)
            if not gates:
                return ir

            result = self._transpiler.optimize(gates)

            # Check if the optimization is confident enough
            confidence = getattr(result, "estimated_fidelity", 1.0)
            if confidence < self._confidence_threshold:
                logger.info(
                    "AI optimizer confidence (%.3f) below threshold (%.3f) "
                    "— returning original IR.",
                    confidence,
                    self._confidence_threshold,
                )
                return ir

            # Convert optimized gates back to IR
            optimized_ir = self._gate_list_to_ir(
                result.circuit,
                num_qubits=ir.num_qubits,
                num_clbits=ir.num_clbits,
                metadata={
                    **ir.metadata,
                    "ai_transpiler": True,
                    "ai_confidence": confidence,
                    "gate_reduction_pct": getattr(
                        result, "gate_reduction_pct", 0.0
                    ),
                    "depth_reduction_pct": getattr(
                        result, "depth_reduction_pct", 0.0
                    ),
                },
            )

            # Record after-optimization data
            if self._collect_data and self._collector is not None:
                depth_before = ir.depth()
                depth_after = optimized_ir.depth()
                reward = (depth_before - depth_after) / max(depth_before, 1)
                self._collector.record(ir, optimized_ir, reward=reward)

            logger.info(
                "AI optimizer reduced gates by %.1f%%, depth by %.1f%%.",
                getattr(result, "gate_reduction_pct", 0.0),
                getattr(result, "depth_reduction_pct", 0.0),
            )

            return optimized_ir

        except Exception as exc:
            logger.warning(
                "AI optimization failed: %s — returning original IR.", exc
            )
            return ir

    def is_idempotent(self) -> bool:
        return self._model_path is None  # Data collection is idempotent

    def __repr__(self) -> str:
        mode = "model" if self._model_path else "data_collection"
        return f"AIOptimizerPass(mode='{mode}', model_path={self._model_path})"
