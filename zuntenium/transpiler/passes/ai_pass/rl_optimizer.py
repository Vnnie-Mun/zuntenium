"""
RLOptimizerPass — RL-based TranspilerPass for the main transpiler pipeline.

Replaces the stub ``AIOptimizerPass`` with a real implementation that
wraps the ``AITranspiler`` Phase 2 RL agent, operating on the existing
``QuantumIR`` / ``IRNode`` contract.

Author:  ZUNTENIUM Core Team
License: Apache 2.0
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from zuntenium.transpiler.passes.base_pass import TranspilerPass

if TYPE_CHECKING:
    from zuntenium.circuit.ir.quantum_ir import QuantumIR


# ── Bridge: QuantumIR ↔ Gate list ────────────────────────────────────────

def _ir_to_gate_list(ir: "QuantumIR") -> list:
    """Convert a ``QuantumIR`` to a list of AI transpiler ``Gate`` objects."""
    from zuntenium.transpiler.ai.ir import Gate

    gates = []
    for node in ir.instructions:
        if not node.is_gate() or node.gate is None:
            continue
        name = node.gate.name
        qubits = tuple(node.qubits)
        params = tuple(float(p) for p in node.params)
        gates.append(Gate(name=name, qubits=qubits, params=params))
    return gates


def _gate_list_to_ir(
    gates: list,
    num_qubits: int,
    num_clbits: int,
    metadata: dict | None = None,
) -> "QuantumIR":
    """Convert an AI transpiler gate list back to ``QuantumIR``."""
    from zuntenium.circuit.ir.ir_node import IRNode
    from zuntenium.circuit.ir.quantum_ir import QuantumIR
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

    return QuantumIR(
        num_qubits=num_qubits,
        num_clbits=num_clbits,
        instructions=nodes,
        metadata=dict(metadata or {}),
    )


class RLOptimizerPass(TranspilerPass):
    """RL-based transpiler pass using the AI Transpiler's Phase 2 agent.

    Converts the ``QuantumIR`` to the AI transpiler's gate-list format,
    runs the full AI optimiser, and converts back.

    Args:
        use_ai:       Enable Phase 2 RL (default ``True``).
        pretrain:     Pre-train the agent on construction (default ``False``).
        num_episodes: Number of pre-training episodes.
    """

    name = "RLOptimizer"
    description = (
        "AI-based circuit optimiser using Phase 1 rule engine + "
        "Phase 2 PPO RL agent."
    )

    def __init__(
        self,
        use_ai: bool = True,
        pretrain: bool = False,
        num_episodes: int = 100,
    ) -> None:
        self._use_ai = use_ai
        self._transpiler = None  # lazy init
        self._pretrain = pretrain
        self._num_episodes = num_episodes

    def _ensure_transpiler(self) -> None:
        if self._transpiler is not None:
            return
        from zuntenium.transpiler.ai.transpiler import AITranspiler
        self._transpiler = AITranspiler(use_ai=self._use_ai)
        if self._pretrain and self._use_ai:
            self._transpiler.pretrain(num_episodes=self._num_episodes, verbose=False)

    def run(self, ir: "QuantumIR", **kwargs: object) -> "QuantumIR":
        """Apply AI transpiler optimisation to *ir*."""
        self._ensure_transpiler()
        gates = _ir_to_gate_list(ir)
        if not gates:
            return ir

        result = self._transpiler.optimize(gates)  # type: ignore[union-attr]
        optimised_ir = _gate_list_to_ir(
            result.circuit,
            num_qubits=ir.num_qubits,
            num_clbits=ir.num_clbits,
            metadata={
                **ir.metadata,
                "ai_transpiler": True,
                "gate_reduction_pct": result.gate_reduction_pct,
                "depth_reduction_pct": result.depth_reduction_pct,
                "estimated_fidelity": result.estimated_fidelity,
            },
        )
        return optimised_ir
